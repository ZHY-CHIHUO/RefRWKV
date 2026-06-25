"""sd2_control_ldm.py — 精简版：纯模型类，所有参数通过 __init__ 直接传入"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, List

from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from RefDiffRWKV import RefDiffRWKV
from sd2_ref_adapter import RWKV_Ref_Adapter
from GlobalSemanticModule import GlobalSemanticModule


class SD2ControlLDM(pl.LightningModule):
    """
    融合 RefDiffRWKV + SD UNet 的扩散超分模型。
    所有超参数通过 __init__ 直接传入，无需配置文件。
    验证时自动计算 IQA 指标（PSNR/SSIM/LPIPS/DISTS）。
    """

    def __init__(
        self,
        # ── Data keys ──
        lr_key:  str = "lr",
        ref_key: str = "ref",
        hr_key:  str = "hr",
        # ── SD ──
        sd_model_path: str = "sd2-community/stable-diffusion-2-1-base",
        use_lora:  bool = True,
        lora_rank: int = 4,
        sd_locked: bool = True,
        # ── RefDiffRWKV ──
        patch_size:    int = 4,
        embed_dim:     int = 384,
        upsample_mode: str = "bilinear",
        # ── GlobalSemantic ──
        use_semantic: bool = True,
        dinov2_model: str = "facebook/dinov2-base",
        # ── Training ──
        learning_rate:       float = 1e-4,
        num_train_timesteps:  int = 1000,
        beta_start:           float = 0.00085,
        beta_end:             float = 0.012,
        beta_schedule:        str = "scaled_linear",
        prediction_type:      str = "epsilon",
        l_simple_weight:      float = 1.0,
        # ── Validation ──
        sample_steps: int = 50,
        fr_metrics:   Optional[List[str]] = None,
        iqa_device:   str = "cpu",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr_key          = lr_key
        self.ref_key         = ref_key
        self.hr_key          = hr_key
        self.sd_locked       = sd_locked
        self.learning_rate   = learning_rate
        self.l_simple_weight = l_simple_weight
        self.sample_steps    = sample_steps
        self.fr_metrics      = fr_metrics or ["psnr", "ssim", "lpips", "dists"]
        self.iqa_device      = iqa_device

        # 1. VAE（冻结）
        self.vae = AutoencoderKL.from_pretrained(sd_model_path, subfolder="vae")
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae_scale_factor = self.vae.config.scaling_factor

        # 2. UNet + LoRA
        self.unet = UNet2DConditionModel.from_pretrained(sd_model_path, subfolder="unet")
        if use_lora:
            self._inject_lora(lora_rank)
        if sd_locked:
            self._freeze_unet_except_attn()

        # 3. Noise Scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )

        # 4. RefDiffRWKV
        self.ref_model = RefDiffRWKV(
            patch_size=patch_size,
            embed_dim=embed_dim,
            channels=3,
            upsample_mode=upsample_mode,
        )

        # 5. Adapter
        self.ref_adapter = RWKV_Ref_Adapter(
            in_dims=(384, 768, 1536),
            out_dims=(320, 640, 1280),
        )

        # 6. GlobalSemantic
        if use_semantic:
            self.global_semantic = GlobalSemanticModule(
                dinov2_model=dinov2_model,
                unet_dim=embed_dim,
            )
            self.ref_model.global_semantic = self.global_semantic
        else:
            self.global_semantic = None

        # 7. IQAEngine
        from RefRWKV.evaluation.eval_pyiqa import IQAEngine
        self.iqa = IQAEngine(
            device=iqa_device,
            nr_metrics=[],
            fr_metrics=self.fr_metrics,
            use_y_channel=True,
            verbose=False,
        )

        # 8. UNet hooks
        self._ref_features_for_hook: Optional[List[torch.Tensor]] = None
        self._hook_idx: int = 0
        self._setup_unet_hooks()

    # ═══════════════════════════════════════════════════════
    #  LoRA & 冻结
    # ═══════════════════════════════════════════════════════

    def _inject_lora(self, rank: int):
        from peft import LoraConfig
        self.unet.add_adapter(LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.0,
        ))
        try:
            from diffusers.utils.peft_utils import set_weights_and_activate_adapters
            set_weights_and_activate_adapters(self.unet)
        except (ImportError, AttributeError):
            pass

    def _freeze_unet_except_attn(self):
        for n, p in self.unet.named_parameters():
            if "attn" not in n and "lora" not in n:
                p.requires_grad = False

    # ═══════════════════════════════════════════════════════
    #  UNet Hook
    # ═══════════════════════════════════════════════════════

    def _setup_unet_hooks(self):
        self._hooks = [
            self.unet.down_blocks[i].register_forward_hook(self._ref_injection_hook)
            for i in [0, 1, 2] if i < len(self.unet.down_blocks)
        ]

    def _ref_injection_hook(self, module, input, output):
        if self._ref_features_for_hook is None:
            return output
        idx = self._hook_idx
        if idx < len(self._ref_features_for_hook):
            feat = self._ref_features_for_hook[idx]
            hidden = output[0]
            if hidden.shape[2:] != feat.shape[2:]:
                feat = F.interpolate(feat, size=hidden.shape[2:], mode="nearest")
            output = (hidden + feat,) + output[1:]
            self._hook_idx += 1
        return output

    # ═══════════════════════════════════════════════════════
    #  VAE helpers
    # ═══════════════════════════════════════════════════════

    @torch.no_grad()
    def encode_latent(self, img):
        return self.vae.encode(img).latent_dist.sample() * self.vae_scale_factor

    @torch.no_grad()
    def decode_latent(self, z):
        return self.vae.decode(z / self.vae_scale_factor).sample

    def _empty_context(self, B, device):
        return torch.zeros(
            B, 77, self.unet.config.cross_attention_dim,
            device=device, dtype=torch.float32,
        )

    def _extract_ref(self, x_t_pixel, lr, ref):
        rf1, rf2, rf3 = self.ref_model.extract_ref_features(
            x_t=x_t_pixel, LR=lr, Ref=ref
        )
        return list(self.ref_adapter(rf1, rf2, rf3))

    # ═══════════════════════════════════════════════════════
    #  Training
    # ═══════════════════════════════════════════════════════

    def training_step(self, batch, batch_idx):
        lr  = batch[self.lr_key].float()
        ref = batch[self.ref_key].float()
        hr  = batch[self.hr_key].float()

        x0 = self.encode_latent(hr.to(self.device))
        B = x0.shape[0]
        t = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device,
        ).long()
        noise = torch.randn_like(x0)
        x_t = self.noise_scheduler.add_noise(x0, noise, t)

        ref_feats = self._extract_ref(
            self.decode_latent(x_t), lr.to(self.device), ref.to(self.device)
        )

        self._ref_features_for_hook = ref_feats
        self._hook_idx = 0
        pred = self.unet(
            x_t, t,
            encoder_hidden_states=self._empty_context(B, self.device),
        ).sample
        self._ref_features_for_hook = None

        loss = F.mse_loss(pred, noise) * self.l_simple_weight
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lr  = batch[self.lr_key].float()
        ref = batch[self.ref_key].float()
        hr  = batch[self.hr_key].float()

        x0 = self.encode_latent(hr.to(self.device))
        B = x0.shape[0]
        t = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device,
        ).long()
        noise = torch.randn_like(x0)
        x_t = self.noise_scheduler.add_noise(x0, noise, t)

        ref_feats = self._extract_ref(
            self.decode_latent(x_t), lr.to(self.device), ref.to(self.device)
        )

        self._ref_features_for_hook = ref_feats
        self._hook_idx = 0
        pred = self.unet(
            x_t, t,
            encoder_hidden_states=self._empty_context(B, self.device),
        ).sample
        self._ref_features_for_hook = None

        self.log("val/loss", F.mse_loss(pred, noise), on_epoch=True)

        if batch_idx == 0:
            self._validate_iqa(batch, lr, ref, hr)

    def configure_optimizers(self):
        params = (
            list(self.ref_model.parameters())
            + list(self.ref_adapter.parameters())
        )
        if self.global_semantic is not None:
            params += list(self.global_semantic.parameters())
        params += [p for p in self.unet.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.learning_rate)

    # ═══════════════════════════════════════════════════════
    #  IQA
    # ═══════════════════════════════════════════════════════

    @torch.no_grad()
    def _validate_iqa(self, batch, lr, ref, hr):
        B = hr.shape[0]
        _, _, H, W = self.encode_latent(hr.to(self.device)).shape

        latents = torch.randn(B, 4, H, W, device=self.device)
        self.noise_scheduler.set_timesteps(self.sample_steps)
        self.unet.eval()

        for ts in self.noise_scheduler.timesteps:
            t_b = torch.full((B,), ts, device=self.device).long()
            x_t_pixel = self.decode_latent(latents)
            ref_feats = self._extract_ref(
                x_t_pixel, lr.to(self.device), ref.to(self.device)
            )
            self._ref_features_for_hook = ref_feats
            self._hook_idx = 0
            pred = self.unet(
                latents, t_b,
                encoder_hidden_states=self._empty_context(B, self.device),
            ).sample
            self._ref_features_for_hook = None
            latents = self.noise_scheduler.step(pred, ts, latents).prev_sample

        sr = ((self.decode_latent(latents) + 1) / 2).clamp(0, 1)
        hr_norm = ((hr.to(self.device) + 1) / 2)
        self.unet.train()

        accum = {m: 0.0 for m in self.fr_metrics}
        for i in range(B):
            r = self.iqa.evaluate_single(
                sr[i].cpu().permute(1, 2, 0).numpy(),
                hr_norm[i].cpu().permute(1, 2, 0).numpy(),
            )
            for k in accum:
                accum[k] += r.get(k, 0.0)
        for k, v in accum.items():
            self.log(f"val/{k}", v / B)

    # ═══════════════════════════════════════════════════════
    #  推理
    # ═══════════════════════════════════════════════════════

    @torch.no_grad()
    def inference(self, lr, ref, steps=None, seed=42):
        steps = steps or self.sample_steps
        lr  = lr.to(self.device).float()
        ref = ref.to(self.device).float()

        th, tw = lr.shape[2] * 10, lr.shape[3] * 10
        if ref.shape[2:] != (th, tw):
            ref = F.interpolate(
                ref, size=(th, tw), mode="bilinear", align_corners=False
            )

        g = torch.Generator(device=self.device).manual_seed(seed)
        self.unet.eval()
        latents = torch.randn(
            1, 4, th // 8, tw // 8,
            generator=g, device=self.device,
        )
        self.noise_scheduler.set_timesteps(steps)

        for ts in self.noise_scheduler.timesteps:
            t_b = torch.full((1,), ts, device=self.device).long()
            x_t_pixel = self.decode_latent(latents)
            ref_feats = self._extract_ref(x_t_pixel, lr, ref)
            self._ref_features_for_hook = ref_feats
            self._hook_idx = 0
            pred = self.unet(
                latents, t_b,
                encoder_hidden_states=self._empty_context(1, self.device),
            ).sample
            self._ref_features_for_hook = None
            latents = self.noise_scheduler.step(pred, ts, latents).prev_sample

        return ((self.decode_latent(latents) + 1) / 2).clamp(0, 1)


# ═══════════════════════════════════════════════════════
#  Smoke test
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SD2ControlLDM(
        sd_model_path="sd2-community/stable-diffusion-2-1-base",
        use_lora=True, lora_rank=4, sd_locked=True,
        use_semantic=True, learning_rate=1e-4, sample_steps=20,
    ).to(device)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {total:,} (~{total/1e6:.1f}M)")

    B, H, W = 2, 480, 480
    batch = {
        "lr":  torch.randn(B, 3, 48, 48, device=device),
        "ref": torch.randn(B, 3, H, W, device=device),
        "hr":  torch.randn(B, 3, H, W, device=device),
    }
    loss = model.training_step(batch, 0)
    print(f"Loss: {loss.item():.4f}")

    sr = model.inference(
        torch.randn(1, 3, 48, 48, device=device),
        torch.randn(1, 3, H, W, device=device),
        steps=20,
    )
    print(f"Inference output: {sr.shape}")
    print("✓ Smoke test passed!")
