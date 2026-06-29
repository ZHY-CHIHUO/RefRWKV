"""sd2_control_ldm.py — 精简版：纯模型类，所有参数通过 __init__ 直接传入
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, List

from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler

import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).parent)
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from RefDiffRWKV import RefDiffRWKV
from sd2_ref_adapter import RWKV_Ref_Adapter
from GlobalSemanticModule import GlobalSemanticModule


class SD2ControlLDM(pl.LightningModule):
    def __init__(
        self,
        lr_key: str = "lr",
        ref_key: str = "ref",
        hr_key: str = "hr",
        sd_model_path: str = "sd2-community/stable-diffusion-2-1-base",
        use_lora: bool = True,
        lora_rank: int = 4,
        lora_target_modules: Optional[List[str]] = None,
        sd_locked: bool = True,
        patch_size: int = 4,
        embed_dim: int = 384,
        upsample_mode: str = "bilinear",
        use_semantic: bool = True,
        dinov2_model_name: str = "facebook/dinov2-base",
        cfg_drop_prob: float = 0.1,
        learning_rate: float = 1e-4,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "epsilon",
        l_simple_weight: float = 1.0,
        weight_decay: float = 1e-3,
        sample_steps: int = 50,
        fr_metrics: Optional[List[str]] = None,
        iqa_device: str = "cpu",
        debug_nan: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr_key = lr_key
        self.ref_key = ref_key
        self.hr_key = hr_key
        self.sd_locked = sd_locked
        self.cfg_drop_prob = cfg_drop_prob
        self.learning_rate = learning_rate
        self.l_simple_weight = l_simple_weight
        self.sample_steps = sample_steps
        self.fr_metrics = fr_metrics or ["psnr", "ssim", "lpips", "dists"]
        self.iqa_device = iqa_device
        self.weight_decay = weight_decay
        self.debug_nan = debug_nan
        self._nan_count = 0

        # 1. VAE
        self.vae = AutoencoderKL.from_pretrained(
            sd_model_path, subfolder="vae", local_files_only=True
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae_scale_factor = self.vae.config.scaling_factor

        # 2. UNet + LoRA
        self.unet = UNet2DConditionModel.from_pretrained(
            sd_model_path, subfolder="unet", local_files_only=True
        )
        self.unet.enable_gradient_checkpointing()
        if use_lora:
            self._inject_lora(lora_rank, lora_target_modules)
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

        # 4. GlobalSemantic
        if use_semantic:
            self.global_semantic = GlobalSemanticModule(
                dinov2_model_name=dinov2_model_name,
                unet_dim=embed_dim,
            )
        else:
            self.global_semantic = None

        # 5. RefDiffRWKV
        self.ref_model = RefDiffRWKV(
            patch_size=patch_size,
            embed_dim=embed_dim,
            channels=3,
            upsample_mode=upsample_mode,
            global_semantic=self.global_semantic,
        )

        # 6. Adapter
        self.ref_adapter = RWKV_Ref_Adapter(
            ref_dims=(384, 768, 1536),
            sd2_dims=(320, 640, 1280),
        )

        # 7. IQA
        from RefRWKV.evaluation.eval_pyiqa import IQAEngine

        self.iqa = IQAEngine(
            device=iqa_device,
            nr_metrics=[],
            fr_metrics=self.fr_metrics,
            use_y_channel=True,
            verbose=False,
        )

        # 8. UNet hooks
        self._injection_blocks: List = []
        self._setup_unet_hooks()

    # ══════════════════════════════════
    #  LoRA & freeze
    # ══════════════════════════════════

    def _inject_lora(self, rank, target_modules=None):
        if target_modules is None:
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        from peft import LoraConfig

        self.unet.add_adapter(
            LoraConfig(
                r=rank, lora_alpha=rank,
                target_modules=target_modules, lora_dropout=0.0,
            )
        )
        try:
            from diffusers.utils.peft_utils import set_weights_and_activate_adapters
            set_weights_and_activate_adapters(self.unet, ["default"], [1.0])
        except (ImportError, AttributeError):
            pass

    def _freeze_unet_except_attn(self):
        for n, p in self.unet.named_parameters():
            if "attn" not in n and "lora" not in n:
                p.requires_grad = False

    # ══════════════════════════════════
    #  UNet hooks — 属性注入，兼容 gradient checkpointing
    # ══════════════════════════════════

    def _setup_unet_hooks(self):
        for h in getattr(self, "_hook_handles", []):
            h.remove()
        self._injection_blocks = []
        self._hook_handles = []
        for i in [0, 1, 2]:
            if i < len(self.unet.down_blocks):
                block = self.unet.down_blocks[i]
                handle = block.register_forward_hook(self._ref_injection_hook)
                self._hook_handles.append(handle)
                self._injection_blocks.append(block)

    def _ref_injection_hook(self, module, input, output):
        feat = getattr(module, "_ref_feat", None)
        if feat is None:
            return output
        hidden = output[0]
        if hidden.shape[2:] != feat.shape[2:]:
            feat = F.interpolate(
                feat, size=hidden.shape[2:], mode="bilinear", align_corners=False
            )
        return (hidden + feat,) + output[1:]

    def _inject_ref_feats(self, ref_feats):
        for block, feat in zip(self._injection_blocks, ref_feats):
            block._ref_feat = feat

    def _clear_ref_feats(self):
        for block in self._injection_blocks:
            if hasattr(block, "_ref_feat"):
                del block._ref_feat

    # ══════════════════════════════════
    #  NaN debug
    # ══════════════════════════════════

    def _check_tensor(self, t, name=""):
        if not self.debug_nan:
            return False
        if torch.isnan(t).any():
            print(f"[NaN] {name} | step={self.global_step} | "
                  f"shape={t.shape} | min={t.min():.6f} max={t.max():.6f}")
            return True
        if torch.isinf(t).any():
            print(f"[Inf] {name} | step={self.global_step} | "
                  f"shape={t.shape} | min={t.min():.6f} max={t.max():.6f}")
            return True
        return False

    # ══════════════════════════════════
    #  VAE helpers
    # ══════════════════════════════════

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

    # ══════════════════════════════════
    #  Ref 特征提取 — 全静态（训练/推理一致）
    # ══════════════════════════════════

    def _extract_ref(self, x_t_pixel, lr, ref):
        """动态提取（保留给特殊场景，当前未使用）。"""
        result = self.ref_model.extract_ref_features(x_t=x_t_pixel, LR=lr, Ref=ref)
        result = [r.clamp(-50, 50) for r in result]
        return list(self.ref_adapter(result[0], result[1], result[2]))

    @torch.no_grad()
    def _extract_ref_static(self, lr, ref, latent_shape):
        """
        从零潜变量解码 VAE 均值图，仅提取一次 ref 特征。
        训推一致：循环外算一次，整个去噪过程复用。
        """
        B, _, H, W = latent_shape
        dummy = torch.zeros(B, 4, H, W, device=self.device)
        dummy_pixel = self.vae.decode(
            dummy / self.vae_scale_factor
        ).sample.clamp(-1, 1)
        return self._extract_ref(dummy_pixel, lr, ref)

    # ══════════════════════════════════
    #  Training
    # ══════════════════════════════════

    def training_step(self, batch, batch_idx):
        lr = batch[self.lr_key].float()
        ref = batch[self.ref_key].float()
        hr = batch[self.hr_key].float()

        if self._check_tensor(lr, "input lr"):
            return None
        if self._check_tensor(ref, "input ref"):
            return None
        if self._check_tensor(hr, "input hr"):
            return None

        x0 = self.encode_latent(hr.to(self.device))
        if self._check_tensor(x0, "x0 (latent)"):
            return None

        B = x0.shape[0]
        t = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device,
        ).long()
        noise = torch.randn_like(x0)
        x_t = self.noise_scheduler.add_noise(x0, noise, t)
        if self._check_tensor(x_t, "x_t (noisy latent)"):
            return None

        ref_input = ref.to(self.device)
        if self.cfg_drop_prob > 0:
            drop_mask = torch.rand(B, device=self.device) < self.cfg_drop_prob
            if drop_mask.any():
                ref_input = ref_input.clone()
                ref_input[drop_mask] = 0.0

        # 静态 ref 特征（不再 decode x_t）
        ref_feats = self._extract_ref_static(
            lr.to(self.device), ref_input, x0.shape
        )
        for i, f in enumerate(ref_feats):
            if self._check_tensor(f, f"ref_feat[{i}]"):
                return None

        self._inject_ref_feats(ref_feats)
        pred = self.unet(
            x_t, t,
            encoder_hidden_states=self._empty_context(B, self.device),
        ).sample

        if self._check_tensor(pred, "pred (UNet output)"):
            return None

        loss = F.mse_loss(pred, noise) * self.l_simple_weight
        if self._check_tensor(loss, "loss"):
            self._nan_count += 1
            print(f"⚠️  NaN loss #{self._nan_count} at step {self.global_step}")
            if self._nan_count >= 10:
                raise RuntimeError(
                    f"连续 {self._nan_count} 步 NaN，终止训练 (step {self.global_step})"
                )
            return None
        else:
            self._nan_count = 0

        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lr = batch[self.lr_key].float()
        ref = batch[self.ref_key].float()
        hr = batch[self.hr_key].float()

        x0 = self.encode_latent(hr.to(self.device))
        B = x0.shape[0]
        t = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device,
        ).long()
        noise = torch.randn_like(x0)
        x_t = self.noise_scheduler.add_noise(x0, noise, t)

        ref_feats = self._extract_ref_static(
            lr.to(self.device), ref.to(self.device), x0.shape
        )
        self._inject_ref_feats(ref_feats)
        pred = self.unet(
            x_t, t,
            encoder_hidden_states=self._empty_context(B, self.device),
        ).sample

        val_loss = F.mse_loss(pred, noise)
        self.log("val/loss", val_loss, on_epoch=True)
        self.log("val_loss", val_loss, on_epoch=True)

        if batch_idx == 0:
            self._validate_iqa(lr, ref, hr)

    def configure_optimizers(self):
        params = list(self.ref_model.parameters()) + list(self.ref_adapter.parameters())
        if self.global_semantic is not None:
            params += list(self.global_semantic.parameters())
        params += [p for p in self.unet.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params, lr=self.learning_rate, weight_decay=self.weight_decay
        )

    # ══════════════════════════════════
    #  IQA（静态 ref 特征，循环外一次提取）
    # ══════════════════════════════════

    @torch.no_grad()
    def _validate_iqa(self, lr, ref, hr):
        B = hr.shape[0]
        latent_shape = self.encode_latent(hr.to(self.device)).shape
        _, _, H, W = latent_shape

        ref_feats = self._extract_ref_static(
            lr.to(self.device), ref.to(self.device), latent_shape
        )
        self._inject_ref_feats(ref_feats)

        latents = torch.randn(B, 4, H, W, device=self.device)
        self.noise_scheduler.set_timesteps(self.sample_steps)
        self.unet.eval()

        try:
            for ts in self.noise_scheduler.timesteps:
                t_b = torch.full((B,), ts, device=self.device).long()
                pred = self.unet(
                    latents, t_b,
                    encoder_hidden_states=self._empty_context(B, self.device),
                ).sample
                latents = self.noise_scheduler.step(pred, ts, latents).prev_sample
                del pred
                torch.cuda.empty_cache()

            sr = ((self.decode_latent(latents) + 1) / 2).clamp(0, 1)

            from torchvision.utils import save_image
            save_image(sr[0:4], f"debug_val_epoch{self.current_epoch}.png")

            del latents
            torch.cuda.empty_cache()

            hr_norm = (hr.to(self.device) + 1) / 2
            accum = {m: 0.0 for m in self.fr_metrics}
            for i in range(B):
                r = self.iqa.evaluate_single(
                    sr[i].cpu().float().permute(1, 2, 0).numpy(),
                    hr_norm[i].cpu().float().permute(1, 2, 0).numpy(),
                )
                for k in accum:
                    accum[k] += r.get(k, 0.0)
            for k, v in accum.items():
                self.log(f"val/{k}", v / B)
        finally:
            self._clear_ref_feats()
            self.unet.train()

    # ══════════════════════════════════
    #  推理（静态 ref 特征）
    # ══════════════════════════════════

    @torch.no_grad()
    def inference(self, lr, ref, steps=None, seed=42):
        steps = steps or self.sample_steps
        lr = lr.to(self.device).float()
        ref = ref.to(self.device).float()

        th, tw = lr.shape[2] * 10, lr.shape[3] * 10
        if ref.shape[2:] != (th, tw):
            ref = F.interpolate(
                ref, size=(th, tw), mode="bilinear", align_corners=False
            )

        latent_shape = (1, 4, th // 8, tw // 8)
        ref_feats = self._extract_ref_static(lr, ref, latent_shape)
        self._inject_ref_feats(ref_feats)

        g = torch.Generator(device=self.device).manual_seed(seed)
        self.unet.eval()
        latents = torch.randn(
            1, 4, th // 8, tw // 8, generator=g, device=self.device,
        )
        self.noise_scheduler.set_timesteps(steps)

        try:
            for ts in self.noise_scheduler.timesteps:
                t_b = torch.full((1,), ts, device=self.device).long()
                pred = self.unet(
                    latents, t_b,
                    encoder_hidden_states=self._empty_context(1, self.device),
                ).sample
                latents = self.noise_scheduler.step(pred, ts, latents).prev_sample
        finally:
            self._clear_ref_feats()

        return ((self.decode_latent(latents) + 1) / 2).clamp(0, 1)

    # ══════════════════════════════════
    #  Checkpoint hooks（最小化）
    # ══════════════════════════════════

    def on_save_checkpoint(self, checkpoint):
        self._clear_ref_feats()

    def on_load_checkpoint(self, checkpoint):
        self._setup_unet_hooks()


# ══════════════════════════════════
#  Smoke test
# ══════════════════════════════════

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
        "lr": torch.randn(B, 3, 48, 48, device=device),
        "ref": torch.randn(B, 3, H, W, device=device),
        "hr": torch.randn(B, 3, H, W, device=device),
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
