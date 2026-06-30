"""
sd2_control_ldm.py — 双路径架构：语义 cross-attention + 纹理 skip connection
训练范式：单步 x0 预测 + 像素空间 loss（L2 + LPIPS + GAN）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import lpips
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
from discriminator import ImageConvNextDiscriminator


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
        lr_D: float = 1e-4,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "epsilon",
        l_simple_weight: float = 1.0,
        lambda_lpips: float = 0.1,
        lambda_gan: float = 0.005,
        model_t: int = 200,
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
        self.lr_D = lr_D
        self.l_simple_weight = l_simple_weight
        self.lambda_lpips = lambda_lpips
        self.lambda_gan = lambda_gan
        self.model_t = model_t
        self.sample_steps = sample_steps
        self.fr_metrics = fr_metrics or ["psnr", "ssim", "lpips", "dists"]
        self.iqa_device = iqa_device
        self.weight_decay = weight_decay
        self.debug_nan = debug_nan
        self._nan_count = 0

        # ── 手动优化：G/D 交替训练 ──
        self.automatic_optimization = False

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

        self.cross_attn_dim = self.unet.config.cross_attention_dim  # 768

        # 4. GlobalSemantic（DINOv2 → RWKV Pyramid）
        if use_semantic:
            self.global_semantic = GlobalSemanticModule(
                dinov2_model_name=dinov2_model_name,
            )
        else:
            self.global_semantic = None
        self.sem_proj: Optional[nn.Linear] = None

        # 5. RefDiffRWKV（纯纹理提取）
        self.ref_model = RefDiffRWKV(
            patch_size=patch_size,
            embed_dim=embed_dim,
            channels=3,
            upsample_mode=upsample_mode,
        )

        # 6. Adapter
        self.ref_adapter = RWKV_Ref_Adapter(
            ref_dims=(384, 768, 1536),
            sd2_dims=(320, 640, 1280),
        )

        # 7. LPIPS 感知损失
        self.net_lpips = lpips.LPIPS(net="vgg", verbose=False)
        self.net_lpips.eval()
        self.net_lpips.requires_grad_(False)

        # 8. 判别器（GAN）
        self.D = ImageConvNextDiscriminator(precision="fp32")

        # 9. IQA
        from RefRWKV.evaluation.eval_pyiqa import IQAEngine

        self.iqa = IQAEngine(
            device=iqa_device,
            nr_metrics=[],
            fr_metrics=self.fr_metrics,
            use_y_channel=True,
            verbose=False,
        )

        # 10. UNet hooks（纹理 skip connection 注入点）
        self._injection_blocks: List = []
        self._hook_handles: List = []
        self._setup_unet_hooks()

    # ════════════════════════════════════════════════════════
    #  LoRA & freeze
    # ════════════════════════════════════════════════════════

    def _inject_lora(self, rank, target_modules=None):
        if target_modules is None:
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        from peft import LoraConfig

        self.unet.add_adapter(
            LoraConfig(
                r=rank,
                lora_alpha=rank,
                target_modules=target_modules,
                lora_dropout=0.0,
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

    # ════════════════════════════════════════════════════════
    #  UNet hooks — 纹理注入（skip connection）
    # ════════════════════════════════════════════════════════

    def _setup_unet_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._injection_blocks = []
        self._hook_handles = []
        for i in [0, 1, 2]:
            if i < len(self.unet.down_blocks):
                block = self.unet.down_blocks[i]
                target = block.resnets[-1]
                handle = target.register_forward_hook(self._ref_injection_hook)
                self._hook_handles.append(handle)
                self._injection_blocks.append(target)

    def _ref_injection_hook(self, module, input, output):
        feat = getattr(module, "_ref_feat", None)
        if feat is None:
            return output

        if isinstance(output, tuple):
            hidden = output[0]
            is_tuple = True
        else:
            hidden = output
            is_tuple = False

        if hidden.shape[2:] != feat.shape[2:]:
            feat = F.interpolate(
                feat, size=hidden.shape[2:], mode="bilinear", align_corners=False
            )
        hidden = hidden + feat

        if is_tuple:
            return (hidden,) + output[1:]
        return hidden

    def _inject_ref_feats(self, ref_feats):
        for block, feat in zip(self._injection_blocks, ref_feats):
            block._ref_feat = feat

    def _clear_ref_feats(self):
        for block in self._injection_blocks:
            if hasattr(block, "_ref_feat"):
                del block._ref_feat

    # ════════════════════════════════════════════════════════
    #  NaN debug
    # ════════════════════════════════════════════════════════

    def _check_tensor(self, t, name=""):
        if not self.debug_nan:
            return False
        if torch.isnan(t).any():
            print(
                f"[NaN] {name} | step={self.global_step} | "
                f"shape={t.shape} | min={t.min():.6f} max={t.max():.6f}"
            )
            return True
        if torch.isinf(t).any():
            print(
                f"[Inf] {name} | step={self.global_step} | "
                f"shape={t.shape} | min={t.min():.6f} max={t.max():.6f}"
            )
            return True
        return False

    # ════════════════════════════════════════════════════════
    #  VAE helpers
    # ════════════════════════════════════════════════════════

    @torch.no_grad()
    def encode_latent(self, img):
        return self.vae.encode(img).latent_dist.sample() * self.vae_scale_factor

    @torch.no_grad()
    def decode_latent(self, z):
        """VAE decode without gradient（inference / IQA 用）。"""
        return self.vae.decode(z / self.vae_scale_factor).sample

    def _decode_latent_train(self, z):
        """VAE decode with gradient flow（training 用，梯度穿过冻结 VAE）。"""
        return self.vae.decode(z / self.vae_scale_factor).sample

    def _empty_context(self, B, device):
        return torch.zeros(
            B, 77, self.cross_attn_dim,
            device=device, dtype=torch.float32,
        )

    # ════════════════════════════════════════════════════════
    #  Ref 特征提取 — 纹理 + 语义两路独立
    # ════════════════════════════════════════════════════════

    def _extract_ref(self, x_t_pixel, lr, ref):
        """纯纹理路径：RefDiffRWKV → Adapter → rf_feats。"""
        rf1, rf2, rf3 = self.ref_model.extract_ref_features(
            x_t=x_t_pixel, LR=lr, Ref=ref
        )
        rf_feats = list(self.ref_adapter(rf1, rf2, rf3))
        return rf_feats

    def _zero_ref_feats(self, x_t_pixel, lr):
        B, _, H, W = x_t_pixel.shape
        p = self.ref_model.patch_size
        h1, w1 = H // p // 2, W // p // 2
        h2, w2 = h1 // 2, w1 // 2
        h3, w3 = h2 // 2, w2 // 2
        rf_feats = [
            torch.zeros(B, 320, h1, w1, device=x_t_pixel.device),
            torch.zeros(B, 640, h2, w2, device=x_t_pixel.device),
            torch.zeros(B, 1280, h3, w3, device=x_t_pixel.device),
        ]
        return rf_feats

    @staticmethod
    def _is_zero_image(ref: torch.Tensor) -> bool:
        return ref.abs().sum(dim=[1, 2, 3]).max().item() < 1e-6

    def _extract_ref_semantic(self, ref):
        if self.global_semantic is None:
            return None
        if self._is_zero_image(ref):
            return None
        sem_pyramid = self.global_semantic(ref)
        sem_tokens = self._build_sem_tokens(sem_pyramid)
        return sem_tokens

    def _extract_ref_static(self, lr, ref, latent_shape):
        B, _, H, W = latent_shape

        with torch.no_grad():
            dummy = torch.zeros(B, 4, H, W, device=self.device)
            dummy_pixel = self.vae.decode(
                dummy / self.vae_scale_factor
            ).sample.clamp(-1, 1)

        if self._is_zero_image(ref):
            rf_feats = self._zero_ref_feats(dummy_pixel, lr)
        else:
            rf_feats = self._extract_ref(dummy_pixel, lr, ref)

        sem_tokens = self._extract_ref_semantic(ref)

        return rf_feats, sem_tokens

    def _build_sem_tokens(self, sem_pyramid):
        if sem_pyramid is None or self.global_semantic is None:
            return None

        layer_tokens = {}
        for key in ["e1", "e2", "e3", "latent"]:
            if key not in sem_pyramid:
                continue
            tokens = sem_pyramid[key]
            layer_tokens[key] = tokens.float()

        if not layer_tokens:
            return None

        if self.sem_proj is None:
            first_tokens = next(iter(layer_tokens.values()))
            base_dim = first_tokens.shape[-1]
            self.sem_proj = nn.Linear(
                base_dim, self.cross_attn_dim
            ).to(first_tokens.device)
            total_tokens = sum(t.shape[1] for t in layer_tokens.values())
            token_sizes = {k: v.shape[1] for k, v in layer_tokens.items()}
            print(
                f"[SemProj] 金字塔语义注入: {token_sizes} "
                f"× {base_dim} → {total_tokens} tokens × {self.cross_attn_dim}"
            )

        projected = []
        for key in ["e1", "e2", "e3", "latent"]:
            if key in layer_tokens:
                proj = self.sem_proj(layer_tokens[key])
                projected.append(proj)

        return torch.cat(projected, dim=1)

    # ════════════════════════════════════════════════════════
    #  语义 context 构建
    # ════════════════════════════════════════════════════════

    def _build_context(self, B, device, sem_tokens=None):
        empty_ctx = self._empty_context(B, device)
        if sem_tokens is not None:
            return torch.cat([empty_ctx, sem_tokens], dim=1)
        return empty_ctx

    # ════════════════════════════════════════════════════════
    #  UNet forward helper（共享给 G step 和 D step）
    # ════════════════════════════════════════════════════════

    def _run_unet(self, lr, ref, hr):
        """
        完整 UNet 前向：HR encode → add noise (t=model_t) → UNet → pred_x0。

        Returns:
            pred_x0:   (B, 4, H_lat, W_lat)  预测的干净 latent
            noise_pred: (B, 4, H_lat, W_lat)  UNet 噪声预测
            noise:      (B, 4, H_lat, W_lat)  实际加入的噪声
        """
        x0 = self.encode_latent(hr.to(self.device))
        B = x0.shape[0]

        t = torch.full((B,), self.model_t, device=self.device).long()
        noise = torch.randn_like(x0)
        x_t = self.noise_scheduler.add_noise(x0, noise, t)

        if self._check_tensor(x_t, "x_t (noisy latent)"):
            return None, None, None

        # CFG drop
        ref_input = ref.to(self.device)
        if self.cfg_drop_prob > 0:
            drop_mask = torch.rand(B, device=self.device) < self.cfg_drop_prob
            if drop_mask.any():
                ref_input = ref_input.clone()
                ref_input[drop_mask] = 0.0

        rf_feats, sem_tokens = self._extract_ref_static(
            lr.to(self.device), ref_input, x0.shape
        )
        for i, f in enumerate(rf_feats):
            if self._check_tensor(f, f"ref_feat[{i}]"):
                return None, None, None

        context = self._build_context(B, self.device, sem_tokens)
        self._inject_ref_feats(rf_feats)

        noise_pred = self.unet(
            x_t, t,
            encoder_hidden_states=context,
        ).sample

        if self._check_tensor(noise_pred, "noise_pred (UNet output)"):
            return None, None, None

        pred_x0 = self.noise_scheduler.step(
            noise_pred, self.model_t, x_t
        ).pred_original_sample

        return pred_x0, noise_pred, noise

    # ════════════════════════════════════════════════════════
    #  Training — G/D 交替（手动优化）
    # ════════════════════════════════════════════════════════

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()

        # 按 global_step 奇偶交替：偶 → Generator，奇 → Discriminator
        is_g_step = (self.trainer.global_step % 2 == 0)

        try:
            if is_g_step:
                return self._generator_step(batch, batch_idx, g_opt)
            else:
                return self._discriminator_step(batch, batch_idx, d_opt)
        finally:
            self._clear_ref_feats()

    def _generator_step(self, batch, batch_idx, g_opt):
        """Generator step: L2 + LPIPS + GAN → 更新 G 参数。"""
        lr = batch[self.lr_key].float()
        ref = batch[self.ref_key].float()
        hr = batch[self.hr_key].float()

        if self._check_tensor(lr, "input lr") or \
           self._check_tensor(ref, "input ref") or \
           self._check_tensor(hr, "input hr"):
            return None

        # ── D 置 eval，冻结 ──
        self.D.eval()
        self.D.requires_grad_(False)

        # ── UNet 前向 ──
        pred_x0, noise_pred, noise = self._run_unet(lr, ref, hr)
        if pred_x0 is None:
            return None

        # ── latent → 像素空间（梯度穿过 VAE decoder）──
        x_pred = self._decode_latent_train(pred_x0)
        if self._check_tensor(x_pred, "x_pred (pixel)"):
            return None

        # ── 三项 loss ──
        loss_l2 = F.mse_loss(x_pred, hr.to(self.device)) * self.l_simple_weight
        loss_lpips = self.net_lpips(x_pred, hr.to(self.device)).mean() * self.lambda_lpips
        loss_gan = self.D(x_pred, for_G=True).mean() * self.lambda_gan

        loss_G = loss_l2 + loss_lpips + loss_gan

        if self._check_tensor(loss_G, "loss_G"):
            self._nan_count += 1
            print(f"⚠️  NaN loss_G #{self._nan_count} at step {self.global_step}")
            if self._nan_count >= 10:
                raise RuntimeError(
                    f"连续 {self._nan_count} 步 NaN，终止训练 (step {self.global_step})"
                )
            return None
        else:
            self._nan_count = 0

        # ── 手动 backward ──
        g_opt.zero_grad()
        self.manual_backward(loss_G)
        self.clip_gradients(g_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        g_opt.step()

        # ── log ──
        self.log_dict({
            "train/G_total": loss_G,
            "train/G_l2": loss_l2,
            "train/G_lpips": loss_lpips,
            "train/G_gan": loss_gan,
        }, prog_bar=True, on_step=True)

        return loss_G

    def _discriminator_step(self, batch, batch_idx, d_opt):
        """Discriminator step: real/fake BCE → 更新 D 参数。"""
        lr = batch[self.lr_key].float()
        ref = batch[self.ref_key].float()
        hr = batch[self.hr_key].float()

        if self._check_tensor(hr, "input hr (D step)"):
            return None

        # ── D 置 train ──
        self.D.train()
        self.D.requires_grad_(True)

        # ── 无梯度生成 x_pred ──
        with torch.no_grad():
            pred_x0, _, _ = self._run_unet(lr, ref, hr)
            if pred_x0 is None:
                return None
            x_pred = self.decode_latent(pred_x0)

        # ── 判别器 loss ──
        loss_D_real = self.D(hr.to(self.device), for_real=True).mean()
        loss_D_fake = self.D(x_pred, for_real=False).mean()
        loss_D = loss_D_real + loss_D_fake

        # ── 手动 backward ──
        d_opt.zero_grad()
        self.manual_backward(loss_D)
        self.clip_gradients(d_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        d_opt.step()

        # ── log ──
        self.log_dict({
            "train/D_total": loss_D,
            "train/D_real": loss_D_real,
            "train/D_fake": loss_D_fake,
        }, prog_bar=True, on_step=True)

        return loss_D

    # ════════════════════════════════════════════════════════
    #  Validation
    # ════════════════════════════════════════════════════════

    def validation_step(self, batch, batch_idx):
        try:
            return self._validation_step_impl(batch, batch_idx)
        finally:
            self._clear_ref_feats()

    def _validation_step_impl(self, batch, batch_idx):
        lr = batch[self.lr_key].float()
        ref = batch[self.ref_key].float()
        hr = batch[self.hr_key].float()

        pred_x0, noise_pred, noise = self._run_unet(lr, ref, hr)
        if pred_x0 is None:
            return None

        # latent 空间 noise loss（与 HYPIR 一致：验证时仍看 denoising 能力）
        val_loss = F.mse_loss(noise_pred, noise)
        self.log("val/loss", val_loss, on_epoch=True)
        self.log("val_loss", val_loss, on_epoch=True)

        if batch_idx == 0:
            self._validate_iqa(lr, ref, hr)

        return val_loss

    # ════════════════════════════════════════════════════════
    #  Optimizer
    # ════════════════════════════════════════════════════════

    def configure_optimizers(self):
        self._ensure_sem_proj()

        # ── Generator 参数 ──
        g_params = list(self.ref_model.parameters()) + list(self.ref_adapter.parameters())
        if self.global_semantic is not None:
            g_params += list(self.global_semantic.parameters())
        if self.sem_proj is not None:
            g_params += list(self.sem_proj.parameters())
        g_params += [p for p in self.unet.parameters() if p.requires_grad]

        g_opt = torch.optim.AdamW(
            g_params, lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # ── Discriminator 参数 ──
        d_params = list(filter(lambda p: p.requires_grad, self.D.parameters()))
        d_opt = torch.optim.AdamW(
            d_params, lr=self.lr_D, weight_decay=self.weight_decay
        )

        return [g_opt, d_opt]

    def _ensure_sem_proj(self):
        if self.global_semantic is None or self.sem_proj is not None:
            return
        dummy_ref = torch.randn(1, 3, 224, 224, device=self.device)
        with torch.no_grad():
            sem_pyramid = self.global_semantic(dummy_ref)
        self._build_sem_tokens(sem_pyramid)
        print("[SemProj] 已在 configure_optimizers 中完成延迟初始化")

    # ════════════════════════════════════════════════════════
    #  IQA
    # ════════════════════════════════════════════════════════

    @torch.no_grad()
    def _validate_iqa(self, lr, ref, hr):
        B = hr.shape[0]
        latent_shape = self.encode_latent(hr.to(self.device)).shape
        _, _, H, W = latent_shape

        rf_feats, sem_tokens = self._extract_ref_static(
            lr.to(self.device), ref.to(self.device), latent_shape
        )
        self._inject_ref_feats(rf_feats)
        context = self._build_context(B, self.device, sem_tokens)

        # 单步出图
        noise = torch.randn(B, 4, H, W, device=self.device)
        alpha_bar = self.noise_scheduler.alphas_cumprod[self.model_t]
        z_t = torch.sqrt(1 - alpha_bar) * noise

        t_b = torch.full((B,), self.model_t, device=self.device).long()
        self.unet.eval()
        noise_pred = self.unet(
            z_t, t_b,
            encoder_hidden_states=context,
        ).sample
        z = self.noise_scheduler.step(
            noise_pred, self.model_t, z_t
        ).pred_original_sample
        sr = ((self.decode_latent(z) + 1) / 2).clamp(0, 1)
        self.unet.train()

        from torchvision.utils import save_image
        save_image(sr[0:4], f"debug_val_epoch{self.current_epoch}.png")

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

        self._clear_ref_feats()

    # ════════════════════════════════════════════════════════
    #  推理（单步）
    # ════════════════════════════════════════════════════════

    @torch.no_grad()
    def inference(self, lr, ref, seed=42):
        lr = lr.to(self.device).float()
        ref = ref.to(self.device).float()

        th, tw = lr.shape[2] * 10, lr.shape[3] * 10
        if ref.shape[2:] != (th, tw):
            ref = F.interpolate(
                ref, size=(th, tw), mode="bilinear", align_corners=False
            )

        latent_shape = (1, 4, th // 8, tw // 8)
        rf_feats, sem_tokens = self._extract_ref_static(lr, ref, latent_shape)
        self._inject_ref_feats(rf_feats)
        context = self._build_context(1, self.device, sem_tokens)

        g = torch.Generator(device=self.device).manual_seed(seed)
        self.unet.eval()
        noise = torch.randn(
            1, 4, th // 8, tw // 8, generator=g, device=self.device,
        )

        # 单步：纯噪声 → 在 t=model_t 噪声水平去噪 → x0
        alpha_bar = self.noise_scheduler.alphas_cumprod[self.model_t]
        z_t = torch.sqrt(1 - alpha_bar) * noise
        t_b = torch.full((1,), self.model_t, device=self.device).long()

        noise_pred = self.unet(
            z_t, t_b,
            encoder_hidden_states=context,
        ).sample
        z = self.noise_scheduler.step(
            noise_pred, self.model_t, z_t
        ).pred_original_sample

        self._clear_ref_feats()
        return ((self.decode_latent(z) + 1) / 2).clamp(0, 1)

    # ════════════════════════════════════════════════════════
    #  Checkpoint hooks
    # ════════════════════════════════════════════════════════

    def on_save_checkpoint(self, checkpoint):
        self._clear_ref_feats()

    def on_load_checkpoint(self, checkpoint):
        self._setup_unet_hooks()
        if self.sem_proj is not None and self.global_semantic is not None:
            print("[SemProj] 已从 checkpoint 恢复")


# ═══════════════════════════════════════════════════════════
#  Smoke test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SD2ControlLDM(
        sd_model_path="sd2-community/stable-diffusion-2-1-base",
        use_lora=True, lora_rank=4, sd_locked=True,
        use_semantic=True,
        learning_rate=1e-4,
        model_t=200,
        lambda_lpips=0.1,
        lambda_gan=0.005,
    ).to(device)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {total:,} (~{total/1e6:.1f}M)")

    B, H, W = 2, 480, 480
    batch = {
        "lr": torch.randn(B, 3, 48, 48, device=device),
        "ref": torch.randn(B, 3, H, W, device=device),
        "hr": torch.randn(B, 3, H, W, device=device),
    }
    # Smoke test: 跑一次 G step
    g_opt, d_opt = model.configure_optimizers()
    # 手动设置 global_step 来触发 G step
    model.trainer = type("Dummy", (), {"global_step": 0})()
    loss = model.training_step(batch, 0)
    if loss is not None:
        print(f"G Loss: {loss.item():.4f}")

    sr = model.inference(
        torch.randn(1, 3, 48, 48, device=device),
        torch.randn(1, 3, H, W, device=device),
        seed=42,
    )
    print(f"Inference output: {sr.shape}")
    print("✓ Smoke test passed!")
