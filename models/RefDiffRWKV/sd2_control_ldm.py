"""
sd2_control_ldm.py — 双路径架构：语义 cross-attention + 纹理 skip connection
训练范式：单步 x0 预测 + 像素空间 loss（L2 + LPIPS + 双 GAN）
双判别器：语义 D（Haar+ConvNeXt）+ 纹理一致性 D（特征差值）
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
from discriminator import (
    ImageConvNextDiscriminator,
    TextureConsistencyDiscriminator,
)


class SD2ControlLDM(pl.LightningModule):
    def __init__(
        self,
        # ═════════════════════════════════════════════
        #  1. 数据键名
        # ═════════════════════════════════════════════
        lr_key: str = "lr",
        ref_key: str = "ref",
        hr_key: str = "hr",
        # ═════════════════════════════════════════════
        #  2. SD2 基础（冻结 backbone）
        # ═════════════════════════════════════════════
        sd_model_path: str = "sd2-community/stable-diffusion-2-1-base",
        use_lora: bool = True,
        lora_rank: int = 64,
        lora_target_modules: Optional[List[str]] = None,
        sd_locked: bool = True,
        # ═════════════════════════════════════════════
        #  3. SR prior / 预训练超分模型
        sr_model: Optional[nn.Module] = None,
        sr_fixed: bool = True,
        # ═════════════════════════════════════════════
        #  4. RefDiffRWKV — 纹理提取
        # ═════════════════════════════════════════════
        patch_size: int = 4,
        embed_dim: int = 384,
        upsample_mode: str = "bilinear",
        # ═════════════════════════════════════════════
        #  4. 语义路径
        # ═════════════════════════════════════════════
        use_semantic: bool = True,
        dinov2_model_name: str = "facebook/dinov2-base",
        cfg_drop_prob: float = 0.1,
        # ═════════════════════════════════════════════
        #  5. Noise Scheduler
        # ═════════════════════════════════════════════
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "epsilon",
        model_t: int = 200,
        # ═════════════════════════════════════════════
        #  6. Loss 权重
        # ═════════════════════════════════════════════
        l_simple_weight: float = 1.0,
        lambda_lpips: float = 0.2,
        lambda_gan: float = 0.2,
        lambda_gan_texture: float = 0.5,
        use_freq: bool = True,
        # ═════════════════════════════════════════════
        #  7. 训练控制
        # ═════════════════════════════════════════════
        learning_rate: float = 1e-4,
        lr_D: float = 5e-6,
        lr_D_texture: float = 1e-5,
        disc_trainable_stages: int = 1,
        use_amp: bool = True,
        weight_decay: float = 1e-3,
        debug_nan: bool = True,
        accumulate_grad_batches: int = 8,
        # ═════════════════════════════════════════════
        #  8. 验证 / 推理
        # ═════════════════════════════════════════════
        sample_steps: int = 50,
        fr_metrics: Optional[List[str]] = None,
        iqa_device: str = "cpu",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["sr_model"])

        # ── 数据键 / 基础开关 ──
        self.lr_key = lr_key
        self.ref_key = ref_key
        self.hr_key = hr_key
        self.sd_locked = sd_locked
        self.cfg_drop_prob = cfg_drop_prob

        # ── SR prior / 预训练超分模型 ──
        self.sr_model = sr_model
        self.sr_fixed = sr_fixed
        if self.sr_model is not None and self.sr_fixed:
            self.sr_model.eval()
            self.sr_model.requires_grad_(False)

        # ── 训练超参 ──
        self.learning_rate = learning_rate
        self.lr_D = lr_D
        self.lr_D_texture = lr_D_texture
        self.disc_trainable_stages = disc_trainable_stages
        self.weight_decay = weight_decay
        self.use_amp = use_amp
        self.debug_nan = debug_nan
        self.accumulate_grad_batches = accumulate_grad_batches

        # ── Loss 权重 ──
        self.l_simple_weight = l_simple_weight
        self.lambda_lpips = lambda_lpips
        self.lambda_gan = lambda_gan
        self.lambda_gan_texture = lambda_gan_texture
        self.use_freq = use_freq

        # ── 推理配置 ──
        self.model_t = model_t
        self.sample_steps = sample_steps
        self.fr_metrics = fr_metrics or ["psnr", "ssim", "lpips", "dists"]
        self.iqa_device = iqa_device

        # ── NaN 追踪 ──
        self._nan_count = 0

        # ══════════════════════════════════════════════
        #  AMP / 手动优化 / 梯度累积
        #  三个独立 GradScaler：各自独立管理 scale factor，
        #  避免多 optimizer 共用时 update() 被多次调用的副作用
        # ══════════════════════════════════════════════
        self.scaler_g = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.scaler_d_sem = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.scaler_d_tex = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.automatic_optimization = False
        self._g_accum_count = 0
        self._d_sem_accum_count = 0
        self._d_tex_accum_count = 0

        # G/D 交替相位: 0=Generator, 1=Discriminator
        # 每个 phase 持续 accumulate_grad_batches 个 training_step 调用，
        # 直到对应 optimizer 真正 step 后才切换到另一 phase
        self._gd_phase = 0

        # ══════════════════════════════════════════════
        #  1. VAE（冻结）
        # ══════════════════════════════════════════════
        self.vae = AutoencoderKL.from_pretrained(
            sd_model_path, subfolder="vae", local_files_only=True
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae_scale_factor = self.vae.config.scaling_factor

        # ══════════════════════════════════════════════
        #  2. UNet + LoRA（可训练 attention）
        # ══════════════════════════════════════════════
        self.unet = UNet2DConditionModel.from_pretrained(
            sd_model_path, subfolder="unet", local_files_only=True
        )
        self.unet.enable_gradient_checkpointing()
        if use_lora:
            self._inject_lora(lora_rank, lora_target_modules)
        if sd_locked:
            self._freeze_unet_except_attn()
        self.cross_attn_dim = self.unet.config.cross_attention_dim

        # ══════════════════════════════════════════════
        #  3. Noise Scheduler
        # ══════════════════════════════════════════════
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )

        # ══════════════════════════════════════════════
        #  4. GlobalSemantic — 语义路径（DINOv2 → RWKV）
        #      冻结 backbone，仅通过 sem_proj 学习投影
        # ══════════════════════════════════════════════
        self.global_semantic = (
            GlobalSemanticModule(dinov2_model_name=dinov2_model_name)
            if use_semantic
            else None
        )
        if self.global_semantic is not None:
            self.global_semantic.eval()
            self.global_semantic.requires_grad_(False)
        self.sem_proj: Optional[nn.Linear] = None

        # ══════════════════════════════════════════════
        #  5. RefDiffRWKV — 纹理路径
        # ══════════════════════════════════════════════
        self.ref_model = RefDiffRWKV(
            patch_size=patch_size,
            embed_dim=embed_dim,
            channels=3,
            upsample_mode=upsample_mode,
        )

        # ══════════════════════════════════════════════
        #  6. Adapter — ref_dims → sd2_dims
        # ══════════════════════════════════════════════
        self.ref_adapter = RWKV_Ref_Adapter(
            ref_dims=(384, 768, 1536),
            sd2_dims=(320, 640, 1280),
        )

        # ══════════════════════════════════════════════
        #  7. LPIPS（冻结权重，不调用 eval 避免被 Lightning 误判）
        # ══════════════════════════════════════════════
        self.net_lpips = lpips.LPIPS(net="vgg", verbose=False)
        # 只冻结权重，不调用 .eval()，避免被 PyTorch Lightning 列入
        # "eval mode modules" 警告，且 LPIPS 内部无 BN/Dropout，train/eval 模式无区别
        for param in self.net_lpips.parameters():
            param.requires_grad = False

        # ══════════════════════════════════════════════
        #  8. 双判别器（GAN）
        # ══════════════════════════════════════════════
        self.D = ImageConvNextDiscriminator(
            precision="fp32",
            use_freq=self.use_freq,
            trainable_stages=self.disc_trainable_stages,
        )

        self.D_texture = TextureConsistencyDiscriminator(
            in_ch=3,
            base_ch=48,
            num_scales=4,
            use_spectral=True,
        )

        # ══════════════════════════════════════════════
        #  9. IQA 引擎
        # ══════════════════════════════════════════════
        from RefRWKV.evaluation.eval_pyiqa import IQAEngine

        self.iqa = IQAEngine(
            device=iqa_device,
            nr_metrics=[],
            fr_metrics=self.fr_metrics,
            use_y_channel=True,
            verbose=False,
        )

        # ══════════════════════════════════════════════
        #  10. UNet hooks — 纹理 skip connection 注入点
        # ══════════════════════════════════════════════
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
        return self.vae.decode(z / self.vae_scale_factor).sample

    def _decode_latent_train(self, z):
        return self.vae.decode(z / self.vae_scale_factor).sample

    def _empty_context(self, B, device):
        return torch.zeros(
            B,
            77,
            self.cross_attn_dim,
            device=device,
            dtype=torch.float32,
        )

    # ════════════════════════════════════════════════════════
    #  Ref 特征提取 — 纹理 + 语义两路独立
    # ════════════════════════════════════════════════════════

    def _extract_ref(self, x_t_pixel, lr, ref):
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
        with torch.no_grad():
            sem_pyramid = self.global_semantic(ref)
        sem_tokens = self._build_sem_tokens(sem_pyramid)
        return sem_tokens

    def _extract_ref_static(self, lr, ref, latent_shape, x_t_pixel=None):
        B, _, H, W = latent_shape

        if x_t_pixel is None:
            with torch.no_grad():
                dummy = torch.zeros(B, 4, H, W, device=self.device)
                x_t_pixel = self.vae.decode(dummy / self.vae_scale_factor).sample.clamp(
                    -1, 1
                )

        if self._is_zero_image(ref):
            rf_feats = self._zero_ref_feats(x_t_pixel, lr)
        else:
            rf_feats = self._extract_ref(x_t_pixel, lr, ref)

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
            self.sem_proj = nn.Linear(base_dim, self.cross_attn_dim).to(
                first_tokens.device
            )
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
    #  UNet forward helper（G/D 共享）
    # ════════════════════════════════════════════════════════

    def _run_unet(self, lr, ref, hr):
        if self.sr_model is not None:
            if self.sr_fixed:
                with torch.no_grad():
                    sr_prior = self.sr_model(lr.to(self.device), ref.to(self.device))
            else:
                sr_prior = self.sr_model(lr.to(self.device), ref.to(self.device))
            x0 = self.encode_latent(sr_prior)
        else:
            with torch.amp.autocast("cuda", enabled=False):
                x0 = self.encode_latent(hr.to(self.device))
        B = x0.shape[0]

        t = torch.full((B,), self.model_t, device=self.device).long()
        noise = torch.randn_like(x0)
        x_t = self.noise_scheduler.add_noise(x0, noise, t)

        if self._check_tensor(x_t, "x_t (noisy latent)"):
            return None, None, None

        with torch.amp.autocast("cuda", enabled=False):
            with torch.no_grad():
                x_t_pixel = self.vae.decode(x_t / self.vae_scale_factor).sample.clamp(
                    -1, 1
                )

        ref_input = ref.to(self.device)
        if self.cfg_drop_prob > 0:
            drop_mask = torch.rand(B, device=self.device) < self.cfg_drop_prob
            if drop_mask.any():
                ref_input = ref_input.clone()
                ref_input[drop_mask] = 0.0

        rf_feats, sem_tokens = self._extract_ref_static(
            lr.to(self.device), ref_input, x0.shape, x_t_pixel=x_t_pixel
        )
        for i, f in enumerate(rf_feats):
            if self._check_tensor(f, f"ref_feat[{i}]"):
                return None, None, None

        context = self._build_context(B, self.device, sem_tokens)
        self._inject_ref_feats(rf_feats)

        noise_pred = self.unet(
            x_t,
            t,
            encoder_hidden_states=context,
        ).sample

        if self._check_tensor(noise_pred, "noise_pred (UNet output)"):
            return None, None, None

        pred_x0 = self.noise_scheduler.step(
            noise_pred, self.model_t, x_t
        ).pred_original_sample

        return pred_x0, noise_pred, noise

    # ════════════════════════════════════════════════════════
    #  Training — G/D 交替（手动优化 + AMP + 梯度累积）
    #
    #  使用 _gd_phase 控制交替，而非 trainer.global_step % 2。
    #  原因：trainer.global_step 每次 training_step 调用就 +1，
    #  但真正 optimizer.step() 每 accumulate_grad_batches 次才执行一次。
    #  用 % 2 会导致同一个梯度累积窗口内混合 G/D 两种 step，
    #  各自只积累了不到 accumulate_grad_batches 的梯度，永远不会触发 step。
    #
    #  现在的逻辑：
    #    _gd_phase=0 → 连续 accumulate_grad_batches 次 G step → 触发 G 更新 → 切到 D
    #    _gd_phase=1 → 连续 accumulate_grad_batches 次 D step → 触发 D 更新 → 切到 G
    # ════════════════════════════════════════════════════════

    def training_step(self, batch, batch_idx):
        g_opt, d_sem_opt, d_tex_opt = self.optimizers()

        # 用 phase 控制，而非 trainer.global_step % 2
        try:
            if self._gd_phase == 0:
                return self._generator_step(batch, batch_idx, g_opt)
            else:
                return self._discriminator_step(batch, batch_idx, d_sem_opt, d_tex_opt)
        finally:
            self._clear_ref_feats()

    # ════════════════════════════════════════════════════════
    #  Generator Step
    # ════════════════════════════════════════════════════════

    def _generator_step(self, batch, batch_idx, g_opt):
        """Generator step: L2 + LPIPS + 语义GAN + 纹理一致性GAN → 更新 G。"""
        lr = batch[self.lr_key].float()
        ref = batch[self.ref_key].float()
        hr = batch[self.hr_key].float()

        if (
            self._check_tensor(lr, "input lr")
            or self._check_tensor(ref, "input ref")
            or self._check_tensor(hr, "input hr")
        ):
            return None

        # ── 冻结所有 D ──
        self.D.eval()
        self.D.requires_grad_(False)
        self.D_texture.eval()
        self.D_texture.requires_grad_(False)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            # ── UNet 前向 ──
            pred_x0, noise_pred, noise = self._run_unet(lr, ref, hr)
            if pred_x0 is None:
                return None

            # ── latent → 像素空间 ──
            x_pred = self._decode_latent_train(pred_x0.float())
            if self._check_tensor(x_pred, "x_pred (pixel)"):
                return None

            # ── L2 + LPIPS ──
            hr_gpu = hr.to(self.device)
            loss_l2 = F.mse_loss(x_pred.float(), hr_gpu) * self.l_simple_weight
            loss_lpips = (
                self.net_lpips(x_pred.float(), hr_gpu).mean() * self.lambda_lpips
            )

        # ── GAN loss：D 全程 fp32，在 autocast 外跑 ──
        with torch.amp.autocast("cuda", enabled=False):
            # 语义 GAN
            loss_gan = self.D(x_pred.float(), for_G=True).mean() * self.lambda_gan

            # 纹理一致性 GAN
            ref_resized = F.interpolate(
                ref.to(self.device), size=x_pred.shape[2:], mode="bilinear"
            )
            fake_logit, _ = self.D_texture(x_pred.float(), ref_resized)
            loss_gan_tex = -fake_logit.mean() * self.lambda_gan_texture

        loss_G = (
            loss_l2 + loss_lpips + loss_gan + loss_gan_tex
        ) / self.accumulate_grad_batches

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
        self.scaler_g.scale(loss_G).backward()
        self._g_accum_count += 1

        if self._g_accum_count >= self.accumulate_grad_batches:
            self.scaler_g.unscale_(g_opt)
            self.clip_gradients(
                g_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
            )
            self.scaler_g.step(g_opt)
            self.scaler_g.update()
            g_opt.zero_grad()
            self._g_accum_count = 0
            # G 权重已更新，切换到 D phase
            self._gd_phase = 1

        # ── log ──
        self.log_dict(
            {
                "train/G_total": loss_G * self.accumulate_grad_batches,
                "train/G_l2": loss_l2,
                "train/G_lpips": loss_lpips,
                "train/G_gan_sem": loss_gan,
                "train/G_gan_tex": loss_gan_tex,
            },
            prog_bar=True,
            on_step=True,
        )

        return loss_G.detach() * self.accumulate_grad_batches

    # ════════════════════════════════════════════════════════
    #  Discriminator Step（双 D 顺序更新，各自独立 GradScaler）
    # ════════════════════════════════════════════════════════

    def _discriminator_step(self, batch, batch_idx, d_sem_opt, d_tex_opt):
        """Discriminator step: 先生成 x_pred（一次），再顺序更新两个 D。

        梯度累积各自独立计数，累积满 self.accumulate_grad_batches 次后执行 step。
        每个 D 使用独立的 GradScaler，避免 scale factor 互相干扰。
        """
        lr = batch[self.lr_key].float()
        ref = batch[self.ref_key].float()
        hr = batch[self.hr_key].float()

        if self._check_tensor(hr, "input hr (D step)"):
            return None

        # ── 无梯度生成 x_pred（两个 D 共享）──
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            with torch.no_grad():
                pred_x0, _, _ = self._run_unet(lr, ref, hr)
                if pred_x0 is None:
                    return None
                x_pred = self.decode_latent(pred_x0.float())

        hr_gpu = hr.to(self.device)
        ref_resized = F.interpolate(
            ref.to(self.device), size=hr_gpu.shape[2:], mode="bilinear"
        )

        # ════════════════════════════════════════════
        #  Part 1: 语义 D（Haar + ConvNeXt）
        # ════════════════════════════════════════════
        self.D.train()
        self.D.requires_grad_(True)
        self.D_texture.eval()
        self.D_texture.requires_grad_(False)

        with torch.amp.autocast("cuda", enabled=False):
            loss_D_sem_real = self.D(hr_gpu, for_real=True).mean()
            loss_D_sem_fake = self.D(x_pred.float(), for_real=False).mean()
            loss_D_sem = (
                loss_D_sem_real + loss_D_sem_fake
            ) / self.accumulate_grad_batches

        self.scaler_d_sem.scale(loss_D_sem).backward()
        self._d_sem_accum_count += 1

        if self._d_sem_accum_count >= self.accumulate_grad_batches:
            self.scaler_d_sem.unscale_(d_sem_opt)
            self.clip_gradients(
                d_sem_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
            )
            self.scaler_d_sem.step(d_sem_opt)
            self.scaler_d_sem.update()
            d_sem_opt.zero_grad()
            self._d_sem_accum_count = 0

        # Part 1 完成后立即冻结语义 D
        self.D.eval()
        self.D.requires_grad_(False)

        # ════════════════════════════════════════════
        #  Part 2: 纹理一致性 D（特征差值）
        # ════════════════════════════════════════════
        self.D_texture.train()
        self.D_texture.requires_grad_(True)

        with torch.amp.autocast("cuda", enabled=False):
            real_logit, _ = self.D_texture(hr_gpu, ref_resized)
            fake_logit, _ = self.D_texture(x_pred.float(), ref_resized)
            loss_D_tex = (
                F.relu(1.0 - real_logit).mean() + F.relu(1.0 + fake_logit).mean()
            ) / self.accumulate_grad_batches

        self.scaler_d_tex.scale(loss_D_tex).backward()

        # ── 记录 D_texture 梯度范数（调试用）──
        with torch.no_grad():
            d_tex_params = [
                p for p in self.D_texture.parameters() if p.grad is not None
            ]
            if d_tex_params:
                d_tex_grad_norm = torch.norm(
                    torch.cat([p.grad.flatten() for p in d_tex_params])
                )
                self.log("train/D_tex_grad_norm", d_tex_grad_norm, on_step=True)

        self._d_tex_accum_count += 1

        if self._d_tex_accum_count >= self.accumulate_grad_batches:
            self.scaler_d_tex.unscale_(d_tex_opt)
            self.clip_gradients(
                d_tex_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
            )
            self.scaler_d_tex.step(d_tex_opt)
            self.scaler_d_tex.update()
            d_tex_opt.zero_grad()
            self._d_tex_accum_count = 0
            # 纹理 D 也已完成一次更新，两个 D 都不会再更新了 → 切回 G phase
            self._gd_phase = 0

        # Part 2 完成后冻结纹理 D（为下一个 G step 做准备）
        self.D_texture.eval()
        self.D_texture.requires_grad_(False)

        # ── log ──
        self.log_dict(
            {
                "train/D_sem_total": loss_D_sem * self.accumulate_grad_batches,
                "train/D_sem_real": loss_D_sem_real,
                "train/D_sem_fake": loss_D_sem_fake,
                "train/D_tex_total": loss_D_tex * self.accumulate_grad_batches,
                "train/D_tex_real": real_logit.mean(),
                "train/D_tex_fake": fake_logit.mean(),
            },
            prog_bar=True,
            on_step=True,
        )

        loss_D_total = (loss_D_sem + loss_D_tex) * self.accumulate_grad_batches
        self.log("train/D_total", loss_D_total, prog_bar=True, on_step=True)
        return loss_D_total.detach()

    # ════════════════════════════════════════════════════════
    #  Validation（像素空间 L2，与训练目标一致）
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

        # ── 像素空间 L2（与训练目标 x0 预测一致）──
        with torch.no_grad():
            x_pred = self.decode_latent(pred_x0.float())
        val_l2 = F.mse_loss(x_pred, hr.to(self.device))

        self.log("val/loss", val_l2, on_epoch=True)
        self.log("val_loss", val_l2, on_epoch=True)

        if batch_idx == 0:
            self._validate_iqa(lr, ref, hr)

        return val_l2

    # ════════════════════════════════════════════════════════
    #  Optimizer（三个优化器 + CosineAnnealingLR）
    # ════════════════════════════════════════════════════════

    def configure_optimizers(self):
        self._ensure_sem_proj()

        # ── Generator 参数 ──
        g_params = list(self.ref_model.parameters()) + list(
            self.ref_adapter.parameters()
        )
        if self.sem_proj is not None:
            g_params += list(self.sem_proj.parameters())
        g_params += [p for p in self.unet.parameters() if p.requires_grad]
        if self.sr_model is not None and not self.sr_fixed:
            g_params += [p for p in self.sr_model.parameters() if p.requires_grad]

        g_opt = torch.optim.AdamW(
            g_params, lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # ── 语义 D 参数 ──
        d_sem_params = list(filter(lambda p: p.requires_grad, self.D.parameters()))
        d_sem_opt = torch.optim.AdamW(
            d_sem_params, lr=self.lr_D, weight_decay=self.weight_decay
        )

        # ── 纹理一致性 D 参数 ──
        d_tex_params = list(self.D_texture.parameters())
        d_tex_opt = torch.optim.AdamW(
            d_tex_params, lr=self.lr_D_texture, weight_decay=self.weight_decay
        )

        # ── Generator 余弦退火调度器 ──
        g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            g_opt, T_max=100000, eta_min=self.learning_rate * 0.1
        )

        return [g_opt, d_sem_opt, d_tex_opt], [
            {"scheduler": g_scheduler, "interval": "step"},
        ]

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

        noise = torch.randn(B, 4, H, W, device=self.device)
        alpha_bar = self.noise_scheduler.alphas_cumprod[self.model_t]
        z_t = torch.sqrt(1 - alpha_bar) * noise
        x_t_pixel = self.decode_latent(z_t.float())

        rf_feats, sem_tokens = self._extract_ref_static(
            lr.to(self.device), ref.to(self.device), latent_shape, x_t_pixel=x_t_pixel
        )
        self._inject_ref_feats(rf_feats)
        context = self._build_context(B, self.device, sem_tokens)

        t_b = torch.full((B,), self.model_t, device=self.device).long()
        self.unet.eval()
        noise_pred = self.unet(
            z_t,
            t_b,
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

        g = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(
            1,
            4,
            th // 8,
            tw // 8,
            generator=g,
            device=self.device,
        )

        if self.sr_model is not None:
            if self.sr_fixed:
                with torch.no_grad():
                    sr_prior = self.sr_model(lr, ref)
            else:
                sr_prior = self.sr_model(lr, ref)
            x0 = self.encode_latent(sr_prior)
            t = torch.full((1,), self.model_t, device=self.device).long()
            x_t = self.noise_scheduler.add_noise(x0, noise, t)
            x_t_pixel = self.decode_latent(x_t / self.vae_scale_factor).sample
        else:
            alpha_bar = self.noise_scheduler.alphas_cumprod[self.model_t]
            z_t = torch.sqrt(1 - alpha_bar) * noise
            x_t = z_t
            x_t_pixel = self.decode_latent(z_t.float())

        rf_feats, sem_tokens = self._extract_ref_static(
            lr, ref, latent_shape, x_t_pixel=x_t_pixel
        )
        self._inject_ref_feats(rf_feats)
        context = self._build_context(1, self.device, sem_tokens)

        t_b = torch.full((1,), self.model_t, device=self.device).long()
        self.unet.eval()
        noise_pred = self.unet(
            x_t,
            t_b,
            encoder_hidden_states=context,
        ).sample
        z = self.noise_scheduler.step(
            noise_pred, self.model_t, x_t
        ).pred_original_sample

        self._clear_ref_feats()
        return ((self.decode_latent(z) + 1) / 2).clamp(0, 1)

    # ════════════════════════════════════════════════════════
    #  Checkpoint hooks（持久化计数器 + sem_proj 重建 + scaler 状态）
    # ════════════════════════════════════════════════════════

    def on_save_checkpoint(self, checkpoint):
        self._clear_ref_feats()
        # 语义 D 的冻结 backbone 不保存（节省磁盘空间）
        keys_to_pop = [k for k in checkpoint["state_dict"] if k.startswith("D.model.")]
        for k in keys_to_pop:
            del checkpoint["state_dict"][k]
        # 纹理一致性 D 全部参数都可训练，全部保存（~3M，不大）

        # 持久化梯度累积计数器
        checkpoint["g_accum_count"] = self._g_accum_count
        checkpoint["d_sem_accum_count"] = self._d_sem_accum_count
        checkpoint["d_tex_accum_count"] = self._d_tex_accum_count
        checkpoint["nan_count"] = self._nan_count
        # 持久化 G/D phase
        checkpoint["gd_phase"] = self._gd_phase

        # 持久化 GradScaler 内部状态（scale factor + growth 计数等）
        checkpoint["scaler_g"] = self.scaler_g.state_dict()
        checkpoint["scaler_d_sem"] = self.scaler_d_sem.state_dict()
        checkpoint["scaler_d_tex"] = self.scaler_d_tex.state_dict()

    def on_load_checkpoint(self, checkpoint):
        self._setup_unet_hooks()

        # ── 重建 sem_proj（解决延迟初始化与 resume 的时序冲突）──
        state_dict = checkpoint.get("state_dict", {})
        if "sem_proj.weight" in state_dict and self.sem_proj is None:
            w = state_dict["sem_proj.weight"]
            b = state_dict["sem_proj.bias"]
            self.sem_proj = nn.Linear(w.shape[1], w.shape[0])
            self.sem_proj.load_state_dict({"weight": w, "bias": b})
            self.sem_proj.to(self.device)
            print(f"[SemProj] 从 checkpoint 重建: {w.shape[1]} → {w.shape[0]}")
        elif self.sem_proj is not None and self.global_semantic is not None:
            print("[SemProj] 已从 checkpoint 恢复")

        # ── 恢复梯度累积计数器 ──
        counter_map = [
            ("g_accum_count", "_g_accum_count"),
            ("d_sem_accum_count", "_d_sem_accum_count"),
            ("d_tex_accum_count", "_d_tex_accum_count"),
            ("nan_count", "_nan_count"),
        ]
        restored = []
        for ckpt_key, attr_name in counter_map:
            if ckpt_key in checkpoint:
                setattr(self, attr_name, checkpoint[ckpt_key])
                restored.append(f"{attr_name}={checkpoint[ckpt_key]}")

        # 恢复 G/D phase
        if "gd_phase" in checkpoint:
            self._gd_phase = checkpoint["gd_phase"]
            restored.append(f"_gd_phase={self._gd_phase}")

        if restored:
            print(f"[AccumCounter] 恢复：{', '.join(restored)}")

        # ── 恢复 GradScaler 状态 ──
        for key in ["scaler_g", "scaler_d_sem", "scaler_d_tex"]:
            if key in checkpoint:
                getattr(self, key).load_state_dict(checkpoint[key])
