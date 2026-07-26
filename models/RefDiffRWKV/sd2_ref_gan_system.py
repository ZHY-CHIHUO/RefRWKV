"""
sd2_ref_gan_system.py — G/D 分离 + 交替训练系统

设计原则：
1. 持有 SD2RefGenerator 和 SD2RefDiscriminator；
2. 手动优化 + AMP + 梯度累积，按 phase 控制 G/D 交替；
3. G step 中扩散 loss 为主，GAN / LPIPS 为辅助；
4. D step 中用单步 pred_x0 生成 fake/real，更新判别器；
5. 所有进入判别器的图像统一保持在 [-1, 1] 值域。
6. 无 adapter 路径复用 generator UNet + 零残差注入，零额外显存开销。
"""

import os
import logging
import shutil
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import lpips
import numpy as np
from PIL import Image

from .sd2_ref_generator import SD2RefGenerator
from .sd2_ref_discriminator import SD2RefDiscriminator

logger = logging.getLogger(__name__)


class SD2RefGANSystem(LightningModule):
    def __init__(
        self,
        generator: SD2RefGenerator,
        discriminator: Optional[SD2RefDiscriminator] = None,
        lambda_gan_semantic: float = 0.3,
        lambda_gan_texture: float = 0.5,
        lambda_lpips: float = 0.3,
        lambda_diff_sr: float = 0.5,
        accumulate_grad_batches: int = 8,
        use_amp: bool = True,
        g_d_ratio: int = 1,
        g_lr: float = 1e-4,
        g_weight_decay: float = 1e-3,
        d_lr_sem: float = 5e-6,
        d_lr_tex: float = 1e-6,
        d_weight_decay: float = 1e-3,
        betas: tuple = (0.5, 0.999),
        sample_steps: int = 50,
        fr_metrics: Optional[List[str]] = None,
        sr_model: Optional[torch.nn.Module] = None,
        sr_fixed: bool = True,
        sr_lr: float = 1e-5,
        t_start: Optional[int] = None,
        guidance_scale: float = 0.0,
        t_stop: int = 200,
        grad_clip_val: float = 1.0,
        grad_warn_threshold: float = 100.0,
        max_consecutive_nan: int = 10,
        gan_enabled: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["generator", "discriminator", "sr_model"])

        self.generator = generator
        self.discriminator = discriminator
        self.sr_model = sr_model

        self.lambda_gan_semantic = lambda_gan_semantic
        self.lambda_gan_texture = lambda_gan_texture
        self.lambda_lpips = lambda_lpips
        self.lambda_diff_sr = lambda_diff_sr

        self.accumulate_grad_batches = accumulate_grad_batches
        self.sample_steps = sample_steps
        self.g_d_ratio = g_d_ratio

        self.t_start = t_start
        self.guidance_scale = guidance_scale
        self.t_stop = t_stop

        self.grad_clip_val = grad_clip_val
        self.grad_warn_threshold = grad_warn_threshold
        self.max_consecutive_nan = max_consecutive_nan

        self.sr_fixed = sr_fixed
        self.sr_lr = sr_lr

        self.gan_enabled = gan_enabled and discriminator is not None

        self._nan_g_count = 0
        self._nan_d_count = 0
        self._consecutive_nan_g = 0
        self._consecutive_nan_d = 0

        self.automatic_optimization = False

        self.use_amp = use_amp
        self.scaler_g = None

        self._g_accum_count = 0
        self._d_sem_accum_count = 0
        self._d_tex_accum_count = 0
        self._gd_phase = 0
        self._g_steps_since_d = 0
        self._opt_idx: dict = {}

        # LPIPS
        self.net_lpips = lpips.LPIPS(net="vgg", verbose=False)
        for p in self.net_lpips.parameters():
            p.requires_grad = False

        # IQA
        self.iqa = None
        try:
            from RefRWKV.evaluation.eval_pyiqa import IQAEngine

            self.iqa = IQAEngine(
                device="cuda" if torch.cuda.is_available() else "cpu",
                nr_metrics=[],
                fr_metrics=fr_metrics or ["psnr", "ssim", "lpips", "dists"],
                use_y_channel=True,
                verbose=False,
            )
        except (ImportError, RuntimeError) as e:
            logger.warning("IQA engine 不可用: %s", e)

    # ═══════════════════════════════════════════════════════
    #  Discriminator 冻结 / 解冻
    # ═══════════════════════════════════════════════════════

    def _freeze_discriminator(self):
        if self.discriminator is not None:
            self.discriminator.eval()
            self.discriminator.requires_grad_(False)

    def _unfreeze_discriminator(self):
        if self.discriminator is not None:
            self.discriminator.train()
            self.discriminator.requires_grad_(True)

    # ═══════════════════════════════════════════════════════
    #  构建全零 down_intrablock 残差（优化：使用 new_zeros）
    # ═══════════════════════════════════════════════════════

    def _build_zero_intrablock(self, x_input: torch.Tensor) -> List[torch.Tensor]:
        bsz, _, latent_h, latent_w = x_input.shape

        def _half(h, w):
            return (h + 1) // 2, (w + 1) // 2

        h0, w0 = latent_h, latent_w
        h1, w1 = _half(h0, w0)
        h2, w2 = _half(h1, w1)
        h3, w3 = _half(h2, w2)

        return [
            x_input.new_zeros(bsz, ch, th, tw)
            for (th, tw), ch in zip(
                [(h0, w0), (h1, w1), (h2, w2), (h3, w3)],
                [320, 640, 1280, 1280],
            )
        ]

    # ═══════════════════════════════════════════════════════
    #  公共 pred_x0 基础逻辑
    # ═══════════════════════════════════════════════════════

    def _pred_x0_base(
        self, latent, sr_latent_cond, t, noise, context, down_intrablock=None
    ):
        x_t = self.generator.noise_scheduler.add_noise(latent, noise, t)
        x_input = self.generator.concat_sr_latent(x_t, sr_latent_cond)

        eps_pred = self.generator.unet(
            x_input,
            t,
            encoder_hidden_states=context,
            down_intrablock_additional_residuals=(
                down_intrablock
                if down_intrablock is not None
                else self._build_zero_intrablock(x_input)
            ),
        ).sample

        pred_x0 = self.generator.predict_x0_from_eps(x_t, t, eps_pred)
        pred_x0 = torch.nan_to_num(pred_x0, nan=0.0, posinf=20.0, neginf=-20.0).clamp(
            -20.0, 20.0
        )
        return self.generator.decode_latent(pred_x0)

    # ═══════════════════════════════════════════════════════
    #  无 adapter 路径（用于生成 real target，始终 no_grad）
    # ═══════════════════════════════════════════════════════

    @torch.no_grad()
    def _no_adapter_pred_x0(self, hr, sr_latent_cond, t, noise):
        bsz = hr.shape[0]
        hr_latent = self.generator.encode_latent(hr)

        null_ctx = torch.zeros(
            bsz,
            self.generator.CROSS_ATTN_CTX_LEN,
            self.generator.cross_attn_dim,
            device=hr.device,
            dtype=torch.float32,
        )

        return self._pred_x0_base(
            latent=hr_latent,
            sr_latent_cond=(
                sr_latent_cond.detach() if sr_latent_cond is not None else None
            ),
            t=t,
            noise=noise,
            context=null_ctx,
            down_intrablock=None,
        )

    # ═══════════════════════════════════════════════════════
    #  有 adapter 路径（G step Phase2 需要梯度穿透到 UNet/Adapter）
    # ═══════════════════════════════════════════════════════

    def _adapter_pred_x0(self, lr, ref, sr_latent_precomputed, t, noise):
        latent_h, latent_w = sr_latent_precomputed.shape[2:]
        bsz = lr.shape[0]

        ref_feats = self.generator.adapter(lr, ref)

        sem_tokens = None
        if self.generator.use_semantic:
            # DINOv2 前向在 GlobalSemanticModule.forward 内部已有 no_grad 保护
            # proj 和 semantic_pyramid 需要梯度
            sem_pyramid = self.generator.global_semantic(ref)
            sem_tokens = self.generator.build_sem_tokens(sem_pyramid)

        context = self.generator.build_context(bsz, sem_tokens)

        down_intrablock = self.generator.build_down_intrablock(
            ref_feats, latent_h, latent_w
        )

        return self._pred_x0_base(
            latent=sr_latent_precomputed,
            sr_latent_cond=sr_latent_precomputed.detach(),
            t=t,
            noise=noise,
            context=context,
            down_intrablock=down_intrablock,
        )

    # ═══════════════════════════════════════════════════════
    #  优化器 / 状态持久化
    # ═══════════════════════════════════════════════════════

    def _get_opt(self, key: str):
        """安全获取优化器（兼容 PL 单优化器返回对象而非列表的行为）。"""
        opts = self.optimizers()
        if not isinstance(opts, list):
            return opts if key == "g" else None
        idx = self._opt_idx.get(key)
        return opts[idx] if idx is not None else None

    def _get_g_opt(self):
        return self._get_opt("g")

    def _get_d_sem_opt(self):
        return self._get_opt("d_sem")

    def _get_d_tex_opt(self):
        return self._get_opt("d_tex")

    def _get_sr_opt(self):
        return self._get_opt("sr")

    def configure_optimizers(self):
        opts = []

        # 收集 Generator 可训练参数，排除 sr_model（由独立 SR 优化器管理）
        sr_param_ids = set()
        if self.sr_model is not None:
            sr_param_ids = {id(p) for p in self.sr_model.parameters()}

        g_params = [
            p
            for p in self.generator.parameters()
            if p.requires_grad and id(p) not in sr_param_ids
        ]

        if not g_params:
            logger.warning("Generator 无可训练参数，使用占位参数")
            if not hasattr(self, "_g_dummy_param"):
                self._g_dummy_param = nn.Parameter(torch.zeros(1))
            g_params = [self._g_dummy_param]

        g_opt = torch.optim.AdamW(
            g_params,
            lr=self.hparams.g_lr,
            weight_decay=self.hparams.g_weight_decay,
        )
        self._opt_idx["g"] = len(opts)
        opts.append(g_opt)

        # SR 模型优化器（仅 sr_fixed=False 时）
        if (
            self.sr_model is not None
            and not self.sr_fixed
            and any(p.requires_grad for p in self.sr_model.parameters())
        ):
            sr_params = [p for p in self.sr_model.parameters() if p.requires_grad]
            sr_opt = torch.optim.AdamW(
                sr_params,
                lr=self.hparams.sr_lr,
                weight_decay=self.hparams.g_weight_decay,
            )
            self._opt_idx["sr"] = len(opts)
            opts.append(sr_opt)
            logger.info(
                "SR 优化器已创建 (lr=%.1e, params=%d)",
                self.hparams.sr_lr,
                sum(p.numel() for p in sr_params),
            )

        # Discriminator 优化器
        if self.discriminator is not None:
            if self.discriminator.use_semantic_d:
                ps = [
                    p for p in self.discriminator.D_sem.parameters() if p.requires_grad
                ]
                if ps:
                    d_opt = torch.optim.AdamW(
                        ps,
                        lr=self.hparams.d_lr_sem,
                        betas=self.hparams.betas,
                        weight_decay=self.hparams.d_weight_decay,
                    )
                    self._opt_idx["d_sem"] = len(opts)
                    opts.append(d_opt)
                else:
                    logger.warning("D_sem 无可训练参数，跳过")

            if self.discriminator.use_texture_d:
                ps = [
                    p for p in self.discriminator.D_tex.parameters() if p.requires_grad
                ]
                if ps:
                    d_opt = torch.optim.AdamW(
                        ps,
                        lr=self.hparams.d_lr_tex,
                        betas=self.hparams.betas,
                        weight_decay=self.hparams.d_weight_decay,
                    )
                    self._opt_idx["d_tex"] = len(opts)
                    opts.append(d_opt)
                else:
                    logger.warning("D_tex 无可训练参数，跳过")

        return opts

    # ═══════════════════════════════════════════════════════
    #  Checkpoint 持久化
    # ═══════════════════════════════════════════════════════

    def on_save_checkpoint(self, checkpoint):
        checkpoint.update(
            {
                "gd_phase": self._gd_phase,
                "g_accum_count": self._g_accum_count,
                "d_sem_accum_count": self._d_sem_accum_count,
                "d_tex_accum_count": self._d_tex_accum_count,
                "g_steps_since_d": self._g_steps_since_d,
            }
        )

    def on_load_checkpoint(self, checkpoint):
        # 强制归零：checkpoint 不保存 .grad，非边界恢复会导致提前 step
        saved_g = checkpoint.get("g_accum_count", 0)
        if saved_g != 0:
            logger.warning(
                "从非累积边界恢复 (g_accum=%d)，梯度已丢失，计数器归零", saved_g
            )
        self._gd_phase = 0
        self._g_accum_count = 0
        self._d_sem_accum_count = 0
        self._d_tex_accum_count = 0
        self._g_steps_since_d = 0

    def load_state_dict(self, state_dict, strict=True):
        # ★ Phase2 从 Phase1 checkpoint 恢复时，discriminator 权重不存在
        if self.discriminator is not None:
            disc_keys_in_ckpt = [
                k for k in state_dict if k.startswith("discriminator.")
            ]
            disc_keys_expected = list(self.discriminator.state_dict().keys())
            if disc_keys_expected and not disc_keys_in_ckpt:
                logger.info(
                    "Discriminator 权重不在 checkpoint 中（%d keys），使用随机初始化",
                    len(disc_keys_expected),
                )
                strict = False

        result = super().load_state_dict(state_dict, strict=strict)
        missing, unexpected = result.missing_keys, result.unexpected_keys

        # 只警告非 discriminator 的缺失 key
        if missing:
            non_disc = [k for k in missing if not k.startswith("discriminator.")]
            if non_disc:
                logger.warning("load_state_dict: %d missing keys", len(non_disc))
                for k in non_disc[:10]:
                    logger.warning("  - %s", k)
        if unexpected:
            logger.warning("load_state_dict: %d unexpected keys", len(unexpected))
            for k in unexpected[:10]:
                logger.warning("  - %s", k)
        if not missing and not unexpected:
            logger.info("load_state_dict: all keys matched")
        return result

    def _override_lr_on_resume(self):
        opts = self.optimizers()
        if not opts:
            return
        optimizers = opts if isinstance(opts, list) else [opts]

        old_g = None
        for pg in optimizers[self._opt_idx["g"]].param_groups:
            old_g = pg["lr"]
            pg["lr"] = self.hparams.g_lr
        if old_g is not None:
            logger.info("G LR: %.1e → %.1e", old_g, self.hparams.g_lr)

        sr_idx = self._opt_idx.get("sr")
        if sr_idx is not None:
            old_sr = None
            for pg in optimizers[sr_idx].param_groups:
                old_sr = pg["lr"]
                pg["lr"] = self.hparams.sr_lr
            if old_sr is not None:
                logger.info("SR LR: %.1e → %.1e", old_sr, self.hparams.sr_lr)

        for key, opt_key in [("d_sem", "d_lr_sem"), ("d_tex", "d_lr_tex")]:
            idx = self._opt_idx.get(key)
            if idx is not None:
                old = None
                for pg in optimizers[idx].param_groups:
                    old = pg["lr"]
                    pg["lr"] = getattr(self.hparams, opt_key)
                if old is not None:
                    logger.info(
                        "%s LR: %.1e → %.1e", key, old, getattr(self.hparams, opt_key)
                    )

    # ═══════════════════════════════════════════════════════
    #  梯度监控
    # ═══════════════════════════════════════════════════════

    def _monitor_grad_norms(self, optimizer, name: str):
        total_norm = (
            sum(
                p.grad.data.norm(2).item() ** 2
                for group in optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            )
            ** 0.5
        )
        if total_norm > self.grad_warn_threshold:
            logger.warning(
                "梯度爆炸警告 [%s]: grad_norm=%.2f > threshold=%.2f",
                name,
                total_norm,
                self.grad_warn_threshold,
            )
        return total_norm

    # ═══════════════════════════════════════════════════════
    #  SR latent 获取
    # ═══════════════════════════════════════════════════════

    @torch.no_grad()
    def _get_sr_latent_precomputed(self, lr, ref):
        """SR latent（无梯度，用于 D step 和 sr_fixed=True 的 G step）。"""
        if self.sr_model is None:
            return None
        with torch.amp.autocast(self.device.type, enabled=False):
            sr_pixel = self.sr_model(lr.float(), ref.float())
            sr_pixel = torch.nan_to_num(
                sr_pixel, nan=0.0, posinf=1.0, neginf=-1.0
            ).clamp(-1.0, 1.0)
            return self.generator.encode_latent(sr_pixel.to(self.generator.vae.dtype))

    def _get_sr_latent_with_grad(self, lr, ref):
        """SR latent（保留计算图，用于 sr_fixed=False 的 G step，梯度反传到 SR 模型）。

        绕过 _compute_sr_prior / encode_latent 的 no_grad，
        让梯度从 loss → UNet → latent → VAE → sr_pixel → SR 模型 完整反传。
        """
        if self.sr_model is None:
            return None
        with torch.amp.autocast(self.device.type, enabled=False):
            sr_pixel = self.sr_model(lr.float(), ref.float())
            sr_pixel = torch.nan_to_num(
                sr_pixel, nan=0.0, posinf=1.0, neginf=-1.0
            ).clamp(-1.0, 1.0)
            return self.generator.encode_latent_with_grad(
                sr_pixel.to(self.generator.vae.dtype)
            )

    # ═══════════════════════════════════════════════════════
    #  Early Stop
    # ═══════════════════════════════════════════════════════

    def _check_early_stop(self, is_g_step: bool):
        cnt = self._consecutive_nan_g if is_g_step else self._consecutive_nan_d
        if cnt >= self.max_consecutive_nan:
            logger.error(
                "连续 %d 步 NaN (%s)，自动停止训练",
                self.max_consecutive_nan,
                "G step" if is_g_step else "D step",
            )
            self.trainer.should_stop = True
            return True
        return False

    # ═══════════════════════════════════════════════════════
    #  Training Step 路由
    # ═══════════════════════════════════════════════════════

    def training_step(self, batch, batch_idx):
        # D 预热：仅在 GAN 启用时
        if self.gan_enabled and self.global_step < 3000:
            return self._discriminator_step(batch, batch_idx)
        # G/D 交替（仅 GAN 启用时切 D phase）
        if self._gd_phase == 0:
            return self._generator_step(batch, batch_idx)
        return self._discriminator_step(batch, batch_idx)

    # ═══════════════════════════════════════════════════════
    #  Generator Step
    # ═══════════════════════════════════════════════════════

    def _generator_step(self, batch, batch_idx):
        g_opt = self._get_g_opt()
        self._freeze_discriminator()

        lr, ref, hr = self.generator.get_input(batch)

        # ── Phase 1: 扩散 ε-prediction loss ──
        try:
            with torch.amp.autocast(
                self.device.type, enabled=self.use_amp, dtype=torch.bfloat16
            ):
                out = self.generator.forward(lr, ref, hr)
                loss = out["loss"]
        except (RuntimeError, TypeError, AttributeError) as e:
            err_msg = str(e)
            is_cuda_error = isinstance(e, RuntimeError) and (
                "CUDA" in err_msg or "cuda" in err_msg or "CUBLAS" in err_msg
            )
            if is_cuda_error:
                logger.warning(
                    "[G step] Phase1 CUDA 异常 (batch=%d): %s，跳过并重置",
                    batch_idx,
                    e,
                    exc_info=True,  # 记录完整 traceback
                )
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except RuntimeError:
                    pass  # CUDA 上下文已损坏，无法清理
                g_opt.zero_grad(set_to_none=True)
                self._g_accum_count = 0
                if self.sr_model is not None:
                    try:
                        self.sr_model.to(self.device)
                    except RuntimeError:
                        pass
                return None
            raise  # 非 CUDA 异常直接抛出，不掩盖 bug

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("[G step] Phase1 loss NaN/Inf, batch=%d", batch_idx)
            self._consecutive_nan_g += 1
            if self._check_early_stop(is_g_step=True):
                return None
            g_opt.zero_grad(set_to_none=True)
            self._g_accum_count = 0
            if self.gan_enabled:
                self._gd_phase = 1
            return None
        self._consecutive_nan_g = 0

        loss_phase1 = loss / self.accumulate_grad_batches
        if self.scaler_g is not None:
            self.scaler_g.scale(loss_phase1).backward()
        else:
            loss_phase1.backward()

        # ── Phase 2: 辅助 loss（diff_sr / LPIPS / GAN）──
        phase2_loss_val = None
        # 像素/感知 loss：不依赖判别器，系数 > 0 即启用
        aux_loss_enabled = self.sr_model is not None and (
            self.lambda_diff_sr > 0 or self.lambda_lpips > 0
        )
        # GAN loss：需要显式开关 + 判别器存在
        gan_active = self.gan_enabled and (
            self.lambda_gan_semantic > 0 or self.lambda_gan_texture > 0
        )

        if aux_loss_enabled or gan_active:
            bsz = lr.shape[0]
            try:
                with torch.amp.autocast(
                    self.device.type, enabled=self.use_amp, dtype=torch.bfloat16
                ):
                    if not self.sr_fixed:
                        sr_latent = self._get_sr_latent_with_grad(lr, ref)
                    else:
                        sr_latent = self._get_sr_latent_precomputed(lr, ref)

                    t_sr = torch.randint(
                        self.generator.t_min,
                        self.generator.t_max + 1,
                        (bsz,),
                        device=lr.device,
                        dtype=torch.long,
                    )
                    noise_sr = torch.randn_like(sr_latent)

                    pred_sr_pixel = self._adapter_pred_x0(
                        lr, ref, sr_latent, t_sr, noise_sr
                    )

                    phase2_loss = 0.0

                    # ── 像素 loss（不依赖判别器）──
                    if self.lambda_diff_sr > 0:
                        loss_diff_sr = F.mse_loss(pred_sr_pixel, hr)
                        phase2_loss = phase2_loss + self.lambda_diff_sr * loss_diff_sr
                        self.log("train/G_diff_sr", loss_diff_sr.detach(), on_step=True)

                    # ── 感知 loss（不依赖判别器）──
                    if self.lambda_lpips > 0:
                        loss_lpips_sr = (
                            self.net_lpips(pred_sr_pixel, hr).mean() * self.lambda_lpips
                        )
                        if not torch.isnan(loss_lpips_sr) and not torch.isinf(
                            loss_lpips_sr
                        ):
                            phase2_loss = phase2_loss + loss_lpips_sr
                            self.log(
                                "train/G_lpips", loss_lpips_sr.detach(), on_step=True
                            )
                        else:
                            self._nan_g_count += 1
                            logger.warning(
                                "[G step] LPIPS NaN/Inf (#%d)，跳过", self._nan_g_count
                            )

                    # ── GAN loss（需要 gan_enabled + 判别器）──
                    if gan_active:
                        with torch.amp.autocast(self.device.type, enabled=False):
                            gan_loss = self.discriminator.compute_g_loss(
                                pred_sr_pixel.float(),
                                ref=ref.float(),
                                lambda_semantic=self.lambda_gan_semantic,
                                lambda_texture=self.lambda_gan_texture,
                            )
                        if not torch.isnan(gan_loss) and not torch.isinf(gan_loss):
                            phase2_loss = phase2_loss + gan_loss
                            self.log("train/G_gan", gan_loss.detach(), on_step=True)
                        else:
                            self._nan_g_count += 1
                            logger.warning(
                                "[G step] GAN NaN/Inf (#%d)", self._nan_g_count
                            )

                    if (
                        isinstance(phase2_loss, torch.Tensor)
                        and phase2_loss.item() != 0
                    ):
                        phase2_loss_val = phase2_loss.detach()
                        phase2_loss_scaled = phase2_loss / self.accumulate_grad_batches
                        if self.scaler_g is not None:
                            self.scaler_g.scale(phase2_loss_scaled).backward()
                        else:
                            phase2_loss_scaled.backward()

            except (RuntimeError, TypeError, AttributeError) as e:
                err_msg = str(e)
                is_cuda_error = isinstance(e, RuntimeError) and (
                    "CUDA" in err_msg or "cuda" in err_msg or "CUBLAS" in err_msg
                )
                if is_cuda_error:
                    logger.warning(
                        "[G step] Phase2 CUDA/设备异常 (batch=%d): %s，跳过 Phase2",
                        batch_idx,
                        e,
                        exc_info=True,
                    )
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    except RuntimeError:
                        pass
                    if self.sr_model is not None:
                        try:
                            self.sr_model.to(self.device)
                        except RuntimeError:
                            pass
                else:
                    raise
        # ── 梯度累积 & Optimizer Step（无论是否有 Phase2 都执行）──
        self._g_accum_count += 1

        if self._g_accum_count >= self.accumulate_grad_batches:
            sr_opt = (
                self._get_sr_opt()
                if (not self.sr_fixed and (aux_loss_enabled or gan_active))
                else None
            )

            # 1. unscale 全部
            if self.scaler_g is not None:
                self.scaler_g.unscale_(g_opt)
                if sr_opt is not None:
                    self.scaler_g.unscale_(sr_opt)

            # 2. clip 全部
            self._monitor_grad_norms(g_opt, "G")
            self.clip_gradients(
                g_opt,
                gradient_clip_val=self.grad_clip_val,
                gradient_clip_algorithm="norm",
            )
            if sr_opt is not None:
                self._monitor_grad_norms(sr_opt, "SR")
                self.clip_gradients(
                    sr_opt,
                    gradient_clip_val=self.grad_clip_val,
                    gradient_clip_algorithm="norm",
                )

            # 3. step 全部 → 4. update 一次
            if self.scaler_g is not None:
                self.scaler_g.step(g_opt)
                if sr_opt is not None:
                    self.scaler_g.step(sr_opt)
                self.scaler_g.update()
            else:
                g_opt.step()
                if sr_opt is not None:
                    sr_opt.step()

            # 5. zero_grad 全部
            g_opt.zero_grad(set_to_none=True)
            if sr_opt is not None:
                sr_opt.zero_grad(set_to_none=True)

            self._g_accum_count = 0
            self._g_steps_since_d += 1

            if self.gan_enabled and self._g_steps_since_d >= self.g_d_ratio:
                self._gd_phase = 1
                self._g_steps_since_d = 0
                self._unfreeze_discriminator()

        # ── 日志 ──
        g_total = loss.detach() + (
            phase2_loss_val if phase2_loss_val is not None else 0.0
        )
        self.log("train/G_total", g_total, on_step=True, prog_bar=True)
        self.log("train/G_diff_hr", out["loss"].detach(), on_step=True, prog_bar=True)

        return g_total

    # ═══════════════════════════════════════════════════════
    #  Discriminator Step
    # ═══════════════════════════════════════════════════════

    def _discriminator_step(self, batch, batch_idx):
        if self.discriminator is None or (
            self.lambda_gan_semantic == 0.0 and self.lambda_gan_texture == 0.0
        ):
            self._gd_phase = 0
            return None

        if self.sr_model is None:
            self._gd_phase = 0
            return None

        self._unfreeze_discriminator()
        d_sem_opt, d_tex_opt = self._get_d_sem_opt(), self._get_d_tex_opt()

        lr, ref, hr = self.generator.get_input(batch)
        bsz = lr.shape[0]

        # ref NaN 检查
        if torch.isnan(ref).any() or torch.isinf(ref).any():
            self._nan_d_count += 1
            self._consecutive_nan_d += 1
            self._check_early_stop(is_g_step=False)
            logger.warning(
                "[D step] ref NaN/Inf (#%d), batch=%d", self._nan_d_count, batch_idx
            )
            if d_sem_opt is not None:
                d_sem_opt.zero_grad(set_to_none=True)
                self._d_sem_accum_count = 0
            if d_tex_opt is not None:
                d_tex_opt.zero_grad(set_to_none=True)
                self._d_tex_accum_count = 0
            self._freeze_discriminator()
            self._gd_phase = 0
            return None
        self._consecutive_nan_d = 0

        # 生成 fake / real（全部 no_grad，D 不需要 G 的梯度）

        try:
            with torch.no_grad():
                with torch.amp.autocast(
                    self.device.type, enabled=self.use_amp, dtype=torch.bfloat16
                ):
                    sr_latent = self._get_sr_latent_precomputed(lr, ref)
                    _num_t = self.generator.noise_scheduler.config.num_train_timesteps
                    t = torch.randint(
                        self.generator.t_min,
                        self.generator.t_max + 1,
                        (bsz,),
                        device=lr.device,
                        dtype=torch.long,
                    )
                    noise = torch.randn_like(sr_latent)
                    pred_hr_pixel = self._no_adapter_pred_x0(hr, sr_latent, t, noise)
                    pred_sr_pixel = self._adapter_pred_x0(lr, ref, sr_latent, t, noise)
                    real, fake = (
                        pred_hr_pixel.detach().float(),
                        pred_sr_pixel.detach().float(),
                    )
        except (RuntimeError, TypeError, AttributeError) as e:
            err_msg = str(e)
            is_cuda_error = isinstance(e, RuntimeError) and (
                "CUDA" in err_msg or "cuda" in err_msg or "CUBLAS" in err_msg
            )
            if is_cuda_error:
                logger.warning(
                    "[D step] CUDA 异常 (batch=%d): %s，跳过",
                    batch_idx,
                    e,
                    exc_info=True,
                )
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
                if d_sem_opt is not None:
                    d_sem_opt.zero_grad(set_to_none=True)
                if d_tex_opt is not None:
                    d_tex_opt.zero_grad(set_to_none=True)
                self._d_sem_accum_count = 0
                self._d_tex_accum_count = 0
                self._freeze_discriminator()
                self._gd_phase = 0
                return None
            raise

        for name, tensor in [("fake", fake), ("real", real)]:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                self._nan_d_count += 1
                self._consecutive_nan_d += 1
                self._check_early_stop(is_g_step=False)
                logger.warning("[D step] %s NaN/Inf (#%d)", name, self._nan_d_count)
                if d_sem_opt is not None:
                    d_sem_opt.zero_grad(set_to_none=True)
                    self._d_sem_accum_count = 0
                if d_tex_opt is not None:
                    d_tex_opt.zero_grad(set_to_none=True)
                    self._d_tex_accum_count = 0
                self._freeze_discriminator()
                self._gd_phase = 0
                return None
        self._consecutive_nan_d = 0

        sem_updated = tex_updated = False

        # ── 语义 D ──
        if (
            self.lambda_gan_semantic > 0
            and self.discriminator.use_semantic_d
            and d_sem_opt is not None
        ):
            with torch.amp.autocast(self.device.type, enabled=False):
                loss_d_sem = self.discriminator.compute_d_loss(
                    real, fake, ref=None, lambda_semantic=1.0, lambda_texture=0.0
                )

            if not torch.isnan(loss_d_sem) and not torch.isinf(loss_d_sem):
                (loss_d_sem / self.accumulate_grad_batches).backward()
                self._d_sem_accum_count += 1
                sem_updated = True
                self.log(
                    "train/D_sem", loss_d_sem.detach(), on_step=True, prog_bar=True
                )

                if self._d_sem_accum_count >= self.accumulate_grad_batches:
                    self._monitor_grad_norms(d_sem_opt, "D_sem")
                    self.clip_gradients(
                        d_sem_opt,
                        gradient_clip_val=self.grad_clip_val,
                        gradient_clip_algorithm="norm",
                    )
                    d_sem_opt.step()
                    d_sem_opt.zero_grad(set_to_none=True)
                    self._d_sem_accum_count = 0
            else:
                self._nan_d_count += 1
                logger.warning(
                    "[D step] loss_d_sem NaN/Inf (#%d)，跳过", self._nan_d_count
                )

        # ── 纹理 D ──
        if (
            self.lambda_gan_texture > 0
            and self.discriminator.use_texture_d
            and d_tex_opt is not None
        ):
            with torch.amp.autocast(self.device.type, enabled=False):
                loss_d_tex = self.discriminator.compute_d_loss(
                    real, fake, ref=ref, lambda_semantic=0.0, lambda_texture=1.0
                )

            if not torch.isnan(loss_d_tex) and not torch.isinf(loss_d_tex):
                (loss_d_tex / self.accumulate_grad_batches).backward()
                self._d_tex_accum_count += 1
                tex_updated = True
                self.log(
                    "train/D_tex", loss_d_tex.detach(), on_step=True, prog_bar=True
                )

                if self._d_tex_accum_count >= self.accumulate_grad_batches:
                    self._monitor_grad_norms(d_tex_opt, "D_tex")
                    self.clip_gradients(
                        d_tex_opt,
                        gradient_clip_val=self.grad_clip_val,
                        gradient_clip_algorithm="norm",
                    )
                    d_tex_opt.step()
                    d_tex_opt.zero_grad(set_to_none=True)
                    self._d_tex_accum_count = 0
            else:
                self._nan_d_count += 1
                logger.warning(
                    "[D step] loss_d_tex NaN/Inf (#%d)，跳过", self._nan_d_count
                )

        # 未参与更新的优化器清零
        if d_sem_opt is not None and not sem_updated:
            d_sem_opt.zero_grad(set_to_none=True)
            self._d_sem_accum_count = 0
        if d_tex_opt is not None and not tex_updated:
            d_tex_opt.zero_grad(set_to_none=True)
            self._d_tex_accum_count = 0

        # 判断是否切回 G phase
        d_sem_done = (
            not self.discriminator.use_semantic_d or self._d_sem_accum_count == 0
        )
        d_tex_done = (
            not self.discriminator.use_texture_d or self._d_tex_accum_count == 0
        )

        if d_sem_done and d_tex_done:
            self._freeze_discriminator()
            self._gd_phase = 0
            self._g_steps_since_d = 0

        return None

    # ═══════════════════════════════════════════════════════
    #  验证 / 推理
    # ═══════════════════════════════════════════════════════

    def validation_step(self, batch, batch_idx):
        import traceback

        try:
            lr, ref, hr = self.generator.get_input(batch)
            loss_diff, _ = self.generator.p_losses(lr, ref, hr)

            self.log(
                "val/loss_diff", loss_diff, on_step=False, on_epoch=True, prog_bar=True
            )
            self.log(
                "val_loss_diff", loss_diff, on_step=False, on_epoch=True, prog_bar=True
            )

            with torch.no_grad():
                val_results = self.generator.log_images(
                    batch,
                    steps=self.sample_steps,
                    sr_model=self.sr_model,
                    t_start=self.t_start,
                    guidance_scale=self.guidance_scale,
                    t_stop=self.t_stop,
                    val_seed=42,
                )

            if self.iqa is not None:
                sr_batch, hq_batch = val_results["samples"], val_results["hq"]
                agg = {}
                iqa_failures = 0
                for i in range(len(sr_batch)):
                    try:
                        m = self.iqa.evaluate_single(
                            sr_batch[i].cpu().numpy(), hq_batch[i].cpu().numpy()
                        )
                        for k, v in m.items():
                            agg[k] = agg.get(k, 0.0) + v
                    except Exception as e:
                        iqa_failures += 1
                        logger.warning("IQA sample %d 失败: %s", i, e)
                n = len(sr_batch) - iqa_failures
                if n > 0:
                    for k, v in agg.items():
                        self.log(f"val/{k}", v / n, on_epoch=True, prog_bar=True)
                        self.log(f"val_{k}", v / n, on_epoch=True)
                else:
                    logger.error(
                        "IQA 全部失败 (%d/%d)，本 epoch 无有效指标",
                        iqa_failures,
                        len(sr_batch),
                    )

            if batch_idx == 0 and self.logger is not None:
                self._save_validation_images(val_results, lr, ref, hr)

            del val_results, lr, ref, hr
            return loss_diff

        except torch.cuda.OutOfMemoryError:
            logger.warning("validation_step OOM (batch=%d)，跳过", batch_idx)
            torch.cuda.empty_cache()
            return None
        except Exception as e:
            logger.error(
                "validation_step 异常 (batch=%d): %s\n%s",
                batch_idx,
                e,
                traceback.format_exc(),
            )
            raise  # 非 OOM 异常不吞掉，让训练者知道

    def _save_validation_images(self, val_results, lr, ref, hr):
        if self.logger is None:
            return

        save_dir = os.path.join(self.logger.log_dir, "validation_tmp")
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            with torch.amp.autocast(self.device.type, enabled=False):
                sr_prior = (
                    self.sr_model(lr.float(), ref.float())
                    if self.sr_model is not None
                    else None
                )
                if sr_prior is not None:
                    sr_prior = torch.nan_to_num(
                        sr_prior, nan=0.0, posinf=1.0, neginf=-1.0
                    ).clamp(-1.0, 1.0)

        images_to_concat = []
        for image_key in ("lq", "ref", "hq", "samples"):
            if image_key not in val_results:
                continue
            img = val_results[image_key][0]
            pil_img = Image.fromarray(
                (img.detach().cpu().permute(1, 2, 0).numpy() * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )
            if image_key == "lq":
                target_size = val_results["samples"].shape[-2:]
                pil_img = pil_img.resize(
                    (target_size[1], target_size[0]), Image.NEAREST
                )
            images_to_concat.append(pil_img)

        if sr_prior is not None:
            sr_img = ((sr_prior + 1.0) / 2.0)[0].detach().cpu().permute(1, 2, 0).numpy()
            sr_img = (
                (np.nan_to_num(sr_img, nan=0.0, posinf=1.0, neginf=0.0) * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )
            images_to_concat.append(Image.fromarray(sr_img))

        if images_to_concat:
            total_w = sum(im.width for im in images_to_concat)
            max_h = max(im.height for im in images_to_concat)
            combined = Image.new("RGB", (total_w, max_h))
            x_offset = 0
            for im in images_to_concat:
                combined.paste(im, (x_offset, 0))
                x_offset += im.width
            combined.save(os.path.join(save_dir, f"step_{self.global_step}.png"))

    # ═══════════════════════════════════════════════════════
    #  Epoch 钩子
    # ═══════════════════════════════════════════════════════

    def on_train_epoch_start(self):
        # 强制冻结模块保持 eval（防止 Lightning 递归 .train() 打开 DropPath）
        if self.sr_model is not None and self.sr_fixed:
            self.sr_model.eval()
        if self.generator.global_semantic is not None:
            # 只冻结 DINOv2 backbone，proj 和 pyramid 保持 train 模式
            self.generator.global_semantic.dinov2.eval()
        if self.generator.vae is not None:
            self.generator.vae.eval()

    def on_validation_epoch_start(self):
        self._freeze_discriminator()
        if self.logger is not None:
            save_dir = os.path.join(self.logger.log_dir, "validation_tmp")
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
                logger.info("已清理 validation_tmp 目录")

    def on_validation_epoch_end(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_train_start(self):
        if self.sr_model is not None:
            self.sr_model.to(self.device)
        self._override_lr_on_resume()
