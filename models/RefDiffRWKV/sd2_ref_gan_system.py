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
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from typing import Optional, List, Dict
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
        # loss 权重
        lambda_gan_semantic: float = 0.3,
        lambda_gan_texture: float = 0.5,
        lambda_lpips: float = 0.3,
        lambda_diff_sr: float = 0.5,
        # 训练控制
        accumulate_grad_batches: int = 8,
        use_amp: bool = True,
        # G:D 训练比例（修复：新增 g_d_ratio）
        g_d_ratio: int = 1,
        # 优化器
        g_lr: float = 1e-4,
        g_weight_decay: float = 1e-3,
        d_lr_sem: float = 5e-6,
        d_lr_tex: float = 1e-6,
        d_weight_decay: float = 1e-3,
        betas: tuple = (0.5, 0.999),
        # 验证
        sample_steps: int = 50,
        fr_metrics: Optional[List[str]] = None,
        # Better Start
        sr_model: Optional[torch.nn.Module] = None,
        sr_fixed: bool = True,
        t_start: Optional[int] = None,
        guidance_scale: float = 0.0,
        t_stop: int = 200,
        # 梯度监控
        grad_clip_val: float = 1.0,
        grad_warn_threshold: float = 100.0,
        # NaN 自动终止（修复：新增）
        max_consecutive_nan: int = 10,
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

        self._nan_g_count = 0
        self._nan_d_count = 0
        self._consecutive_nan_g = 0  # 修复：连续 NaN 计数
        self._consecutive_nan_d = 0

        # 手动优化
        self.automatic_optimization = False
        self.use_amp = use_amp
        if use_amp and torch.cuda.is_available():
            self.scaler_g = torch.amp.GradScaler("cuda", enabled=True)
        else:
            self.scaler_g = None

        self._g_accum_count = 0
        self._d_sem_accum_count = 0
        self._d_tex_accum_count = 0
        self._gd_phase = 0
        self._g_steps_since_d = 0  # 修复：g_d_ratio 计数器

        self._opt_idx: Dict[str, int] = {}

        # LPIPS
        self.net_lpips = lpips.LPIPS(net="vgg", verbose=False)
        for p in self.net_lpips.parameters():
            p.requires_grad = False

        # IQA
        self.iqa = None
        try:
            from RefRWKV.evaluation.eval_pyiqa import IQAEngine

            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            self.iqa = IQAEngine(
                device=device_str,
                nr_metrics=[],
                fr_metrics=fr_metrics or ["psnr", "ssim", "lpips", "dists"],
                use_y_channel=True,
                verbose=False,
            )
        except (ImportError, RuntimeError) as e:
            logger.warning("IQA engine 不可用: %s", e)

    # ═══════════════════════════════════════════════════════
    #  构建全零 down_intrablock 残差
    # ═══════════════════════════════════════════════════════

    def _build_zero_intrablock(self, x_input: torch.Tensor) -> List[torch.Tensor]:
        _, _, latent_h, latent_w = x_input.shape

        def _half(h, w):
            return (h + 1) // 2, (w + 1) // 2

        h0, w0 = latent_h, latent_w
        h1, w1 = _half(h0, w0)
        h2, w2 = _half(h1, w1)
        h3, w3 = _half(h2, w2)

        target_sizes = [(h0, w0), (h1, w1), (h2, w2), (h3, w3)]
        channels = [320, 640, 1280, 1280]

        residuals = []
        for (th, tw), ch in zip(target_sizes, channels):
            residuals.append(
                torch.zeros(
                    x_input.shape[0],
                    ch,
                    th,
                    tw,
                    device=x_input.device,
                    dtype=x_input.dtype,
                )
            )
        return residuals

    # ═══════════════════════════════════════════════════════
    #  公共 pred_x0 基础逻辑
    # ═══════════════════════════════════════════════════════

    def _pred_x0_base(
        self,
        latent: torch.Tensor,
        sr_latent_cond: Optional[torch.Tensor],
        t: torch.Tensor,
        noise: torch.Tensor,
        context: torch.Tensor,
        down_intrablock: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        x_t = self.generator.noise_scheduler.add_noise(latent, noise, t)
        x_input = self.generator._concat_sr_latent(x_t, sr_latent_cond)

        if down_intrablock is None:
            zero_residuals = self._build_zero_intrablock(x_input)
            eps_pred = self.generator.unet(
                x_input,
                t,
                encoder_hidden_states=context,
                down_intrablock_additional_residuals=zero_residuals,
            ).sample
        else:
            eps_pred = self.generator.unet(
                x_input,
                t,
                encoder_hidden_states=context,
                down_intrablock_additional_residuals=down_intrablock,
            ).sample

        pred_x0 = self.generator._predict_x0_from_eps(x_t, t, eps_pred)
        pred_x0 = torch.nan_to_num(pred_x0, nan=0.0, posinf=20.0, neginf=-20.0).clamp(
            -20.0, 20.0
        )
        pred_pixel = self.generator.decode_latent(pred_x0)
        return pred_pixel

    # ═══════════════════════════════════════════════════════
    #  无 adapter 路径（HR 去噪标杆，修复：移除 empty_cache）
    # ═══════════════════════════════════════════════════════

    @torch.no_grad()
    def _no_adapter_pred_x0(self, hr, sr_latent_cond, t, noise):
        bsz = hr.shape[0]
        device = hr.device
        hr_latent = self.generator.encode_latent(hr)
        null_ctx = torch.zeros(
            bsz,
            77,
            self.generator.cross_attn_dim,
            device=device,
            dtype=torch.float32,
        )
        pred_pixel = self._pred_x0_base(
            latent=hr_latent,
            sr_latent_cond=(
                sr_latent_cond.detach() if sr_latent_cond is not None else None
            ),
            t=t,
            noise=noise,
            context=null_ctx,
            down_intrablock=None,
        )
        return pred_pixel

    # ═══════════════════════════════════════════════════════
    #  有 adapter 路径（SR 去噪，可梯度，修复：从 latent 形状推导尺寸）
    # ═══════════════════════════════════════════════════════

    def _adapter_pred_x0(self, lr, ref, sr_latent_precomputed, t, noise):
        bsz = lr.shape[0]
        # [FIX] 直接从 latent 形状推导尺寸，避免重复计算 x_t/x_input
        latent_h, latent_w = sr_latent_precomputed.shape[2:]

        ref_feats = self.generator.adapter(lr, ref)
        sem_tokens = None
        if self.generator.use_semantic:
            with torch.no_grad():
                sem_pyramid = self.generator.global_semantic(ref)
            sem_tokens = self.generator.build_sem_tokens(sem_pyramid)

        context = self.generator._build_context(bsz, sem_tokens)
        down_intrablock = self.generator._build_down_intrablock(
            ref_feats, latent_h, latent_w
        )

        pred_pixel = self._pred_x0_base(
            latent=sr_latent_precomputed,
            sr_latent_cond=sr_latent_precomputed.detach(),
            t=t,
            noise=noise,
            context=context,
            down_intrablock=down_intrablock,
        )
        return pred_pixel

    # ═══════════════════════════════════════════════════════
    #  优化器 / 状态持久化
    # ═══════════════════════════════════════════════════════

    def _get_g_opt(self):
        return self.optimizers()[self._opt_idx["g"]]

    def _get_d_sem_opt(self):
        idx = self._opt_idx.get("d_sem", None)
        return self.optimizers()[idx] if idx is not None else None

    def _get_d_tex_opt(self):
        idx = self._opt_idx.get("d_tex", None)
        return self.optimizers()[idx] if idx is not None else None

    def configure_optimizers(self):
        opts = []
        g_opt = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.hparams.g_lr,
            weight_decay=self.hparams.g_weight_decay,
        )
        self._opt_idx["g"] = len(opts)
        opts.append(g_opt)

        if self.discriminator is not None:
            if self.discriminator.use_semantic_d:
                d_sem_opt = torch.optim.AdamW(
                    [
                        p
                        for p in self.discriminator.D_sem.parameters()
                        if p.requires_grad
                    ],
                    lr=self.hparams.d_lr_sem,
                    betas=self.hparams.betas,
                    weight_decay=self.hparams.d_weight_decay,
                )
                self._opt_idx["d_sem"] = len(opts)
                opts.append(d_sem_opt)
            if self.discriminator.use_texture_d:
                d_tex_opt = torch.optim.AdamW(
                    list(self.discriminator.D_tex.parameters()),
                    lr=self.hparams.d_lr_tex,
                    betas=self.hparams.betas,
                    weight_decay=self.hparams.d_weight_decay,
                )
                self._opt_idx["d_tex"] = len(opts)
                opts.append(d_tex_opt)
        return opts

    def on_save_checkpoint(self, checkpoint):
        checkpoint["gd_phase"] = self._gd_phase
        checkpoint["g_accum_count"] = self._g_accum_count
        checkpoint["d_sem_accum_count"] = self._d_sem_accum_count
        checkpoint["d_tex_accum_count"] = self._d_tex_accum_count
        checkpoint["g_steps_since_d"] = self._g_steps_since_d

    def on_load_checkpoint(self, checkpoint):
        self._gd_phase = checkpoint.get("gd_phase", 0)
        self._g_accum_count = checkpoint.get("g_accum_count", 0)
        self._d_sem_accum_count = checkpoint.get("d_sem_accum_count", 0)
        self._d_tex_accum_count = checkpoint.get("d_tex_accum_count", 0)
        self._g_steps_since_d = checkpoint.get("g_steps_since_d", 0)

    def load_state_dict(self, state_dict, strict=True):
        result = super().load_state_dict(state_dict, strict=False)
        missing = getattr(result, "missing_keys", []) or []
        unexpected = getattr(result, "unexpected_keys", []) or []
        if missing:
            logger.warning("load_state_dict: %d missing keys", len(missing))
            for k in missing[:10]:
                logger.warning("  - %s", k)
            if len(missing) > 10:
                logger.warning("  ... and %d more", len(missing) - 10)
        if unexpected:
            logger.warning("load_state_dict: %d unexpected keys", len(unexpected))
            for k in unexpected[:10]:
                logger.warning("  - %s", k)
            if len(unexpected) > 10:
                logger.warning("  ... and %d more", len(unexpected) - 10)
        if not missing and not unexpected:
            logger.info("load_state_dict: all keys matched")
        return result

    def _override_lr_on_resume(self):
        optimizers = self.optimizers()
        if not optimizers:
            return
        target_g_lr = self.hparams.g_lr
        for pg in optimizers[self._opt_idx["g"]].param_groups:
            old = pg["lr"]
            pg["lr"] = target_g_lr
        logger.info("G LR: %.1e → %.1e", old, target_g_lr)

        idx_sem = self._opt_idx.get("d_sem")
        if idx_sem is not None:
            for pg in optimizers[idx_sem].param_groups:
                old = pg["lr"]
                pg["lr"] = self.hparams.d_lr_sem
            logger.info("D_sem LR: %.1e → %.1e", old, self.hparams.d_lr_sem)

        idx_tex = self._opt_idx.get("d_tex")
        if idx_tex is not None:
            for pg in optimizers[idx_tex].param_groups:
                old = pg["lr"]
                pg["lr"] = self.hparams.d_lr_tex
            logger.info("D_tex LR: %.1e → %.1e", old, self.hparams.d_lr_tex)

    # ═══════════════════════════════════════════════════════
    #  梯度监控
    # ═══════════════════════════════════════════════════════

    def _monitor_grad_norms(self, optimizer, name: str):
        total_norm = 0.0
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm**2
        total_norm = total_norm**0.5
        if total_norm > self.grad_warn_threshold:
            logger.warning(
                "梯度爆炸警告 [%s]: grad_norm=%.2f > threshold=%.2f",
                name,
                total_norm,
                self.grad_warn_threshold,
            )
        return total_norm

    # ═══════════════════════════════════════════════════════
    #  Helper: sr_latent 预计算
    # ═══════════════════════════════════════════════════════

    @torch.no_grad()
    def _get_sr_latent_precomputed(self, lr, ref):
        if self.sr_model is None:
            return None
        with torch.amp.autocast("cuda", enabled=False):
            sr_pixel = self.sr_model(lr.float(), ref.float())
            sr_pixel = torch.nan_to_num(sr_pixel, nan=0.0, posinf=1.0, neginf=-1.0)
            sr_pixel = sr_pixel.clamp(-1.0, 1.0)
            return self.generator.encode_latent(sr_pixel.to(self.generator.vae.dtype))

    # ═══════════════════════════════════════════════════════
    #  NaN 自动终止检测
    # ═══════════════════════════════════════════════════════

    def _check_early_stop(self, is_g_step: bool):
        cnt_name = "_consecutive_nan_g" if is_g_step else "_consecutive_nan_d"
        cnt = getattr(self, cnt_name, 0)
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
    #  训练入口
    # ═══════════════════════════════════════════════════════

    def training_step(self, batch, batch_idx):
        if self._gd_phase == 0:
            return self._generator_step(batch, batch_idx)
        else:
            return self._discriminator_step(batch, batch_idx)

    # ═══════════════════════════════════════════════════════
    #  Generator Step（修复：D 状态切换与累积绑定）
    # ═══════════════════════════════════════════════════════

    def _generator_step(self, batch, batch_idx):
        g_opt = self._get_g_opt()
        lr, ref, hr = self.generator.get_input(batch)

        # Phase 1: 扩散去噪
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            out = self.generator.forward(lr, ref, hr)
            loss = out["loss"]

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("[G step] Phase1 loss NaN/Inf, batch=%d", batch_idx)
            self._consecutive_nan_g += 1
            if self._check_early_stop(is_g_step=True):
                return None
            g_opt.zero_grad(set_to_none=True)
            self._g_accum_count = 0
            self._gd_phase = 1
            if self.discriminator is not None:
                self.discriminator.eval()
                self.discriminator.requires_grad_(False)
            return None
        self._consecutive_nan_g = 0

        loss_phase1 = loss / self.accumulate_grad_batches
        if self.scaler_g is not None:
            self.scaler_g.scale(loss_phase1).backward()
        else:
            loss_phase1.backward()

        # Phase 2: SR 辅助 loss
        phase2_loss_val = None
        if (
            self.sr_model is not None
            and self.discriminator is not None
            and (
                self.lambda_gan_semantic > 0
                or self.lambda_gan_texture > 0
                or self.lambda_diff_sr > 0
                or self.lambda_lpips > 0
            )
        ):
            self.discriminator.eval()
            self.discriminator.requires_grad_(False)
            bsz = lr.shape[0]
            device = lr.device

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                sr_latent = self._get_sr_latent_precomputed(lr, ref)
                t_sr = torch.randint(
                    self.generator.t_min,
                    self.generator.t_max + 1,
                    (bsz,),
                    device=device,
                    dtype=torch.long,
                )
                noise_sr = torch.randn_like(sr_latent)

                with torch.no_grad():
                    pred_hr_pixel = self._no_adapter_pred_x0(
                        hr, sr_latent, t_sr, noise_sr
                    )

                pred_sr_pixel = self._adapter_pred_x0(
                    lr, ref, sr_latent, t_sr, noise_sr
                )

                phase2_loss = 0.0

                if self.lambda_diff_sr > 0:
                    loss_diff_sr = F.mse_loss(pred_sr_pixel, pred_hr_pixel.detach())
                    phase2_loss = phase2_loss + self.lambda_diff_sr * loss_diff_sr
                    self.log("train/G_diff_sr", loss_diff_sr.detach(), on_step=True)

                if self.lambda_lpips > 0:
                    loss_lpips_sr = (
                        self.net_lpips(pred_sr_pixel, hr).mean() * self.lambda_lpips
                    )
                    if not torch.isnan(loss_lpips_sr) and not torch.isinf(
                        loss_lpips_sr
                    ):
                        phase2_loss = phase2_loss + loss_lpips_sr
                        self.log("train/G_lpips", loss_lpips_sr.detach(), on_step=True)
                    else:
                        self._nan_g_count += 1
                        logger.warning(
                            "[G step] LPIPS(SR) NaN/Inf (#%d)，跳过", self._nan_g_count
                        )

                with torch.amp.autocast("cuda", enabled=False):
                    gan_loss = self.discriminator.compute_g_loss(
                        fake=pred_sr_pixel.float(),
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
                        "[G step] GAN NaN/Inf (#%d)，pred_sr_pixel range=[%.2f, %.2f]",
                        self._nan_g_count,
                        pred_sr_pixel.min(),
                        pred_sr_pixel.max(),
                    )

            if isinstance(phase2_loss, torch.Tensor) and phase2_loss.item() != 0:
                phase2_loss_val = phase2_loss.detach()
                phase2_loss_scaled = phase2_loss / self.accumulate_grad_batches
                if self.scaler_g is not None:
                    self.scaler_g.scale(phase2_loss_scaled).backward()
                else:
                    phase2_loss_scaled.backward()

        # 累积 → step
        self._g_accum_count += 1
        if self._g_accum_count >= self.accumulate_grad_batches:
            self._monitor_grad_norms(g_opt, "G")
            if self.scaler_g is not None:
                self.scaler_g.unscale_(g_opt)
            self.clip_gradients(
                g_opt,
                gradient_clip_val=self.grad_clip_val,
                gradient_clip_algorithm="norm",
            )
            if self.scaler_g is not None:
                self.scaler_g.step(g_opt)
                self.scaler_g.update()
            else:
                g_opt.step()
            g_opt.zero_grad(set_to_none=True)
            self._g_accum_count = 0

            # [FIX] g_d_ratio 控制：累积 1 次 G 后才切换 phase
            self._g_steps_since_d += 1
            if self._g_steps_since_d >= self.g_d_ratio:
                self._gd_phase = 1
                self._g_steps_since_d = 0
                # [FIX] 只在切换 phase 时恢复 D 训练状态
                if self.discriminator is not None:
                    self.discriminator.train()
                    self.discriminator.requires_grad_(True)
        else:
            # [FIX] 累积未完成时，保持 D 冻结
            pass

        g_total = loss.detach()
        if phase2_loss_val is not None:
            g_total = g_total + phase2_loss_val
        self.log("train/G_total", g_total, on_step=True, prog_bar=True)
        self.log("train/G_diff_hr", out["loss"].detach(), on_step=True, prog_bar=True)
        return g_total

    # ═══════════════════════════════════════════════════════
    #  Discriminator Step（修复：条件 phase 切换）
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

        self.discriminator.train()
        self.discriminator.requires_grad_(True)

        d_sem_opt = self._get_d_sem_opt()
        d_tex_opt = self._get_d_tex_opt()
        lr, ref, hr = self.generator.get_input(batch)
        bsz = lr.shape[0]
        device = lr.device

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
            self.discriminator.eval()
            self.discriminator.requires_grad_(False)
            self._gd_phase = 0
            return None
        self._consecutive_nan_d = 0

        # 单步 pred_x0
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                sr_latent = self._get_sr_latent_precomputed(lr, ref)
                t = torch.randint(0, 999, (bsz,), device=device, dtype=torch.long)
                noise = torch.randn_like(sr_latent)
                pred_hr_pixel = self._no_adapter_pred_x0(hr, sr_latent, t, noise)
                pred_sr_pixel = self._adapter_pred_x0(lr, ref, sr_latent, t, noise)

        real = pred_hr_pixel.detach().float()
        fake = pred_sr_pixel.detach().float()

        # NaN 检查
        if torch.isnan(fake).any() or torch.isinf(fake).any():
            self._nan_d_count += 1
            self._consecutive_nan_d += 1
            self._check_early_stop(is_g_step=False)
            logger.warning(
                "[D step] fake NaN/Inf (#%d), batch=%d", self._nan_d_count, batch_idx
            )
            if d_sem_opt is not None:
                d_sem_opt.zero_grad(set_to_none=True)
                self._d_sem_accum_count = 0
            if d_tex_opt is not None:
                d_tex_opt.zero_grad(set_to_none=True)
                self._d_tex_accum_count = 0
            self.discriminator.eval()
            self.discriminator.requires_grad_(False)
            self._gd_phase = 0
            return None
        self._consecutive_nan_d = 0

        if torch.isnan(real).any() or torch.isinf(real).any():
            self._nan_d_count += 1
            self._consecutive_nan_d += 1
            self._check_early_stop(is_g_step=False)
            logger.warning("[D step] real(no-adapter) NaN/Inf (#%d)", self._nan_d_count)
            if d_sem_opt is not None:
                d_sem_opt.zero_grad(set_to_none=True)
                self._d_sem_accum_count = 0
            if d_tex_opt is not None:
                d_tex_opt.zero_grad(set_to_none=True)
                self._d_tex_accum_count = 0
            self.discriminator.eval()
            self.discriminator.requires_grad_(False)
            self._gd_phase = 0
            return None
        self._consecutive_nan_d = 0

        sem_updated_this_step = False
        tex_updated_this_step = False

        # 语义 D
        if (
            self.lambda_gan_semantic > 0
            and self.discriminator.use_semantic_d
            and d_sem_opt is not None
        ):
            with torch.amp.autocast("cuda", enabled=False):
                loss_d_sem = self.discriminator.compute_d_loss(
                    real=real,
                    fake=fake,
                    ref=None,
                    lambda_semantic=1.0,
                    lambda_texture=0.0,
                )
            if not torch.isnan(loss_d_sem) and not torch.isinf(loss_d_sem):
                (loss_d_sem / self.accumulate_grad_batches).backward()
                self._d_sem_accum_count += 1
                sem_updated_this_step = True
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
                    "[D step] loss_d_sem NaN/Inf (#%d)，跳过 D_sem", self._nan_d_count
                )

        # 纹理 D
        if (
            self.lambda_gan_texture > 0
            and self.discriminator.use_texture_d
            and d_tex_opt is not None
        ):
            with torch.amp.autocast("cuda", enabled=False):
                loss_d_tex = self.discriminator.compute_d_loss(
                    real=real,
                    fake=fake,
                    ref=ref,
                    lambda_semantic=0.0,
                    lambda_texture=1.0,
                )
            if not torch.isnan(loss_d_tex) and not torch.isinf(loss_d_tex):
                (loss_d_tex / self.accumulate_grad_batches).backward()
                self._d_tex_accum_count += 1
                tex_updated_this_step = True
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
                    "[D step] loss_d_tex NaN/Inf (#%d)，跳过 D_tex", self._nan_d_count
                )

        # 未参与更新的优化器清零累积
        if d_sem_opt is not None and not sem_updated_this_step:
            d_sem_opt.zero_grad(set_to_none=True)
            self._d_sem_accum_count = 0
        if d_tex_opt is not None and not tex_updated_this_step:
            d_tex_opt.zero_grad(set_to_none=True)
            self._d_tex_accum_count = 0

        # [FIX] 条件 phase 切换：仅当所有 D 都完成累积时才切换
        d_sem_done = (
            not self.discriminator.use_semantic_d or self._d_sem_accum_count == 0
        )
        d_tex_done = (
            not self.discriminator.use_texture_d or self._d_tex_accum_count == 0
        )
        if d_sem_done and d_tex_done:
            self.discriminator.eval()
            self.discriminator.requires_grad_(False)
            self._gd_phase = 0
            self._g_steps_since_d = 0

        return None

    # ═══════════════════════════════════════════════════════
    #  验证 / 推理
    # ═══════════════════════════════════════════════════════

    def validation_step(self, batch, batch_idx):
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
                )

            if self.iqa is not None:
                try:
                    sr_batch = val_results["samples"]
                    hq_batch = val_results["hq"]
                    agg = {}
                    for i in range(len(sr_batch)):
                        sr_np = sr_batch[i].cpu().numpy()
                        hq_np = hq_batch[i].cpu().numpy()
                        m = self.iqa.evaluate_single(sr_np, hq_np)
                        for k, v in m.items():
                            agg[k] = agg.get(k, 0.0) + v
                    n = len(sr_batch)
                    for k, v in agg.items():
                        self.log(f"val/{k}", v / n, on_epoch=True, prog_bar=True)
                        if k == "psnr":
                            self.log("val_psnr", v / n, on_epoch=True)
                except Exception as e:
                    logger.warning("IQA 评估失败: %s", e)

            if batch_idx % 4 == 0:
                save_dir = os.path.join(self.logger.save_dir, "validation_tmp")
                os.makedirs(save_dir, exist_ok=True)
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=False):
                        sr_prior = self.sr_model(lr.float(), ref.float())
                        sr_prior = torch.nan_to_num(
                            sr_prior, nan=0.0, posinf=1.0, neginf=-1.0
                        ).clamp(-1.0, 1.0)
                sr_prior_01 = (sr_prior + 1.0) / 2.0

                images_to_concat = []
                for image_key in ["lq", "ref", "hq", "samples"]:
                    if image_key not in val_results:
                        continue
                    img = val_results[image_key][0]
                    img = img.detach().cpu().permute(1, 2, 0).numpy()
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(img)
                    if image_key == "lq":
                        target_size = val_results["samples"].shape[-2:]
                        pil_img = pil_img.resize(
                            (target_size[1], target_size[0]), Image.NEAREST
                        )
                    images_to_concat.append(pil_img)

                sr_img = sr_prior_01[0].detach().cpu().permute(1, 2, 0).numpy()
                sr_img = np.nan_to_num(sr_img, nan=0.0, posinf=1.0, neginf=0.0)
                sr_img = (sr_img * 255).clip(0, 255).astype(np.uint8)
                images_to_concat.append(Image.fromarray(sr_img))

                if images_to_concat:
                    total_w = sum(im.width for im in images_to_concat)
                    max_h = max(im.height for im in images_to_concat)
                    combined = Image.new("RGB", (total_w, max_h))
                    x_offset = 0
                    for im in images_to_concat:
                        combined.paste(im, (x_offset, 0))
                        x_offset += im.width
                    combined.save(os.path.join(save_dir, f"b{batch_idx}.png"))

            del val_results, lr, ref, hr
            torch.cuda.empty_cache()
            return loss_diff
        except Exception as e:
            logger.warning("validation_step 异常: %s", e)
            torch.cuda.empty_cache()
            return torch.tensor(0.0, device=self.device)

    def on_validation_epoch_start(self):
        if self.discriminator is not None:
            self.discriminator.eval()
        import shutil

        save_dir = os.path.join(self.logger.save_dir, "validation_tmp")
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            logger.info("已清理 validation_tmp 目录")

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()

    def on_train_start(self):
        if self.sr_model is not None:
            self.sr_model.to(self.device)
        self._override_lr_on_resume()
