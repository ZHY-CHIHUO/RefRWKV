"""
sd2_ref_gan_system.py — G/D 分离 + 交替训练系统

设计原则：
  1. 持有 SD2RefGenerator 和 SD2RefDiscriminator；
  2. 手动优化 + AMP + 梯度累积，按 phase 控制 G/D 交替；
  3. G step 中扩散 loss 为主，GAN / LPIPS 为辅助；
  4. D step 中用单步 pred_x0 生成 fake/real（无adapter vs 有adapter），更新判别器；
  5. 所有进入判别器的图像统一保持在 [-1, 1] 值域。
  6. 复用 RefGAN 的 UNet：HR 路径不注入 adapter/lr/ref，SR 路径注入。
     D 比较同一个 UNet "有/无 adapter 增强" 的去噪质量差异。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from typing import Optional, List, Dict
import lpips
import numpy as np
import math
from PIL import Image

from .sd2_ref_generator import SD2RefGenerator
from .sd2_ref_discriminator import SD2RefDiscriminator


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
        # Better Start：推理时用 SR prior 做 warm-start 初始化
        sr_model: Optional[torch.nn.Module] = None,
        sr_fixed: bool = True,
        # Better Start + MSE Guidance 推理参数 ──
        t_start: Optional[int] = None,
        guidance_scale: float = 0.0,
        t_stop: int = 200,
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

        self.t_start = t_start
        self.guidance_scale = guidance_scale
        self.t_stop = t_stop

        # ── NaN 防护计数 ──
        self._nan_g_count = 0
        self._nan_d_count = 0

        # ═══════════════════════════════════════
        #  手动优化 + AMP + 梯度累积
        # ═══════════════════════════════════════
        self.automatic_optimization = False
        self.use_amp = use_amp
        self.scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)

        self._g_accum_count = 0
        self._d_sem_accum_count = 0
        self._d_tex_accum_count = 0
        self._gd_phase = 0

        self._opt_idx: Dict[str, int] = {}

        # ═══════════════════════════════════════
        #  LPIPS（用于 G 的辅助 loss）
        # ═══════════════════════════════════════
        self.net_lpips = lpips.LPIPS(net="vgg", verbose=False)
        for p in self.net_lpips.parameters():
            p.requires_grad = False

        # IQA engine
        try:
            from RefRWKV.evaluation.eval_pyiqa import IQAEngine

            self.iqa = IQAEngine(
                device="cuda",
                nr_metrics=[],
                fr_metrics=fr_metrics or ["psnr", "ssim", "lpips", "dists"],
                use_y_channel=True,
                verbose=False,
            )
        except ImportError:
            self.iqa = None

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

    def on_load_checkpoint(self, checkpoint):
        self._gd_phase = checkpoint.get("gd_phase", 0)
        self._g_accum_count = checkpoint.get("g_accum_count", 0)
        self._d_sem_accum_count = checkpoint.get("d_sem_accum_count", 0)
        self._d_tex_accum_count = checkpoint.get("d_tex_accum_count", 0)

    def load_state_dict(self, state_dict, strict=True):
        """Load checkpoint weights. sr_fixed 只控制冻结，不影响加载。"""
        return super().load_state_dict(state_dict, strict=False)

    def _override_lr_on_resume(self):
        optimizers = self.optimizers()
        if not optimizers:
            return

        target_g_lr = self.hparams.g_lr
        for pg in optimizers[self._opt_idx["g"]].param_groups:
            old = pg["lr"]
            pg["lr"] = target_g_lr
        print(f"🔧 G  LR: {old:.1e} → {target_g_lr:.1e}")

        idx_sem = self._opt_idx.get("d_sem")
        if idx_sem is not None:
            for pg in optimizers[idx_sem].param_groups:
                old = pg["lr"]
                pg["lr"] = self.hparams.d_lr_sem
            print(f"🔧 D_sem LR: {old:.1e} → {self.hparams.d_lr_sem:.1e}")

        idx_tex = self._opt_idx.get("d_tex")
        if idx_tex is not None:
            for pg in optimizers[idx_tex].param_groups:
                old = pg["lr"]
                pg["lr"] = self.hparams.d_lr_tex
            print(f"🔧 D_tex LR: {old:.1e} → {self.hparams.d_lr_tex:.1e}")

    # ═══════════════════════════════════════════════════════
    #  Helper: 共享的 sr_latent 预计算
    # ═══════════════════════════════════════════════════════
    @torch.no_grad()
    def _get_sr_latent_precomputed(self, lr, ref):
        """预先计算 sr_latent，避免 Phase 2 和 D step 各调一次 SR model。"""
        if self.sr_model is None:
            return None
        sr_pixel = self.sr_model(lr, ref)
        return self.generator.encode_latent(sr_pixel)

    # ═══════════════════════════════════════════════════════
    #  Helper: 无 adapter 路径（HR 去噪标杆）
    #  复用 generator 的 UNet / VAE / scheduler，不注入 lr/ref/adapter
    # ═══════════════════════════════════════════════════════
    @torch.no_grad()
    def _no_adapter_pred_x0(self, hr, sr_latent_cond, t, noise):
        """
        HR → VAE → latent → add_noise → UNet(no adapter) → pred_x0 → decode

        Args:
            hr: [B, 3, H, W] in [-1, 1]
            sr_latent_cond: [B, 4, h, w]，共享的 sr_latent 条件
            t: [B] timestep, long
            noise: [B, 4, h, w] noise to add
        Returns:
            pred_pixel: [B, 3, H, W] in [-1, 1]
        """
        bsz = hr.shape[0]
        device = hr.device

        # HR → latent（复用 generator 的 VAE）
        hr_latent = self.generator.encode_latent(hr)
        x_t = self.generator.noise_scheduler.add_noise(hr_latent, noise, t)

        # 拼接 sr_latent_cond（和 adapter 路径一致）──
        x_input = self.generator._concat_sr_latent(x_t, sr_latent_cond.detach())

        # 全零 context + 无 down_intrablock（不注入任何 adapter 特征）
        null_ctx = torch.zeros(
            bsz,
            77,
            self.generator.cross_attn_dim,
            device=device,
            dtype=torch.float32,
        )

        eps_pred = self.generator.unet(
            x_input,
            t,
            encoder_hidden_states=null_ctx,
            # 不传 down_intrablock_additional_residuals
        ).sample

        pred_x0 = self.generator._predict_x0_from_eps(x_t, t, eps_pred)
        pred_x0 = torch.nan_to_num(pred_x0, nan=0.0, posinf=20.0, neginf=-20.0).clamp(
            -20.0, 20.0
        )

        pred_pixel = self.generator.decode_latent(pred_x0)
        return pred_pixel

    # ═══════════════════════════════════════════════════════
    #  Helper: 有 adapter 路径（SR 去噪，可梯度）
    #  复用 generator 的 UNet，注入 lr/ref/adapter 特征
    # ═══════════════════════════════════════════════════════
    def _adapter_pred_x0(self, lr, ref, sr_latent_precomputed, t, noise):
        """
        SR latent → add_noise → UNet(with adapter) → pred_x0 → decode

        Args:
            lr, ref: 输入图像
            sr_latent_precomputed: [B, 4, h, w]，预先计算的 sr_latent
            t: [B] timestep, long
            noise: [B, 4, h, w] noise to add
        Returns:
            pred_pixel: [B, 3, H, W] in [-1, 1]
        """
        x_t = self.generator.noise_scheduler.add_noise(sr_latent_precomputed, noise, t)

        x_input = self.generator._concat_sr_latent(x_t, sr_latent_precomputed.detach())
        noise_pred = self.generator.apply_model(x_input, t, lr, ref)
        pred_x0 = self.generator._predict_x0_from_eps(x_t, t, noise_pred)

        pred_x0 = torch.nan_to_num(pred_x0, nan=0.0, posinf=20.0, neginf=-20.0).clamp(
            -20.0, 20.0
        )

        pred_pixel = self.generator.decode_latent(pred_x0)
        return pred_pixel

    # ═══════════════════════════════════════════════════════
    #  构建两路 pred_x0（共享 t 和 noise，预计算 sr_latent）
    #  返回 (pred_hr_pixel, pred_sr_pixel, sr_latent)
    #  调用方负责 torch.no_grad() 或设置 requires_grad
    # ═══════════════════════════════════════════════════════
    def _build_two_path_preds(self, lr, ref, hr):
        """构建两路单步 pred_x0，返回 (pred_hr, pred_sr, sr_latent, t, noise)。"""
        bsz = lr.shape[0]
        device = lr.device

        sr_latent = self._get_sr_latent_precomputed(lr, ref)
        if sr_latent is None:
            return None, None, None, None, None

        t = torch.randint(
            0,
            999,
            (bsz,),
            device=device,
            dtype=torch.long,
        )
        noise = torch.randn_like(sr_latent)

        pred_hr = self._no_adapter_pred_x0(hr, sr_latent, t, noise)
        # pred_sr 由调用方控制 grad（G step 需要 grad，D step 不需要）
        pred_sr = None  # 调用方自行调用 _adapter_pred_x0

        return pred_hr, sr_latent, t, noise

    # ═══════════════════════════════════════════════════════
    #  训练入口：按 phase 交替 G / D
    # ═══════════════════════════════════════════════════════
    def training_step(self, batch, batch_idx):
        if self._gd_phase == 0:
            return self._generator_step(batch, batch_idx)
        else:
            return self._discriminator_step(batch, batch_idx)

    # ═══════════════════════════════════════════════════════
    #  Generator Step
    # ═══════════════════════════════════════════════════════
    def _generator_step(self, batch, batch_idx):
        g_opt = self._get_g_opt()
        lr, ref, hr = self.generator.get_input(batch)

        # ═══════════════════════════════════════════════════
        # Phase 1: MSE(noise_pred, noise) — 纯扩散去噪
        # ═══════════════════════════════════════════════════
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            out = self.generator.forward(lr, ref, hr)
            loss = out["loss"]  # MSE(noise_pred, noise)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ [G step] Phase1 loss NaN/Inf，跳过。batch={batch_idx}")
            self._gd_phase = 1
            if self.discriminator is not None:
                self.discriminator.train()
                self.discriminator.requires_grad_(True)
            return None

        # Phase 1 backward → 释放 forward() 的 UNet 激活
        loss_phase1 = loss / self.accumulate_grad_batches
        self.scaler_g.scale(loss_phase1).backward()

        # ═══════════════════════════════════════════════════
        # Phase 2: SR 路径 — MSE(SR) + LPIPS(SR) + GAN
        # 无 adapter vs 有 adapter（共享 t、noise、sr_latent）
        # 新的 UNet 前向复用 Phase 1 释放的显存 — 峰值不叠加
        # ═══════════════════════════════════════════════════
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
                # ── 预计算 sr_latent + 共享 t 和 noise ──
                sr_latent = self._get_sr_latent_precomputed(lr, ref)
                t_sr = torch.randint(
                    0,
                    999,
                    (bsz,),
                    device=device,
                    dtype=torch.long,
                )
                noise_sr = torch.randn_like(sr_latent)

                # ── HR 路径（无 adapter，no grad）──
                with torch.no_grad():
                    pred_hr_pixel = self._no_adapter_pred_x0(
                        hr, sr_latent, t_sr, noise_sr
                    )

                # ── SR 路径（有 adapter，需要 grad）──
                pred_sr_pixel = self._adapter_pred_x0(
                    lr, ref, sr_latent, t_sr, noise_sr
                )

                phase2_loss = 0.0

                # MSE(SR pixel vs HR path pixel)
                if self.lambda_diff_sr > 0:
                    loss_diff_sr = F.mse_loss(pred_sr_pixel, pred_hr_pixel.detach())
                    phase2_loss = phase2_loss + self.lambda_diff_sr * loss_diff_sr
                    self.log("train/G_diff_sr", loss_diff_sr.detach(), on_step=True)

                # LPIPS(SR pixel vs HR path pixel)
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
                        print(
                            f"⚠️ [G step] LPIPS(SR) NaN/Inf "
                            f"(#{self._nan_g_count})，跳过"
                        )

                # GAN：D 评估 adapter 是否带来质量提升
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
                    print(
                        f"⚠️ [G step] GAN NaN/Inf (#{self._nan_g_count})，"
                        f"pred_sr_pixel range=[{pred_sr_pixel.min():.2f}, "
                        f"{pred_sr_pixel.max():.2f}]"
                    )

            if isinstance(phase2_loss, torch.Tensor) and phase2_loss.item() != 0:
                phase2_loss_val = phase2_loss.detach()
                phase2_loss_scaled = phase2_loss / self.accumulate_grad_batches
                self.scaler_g.scale(phase2_loss_scaled).backward()

        # ═══════════════════════════════════════════════════
        # 累积 → step optimizer
        # ═══════════════════════════════════════════════════
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
            self._gd_phase = 1

        if self.discriminator is not None:
            self.discriminator.train()
            self.discriminator.requires_grad_(True)

        # log
        g_total = loss.detach()
        if phase2_loss_val is not None:
            g_total = g_total + phase2_loss_val
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

        self.discriminator.train()
        self.discriminator.requires_grad_(True)

        d_sem_opt = self._get_d_sem_opt()
        d_tex_opt = self._get_d_tex_opt()
        lr, ref, hr = self.generator.get_input(batch)
        bsz = lr.shape[0]
        device = lr.device

        # ═══════════════════════════════════════════════════
        #  单步 pred_x0：无 adapter vs 有 adapter
        #  两路共享 sr_latent、t、noise
        # ═══════════════════════════════════════════════════
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                sr_latent = self._get_sr_latent_precomputed(lr, ref)
                t = torch.randint(
                    0,
                    999,
                    (bsz,),
                    device=device,
                    dtype=torch.long,
                )
                noise = torch.randn_like(sr_latent)

                # 无 adapter（质量标杆）
                pred_hr_pixel = self._no_adapter_pred_x0(hr, sr_latent, t, noise)

                # 有 adapter（需要追赶）
                pred_sr_pixel = self._adapter_pred_x0(lr, ref, sr_latent, t, noise)

        real = pred_hr_pixel.detach().float()
        fake = pred_sr_pixel.detach().float()

        # ── NaN 检查 ──
        if torch.isnan(fake).any() or torch.isinf(fake).any():
            self._nan_d_count += 1
            print(
                f"⚠️ [D step] fake NaN/Inf (#{self._nan_d_count})，"
                f"batch={batch_idx}，跳过"
            )
            self.discriminator.eval()
            self.discriminator.requires_grad_(False)
            self._gd_phase = 0
            return None

        if torch.isnan(real).any() or torch.isinf(real).any():
            self._nan_d_count += 1
            print(
                f"⚠️ [D step] real(no-adapter) NaN/Inf " f"(#{self._nan_d_count})，跳过"
            )
            self.discriminator.eval()
            self.discriminator.requires_grad_(False)
            self._gd_phase = 0
            return None

        # ── 语义 D ──
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
                self.log(
                    "train/D_sem",
                    loss_d_sem.detach(),
                    on_step=True,
                    prog_bar=True,
                )
                if self._d_sem_accum_count >= self.accumulate_grad_batches:
                    self.clip_gradients(
                        d_sem_opt,
                        gradient_clip_val=1.0,
                        gradient_clip_algorithm="norm",
                    )
                    d_sem_opt.step()
                    d_sem_opt.zero_grad()
                    self._d_sem_accum_count = 0
            else:
                self._nan_d_count += 1
                print(
                    f"⚠️ [D step] loss_d_sem NaN/Inf "
                    f"(#{self._nan_d_count})，跳过 D_sem"
                )

        # ── 纹理 D ──
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
                self.log(
                    "train/D_tex",
                    loss_d_tex.detach(),
                    on_step=True,
                    prog_bar=True,
                )
                if self._d_tex_accum_count >= self.accumulate_grad_batches:
                    self.clip_gradients(
                        d_tex_opt,
                        gradient_clip_val=1.0,
                        gradient_clip_algorithm="norm",
                    )
                    d_tex_opt.step()
                    d_tex_opt.zero_grad()
                    self._d_tex_accum_count = 0
            else:
                self._nan_d_count += 1
                print(
                    f"⚠️ [D step] loss_d_tex NaN/Inf "
                    f"(#{self._nan_d_count})，跳过 D_tex"
                )

        self.discriminator.eval()
        self.discriminator.requires_grad_(False)
        self._gd_phase = 0
        return None

    # ═══════════════════════════════════════════════════════
    #  验证 / 推理
    # ═══════════════════════════════════════════════════════
    def validation_step(self, batch, batch_idx):
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
                    self.log(f"val_psnr", v / n, on_epoch=True)

        if batch_idx % 4 == 0:
            import shutil

            save_dir = os.path.join(self.logger.save_dir, "validation_tmp")
            if batch_idx == 0 and os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir, exist_ok=True)

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    sr_prior = self.sr_model(lr, ref)
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
                        (target_size[1], target_size[0]),
                        Image.NEAREST,
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

        return loss_diff

    def on_validation_epoch_start(self):
        if self.discriminator is not None:
            self.discriminator.eval()

    def on_train_start(self):
        if self.sr_model is not None:
            self.sr_model.to(self.device)
        self._override_lr_on_resume()
