"""
sd2_ref_gan_system.py — G/D 分离 + 交替训练系统

设计原则：
  1. 持有 SD2RefGenerator 和 SD2RefDiscriminator；
  2. 手动优化 + AMP + 梯度累积，按 phase 控制 G/D 交替；
  3. G step 中扩散 loss 为主，GAN / LPIPS 为辅助；
  4. D step 中用 generator.generate_sr 生成 fake，更新判别器；
  5. 支持 Better Start：训练/推理均传入 sr_model。
"""

import os
import math
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


class SD2RefGANSystem(LightningModule):
    def __init__(
        self,
        generator: SD2RefGenerator,
        discriminator: Optional[SD2RefDiscriminator] = None,
        sr_model: Optional[nn.Module] = None,
        # Better Start
        better_start_prob: float = 0.5,
        t_max_better: int = 200,
        # loss 权重
        lambda_gan_semantic: float = 0.0,
        lambda_gan_texture: float = 0.0,
        lambda_lpips: float = 0.0,
        lambda_hf: float = 0.0,
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
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["generator", "discriminator", "sr_model"])

        self.generator = generator
        self.discriminator = discriminator
        self.sr_model = sr_model

        self.better_start_prob = better_start_prob
        self.t_max_better = t_max_better

        self.lambda_gan_semantic = lambda_gan_semantic
        self.lambda_gan_texture = lambda_gan_texture
        self.lambda_lpips = lambda_lpips
        self.lambda_hf = lambda_hf
        self.accumulate_grad_batches = accumulate_grad_batches
        self.sample_steps = sample_steps

        # ═══════════════════════════════════════
        #  手动优化 + AMP + 梯度累积
        # ═══════════════════════════════════════
        self.automatic_optimization = False
        self.use_amp = use_amp
        self.scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.scaler_d_sem = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.scaler_d_tex = torch.amp.GradScaler("cuda", enabled=use_amp)

        self._g_accum_count = 0
        self._d_sem_accum_count = 0
        self._d_tex_accum_count = 0
        self._gd_phase = 0  # 0: G, 1: D

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
                device="cpu",
                nr_metrics=[],
                fr_metrics=fr_metrics or ["psnr", "ssim", "lpips", "dists"],
                use_y_channel=True,
                verbose=False,
            )
        except ImportError:
            self.iqa = None

    # ═══════════════════════════════════════════════════════
    #  优化器
    # ═══════════════════════════════════════════════════════
    def _get_g_opt(self):
        return self.optimizers()[0]

    def _get_d_sem_opt(self):
        return self.optimizers()[1]

    def _get_d_tex_opt(self):
        return self.optimizers()[2]

    def configure_optimizers(self):
        g_opt = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.hparams.g_lr,
            weight_decay=self.hparams.g_weight_decay,
        )
        opts = [g_opt]

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
                opts.append(d_sem_opt)

            if self.discriminator.use_texture_d:
                d_tex_opt = torch.optim.AdamW(
                    list(self.discriminator.D_tex.parameters()),
                    lr=self.hparams.d_lr_tex,
                    betas=self.hparams.betas,
                    weight_decay=self.hparams.d_weight_decay,
                )
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

    # ═══════════════════════════════════════════════════════
    #  训练入口：按 phase 交替 G / D
    # ═══════════════════════════════════════════════════════
    def training_step(self, batch, batch_idx):
        try:
            if self._gd_phase == 0:
                return self._generator_step(batch, batch_idx)
            else:
                return self._discriminator_step(batch, batch_idx)
        finally:
            pass

    # ═══════════════════════════════════════════════════════
    #  Generator Step
    # ═══════════════════════════════════════════════════════
    def _high_frequency_loss(self, sr, hr):
        """简单 Laplacian 高频 L2 loss。"""
        laplacian_kernel = (
            torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                dtype=sr.dtype,
                device=sr.device,
            )
            .view(1, 1, 3, 3)
            .repeat(3, 1, 1, 1)
        )
        sr_hf = F.conv2d(sr, laplacian_kernel, padding=1, groups=3)
        hr_hf = F.conv2d(hr, laplacian_kernel, padding=1, groups=3)
        return F.mse_loss(sr_hf, hr_hf)

    def _generator_step(self, batch, batch_idx):
        g_opt = self._get_g_opt()
        lr, ref, hr = self.generator.get_input(batch)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            out = self.generator.forward(
                lr,
                ref,
                hr,
                sr_model=self.sr_model,
                better_start_prob=self.better_start_prob,
                t_max_better=self.t_max_better,
            )
            loss = out["loss"]

            # 预测 x0 并 decode 到 pixel（用于 GAN / LPIPS / HF）
            pred_x0_latent = out["pred_x0_latent"]
            sr_pixel = self.generator.decode_latent(pred_x0_latent)

            # 归一化到 [0, 1]
            sr_pixel_01 = (sr_pixel + 1.0) / 2.0
            hr_01 = (hr + 1.0) / 2.0

            # LPIPS 辅助 loss
            if self.lambda_lpips > 0:
                loss_lpips = (
                    self.net_lpips(sr_pixel_01, hr_01).mean() * self.lambda_lpips
                )
                loss = loss + loss_lpips
                self.log("train/G_lpips", loss_lpips.detach(), on_step=True)

            # 高频辅助 loss
            if self.lambda_hf > 0:
                loss_hf = self._high_frequency_loss(sr_pixel_01, hr_01) * self.lambda_hf
                loss = loss + loss_hf
                self.log("train/G_hf", loss_hf.detach(), on_step=True)

            # GAN 辅助 loss
            if self.discriminator is not None and (
                self.lambda_gan_semantic > 0 or self.lambda_gan_texture > 0
            ):
                self.discriminator.eval()
                self.discriminator.requires_grad_(False)

                gan_loss = self.discriminator.compute_g_loss(
                    fake=sr_pixel_01,
                    ref=ref,
                    lambda_semantic=self.lambda_gan_semantic,
                    lambda_texture=self.lambda_gan_texture,
                )
                loss = loss + gan_loss
                self.log("train/G_gan", gan_loss.detach(), on_step=True)

        loss = loss / self.accumulate_grad_batches

        self.scaler_g.scale(loss).backward()
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

        self.log(
            "train/G_total",
            loss * self.accumulate_grad_batches,
            on_step=True,
            prog_bar=True,
        )
        self.log("train/G_diff", out["loss"].detach(), on_step=True, prog_bar=True)
        self.log(
            "train/better_start_ratio", float(out["use_better_start"]), on_step=True
        )

        return loss.detach() * self.accumulate_grad_batches

    # ═══════════════════════════════════════════════════════
    #  Discriminator Step
    # ═══════════════════════════════════════════════════════
    def _discriminator_step(self, batch, batch_idx):
        if self.discriminator is None:
            self._gd_phase = 0
            return None

        d_sem_opt = self._get_d_sem_opt()
        d_tex_opt = self._get_d_tex_opt()
        lr, ref, hr = self.generator.get_input(batch)

        # 生成 fake（无梯度，节省显存）
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                fake = self.generator.generate_sr(
                    lr, ref, steps=self.sample_steps, sr_model=self.sr_model
                )

        real = hr
        fake = fake.detach()
        real_01 = (real + 1.0) / 2.0
        fake_01 = fake

        # 语义 D
        if self.lambda_gan_semantic > 0 and self.discriminator.use_semantic_d:
            with torch.amp.autocast("cuda", enabled=False):
                loss_d_sem = self.discriminator.compute_d_loss(
                    real=real_01,
                    fake=fake_01,
                    ref=None,
                    lambda_semantic=1.0,
                    lambda_texture=0.0,
                )

            loss_d_sem = loss_d_sem / self.accumulate_grad_batches

            self.scaler_d_sem.scale(loss_d_sem).backward()
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

            self.log(
                "train/D_sem",
                loss_d_sem.detach() * self.accumulate_grad_batches,
                on_step=True,
                prog_bar=True,
            )

        # 纹理 D
        if self.lambda_gan_texture > 0 and self.discriminator.use_texture_d:
            with torch.amp.autocast("cuda", enabled=False):
                loss_d_tex = self.discriminator.compute_d_loss(
                    real=real_01,
                    fake=fake_01,
                    ref=ref,
                    lambda_semantic=0.0,
                    lambda_texture=1.0,
                )

            loss_d_tex = loss_d_tex / self.accumulate_grad_batches

            self.scaler_d_tex.scale(loss_d_tex).backward()
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

            self.log(
                "train/D_tex",
                loss_d_tex.detach() * self.accumulate_grad_batches,
                on_step=True,
                prog_bar=True,
            )

        # D 阶段结束，切回 G
        self._gd_phase = 0

        return None

    # ═══════════════════════════════════════════════════════
    #  验证 / 推理
    # ═══════════════════════════════════════════════════════
    def validation_step(self, batch, batch_idx):
        lr, ref, hr = self.generator.get_input(batch)

        with torch.no_grad():
            val_results = self.generator.log_images(
                batch, steps=self.sample_steps, sr_model=self.sr_model
            )

        # 计算指标
        if self.iqa is not None:
            sr = val_results["samples"]
            hq = val_results["hq"]
            metrics = self.iqa(sr, hq)
            for k, v in metrics.items():
                self.log(f"val/{k}", v, on_epoch=True, prog_bar=True)

        # 保存图像
        save_dir = os.path.join(
            self.logger.save_dir, "validation", f"step--{self.global_step}"
        )
        os.makedirs(save_dir, exist_ok=True)

        hr_batch_tensor = val_results["hq"].detach().cpu()
        sr_batch_tensor = val_results["samples"].detach().cpu()
        this_psnr = 0.0
        for i in range(len(hr_batch_tensor)):
            curr_hr = hr_batch_tensor[i].numpy().astype(np.float64)
            curr_sr = sr_batch_tensor[i].numpy().astype(np.float64)
            curr_psnr = 20 * math.log10(
                1.0 / math.sqrt(np.mean((curr_hr - curr_sr) ** 2))
            )
            this_psnr += curr_psnr
        this_psnr /= len(hr_batch_tensor)
        self.log("val/psnr", this_psnr, on_epoch=True, prog_bar=True)

        for image_key in val_results:
            os.makedirs(os.path.join(save_dir, image_key), exist_ok=True)
            image = val_results[image_key].detach().cpu()
            for i in range(len(image)):
                curr_img = image[i]
                curr_img = curr_img.permute(1, 2, 0).numpy()
                curr_img = (curr_img * 255).clip(0, 255).astype(np.uint8)
                filename = f"{batch_idx}_{i}_{image_key}.png"
                path = os.path.join(save_dir, image_key, filename)
                Image.fromarray(curr_img).save(path)

        return

    def on_validation_epoch_start(self):
        if self.discriminator is not None:
            self.discriminator.eval()
