"""
sd2_ref_gan_system.py — G/D 分离 + 交替训练系统

设计原则：
  1. 持有 SD2RefGenerator 和 SD2RefDiscriminator；
  2. 手动优化 + AMP + 梯度累积，按 phase 控制 G/D 交替；
  3. G step 中扩散 loss 为主，GAN / LPIPS 为辅助；
  4. D step 中用 generator.generate_sr 生成 fake，更新判别器。
  5. 所有进入判别器的图像统一保持在 [-1, 1] 值域。
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
        # Better Start + MSE Guidance 推理参数 ──
        t_start: Optional[int] = None,  # Better Start 加噪目标时间步（None=纯噪声起点）
        guidance_scale: float = 0.0,  # MSE Guidance 引导强度（0=关闭）
        t_stop: int = 200,  # MSE Guidance 仅在 t > t_stop 时启用
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["generator", "discriminator", "sr_model"])

        self.generator = generator
        self.discriminator = discriminator
        self.sr_model = sr_model

        self.lambda_gan_semantic = lambda_gan_semantic
        self.lambda_gan_texture = lambda_gan_texture
        self.lambda_lpips = lambda_lpips
        self.accumulate_grad_batches = accumulate_grad_batches
        self.sample_steps = sample_steps

        # ── NEW ──
        self.t_start = t_start
        self.guidance_scale = guidance_scale
        self.t_stop = t_stop

        # ═══════════════════════════════════════
        #  手动优化 + AMP + 梯度累积
        # ═══════════════════════════════════════
        self.automatic_optimization = False
        self.use_amp = use_amp
        # 仅 G 段使用 AMP + scaler；D 段强制 fp32，不用 scaler
        self.scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)

        self._g_accum_count = 0
        self._d_sem_accum_count = 0
        self._d_tex_accum_count = 0
        self._gd_phase = 0  # 0: G, 1: D

        # 优化器索引（在 configure_optimizers 中按实际启用情况填充）
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
        filtered = {
            k: v for k, v in state_dict.items() if not k.startswith("sr_model.")
        }
        skipped = len(state_dict) - len(filtered)
        if skipped:
            print(f"🔒 跳过 {skipped} 个 sr_model 键，使用 build_sr_model 加载的权重")
        return super().load_state_dict(filtered, strict=False)

    def _override_lr_on_resume(self):
        """从 checkpoint 恢复后，用 hparams 中的 LR 覆盖 optimizer。"""
        optimizers = self.optimizers()
        if not optimizers:
            return

        # G optimizer（idx 0）
        target_g_lr = self.hparams.g_lr
        for pg in optimizers[self._opt_idx["g"]].param_groups:
            old = pg["lr"]
            pg["lr"] = target_g_lr
        print(f"🔧 G  LR: {old:.1e} → {target_g_lr:.1e}")

        # D_sem optimizer（如果有）
        idx_sem = self._opt_idx.get("d_sem")
        if idx_sem is not None:
            for pg in optimizers[idx_sem].param_groups:
                old = pg["lr"]
                pg["lr"] = self.hparams.d_lr_sem
            print(f"🔧 D_sem LR: {old:.1e} → {self.hparams.d_lr_sem:.1e}")

        # D_tex optimizer（如果有）
        idx_tex = self._opt_idx.get("d_tex")
        if idx_tex is not None:
            for pg in optimizers[idx_tex].param_groups:
                old = pg["lr"]
                pg["lr"] = self.hparams.d_lr_tex
            print(f"🔧 D_tex LR: {old:.1e} → {self.hparams.d_lr_tex:.1e}")

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
        # 此时 lr, ref, hr 均在 [-1, 1]（你的数据集）

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            out = self.generator.forward(lr, ref, hr)
            loss = out["loss"]

            # 预测 x0 并 decode 到 pixel（[-1, 1]），用于 GAN / LPIPS
            pred_x0_latent = out["pred_x0_latent"]
            sr_pixel = self.generator.decode_latent(pred_x0_latent)
            # sr_pixel ∈ [-1, 1]，与 hr / ref 一致

            # LPIPS 辅助 loss（LPIPS 期望 [-1, 1]）
            if self.lambda_lpips > 0:
                loss_lpips = self.net_lpips(sr_pixel, hr).mean() * self.lambda_lpips
                loss = loss + loss_lpips
                self.log("train/G_lpips", loss_lpips.detach(), on_step=True)

            # GAN 辅助 loss：先冻结 D，只回传到 G
            # sr_pixel ∈ [-1, 1]，直接送入，判别器内部会处理
            if self.discriminator is not None and (
                self.lambda_gan_semantic > 0 or self.lambda_gan_texture > 0
            ):
                self.discriminator.eval()
                self.discriminator.requires_grad_(False)

                with torch.amp.autocast("cuda", enabled=False):
                    gan_loss = self.discriminator.compute_g_loss(
                        fake=sr_pixel.float(),
                        ref=ref.float(),
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
            self._gd_phase = 1  # 攒够一次 G 更新后切到 D

        # G step 结束：恢复 D 到可训练状态，为 D step 做准备
        if self.discriminator is not None:
            self.discriminator.train()
            self.discriminator.requires_grad_(True)

        self.log(
            "train/G_total",
            loss.detach() * self.accumulate_grad_batches,
            on_step=True,
            prog_bar=True,
        )
        self.log("train/G_diff", out["loss"].detach(), on_step=True, prog_bar=True)
        return loss.detach() * self.accumulate_grad_batches

    # ═══════════════════════════════════════════════════════
    #  Discriminator Step  （不变——保持旧路径，不加 Better Start / Guidance）
    # ═══════════════════════════════════════════════════════
    def _discriminator_step(self, batch, batch_idx):
        # 如果 D 为空或 GAN loss 都没开，直接切回 G 阶段 ──
        if self.discriminator is None or (
            self.lambda_gan_semantic == 0.0 and self.lambda_gan_texture == 0.0
        ):
            self._gd_phase = 0
            return None

        # 关键：把 D 恢复为 train + requires_grad(True)，否则 backward 无梯度
        self.discriminator.train()
        self.discriminator.requires_grad_(True)

        d_sem_opt = self._get_d_sem_opt()
        d_tex_opt = self._get_d_tex_opt()
        lr, ref, hr = self.generator.get_input(batch)
        # lr, ref, hr ∈ [-1, 1]

        # 生成 fake（无梯度，节省显存）
        # 去掉 self.generator.eval() 防止触发底层 SystemError
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                fake = self.generator.generate_sr(
                    lr, ref, steps=self.sample_steps, sr_model=self.sr_model, hr=hr
                )

        # generate_sr 始终返回 [0, 1] → 映射到 [-1, 1]，与 real/ref 对齐
        fake = fake.detach().float() * 2.0 - 1.0
        real = hr  # 已在 [-1, 1]

        # 语义 D（fp32，不用 scaler）
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
            (loss_d_sem / self.accumulate_grad_batches).backward()
            self._d_sem_accum_count += 1

            if self._d_sem_accum_count >= self.accumulate_grad_batches:
                self.clip_gradients(
                    d_sem_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
                )
                d_sem_opt.step()
                d_sem_opt.zero_grad()
                self._d_sem_accum_count = 0

            self.log("train/D_sem", loss_d_sem.detach(), on_step=True, prog_bar=True)

        # 纹理 D（fp32，不用 scaler）
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
            (loss_d_tex / self.accumulate_grad_batches).backward()
            self._d_tex_accum_count += 1

            if self._d_tex_accum_count >= self.accumulate_grad_batches:
                self.clip_gradients(
                    d_tex_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
                )
                d_tex_opt.step()
                d_tex_opt.zero_grad()
                self._d_tex_accum_count = 0

            self.log("train/D_tex", loss_d_tex.detach(), on_step=True, prog_bar=True)

        # D 阶段结束：把 D 设回 eval（G step 里会再控制），切回 G
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
            # 传入 Better Start + MSE Guidance 参数 ──
            val_results = self.generator.log_images(
                batch,
                steps=self.sample_steps,
                sr_model=self.sr_model,
                t_start=self.t_start,
                guidance_scale=self.guidance_scale,
                t_stop=self.t_stop,
            )

        # 计算指标 — 扩散采样结果
        if self.iqa is not None:
            sr_batch = val_results["samples"]  # (B, C, H, W), [0,1]
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

        # ── 每 4 张保存一张到临时目录 ──
        if batch_idx % 4 == 0:
            import shutil

            save_dir = os.path.join(self.logger.save_dir, "validation_tmp")
            if batch_idx == 0 and os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir, exist_ok=True)

            for image_key in val_results:
                os.makedirs(os.path.join(save_dir, image_key), exist_ok=True)
                image = val_results[image_key].detach().cpu()
                for i in range(len(image)):
                    curr_img = image[i].permute(1, 2, 0).numpy()
                    curr_img = (curr_img * 255).clip(0, 255).astype(np.uint8)
                    filename = f"b{batch_idx}_{i}_{image_key}.png"
                    path = os.path.join(save_dir, image_key, filename)
                    Image.fromarray(curr_img).save(path)

        return loss_diff

    def on_validation_epoch_start(self):
        if self.discriminator is not None:
            self.discriminator.eval()

    def on_train_start(self):
        if self.sr_model is not None:
            self.sr_model.to(self.device)
        self._override_lr_on_resume()
