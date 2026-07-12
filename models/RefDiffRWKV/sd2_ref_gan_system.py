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
        """Load checkpoint weights.

        When sr_fixed=True:  skip sr_model keys (weights come from build_sr_model).
        When sr_fixed=False: load sr_model keys from checkpoint (retain fine-tuned weights).
        """
        if self.hparams.get("sr_fixed", True):
            filtered = {
                k: v for k, v in state_dict.items() if not k.startswith("sr_model.")
            }
            skipped = len(state_dict) - len(filtered)
            if skipped:
                print(
                    f"🔒 跳过 {skipped} 个 sr_model 键，使用 build_sr_model 加载的权重"
                )
        else:
            filtered = state_dict
            sr_keys = [k for k in state_dict if k.startswith("sr_model.")]
            if sr_keys:
                print(
                    f"🔧 sr_fixed=False，从 checkpoint 恢复 {len(sr_keys)} 个 sr_model 权重"
                )
        return super().load_state_dict(filtered, strict=False)

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
        # Phase 1: diff loss + LPIPS（HR 路径）
        # backward 后 UNet 激活被释放，显存回收
        # ═══════════════════════════════════════════════════
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            out = self.generator.forward(lr, ref, hr)
            loss = out["loss"]  # 全局 MSE

            # 预测 x0 并 decode — 用于 LPIPS
            pred_x0_latent = out["pred_x0_latent"]
            pred_x0_latent = torch.nan_to_num(
                pred_x0_latent, nan=0.0, posinf=20.0, neginf=-20.0
            )
            pred_x0_latent = pred_x0_latent.clamp(-20.0, 20.0)
            sr_pixel = self.generator.decode_latent(pred_x0_latent)
            # sr_pixel ∈ [-1, 1]

            # LPIPS
            if self.lambda_lpips > 0:
                loss_lpips = self.net_lpips(sr_pixel, hr).mean() * self.lambda_lpips
                if not torch.isnan(loss_lpips) and not torch.isinf(loss_lpips):
                    loss = loss + loss_lpips
                    self.log("train/G_lpips", loss_lpips.detach(), on_step=True)
                else:
                    self._nan_g_count += 1
                    print(
                        f"⚠️ [G step] LPIPS loss 为 NaN/Inf (#{self._nan_g_count})，跳过"
                    )

        # ── Phase 1 NaN 保护 ──
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ [G step] G_total 为 NaN/Inf，跳过本次更新。batch={batch_idx}")
            self._gd_phase = 1
            if self.discriminator is not None:
                self.discriminator.train()
                self.discriminator.requires_grad_(True)
            return None

        # ═══════════════════════════════════════════════════
        # Phase 1 backward → 释放 forward() 的 UNet 激活
        # ═══════════════════════════════════════════════════
        loss_phase1 = loss / self.accumulate_grad_batches
        self.scaler_g.scale(loss_phase1).backward()

        # ═══════════════════════════════════════════════════
        # Phase 2: SR 路径 GAN（SR prior latent 单步 pred_x0）
        # Phase 1 的 UNet 激活已被 backward 释放，
        # 新的 UNet 前向复用刚释放的显存 — 峰值不叠加
        # ═══════════════════════════════════════════════════
        gan_loss_val = None
        if self.discriminator is not None and (
            self.lambda_gan_semantic > 0 or self.lambda_gan_texture > 0
        ):
            self.discriminator.eval()
            self.discriminator.requires_grad_(False)
            bsz = lr.shape[0]

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                # SR prior → latent
                sr_latent = self.generator._get_sr_latent_cond(lr, ref)
                if sr_latent is None:
                    with torch.no_grad():
                        sr_prior_gan = self.sr_model(lr, ref)
                    sr_latent = self.generator.encode_latent(sr_prior_gan)

                # 加噪 + 单步反推 → SR 路径 pred_x0
                t_sr = torch.randint(
                    0,
                    999,
                    (bsz,),
                    device=sr_latent.device,
                    dtype=torch.long,
                )
                noise_sr = torch.randn_like(sr_latent)
                x_t_sr = self.generator.noise_scheduler.add_noise(
                    sr_latent, noise_sr, t_sr
                )

                x_input_sr = self.generator._concat_sr_latent(
                    x_t_sr,
                    sr_latent.detach(),
                )
                noise_pred_sr = self.generator.apply_model(
                    x_input_sr,
                    t_sr,
                    lr,
                    ref,
                )
                pred_x0_sr = self.generator._predict_x0_from_eps(
                    x_t_sr,
                    t_sr,
                    noise_pred_sr,
                )
                pred_x0_sr = torch.nan_to_num(
                    pred_x0_sr,
                    nan=0.0,
                    posinf=20.0,
                    neginf=-20.0,
                )
                pred_x0_sr = pred_x0_sr.clamp(-20.0, 20.0)

                sr_pixel_gan = self.generator.decode_latent(pred_x0_sr)

                with torch.amp.autocast("cuda", enabled=False):
                    gan_loss = self.discriminator.compute_g_loss(
                        fake=sr_pixel_gan.float(),
                        ref=ref.float(),
                        lambda_semantic=self.lambda_gan_semantic,
                        lambda_texture=self.lambda_gan_texture,
                    )

            if not torch.isnan(gan_loss) and not torch.isinf(gan_loss):
                gan_loss_val = gan_loss.detach()
                gan_loss_scaled = gan_loss / self.accumulate_grad_batches
                self.scaler_g.scale(gan_loss_scaled).backward()
                self.log("train/G_gan", gan_loss_val, on_step=True)
            else:
                self._nan_g_count += 1
                print(
                    f"⚠️ [G step] SR GAN loss 为 NaN/Inf (#{self._nan_g_count})，"
                    f"sr_pixel_gan range=[{sr_pixel_gan.min():.2f}, "
                    f"{sr_pixel_gan.max():.2f}]"
                )

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

        # log G_total = diff + lpips + gan
        g_total = loss.detach()
        if gan_loss_val is not None:
            g_total = g_total + gan_loss_val
        self.log("train/G_total", g_total, on_step=True, prog_bar=True)
        self.log("train/G_diff", out["loss"].detach(), on_step=True, prog_bar=True)
        return g_total

    # ═══════════════════════════════════════════════════════
    #  Discriminator Step
    # ═══════════════════════════════════════════════════════
    def _discriminator_step(self, batch, batch_idx):
        # 如果 D 为空或 GAN loss 都没开，直接切回 G 阶段 ──
        if self.discriminator is None or (
            self.lambda_gan_semantic == 0.0 and self.lambda_gan_texture == 0.0
        ):
            self._gd_phase = 0
            return None

        self.discriminator.train()
        self.discriminator.requires_grad_(True)

        d_sem_opt = self._get_d_sem_opt()
        d_tex_opt = self._get_d_tex_opt()
        lr, ref, hr = self.generator.get_input(batch)

        # 生成 fake（无梯度，节省显存）
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                fake = self.generator.generate_sr(
                    lr,
                    ref,
                    steps=self.sample_steps,
                    sr_model=self.sr_model,
                    hr=hr,
                    # ── 传入 Better Start 参数 ──
                    t_start=self.t_start,
                    guidance_scale=self.guidance_scale,
                    t_stop=self.t_stop,
                )

        fake = fake.detach().float() * 2.0 - 1.0
        real = hr  # 已在 [-1, 1]

        # ── NaN 检查：fake/real 含 NaN 则跳过 ──
        if torch.isnan(fake).any() or torch.isinf(fake).any():
            self._nan_d_count += 1
            print(
                f"⚠️ [D step] fake 含 NaN/Inf (#{self._nan_d_count})，"
                f"batch={batch_idx}，跳过本次 D step"
            )
            self.discriminator.eval()
            self.discriminator.requires_grad_(False)
            self._gd_phase = 0
            return None

        if torch.isnan(real).any() or torch.isinf(real).any():
            self._nan_d_count += 1
            print(f"⚠️ [D step] real(hr) 含 NaN/Inf (#{self._nan_d_count})，跳过")
            self.discriminator.eval()
            self.discriminator.requires_grad_(False)
            self._gd_phase = 0
            return None

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
            if not torch.isnan(loss_d_sem) and not torch.isinf(loss_d_sem):
                (loss_d_sem / self.accumulate_grad_batches).backward()
                self._d_sem_accum_count += 1
                self.log(
                    "train/D_sem", loss_d_sem.detach(), on_step=True, prog_bar=True
                )

                if self._d_sem_accum_count >= self.accumulate_grad_batches:
                    self.clip_gradients(
                        d_sem_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
                    )
                    d_sem_opt.step()
                    d_sem_opt.zero_grad()
                    self._d_sem_accum_count = 0
            else:
                self._nan_d_count += 1
                print(
                    f"⚠️ [D step] loss_d_sem 为 NaN/Inf (#{self._nan_d_count})，跳过 D_sem 更新"
                )

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
            if not torch.isnan(loss_d_tex) and not torch.isinf(loss_d_tex):
                (loss_d_tex / self.accumulate_grad_batches).backward()
                self._d_tex_accum_count += 1
                self.log(
                    "train/D_tex", loss_d_tex.detach(), on_step=True, prog_bar=True
                )

                if self._d_tex_accum_count >= self.accumulate_grad_batches:
                    self.clip_gradients(
                        d_tex_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
                    )
                    d_tex_opt.step()
                    d_tex_opt.zero_grad()
                    self._d_tex_accum_count = 0
            else:
                self._nan_d_count += 1
                print(
                    f"⚠️ [D step] loss_d_tex 为 NaN/Inf (#{self._nan_d_count})，跳过 D_tex 更新"
                )

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

        # ── 每 4 个 batch 保存一张拼接对比图 ──
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
