"""
sd2_control_ldm.py — HYPIR + CRefDiff + RefDiffRWKV 融合训练框架

Copyright (c) Shanghai AI Lab. All rights reserved.
原始 ControlLDM 版权归 CRefDiff 作者所有。
本文件为 HYPIR 融合方案的训练框架。
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from PIL import Image
from typing import Dict, Optional, List, Tuple

# ── Diffusers ──
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils.peft_utils import set_weights_and_activate_adapters

# ── 项目内部模块 ──
from RefDiffRWKV import RefDiffRWKV
from sd2_ref_adapter import RWKV_Ref_Adapter
from GlobalSemanticModule import GlobalSemanticModule
import lpips


# ══════════════════════════════════════════════════════════════════════
#  SD2ControlLDM — 训练框架
# ══════════════════════════════════════════════════════════════════════

class SD2ControlLDM(pl.LightningModule):
    """
    基于 diffusers SD2 UNet 的 Reference-guided 扩散训练框架。

    数据流 (Training):
        batch = {lr, ref, hr}
        1.  hr → VAE.encode → latent_hr
        2.  latent_hr + noise → latent_noisy (via DDPMScheduler)
        3.  latent_noisy → VAE.decode → x_t (pixel, 供 RefDiffRWKV 使用)
        4.  x_t + lr + ref → RefDiffRWKV → rf1, rf2, rf3
        5.  rf1-3 → RWKV_Ref_Adapter → adapted_features
        6.  latent_noisy + t + adapted_features → UNet → noise_pred
        7.  loss = MSE(noise_pred, noise)
    """

    def __init__(
        self,
        # ── 数据键名 ──
        lr_key: str = "lr",
        ref_key: str = "ref",
        hr_key: str = "hr",

        # ── SD2 配置 ──
        sd_model_path: str = "runwayml/stable-diffusion-v1-5",
        use_lora: bool = True,
        lora_rank: int = 4,
        sd_locked: bool = True,

        # ── RefDiffRWKV 配置 ──
        ref_patch_size: int = 4,
        ref_embed_dim: int = 384,
        upsample_mode: str = "bilinear",

        # ── GlobalSemantic 配置 ──
        use_global_semantic: bool = True,
        dinov2_model: str = "facebook/dinov2-base",

        # ── 训练配置 ──
        learning_rate: float = 1e-4,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "epsilon",

        # ── Loss 权重 ──
        l_simple_weight: float = 1.0,

        *args, **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['*args', '**kwargs'])

        # ── 数据键 ──
        self.lr_key = lr_key
        self.ref_key = ref_key
        self.hr_key = hr_key
        self.learning_rate = learning_rate
        self.sd_locked = sd_locked
        self.l_simple_weight = l_simple_weight

        # ═══════════════════════════════════════════════════════
        #  1. 加载 SD2 基础组件
        # ═══════════════════════════════════════════════════════

        # VAE (冻结)
        self.vae = AutoencoderKL.from_pretrained(
            sd_model_path, subfolder="vae"
        )
        self.vae.requires_grad_(False)
        self.vae.eval()

        # VAE 缩放因子 (SD2: 0.18215)
        self.vae_scale_factor = self.vae.config.scaling_factor

        # UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            sd_model_path, subfolder="unet"
        )

        # LoRA 注入
        if use_lora:
            self.unet.add_adapter(self._make_lora_config(lora_rank))
            set_weights_and_activate_adapters(self.unet)

        # 冻结控制
        if sd_locked:
            self._freeze_unet_except_attn()

        # Noise Scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )

        # ═══════════════════════════════════════════════════════
        #  2. RefDiffRWKV (ref 特征提取管线)
        # ═══════════════════════════════════════════════════════
        self.ref_model = RefDiffRWKV(
            patch_size=ref_patch_size,
            embed_dim=ref_embed_dim,
            channels=3,
            upsample_mode=upsample_mode,
        )

        # ═══════════════════════════════════════════════════════
        #  3. RWKV_Ref_Adapter (通道映射)
        #     rf1(384) → 320 (SD2 d1)
        #     rf2(768) → 640 (SD2 d2)
        #     rf3(1536)→ 1280 (SD2 d3)
        # ═══════════════════════════════════════════════════════
        self.ref_adapter = RWKV_Ref_Adapter(
            in_dims=(384, 768, 1536),
            out_dims=(320, 640, 1280),
        )

        # ═══════════════════════════════════════════════════════
        #  4. GlobalSemanticModule (可选)
        # ═══════════════════════════════════════════════════════
        if use_global_semantic:
            self.global_semantic = GlobalSemanticModule(
                dinov2_model=dinov2_model,
                unet_dim=ref_embed_dim,
            )
            # 注入 RefDiffRWKV
            self.ref_model.global_semantic = self.global_semantic
        else:
            self.global_semantic = None

        # ═══════════════════════════════════════════════════════
        #  5. 指标 & 其他
        # ═══════════════════════════════════════════════════════
        self.lpips_metric = lpips.LPIPS(net="alex").to(self.device)

        # Hook 注入状态
        self._ref_features_for_hook: Optional[List[torch.Tensor]] = None
        self._hook_idx: int = 0
        self._setup_unet_hooks()

        # 验证指标累计
        self.val_psnr = 0.0
        self.val_lpips = 0.0

    # ──────────────────────────────────────────────────────────
    #  UNet Hook 机制: 在每个 down_block 输出后注入 ref 特征
    # ──────────────────────────────────────────────────────────

    def _setup_unet_hooks(self):
        """在 SD2 UNet 的 3 个 down_blocks 上注册 forward hook。"""
        self._hooks = []
        for block in self.unet.down_blocks:
            hook = block.register_forward_hook(self._ref_injection_hook)
            self._hooks.append(hook)

    def _ref_injection_hook(self, module, input, output):
        """
        UNet down_block 前向完成后触发。
        output 是 (hidden_states, res_samples) 的 tuple。
        """
        if self._ref_features_for_hook is None:
            return output

        idx = self._hook_idx
        if idx < len(self._ref_features_for_hook):
            feat = self._ref_features_for_hook[idx]
            # output[0] 是 hidden_states
            hidden_states = output[0]
            # 尺寸对齐
            if hidden_states.shape[2:] != feat.shape[2:]:
                feat = F.interpolate(
                    feat, size=hidden_states.shape[2:], mode="nearest"
                )
            hidden_states = hidden_states + feat
            output = (hidden_states,) + output[1:]
            self._hook_idx += 1
        return output

    # ──────────────────────────────────────────────────────────
    #  LoRA 配置
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _make_lora_config(rank: int):
        from peft import LoraConfig
        return LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.0,
        )

    def _freeze_unet_except_attn(self):
        """冻结 SD2 UNet 中除 attention 层外的所有参数。"""
        for name, param in self.unet.named_parameters():
            if "attn" not in name:
                param.requires_grad = False

    # ──────────────────────────────────────────────────────────
    #  VAE 编解码 (冻结, 无梯度)
    # ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_latent(self, image: torch.Tensor) -> torch.Tensor:
        """image: (B, 3, H, W) [-1, 1] → latent: (B, 4, H/8, W/8)"""
        latent = self.vae.encode(image).latent_dist.sample()
        return latent * self.vae_scale_factor

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: (B, 4, H/8, W/8) → image: (B, 3, H, W) [-1, 1]"""
        latent = latent / self.vae_scale_factor
        return self.vae.decode(latent).sample

    # ═══════════════════════════════════════════════════════════
    #  get_input: 数据预处理 (骨架来自 ControlLDM)
    # ═══════════════════════════════════════════════════════════

    def get_input(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        """
        处理一个 batch, 返回:
            x_start: (B, 4, h, w) HR latent
            cond:    dict, 包含 ref 特征、文本嵌入、原始图像引用
        """
        # ── 提取 batch 数据 ──
        lr_cond = batch[self.lr_key].to(self.device).float()
        ref_cond = batch[self.ref_key].to(self.device).float()
        hr_cond = batch[self.hr_key].to(self.device).float()

        # 确保 contiguous
        lr_cond = lr_contiguous(lr_cond)
        ref_cond = ref_contiguous(ref_cond)
        hr_cond = hr_contiguous(hr_cond)

        # ── HR → latent ──
        x_start = self.encode_latent(hr_cond)

        # ── Ref 特征提取需要 pixel 空间输入 ──
        # 训练开始时还没有 noisy_latent, 用 clean latent decode 作为初始 x_t
        # 注: 实际训练中 x_t 会在 p_losses 中更新
        x_t_pixel = hr_cond  # 训练时直接用 HR 作为初始 pixel 代理

        # ── RefDiffRWKV: 提取多尺度 ref 特征 ──
        if self.global_semantic is not None:
            rf1, rf2, rf3, _sem = self.ref_model.extract_ref_features(
                x_t=x_t_pixel, LR=lr_cond, Ref=ref_cond
            )
        else:
            rf1, rf2, rf3 = self.ref_model.extract_ref_features(
                x_t=x_t_pixel, LR=lr_cond, Ref=ref_cond
            )

        # ── Adapter: 通道映射 ──
        adapted = self.ref_adapter(rf1, rf2, rf3)

        # ── 空文本嵌入 (SD2 UNet 需要 cross-attn context) ──
        B = hr_cond.shape[0]
        encoder_hidden_states = torch.zeros(
            B, 77, self.unet.config.cross_attention_dim,
            device=self.device, dtype=torch.float32,
        )

        cond = dict(
            ref_features=adapted,               # [feat_d1, feat_d2, feat_d3]
            encoder_hidden_states=encoder_hidden_states,  # (B, 77, 768)
            lq=[lr_cond],
            ref=[ref_cond],
            hr=[hr_cond],
        )

        return x_start, cond

    # ═══════════════════════════════════════════════════════════
    #  apply_model: UNet 前向 (带 ref 注入)
    # ═══════════════════════════════════════════════════════════

    def apply_model(
        self,
        x_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        cond: dict,
    ) -> torch.Tensor:
        """
        SD2 UNet 前向传播, 通过 hook 自动注入 ref 特征。

        Args:
            x_noisy:    (B, 4, h, w) noisy latent
            timesteps:  (B,) 扩散时间步
            cond:       get_input 返回的 cond dict

        Returns:
            noise_pred: (B, 4, h, w) 与 x_noisy 同形状
        """
        # 设置 hook 状态
        self._ref_features_for_hook = list(cond["ref_features"])
        self._hook_idx = 0

        # SD2 UNet forward
        noise_pred = self.unet(
            sample=x_noisy,
            timestep=timesteps,
            encoder_hidden_states=cond["encoder_hidden_states"],
        ).sample

        # 清理 hook 状态
        self._ref_features_for_hook = None

        return noise_pred

    # ═══════════════════════════════════════════════════════════
    #  p_losses: 扩散损失 (骨架来自 ControlLDM)
    # ═══════════════════════════════════════════════════════════

    def p_losses(
        self,
        x_start: torch.Tensor,
        cond: dict,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算扩散损失。

        Args:
            x_start: (B, 4, h, w) clean latent
            cond:    condition dict
            t:       (B,) timestep
            noise:   (B, 4, h, w) 预生成噪声 (None 则随机)

        Returns:
            loss:      scalar
            loss_dict: 各 loss 分量
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # 加噪
        x_noisy = self.noise_scheduler.add_noise(x_start, noise, t)

        # UNet 预测
        model_output = self.apply_model(x_noisy, t, cond)

        # Loss
        prefix = "train" if self.training else "val"
        loss_simple = F.mse_loss(model_output, noise, reduction="none").mean([1, 2, 3])
        loss = self.l_simple_weight * loss_simple.mean()

        loss_dict = {
            f"{prefix}/loss_simple": loss_simple.mean().detach(),
            f"{prefix}/loss": loss.detach(),
        }

        return loss, loss_dict

    # ═══════════════════════════════════════════════════════════
    #  PyTorch Lightning 标准方法
    # ═══════════════════════════════════════════════════════════

    def training_step(self, batch, batch_idx):
        x_start, cond = self.get_input(batch)
        B = x_start.shape[0]
        t = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()
        loss, loss_dict = self.p_losses(x_start, cond, t)
        self.log_dict(loss_dict, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_start, cond = self.get_input(batch)
        B = x_start.shape[0]
        t = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()
        loss, loss_dict = self.p_losses(x_start, cond, t)
        self.log_dict(loss_dict, prog_bar=False, on_epoch=True)

        # ── 采样 + 指标 (每 N 步做一次) ──
        if batch_idx == 0:
            val_results = self.log_images(batch)
            self._compute_validation_metrics(val_results, batch)

    def configure_optimizers(self):
        # 分组: Adapter + RefDiffRWKV (可训练) + UNet attention/LoRA
        params = (
            list(self.ref_model.parameters())
            + list(self.ref_adapter.parameters())
        )
        if self.global_semantic is not None:
            params += list(self.global_semantic.parameters())
        if not self.sd_locked:
            params += list(self.unet.parameters())
        else:
            # 只训练 UNet 中可训练的参数 (attn / LoRA)
            params += [p for p in self.unet.parameters() if p.requires_grad]

        opt = torch.optim.AdamW(params, lr=self.learning_rate)
        return opt

    # ═══════════════════════════════════════════════════════════
    #  采样 & 验证 (骨架来自 ControlLDM)
    # ═══════════════════════════════════════════════════════════

    @torch.no_grad()
    def log_images(self, batch: dict, sample_steps: int = 50) -> dict:
        """采样并返回可视化结果。"""
        log = {}
        log["hq"] = (batch[self.hr_key] + 1) / 2
        log["lq"] = (batch[self.lr_key] + 1) / 2
        log["ref"] = (batch[self.ref_key] + 1) / 2

        # 用随机噪声初始化 latent
        _, cond = self.get_input(batch)
        b, _, h, w = cond["ref_features"][0].shape
        latent_shape = (b, 4, h, w)

        samples = self.sample_ddpm(
            shape=latent_shape,
            cond=cond,
            steps=sample_steps,
        )
        log["samples"] = samples
        return log

    @torch.no_grad()
    def sample_ddpm(
        self,
        shape: tuple,
        cond: dict,
        steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM/DDPM 采样。

        Args:
            shape: (B, C, H, W) latent 形状
            cond:  condition dict
            steps: 采样步数
            eta:   0=DDIM, 1=DDPM

        Returns:
            images: (B, 3, H*8, W*8) pixel space, [0, 1]
        """
        self.unet.eval()

        # 起始噪声
        latents = torch.randn(shape, device=self.device, dtype=torch.float32)

        # 时间步 (等间距)
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(steps)

        for t in scheduler.timesteps:
            t_batch = torch.full((shape[0],), t, device=self.device).long()
            noise_pred = self.apply_model(latents, t_batch, cond)
            latents = scheduler.step(noise_pred, t, latents, eta=eta).prev_sample

        # VAE decode
        images = self.decode_latent(latents)
        images = (images + 1) / 2  # [-1,1] → [0,1]
        images = images.clamp(0, 1)

        self.unet.train()
        return images

    def _compute_validation_metrics(self, val_results: dict, batch: dict):
        """计算 PSNR / LPIPS 并保存图像。"""
        save_dir = os.path.join(
            self.logger.save_dir, "validation", f"step--{self.global_step}"
        )
        os.makedirs(save_dir, exist_ok=True)

        hr = val_results["hq"].detach().cpu()
        sr = val_results["samples"].detach().cpu()

        # PSNR
        psnr = 0.0
        for i in range(len(hr)):
            curr_hr = hr[i].numpy().astype(np.float64)
            curr_sr = sr[i].numpy().astype(np.float64)
            mse = np.mean((curr_hr - curr_sr) ** 2)
            psnr += 20 * math.log10(1.0 / math.sqrt(mse + 1e-8))
        psnr /= len(hr)
        self.log("val_psnr", psnr)

        # LPIPS
        lpips_val = self.lpips_metric(
            hr.clamp(0, 1), sr.clamp(0, 1)
        ).sum().item() / len(hr)
        self.log("val_lpips", lpips_val)

        # 保存图像
        for image_key in val_results:
            os.makedirs(os.path.join(save_dir, image_key), exist_ok=True)
            images = val_results[image_key].detach().cpu()
            for i in range(len(images)):
                if "path" in batch:
                    img_name = os.path.splitext(os.path.basename(batch["path"][i]))[0]
                else:
                    img_name = f"{i:04d}"
                curr_img = images[i].permute(1, 2, 0).numpy()
                curr_img = (curr_img * 255).clip(0, 255).astype(np.uint8)
                path = os.path.join(
                    save_dir, image_key, f"{img_name}_{image_key}.png"
                )
                os.makedirs(os.path.dirname(path), exist_ok=True)
                Image.fromarray(curr_img).save(path)

    # ═══════════════════════════════════════════════════════════
    #  推理接口
    # ═══════════════════════════════════════════════════════════

    @torch.no_grad()
    def inference(
        self,
        lr: torch.Tensor,
        ref: torch.Tensor,
        steps: int = 50,
        seed: int = 42,
    ) -> torch.Tensor:
        """
        单图推理。

        Args:
            lr:   (1, 3, H_lr, W_lr) 低分辨率输入
            ref:  (1, 3, H, W) 参考图像
            steps: 采样步数
            seed:  随机种子

        Returns:
            (1, 3, H*8, W*8) SR 输出 [0, 1]
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # LR 上采样到 target 分辨率 (10×)
        _, _, h_lr, w_lr = lr.shape
        lr_up = F.interpolate(
            lr, size=(h_lr * 10, w_lr * 10),
            mode="bilinear", align_corners=False
        )

        # 确保 Ref 与 LR_up 同分辨率
        if ref.shape[2:] != lr_up.shape[2:]:
            ref = F.interpolate(
                ref, size=lr_up.shape[2:],
                mode="bilinear", align_corners=False
            )

        # 构建 batch (推理时 x_t 用 LR_up 近似)
        batch = {
            self.lr_key: lr,
            self.ref_key: ref,
            self.hr_key: lr_up,  # 占位
        }
        _, cond = self.get_input(batch)

        # Latent 形状
        _, _, H, W = lr_up.shape
        latent_shape = (1, 4, H // 8, W // 8)

        # 采样
        samples = self.sample_ddpm(
            shape=latent_shape,
            cond=cond,
            steps=steps,
            eta=0.0,
        )
        return samples


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

def lr_contiguous(t: torch.Tensor) -> torch.Tensor:
    return t.to(memory_format=torch.contiguous_format)

def ref_contiguous(t: torch.Tensor) -> torch.Tensor:
    return t.to(memory_format=torch.contiguous_format)

def hr_contiguous(t: torch.Tensor) -> torch.Tensor:
    return t.to(memory_format=torch.contiguous_format)


# ══════════════════════════════════════════════════════════════════════
#  测试
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SD2ControlLDM(
        lr_key="lr",
        ref_key="ref",
        hr_key="hr",
        sd_model_path="runwayml/stable-diffusion-v1-5",
        use_lora=True,
        lora_rank=4,
        use_global_semantic=True,
    ).to(device)

    # 统计参数量
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total:,} (~{total/1e6:.1f}M)")

    # Smoke test: 前向传播
    B, H, W = 2, 480, 480
    batch = {
        "lr": torch.randn(B, 3, 48, 48).to(device),
        "ref": torch.randn(B, 3, H, W).to(device),
        "hr": torch.randn(B, 3, H, W).to(device),
    }

    x_start, cond = model.get_input(batch)
    print(f"\nLatent shape: {x_start.shape}")  # (2, 4, 60, 60)
    print(f"Ref features: {[f.shape for f in cond['ref_features']]}")

    t = torch.randint(0, 1000, (B,), device=device).long()
    loss, _ = model.p_losses(x_start, cond, t)
    print(f"Loss: {loss.item():.4f}")

    print("\n✓ Smoke test passed!")
