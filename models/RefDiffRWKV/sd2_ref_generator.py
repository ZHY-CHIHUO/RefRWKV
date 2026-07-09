# sd2_ref_generator.py
"""
sd2_ref_generator.py — SD2 Ref-guided Generator (latent ε-prediction)

设计原则：
  1. 只负责 diffusion 前向 / 采样 / 生成 pixel；
  2. 使用 diffusers 标准 SD2 UNet + T2I-Adapter 风格注入
     (down_intrablock_additional_residuals)，无需自定义 UNet；
  3. 借鉴 ControlLDM 的 get_input / p_losses / sample_log / log_images 接口。
"""

import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from typing import Optional, List, Tuple, Dict

# 兼容“作为包导入”和“直接 python 运行本文件”两种方式
try:
    from .sd2_ref_adapter import SD2_RefAdapter
    from .globalsemanticmodule import GlobalSemanticModule
except ImportError:
    from sd2_ref_adapter import SD2_RefAdapter

    try:
        from globalsemanticmodule import GlobalSemanticModule
    except ImportError:
        GlobalSemanticModule = None


class SD2RefGenerator(LightningModule):
    def __init__(
        self,
        strategy: str = "rwkv",
        sd_model_path: str = "sd2-community/stable-diffusion-2-1-base",
        use_lora: bool = True,
        lora_rank: int = 64,
        lora_target_modules: Optional[List[str]] = None,
        sd_locked: bool = True,
        rwkv_cfg: Optional[dict] = None,
        use_semantic: bool = True,
        dinov2_model_name: str = "facebook/dinov2-base",
        sem_base_dim: int = 768,  # DINOv2 token 维度（base=768, large=1024）
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "epsilon",
        t_min: int = 300,
        t_max: int = 700,
        cfg_drop_prob: float = 0.1,
        control_scale: float = 1.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        lr_key: str = "lr",
        ref_key: str = "ref",
        hr_key: str = "hr",
        normalize_input: bool = False,  # True 时 get_input 内做 [0,1]->[-1,1]
        local_files_only: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr_key = lr_key
        self.ref_key = ref_key
        self.hr_key = hr_key

        self.t_min = t_min
        self.t_max = t_max
        self.cfg_drop_prob = cfg_drop_prob
        self.control_scale = control_scale
        self.normalize_input = normalize_input

        # ═══════════════════════════════════════
        #  VAE（冻结）
        # ═══════════════════════════════════════
        self.vae = AutoencoderKL.from_pretrained(
            sd_model_path, subfolder="vae", local_files_only=local_files_only
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae_scale_factor = self.vae.config.scaling_factor

        # ═══════════════════════════════════════
        #  UNet + LoRA
        # ═══════════════════════════════════════
        self.unet = UNet2DConditionModel.from_pretrained(
            sd_model_path, subfolder="unet", local_files_only=local_files_only
        )
        self.unet.enable_gradient_checkpointing()
        if use_lora:
            self._inject_lora(lora_rank, lora_target_modules)
        if sd_locked:
            self._freeze_unet_except_attn()
        self.cross_attn_dim = self.unet.config.cross_attention_dim

        # ═══════════════════════════════════════
        #  Scheduler
        # ═══════════════════════════════════════
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )

        # ═══════════════════════════════════════
        #  Adapter（输出 f320 / f640 / f1280 三尺度）
        # ═══════════════════════════════════════
        self.adapter = SD2_RefAdapter(strategy=strategy, rwkv_cfg=rwkv_cfg)

        # ═══════════════════════════════════════
        #  DINOv2 语义路径
        # ═══════════════════════════════════════
        self.use_semantic = use_semantic and (GlobalSemanticModule is not None)
        self.global_semantic = (
            GlobalSemanticModule(dinov2_model_name=dinov2_model_name)
            if self.use_semantic
            else None
        )
        if self.global_semantic is not None:
            self.global_semantic.eval()
            self.global_semantic.requires_grad_(False)

        # sem_proj 在 __init__ 就显式建好，保证被 configure_optimizers 纳入、DDP 同步
        self.sem_proj: Optional[nn.Linear] = (
            nn.Linear(sem_base_dim, self.cross_attn_dim) if self.use_semantic else None
        )

        # ═══════════════════════════════════════
        #  Optimizer
        # ═══════════════════════════════════════
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    # ═══════════════════════════════════════════════════════
    #  LoRA & Freeze
    # ═══════════════════════════════════════════════════════
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

    # ═══════════════════════════════════════════════════════
    #  VAE helpers
    # ═══════════════════════════════════════════════════════
    @torch.no_grad()
    def encode_latent(self, img: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(img).latent_dist.sample() * self.vae_scale_factor

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """可梯度 decode，GAN 辅助 loss 需要。"""
        return self.vae.decode(z / self.vae_scale_factor).sample

    @torch.no_grad()
    def decode_latent_eval(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode_latent(z)

    # ═══════════════════════════════════════════════════════
    #  语义 token 构建
    # ═══════════════════════════════════════════════════════
    def build_sem_tokens(
        self, sem_pyramid: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if sem_pyramid is None or self.sem_proj is None:
            return None

        layer_tokens = {}
        for key in ["e1", "e2", "e3", "latent"]:
            if key in sem_pyramid:
                layer_tokens[key] = sem_pyramid[key].float()

        if not layer_tokens:
            return None

        projected = [
            self.sem_proj(layer_tokens[k])
            for k in ["e1", "e2", "e3", "latent"]
            if k in layer_tokens
        ]
        return torch.cat(projected, dim=1)

    def _build_context(self, bsz: int, sem_tokens: Optional[torch.Tensor] = None):
        empty_ctx = torch.zeros(
            bsz, 77, self.cross_attn_dim, device=self.device, dtype=torch.float32
        )
        if sem_tokens is not None:
            return torch.cat([empty_ctx, sem_tokens], dim=1)
        return empty_ctx

    # ═══════════════════════════════════════════════════════
    #  adapter 三尺度特征 -> T2I-Adapter 风格 4 个 down 注入项
    #  通道 [320, 640, 1280, 1280] / 分辨率 [H, H/2, H/4, H/8]
    #  正好对应 SD2 UNet 的 4 个 down block 输出
    # ═══════════════════════════════════════════════════════
    def _build_down_intrablock(
        self,
        ref_feats: List[torch.Tensor],
        latent_h: int,
        latent_w: int,
    ) -> List[torch.Tensor]:
        """
        adapter 输出 3 尺度: f320 / f640 / f1280。
        SD2 UNet 4 个 down block 分辨率按 ceil 减半推导：
            latent=60 -> [60, 30, 15, 8]
        通道 [320, 640, 1280, 1280]。
        """
        f320, f640, f1280 = ref_feats

        # 按 UNet 的 ceil 减半规则推导 4 个 down stage 的目标尺寸
        def _half(h, w):
            return (h + 1) // 2, (w + 1) // 2

        h0, w0 = latent_h, latent_w  # 60
        h1, w1 = _half(h0, w0)  # 30
        h2, w2 = _half(h1, w1)  # 15
        h3, w3 = _half(h2, w2)  # 8

        target_sizes = [(h0, w0), (h1, w1), (h2, w2), (h3, w3)]
        feats = [f320, f640, f1280, f1280]  # 第 4 个复用 f1280 再降采样

        residuals = []
        for feat, (th, tw) in zip(feats, target_sizes):
            if feat.shape[-2] != th or feat.shape[-1] != tw:
                feat = F.interpolate(
                    feat, size=(th, tw), mode="bilinear", align_corners=False
                )
            residuals.append(feat * self.control_scale)

        return residuals

    # ═══════════════════════════════════════════════════════
    #  核心 UNet 前向：latent ε-prediction
    # ═══════════════════════════════════════════════════════
    def apply_model(self, x_t, t, lr, ref, ref_input=None):
        bsz = x_t.shape[0]
        if ref_input is None:
            ref_input = ref

        # adapter 纹理特征 [f320, f640, f1280]
        ref_feats = self.adapter(lr, ref_input)

        # DINOv2 语义 token
        sem_tokens = None
        if self.use_semantic:
            with torch.no_grad():
                sem_pyramid = self.global_semantic(ref_input)
            sem_tokens = self.build_sem_tokens(sem_pyramid)

        context = self._build_context(bsz, sem_tokens)

        _, _, latent_h, latent_w = x_t.shape
        down_intrablock = self._build_down_intrablock(ref_feats, latent_h, latent_w)

        # 注意：diffusers 内部会对该 list 做 pop，需每次传新 list
        noise_pred = self.unet(
            x_t,
            t,
            encoder_hidden_states=context,
            down_intrablock_additional_residuals=list(down_intrablock),
        ).sample

        return noise_pred

    def get_input(self, batch, bs=None, *args, **kwargs):
        """返回: lr, ref, hr，均已 to device / float；normalize_input=True 时映射到 [-1,1]。"""
        lr = batch[self.lr_key]
        ref = batch[self.ref_key]
        hr = batch[self.hr_key]

        if bs is not None:
            lr, ref, hr = lr[:bs], ref[:bs], hr[:bs]

        lr = lr.to(self.device).float()
        ref = ref.to(self.device).float()
        hr = hr.to(self.device).float()

        if self.normalize_input:
            lr = lr * 2.0 - 1.0
            ref = ref * 2.0 - 1.0
            hr = hr * 2.0 - 1.0

        return lr, ref, hr

    # ═══════════════════════════════════════════════════════
    #  按 batch 手动反推 pred_x0（epsilon 预测）
    #  避免把 (B,) 的 t 传入只接受标量的 DDPMScheduler.step
    # ═══════════════════════════════════════════════════════
    def _predict_x0_from_eps(self, x_t, t, noise_pred):
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(x_t.device)
        a_bar = alphas_cumprod[t].float().view(-1, 1, 1, 1)
        return (x_t - (1.0 - a_bar).sqrt() * noise_pred) / a_bar.sqrt()

    def forward(self, lr, ref, hr):
        bsz = lr.shape[0]

        with torch.no_grad():
            hr_latent = self.encode_latent(hr)

        noise = torch.randn_like(hr_latent)
        t = torch.randint(
            self.t_min, self.t_max + 1, (bsz,), device=self.device, dtype=torch.long
        )
        x_t = self.noise_scheduler.add_noise(hr_latent, noise, t)

        # CFG drop
        ref_input = ref
        if self.cfg_drop_prob > 0:
            mask = torch.rand(bsz, device=self.device) < self.cfg_drop_prob
            if mask.any():
                ref_input = ref_input.clone()
                ref_input[mask] = 0.0

        noise_pred = self.apply_model(x_t, t, lr, ref, ref_input=ref_input)
        loss = F.mse_loss(noise_pred, noise)
        pred_x0_latent = self._predict_x0_from_eps(x_t, t, noise_pred)

        return {
            "loss": loss,
            "noise_pred": noise_pred,
            "noise": noise,
            "x_t": x_t,
            "hr_latent": hr_latent,
            "pred_x0_latent": pred_x0_latent,
        }

    def p_losses(self, lr, ref, hr):
        out = self.forward(lr, ref, hr)
        return out["loss"], {"train/loss_diff": out["loss"].detach()}

    # ═══════════════════════════════════════════════════════
    #  采样 / 推理
    # ═══════════════════════════════════════════════════════
    def _infer_latent_size(self, ref, hr=None):
        """latent 尺寸与训练时 HR latent 对齐（目标分辨率 // 8），而非 LR // 8。"""
        target = hr if hr is not None else ref
        return target.shape[-2] // 8, target.shape[-1] // 8

    @torch.no_grad()
    def sample_log(
        self,
        lr,
        ref,
        steps=50,
        sr_model=None,
        hr=None,
        t_start=None,  # Better Start: 加噪目标时间步（None=不用，纯噪声）
        guidance_scale=0.0,  # MSE Guidance: 引导强度（0=不用）
        t_stop=200,  # MSE Guidance: 只在前几步（t > t_stop）引导
    ):
        bsz = lr.shape[0]
        device = self.device

        # ═══════════════════════════════════════
        # Step 1: 初始化 x_t
        # ═══════════════════════════════════════
        sr_target = None  # MSE Guidance 的 target latent

        if sr_model is not None and t_start is not None:
            # ── Better Start: SR prior → 加噪到 t_start ──
            sr_prior = sr_model(lr, ref)
            sr_latent = self.encode_latent(sr_prior)  # 干净 latent
            sr_target = sr_latent.detach().clone()  # 保存为 guidance target

            noise = torch.randn_like(sr_latent)
            t_tensor = torch.full((bsz,), t_start, device=device, dtype=torch.long)
            x_t = self.noise_scheduler.add_noise(sr_latent, noise, t_tensor)

        elif sr_model is not None:
            # ── 仅 MSE Guidance（无 Better Start）: 纯噪声起点 ──
            sr_prior = sr_model(lr, ref)
            sr_target = self.encode_latent(sr_prior).detach()

            latent_h, latent_w = self._infer_latent_size(ref, hr)
            x_t = torch.randn(
                bsz,
                self.unet.config.in_channels,
                latent_h,
                latent_w,
                device=device,
                dtype=lr.dtype,
            )

        else:
            # ── 原始纯噪声路径 ──
            latent_h, latent_w = self._infer_latent_size(ref, hr)
            x_t = torch.randn(
                bsz,
                self.unet.config.in_channels,
                latent_h,
                latent_w,
                device=device,
                dtype=lr.dtype,
            )

        # ═══════════════════════════════════════
        # Step 2: 去噪循环
        # ═══════════════════════════════════════
        self.noise_scheduler.set_timesteps(steps, device=device)

        # 如果有 Better Start，只取 t <= t_start 的时间步
        timesteps = self.noise_scheduler.timesteps
        if t_start is not None:
            timesteps = [t for t in timesteps if t <= t_start]

        for t in timesteps:
            t_tensor = torch.full((bsz,), int(t), device=device, dtype=torch.long)

            # ── MSE Guidance: 用 SR target 修正去噪方向 ──
            if sr_target is not None and guidance_scale > 0 and t > t_stop:
                with torch.enable_grad():
                    x_t.requires_grad_(True)
                    noise_pred = self.apply_model(x_t, t_tensor, lr, ref)
                    pred_x0 = self._predict_x0_from_eps(x_t, t_tensor, noise_pred)

                    loss_guidance = F.mse_loss(pred_x0, sr_target)
                    grad = torch.autograd.grad(loss_guidance, x_t)[0]
                    x_t = x_t.detach() - guidance_scale * grad
                noise_pred = noise_pred.detach()
            else:
                noise_pred = self.apply_model(x_t, t_tensor, lr, ref)

            x_t = self.noise_scheduler.step(noise_pred, t, x_t).prev_sample

        return self.decode_latent_eval(x_t)

    @torch.no_grad()
    def visual_steps(self, lr, ref, steps=50, sr_model=None, hr=None):
        """返回采样过程中每一步的中间图像 [0, 1]，用于排查蓝色偏等质量问题。"""
        bsz = lr.shape[0]
        device = self.device

        if sr_model is not None:
            sr_prior = sr_model(lr, ref)
            x_t = self.encode_latent(sr_prior)
        else:
            latent_h, latent_w = self._infer_latent_size(ref, hr)
            x_t = torch.randn(
                bsz,
                self.unet.config.in_channels,
                latent_h,
                latent_w,
                device=device,
                dtype=lr.dtype,
            )

        self.noise_scheduler.set_timesteps(steps, device=device)
        pixel_each_step = []

        for t in self.noise_scheduler.timesteps:
            t_tensor = torch.full((bsz,), int(t), device=device, dtype=torch.long)
            noise_pred = self.apply_model(x_t, t_tensor, lr, ref)
            x_t = self.noise_scheduler.step(noise_pred, t, x_t).prev_sample
            # 解码当前步的 latent → pixel
            current_pixel = self.decode_latent_eval(x_t)
            current_pixel_01 = torch.clamp((current_pixel + 1.0) / 2.0, 0.0, 1.0)
            pixel_each_step.append(current_pixel_01)

        return pixel_each_step  # list of (B,C,H,W) tensors, 从第一步到最后一步

    @torch.no_grad()
    def validation_inference(self, batch, save_dir, steps=50, sr_model=None):
        """验证推理，记录推理时间和显存，保存结果图。"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        lr, ref, hr = self.get_input(batch)

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        starter.record()
        val_results = self.log_images(batch, steps=steps, sr_model=sr_model)
        ender.record()
        torch.cuda.synchronize()
        elapsed_time = starter.elapsed_time(ender)  # ms
        max_memory = torch.cuda.memory_allocated() / 1024**2  # MB

        print(f"[Inference Time] {elapsed_time:.2f} ms")
        print(f"[Current Memory] {max_memory:.2f} MB")

        os.makedirs(save_dir, exist_ok=True)
        for image_key in val_results:
            image = val_results[image_key].detach().cpu()
            for i in range(len(image)):
                curr_img = image[i].permute(1, 2, 0).numpy()
                curr_img = (curr_img * 255).clip(0, 255).astype(np.uint8)
                filename = f"{i}_{image_key}.png"
                path = os.path.join(save_dir, filename)
                Image.fromarray(curr_img).save(path)

    @torch.no_grad()
    def log_images(self, batch, steps=50, sr_model=None):
        lr, ref, hr = self.get_input(batch)
        samples = self.sample_log(lr, ref, steps=steps, sr_model=sr_model, hr=hr)
        return {
            "lq": (lr + 1.0) / 2.0,
            "ref": (ref + 1.0) / 2.0,
            "hq": (hr + 1.0) / 2.0,
            "samples": torch.clamp((samples + 1.0) / 2.0, 0.0, 1.0),
        }

    @torch.no_grad()
    def generate_sr(self, lr, ref, steps=50, sr_model=None, hr=None):
        """返回 pixel 空间 SR 图像 [0, 1]。"""
        samples = self.sample_log(lr, ref, steps=steps, sr_model=sr_model, hr=hr)
        return (samples + 1.0) / 2.0

    # ═══════════════════════════════════════════════════════
    #  Lightning 接口
    # ═══════════════════════════════════════════════════════
    def training_step(self, batch, batch_idx):
        lr, ref, hr = self.get_input(batch)
        loss, log_dict = self.p_losses(lr, ref, hr)
        self.log_dict(log_dict, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lr, ref, hr = self.get_input(batch)
        loss, _ = self.p_losses(lr, ref, hr)
        self.log("val/loss_diff", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = []
        params += [p for p in self.adapter.parameters() if p.requires_grad]
        params += [p for p in self.unet.parameters() if p.requires_grad]
        if self.use_semantic and self.sem_proj is not None:
            params += list(self.sem_proj.parameters())

        return torch.optim.AdamW(
            params, lr=self.learning_rate, weight_decay=self.weight_decay
        )


# ══════════════════════════════════════════════════════════════
#  测试 main：验证 forward / sample_log / control 注入是否跑通
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sd_path",
        type=str,
        default="sd2-community/stable-diffusion-2-1-base",
        help="SD2 权重路径（含 vae / unet 子目录）",
    )
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--hr_size", type=int, default=480)
    parser.add_argument("--lr_size", type=int, default=48)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("  SD2RefGenerator — down_intrablock (T2I-Adapter 风格) 注入测试")
    print("=" * 80)

    # 关闭 DINOv2 语义路径，避免额外依赖；策略用 rwkv
    model = SD2RefGenerator(
        strategy="rwkv",
        sd_model_path=args.sd_path,
        use_lora=True,
        sd_locked=True,
        use_semantic=False,
        cfg_drop_prob=0.0,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    B = 1
    hr = torch.rand(B, 3, args.hr_size, args.hr_size, device=device) * 2 - 1
    ref = torch.rand(B, 3, args.hr_size, args.hr_size, device=device) * 2 - 1
    lr = torch.rand(B, 3, args.lr_size, args.lr_size, device=device) * 2 - 1
    print(f"Input : LR={list(lr.shape)}, Ref={list(ref.shape)}, HR={list(hr.shape)}")

    # 1) 训练前向（含 loss / pred_x0）
    out = model.forward(lr, ref, hr)
    print("\n--- forward ---")
    print(f"  loss           = {out['loss'].item():.4f}")
    print(f"  noise_pred     = {list(out['noise_pred'].shape)}")
    print(f"  hr_latent      = {list(out['hr_latent'].shape)}")
    print(f"  pred_x0_latent = {list(out['pred_x0_latent'].shape)}")
    assert out["noise_pred"].shape == out["hr_latent"].shape
    print("  ✅ forward passed!")

    # 2) 采样（少量步数验证 latent 尺寸与注入尺寸一致）
    print("\n--- sample_log (steps=2) ---")
    with torch.no_grad():
        samples = model.sample_log(lr, ref, steps=2, hr=hr)
    print(f"  samples(pixel) = {list(samples.shape)}")
    assert samples.shape[-1] == args.hr_size, "采样输出分辨率应等于 HR 尺寸"
    print("  ✅ sample_log passed!")

    print(
        f"\n  Trainable params: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print("=" * 80)
    print("  ✅ 所有测试通过")
    print("=" * 80)
