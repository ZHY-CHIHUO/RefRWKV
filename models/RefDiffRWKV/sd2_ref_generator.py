# sd2_ref_generator.py
"""
sd2_ref_generator.py — SD2 Ref-guided Generator (latent ε-prediction)

设计原则：
  1. 只负责 diffusion 前向 / 采样 / 生成 pixel；
  2. 使用 diffusers 标准 SD2 UNet + T2I-Adapter 风格注入
     (down_intrablock_additional_residuals)，无需自定义 UNet；
  3. 借鉴 ControlLDM 的 get_input / p_losses / sample_log / log_images 接口。
  4. UNet conv_in 扩展为 8 通道：前 4 为 noisy_latent，后 4 为 sr_latent（条件）。
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

# 兼容"作为包导入"和"直接 python 运行本文件"两种方式
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
        # SR prior 注入 ──
        sr_model: Optional[torch.nn.Module] = None,
        use_sr_latent_cond: bool = True,  # 是否在训练时把 sr_latent 拼接到 UNet 输入
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["sr_model"])

        self.lr_key = lr_key
        self.ref_key = ref_key
        self.hr_key = hr_key

        self.t_min = t_min
        self.t_max = t_max
        self.cfg_drop_prob = cfg_drop_prob
        self.control_scale = control_scale
        self.normalize_input = normalize_input

        self.sr_model = sr_model
        self.use_sr_latent_cond = use_sr_latent_cond

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

        # 先扩展 conv_in 再注入 LoRA：LoRA 只钩 attention，不影响 conv_in
        # 但 conv_in 的替换需要在 freeze 之前完成
        self._expand_conv_in_for_sr_latent()

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

        self.sem_proj: Optional[nn.Linear] = (
            nn.Linear(sem_base_dim, self.cross_attn_dim) if self.use_semantic else None
        )

        # ═══════════════════════════════════════
        #  Optimizer
        # ═══════════════════════════════════════
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    # ═══════════════════════════════════════════════════════
    #  UNet conv_in 4→8 通道扩展
    # ═══════════════════════════════════════════════════════
    def _expand_conv_in_for_sr_latent(self):
        """把 UNet conv_in 从 4 通道扩展到 8 通道。
        前 4 通道保留 SD2 预训练权重，后 4 通道零初始化（InstructPix2Pix 范式）。
        """
        old_conv = self.unet.conv_in
        old_weight = old_conv.weight.data  # [320, 4, 3, 3]

        new_conv = nn.Conv2d(
            8,
            old_weight.shape[0],
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        )

        with torch.no_grad():
            new_conv.weight[:, :4] = old_weight  # 前 4 通道：原始预训练
            new_conv.weight[:, 4:] = 0.0  # 后 4 通道：零初始化
            new_conv.bias.copy_(old_conv.bias)

        self.unet.conv_in = new_conv

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

        def _half(h, w):
            return (h + 1) // 2, (w + 1) // 2

        h0, w0 = latent_h, latent_w
        h1, w1 = _half(h0, w0)
        h2, w2 = _half(h1, w1)
        h3, w3 = _half(h2, w2)

        target_sizes = [(h0, w0), (h1, w1), (h2, w2), (h3, w3)]
        feats = [f320, f640, f1280, f1280]

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
    #  接受 [B, 8, H, W] 输入（noisy_latent + sr_latent）
    # ═══════════════════════════════════════════════════════
    def apply_model(self, x_input, t, lr, ref, ref_input=None):
        """x_input: [B, 8, 60, 60] 或 [B, 4, 60, 60]（回退兼容）。"""
        bsz = x_input.shape[0]
        if ref_input is None:
            ref_input = ref

        ref_feats = self.adapter(lr, ref_input)

        sem_tokens = None
        if self.use_semantic:
            with torch.no_grad():
                sem_pyramid = self.global_semantic(ref_input)
            sem_tokens = self.build_sem_tokens(sem_pyramid)

        context = self._build_context(bsz, sem_tokens)

        _, _, latent_h, latent_w = x_input.shape
        down_intrablock = self._build_down_intrablock(ref_feats, latent_h, latent_w)

        noise_pred = self.unet(
            x_input,
            t,
            encoder_hidden_states=context,
            down_intrablock_additional_residuals=list(down_intrablock),
        ).sample

        return noise_pred

    def get_input(self, batch, bs=None, *args, **kwargs):
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
    #  反推 pred_x0（epsilon 预测）
    # ═══════════════════════════════════════════════════════
    def _predict_x0_from_eps(self, x_t, t, noise_pred):
        """x_t 是 4 通道 noisy latent（不含拼接的 sr_latent）。"""
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(x_t.device)
        a_bar = alphas_cumprod[t].float().view(-1, 1, 1, 1)
        return (x_t - (1.0 - a_bar).sqrt() * noise_pred) / a_bar.sqrt()

    # ═══════════════════════════════════════════════════════
    #  训练前向
    # ═══════════════════════════════════════════════════════
    def forward(self, lr, ref, hr):
        bsz = lr.shape[0]

        with torch.no_grad():
            hr_latent = self.encode_latent(hr)

        noise = torch.randn_like(hr_latent)
        t = torch.randint(
            self.t_min, self.t_max + 1, (bsz,), device=self.device, dtype=torch.long
        )
        x_t = self.noise_scheduler.add_noise(hr_latent, noise, t)

        # SR prior latent 作为条件（每一步 UNet 都看到目标方向）──
        sr_latent_cond = self._get_sr_latent_cond(lr, ref)

        # CFG drop（只影响 adapter 的 ref 输入，不影响 sr_latent_cond）
        ref_input = ref
        if self.cfg_drop_prob > 0:
            mask = torch.rand(bsz, device=self.device) < self.cfg_drop_prob
            if mask.any():
                ref_input = ref_input.clone()
                ref_input[mask] = 0.0

        # 拼接 sr_latent 到 UNet 输入 ──
        x_t_input = self._concat_sr_latent(x_t, sr_latent_cond)
        noise_pred = self.apply_model(x_t_input, t, lr, ref, ref_input=ref_input)

        loss = F.mse_loss(noise_pred, noise)

        pred_x0_latent = self._predict_x0_from_eps(x_t, t, noise_pred)
        pred_x0_latent = torch.nan_to_num(
            pred_x0_latent, nan=0.0, posinf=20.0, neginf=-20.0
        )
        pred_x0_latent = pred_x0_latent.clamp(-20.0, 20.0)

        return {
            "loss": loss,
            "noise_pred": noise_pred,
            "noise": noise,
            "x_t": x_t,
            "hr_latent": hr_latent,
            "pred_x0_latent": pred_x0_latent,
            "t": t,
        }

    def p_losses(self, lr, ref, hr):
        out = self.forward(lr, ref, hr)
        return out["loss"], {"train/loss_diff": out["loss"].detach()}

    # ═══════════════════════════════════════════════════════
    #  sr_latent 条件生成
    # ═══════════════════════════════════════════════════════
    def _get_sr_latent_cond(self, lr, ref) -> Optional[torch.Tensor]:
        actual_sr = self.sr_model if self.sr_model is not None else None
        if actual_sr is None or not self.use_sr_latent_cond:
            return None
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=False):
                sr_prior = actual_sr(lr.float(), ref.float())
                sr_prior = torch.nan_to_num(sr_prior, nan=0.0, posinf=1.0, neginf=-1.0)
                sr_prior = sr_prior.clamp(-1.0, 1.0)
                return self.encode_latent(sr_prior.to(self.vae.dtype))

    def _concat_sr_latent(
        self, x_t: torch.Tensor, sr_latent_cond: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """把 4 通道 x_t 和可选的 sr_latent_cond 拼成 8 通道输入。"""
        if sr_latent_cond is not None:
            return torch.cat([x_t, sr_latent_cond], dim=1)
        else:
            return torch.cat([x_t, torch.zeros_like(x_t)], dim=1)

    # ═══════════════════════════════════════════════════════
    #  采样 / 推理
    # ═══════════════════════════════════════════════════════
    def _infer_latent_size(self, ref, hr=None):
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
        t_start=None,
        guidance_scale=0.0,
        t_stop=200,
    ):
        bsz = lr.shape[0]
        device = self.device

        # ── 选择 SR model：参数优先，fallback 到存储的 ──
        actual_sr = sr_model if sr_model is not None else self.sr_model

        # ═══════════════════════════════════════
        # Step 1: sr_latent 条件（全去噪循环复用）
        # ═══════════════════════════════════════
        sr_latent_cond = None
        if actual_sr is not None:
            with torch.amp.autocast("cuda", enabled=False):
                sr_prior_for_cond = actual_sr(lr.float(), ref.float())
                sr_prior_for_cond = torch.nan_to_num(
                    sr_prior_for_cond, nan=0.0, posinf=1.0, neginf=-1.0
                )
                sr_prior_for_cond = sr_prior_for_cond.clamp(-1.0, 1.0)
                sr_latent_cond = self.encode_latent(
                    sr_prior_for_cond.to(self.vae.dtype)
                )

        # ═══════════════════════════════════════
        # Step 2: 初始化 x_t
        # ═══════════════════════════════════════
        sr_target = None

        if actual_sr is not None and t_start is not None:
            # ───────────────────────────────────
            # Better Start + 空间自适应加噪
            # ───────────────────────────────────
            sr_prior = sr_prior_for_cond
            sr_latent = (
                sr_latent_cond
                if sr_latent_cond is not None
                else self.encode_latent(sr_prior)
            )
            sr_target = sr_latent.detach().clone()

            sr_pixel_01 = (sr_prior + 1.0) / 2.0
            ref_pixel_01 = (ref + 1.0) / 2.0
            tex_map = (sr_pixel_01 - ref_pixel_01).abs().mean(dim=1, keepdim=True)

            tex_map_latent = F.interpolate(
                tex_map,
                size=sr_latent.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

            tex_norm = tex_map_latent.clone()
            for b in range(bsz):
                tmin = tex_norm[b].min()
                tmax = tex_norm[b].max()
                if tmax > tmin:
                    tex_norm[b] = (tex_norm[b] - tmin) / (tmax - tmin + 1e-8)

            t_flat = max(200, t_start - 500)
            t_detail = min(999, t_start + 200)

            noise = torch.randn_like(sr_latent)
            t_flat_t = torch.full((bsz,), t_flat, device=device, dtype=torch.long)
            t_detail_t = torch.full((bsz,), t_detail, device=device, dtype=torch.long)

            x_t_flat = self.noise_scheduler.add_noise(sr_latent, noise, t_flat_t)
            x_t_detail = self.noise_scheduler.add_noise(sr_latent, noise, t_detail_t)

            x_t = tex_norm * x_t_detail + (1.0 - tex_norm) * x_t_flat

        elif actual_sr is not None:
            sr_prior = sr_prior_for_cond
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
        # Step 3: 去噪循环（每步拼接 sr_latent_cond）
        # ═══════════════════════════════════════
        self.noise_scheduler.set_timesteps(steps, device=device)

        timesteps = self.noise_scheduler.timesteps
        if t_start is not None:
            timesteps = [t for t in timesteps if t <= t_start]

        for t in timesteps:
            t_tensor = torch.full((bsz,), int(t), device=device, dtype=torch.long)

            # ── 拼接 sr_latent_cond ──
            x_t_input = self._concat_sr_latent(x_t, sr_latent_cond)

            if sr_target is not None and guidance_scale > 0 and t > t_stop:
                with torch.enable_grad():
                    x_t.requires_grad_(True)
                    x_t_input = self._concat_sr_latent(x_t, sr_latent_cond)
                    noise_pred = self.apply_model(x_t_input, t_tensor, lr, ref)
                    pred_x0 = self._predict_x0_from_eps(x_t, t_tensor, noise_pred)

                    loss_guidance = F.mse_loss(pred_x0, sr_target)
                    grad = torch.autograd.grad(loss_guidance, x_t)[0]
                    x_t = x_t.detach() - guidance_scale * grad
                noise_pred = noise_pred.detach()
            else:
                noise_pred = self.apply_model(x_t_input, t_tensor, lr, ref)

            x_t = self.noise_scheduler.step(noise_pred, t, x_t).prev_sample

        return self.decode_latent_eval(x_t)

    @torch.no_grad()
    def visual_steps(self, lr, ref, steps=50, sr_model=None, hr=None):
        bsz = lr.shape[0]
        device = self.device
        actual_sr = sr_model if sr_model is not None else self.sr_model

        if actual_sr is not None:
            with torch.amp.autocast("cuda", enabled=False):
                sr_prior = actual_sr(lr.float(), ref.float())
                sr_prior = torch.nan_to_num(
                    sr_prior, nan=0.0, posinf=1.0, neginf=-1.0
                ).clamp(-1.0, 1.0)
                x_t = self.encode_latent(sr_prior.to(self.vae.dtype))
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

        sr_latent_cond = (
            self._get_sr_latent_cond(lr, ref) if actual_sr is not None else None
        )

        for t in self.noise_scheduler.timesteps:
            t_tensor = torch.full((bsz,), int(t), device=device, dtype=torch.long)
            x_t_input = self._concat_sr_latent(x_t, sr_latent_cond)
            noise_pred = self.apply_model(x_t_input, t_tensor, lr, ref)
            x_t = self.noise_scheduler.step(noise_pred, t, x_t).prev_sample
            current_pixel = self.decode_latent_eval(x_t)
            current_pixel_01 = torch.clamp((current_pixel + 1.0) / 2.0, 0.0, 1.0)
            pixel_each_step.append(current_pixel_01)

        return pixel_each_step

    @torch.no_grad()
    def validation_inference(self, batch, save_dir, steps=50, sr_model=None):
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
        elapsed_time = starter.elapsed_time(ender)
        max_memory = torch.cuda.memory_allocated() / 1024**2

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
    def log_images(
        self,
        batch,
        steps=50,
        sr_model=None,
        t_start=None,
        guidance_scale=0.0,
        t_stop=200,
    ):
        lr, ref, hr = self.get_input(batch)
        samples = self.sample_log(
            lr,
            ref,
            steps=steps,
            sr_model=sr_model,
            hr=hr,
            t_start=t_start,
            guidance_scale=guidance_scale,
            t_stop=t_stop,
        )
        return {
            "lq": (lr + 1.0) / 2.0,
            "ref": (ref + 1.0) / 2.0,
            "hq": (hr + 1.0) / 2.0,
            "samples": torch.clamp((samples + 1.0) / 2.0, 0.0, 1.0),
        }

    @torch.no_grad()
    def generate_sr(
        self,
        lr,
        ref,
        steps=50,
        sr_model=None,
        hr=None,
        t_start=None,
        guidance_scale=0.0,
        t_stop=200,
    ):
        samples = self.sample_log(
            lr,
            ref,
            steps=steps,
            sr_model=sr_model,
            hr=hr,
            t_start=t_start,
            guidance_scale=guidance_scale,
            t_stop=t_stop,
        )
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
#  测试 main
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
    print("  SD2RefGenerator — 8ch conv_in (sr_latent 条件) 测试")
    print("=" * 80)

    model = SD2RefGenerator(
        strategy="rwkv",
        sd_model_path=args.sd_path,
        use_lora=True,
        sd_locked=True,
        use_semantic=False,
        cfg_drop_prob=0.0,
        local_files_only=args.local_files_only,
        sr_model=None,
    ).to(device)
    model.eval()

    # 验证 conv_in 是 8 通道
    assert (
        model.unet.conv_in.in_channels == 8
    ), f"Expected 8-channel conv_in, got {model.unet.conv_in.in_channels}"
    print("✅ conv_in 8 通道验证通过")

    # 验证前 4 通道 = 原始 SD2 权重（非全零）
    assert (
        model.unet.conv_in.weight[:, :4].abs().sum() > 0
    ), "前 4 通道应为非零预训练权重"
    print("✅ 前 4 通道预训练权重保留")

    # 验证后 4 通道 = 零初始化
    assert model.unet.conv_in.weight[:, 4:].abs().sum() == 0, "后 4 通道应全为零"
    print("✅ 后 4 通道零初始化")

    B = 1
    hr = torch.rand(B, 3, args.hr_size, args.hr_size, device=device) * 2 - 1
    ref = torch.rand(B, 3, args.hr_size, args.hr_size, device=device) * 2 - 1
    lr = torch.rand(B, 3, args.lr_size, args.lr_size, device=device) * 2 - 1
    print(f"Input : LR={list(lr.shape)}, Ref={list(ref.shape)}, HR={list(hr.shape)}")

    # 1) 训练前向（sr_model=None，回退拼零）
    out = model.forward(lr, ref, hr)
    print("\n--- forward (sr_model=None) ---")
    print(f"  loss           = {out['loss'].item():.4f}")
    print(f"  noise_pred     = {list(out['noise_pred'].shape)}")
    assert out["noise_pred"].shape == out["hr_latent"].shape
    print("  ✅ forward passed!")

    # 2) 采样
    print("\n--- sample_log (steps=2) ---")
    with torch.no_grad():
        samples = model.sample_log(lr, ref, steps=2, hr=hr)
    print(f"  samples(pixel) = {list(samples.shape)}")
    assert samples.shape[-1] == args.hr_size
    print("  ✅ sample_log passed!")

    print(
        f"\n  Trainable params: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print("=" * 80)
    print("  ✅ 所有测试通过 — UNet 8 通道扩展成功")
    print("=" * 80)
