"""
sd2_ref_generator.py — SD2 Ref-guided Generator (latent ε-prediction)

设计原则：
  1. 只负责 diffusion 前向 / 采样 / 生成 pixel；
  2. 使用 diffusers SD2 UNet + control list 注入，与 CRefDiff 对齐；
  3. 借鉴 ControlLDM 的 get_input / p_losses / sample_log / log_images 接口；
  4. 支持 Better Start：训练时随机以 SR prior 为起点，低噪声区域 refine 高频。
"""

import os
import math
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

from .sd2_ref_adapter import SD2_RefAdapter
from .globalsemanticmodule import GlobalSemanticModule


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
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "epsilon",
        t_min: int = 0,
        t_max: int = 999,
        cfg_drop_prob: float = 0.1,
        control_scale: float = 1.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        lr_key: str = "lr",
        ref_key: str = "ref",
        hr_key: str = "hr",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr_key = lr_key
        self.ref_key = ref_key
        self.hr_key = hr_key

        self.t_min = t_min
        self.t_max = min(t_max, num_train_timesteps - 1)
        self.cfg_drop_prob = cfg_drop_prob
        self.control_scale = control_scale

        # ═══════════════════════════════════════
        #  VAE（冻结）
        # ═══════════════════════════════════════
        self.vae = AutoencoderKL.from_pretrained(
            sd_model_path, subfolder="vae", local_files_only=True
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae_scale_factor = self.vae.config.scaling_factor

        # ═══════════════════════════════════════
        #  UNet + LoRA
        # ═══════════════════════════════════════
        self.unet = UNet2DConditionModel.from_pretrained(
            sd_model_path, subfolder="unet", local_files_only=True
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
        #  Adapter
        # ═══════════════════════════════════════
        self.adapter = SD2_RefAdapter(strategy=strategy, rwkv_cfg=rwkv_cfg)

        # ═══════════════════════════════════════
        #  DINOv2 语义路径
        # ═══════════════════════════════════════
        self.use_semantic = use_semantic
        self.global_semantic = (
            GlobalSemanticModule(dinov2_model_name=dinov2_model_name)
            if use_semantic
            else None
        )
        if self.global_semantic is not None:
            self.global_semantic.eval()
            self.global_semantic.requires_grad_(False)
        self.sem_proj: Optional[nn.Linear] = None

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
    def build_sem_tokens(self, sem_pyramid: Dict[str, torch.Tensor]) -> torch.Tensor:
        if sem_pyramid is None or self.global_semantic is None:
            return None

        layer_tokens = {}
        for key in ["e1", "e2", "e3", "latent"]:
            if key in sem_pyramid:
                layer_tokens[key] = sem_pyramid[key].float()

        if not layer_tokens:
            return None

        if self.sem_proj is None:
            first_tokens = next(iter(layer_tokens.values()))
            base_dim = first_tokens.shape[-1]
            self.sem_proj = nn.Linear(base_dim, self.cross_attn_dim).to(
                first_tokens.device
            )
            total_tokens = sum(t.shape[1] for t in layer_tokens.values())
            print(
                f"[SemProj] layers={list(layer_tokens.keys())} "
                f"-> {total_tokens} tokens x {self.cross_attn_dim}"
            )

        projected = []
        for key in ["e1", "e2", "e3", "latent"]:
            if key in layer_tokens:
                projected.append(self.sem_proj(layer_tokens[key]))

        return torch.cat(projected, dim=1)

    def _build_context(self, bsz: int, sem_tokens: Optional[torch.Tensor] = None):
        empty_ctx = torch.zeros(
            bsz, 77, self.cross_attn_dim, device=self.device, dtype=torch.float32
        )
        if sem_tokens is not None:
            return torch.cat([empty_ctx, sem_tokens], dim=1)
        return empty_ctx

    # ═══════════════════════════════════════════════════════
    #  把 adapter 3 尺度特征扩展为 SD2 UNet 需要的 4 + 1 + 4
    # ═══════════════════════════════════════════════════════
    def _build_control_lists(
        self,
        ref_feats: List[torch.Tensor],
        latent_h: int,
        latent_w: int,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        f320, f640, f1280 = ref_feats

        # down blocks: H, H/2, H/4, H/8
        f1280_down = F.interpolate(
            f1280,
            size=(latent_h // 8, latent_w // 8),
            mode="bilinear",
            align_corners=False,
        )
        down_block_additional_residuals = [f320, f640, f1280, f1280_down]
        down_block_additional_residuals = [
            x * self.control_scale for x in down_block_additional_residuals
        ]

        # mid block: H/8
        mid_block_additional_residual = f1280_down * self.control_scale

        # up blocks: H/8, H/4, H/2, H
        f1280_up = F.interpolate(
            f1280,
            size=(latent_h // 4, latent_w // 4),
            mode="bilinear",
            align_corners=False,
        )
        f640_up = F.interpolate(
            f640,
            size=(latent_h // 2, latent_w // 2),
            mode="bilinear",
            align_corners=False,
        )
        up_block_additional_residuals = [f1280, f1280_up, f640_up, f320]
        up_block_additional_residuals = [
            x * self.control_scale for x in up_block_additional_residuals
        ]

        return (
            down_block_additional_residuals,
            mid_block_additional_residual,
            up_block_additional_residuals,
        )

    # ═══════════════════════════════════════════════════════
    #  核心 UNet 前向
    # ═══════════════════════════════════════════════════════
    def apply_model(self, x_t, t, lr, ref, ref_input=None):
        bsz = x_t.shape[0]
        if ref_input is None:
            ref_input = ref

        # adapter 纹理特征 [f320, f640, f1280]
        ref_feats = self.adapter(lr=lr, ref=ref_input)

        # DINOv2 语义 token
        sem_tokens = None
        if self.use_semantic:
            with torch.no_grad():
                sem_pyramid = self.global_semantic(ref_input)
            sem_tokens = self.build_sem_tokens(sem_pyramid)

        context = self._build_context(bsz, sem_tokens)

        _, _, latent_h, latent_w = x_t.shape
        down_res, mid_res, up_res = self._build_control_lists(ref_feats, latent_h, latent_w)

        noise_pred = self.unet(
            x_t,
            t,
            encoder_hidden_states=context,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
            up_block_additional_residuals=up_res,
        ).sample

        return noise_pred

    def get_input(self, batch, bs=None, *args, **kwargs):
        """
        借鉴 ControlLDM.get_input。
        返回: lr, ref, hr，均已 to device / float / [-1, 1]。
        """
        lr = batch[self.lr_key]
        ref = batch[self.ref_key]
        hr = batch[self.hr_key]

        if bs is not None:
            lr = lr[:bs]
            ref = ref[:bs]
            hr = hr[:bs]

        lr = lr.to(self.device).float()
        ref = ref.to(self.device).float()
        hr = hr.to(self.device).float()

        return lr, ref, hr

    def forward(
        self,
        lr,
        ref,
        hr,
        sr_model=None,
        better_start_prob: float = 0.5,
        t_max_better: int = 200,
    ):
        """
        训练前向。

        分支策略：
          - 标准扩散 (probability 1 - better_start_prob): x0 = HR latent, t ~ [0, T-1]
          - Better Start (probability better_start_prob): x0 = SR prior latent, t ~ [0, t_max_better]
        """
        bsz = lr.shape[0]

        use_better_start = (sr_model is not None) and (torch.rand(1).item() < better_start_prob)

        # 起点 latent
        if use_better_start:
            with torch.no_grad():
                sr_prior = sr_model(lr, ref)
            x0 = self.encode_latent(sr_prior)
            t = torch.randint(
                0,
                min(t_max_better + 1, self.noise_scheduler.num_train_timesteps),
                (bsz,),
                device=self.device,
                dtype=torch.long,
            )
        else:
            with torch.no_grad():
                x0 = self.encode_latent(hr)
            t = torch.randint(
                self.t_min,
                self.t_max + 1,
                (bsz,),
                device=self.device,
                dtype=torch.long,
            )

        # 加噪
        noise = torch.randn_like(x0)
        x_t = self.noise_scheduler.add_noise(x0, noise, t)

        # CFG drop
        ref_input = ref
        if self.cfg_drop_prob > 0:
            mask = torch.rand(bsz, device=self.device) < self.cfg_drop_prob
            if mask.any():
                ref_input = ref_input.clone()
                ref_input[mask] = 0.0

        # UNet 预测噪声
        noise_pred = self.apply_model(x_t, t, lr, ref, ref_input=ref_input)

        # latent diffusion loss
        loss = F.mse_loss(noise_pred, noise)

        # 预测 x0 latent（用于 GAN / 可视化）
        pred_x0_latent = self.noise_scheduler.step(noise_pred, t, x_t).pred_original_sample

        return {
            "loss": loss,
            "noise_pred": noise_pred,
            "noise": noise,
            "x_t": x_t,
            "pred_x0_latent": pred_x0_latent,
            "use_better_start": use_better_start,
        }

    def p_losses(self, lr, ref, hr, sr_model=None, better_start_prob: float = 0.5, t_max_better: int = 200):
        """
        借鉴 ControlLDM.p_losses，返回 loss 和 log dict。
        """
        out = self.forward(lr, ref, hr, sr_model=sr_model, better_start_prob=better_start_prob, t_max_better=t_max_better)
        log_dict = {
            "train/loss_diff": out["loss"].detach(),
            "train/better_start": float(out["use_better_start"]),
        }
        return out["loss"], log_dict

    # ═══════════════════════════════════════════════════════
    #  采样 / 推理
    # ═══════════════════════════════════════════════════════
    @torch.no_grad()
    def sample_log(self, lr, ref, steps=50, sr_model=None):
        """
        借鉴 ControlLDM.sample_log。
        如果提供 sr_model，使用 Better Start：sr_prior 作为 x0 起点。
        """
        bsz = lr.shape[0]
        device = self.device

        if sr_model is not None:
            sr_prior = sr_model(lr, ref)
            x_t = self.encode_latent(sr_prior)
        else:
            x_t = torch.randn(
                bsz,
                self.unet.config.in_channels,
                lr.shape[-2] // 8,
                lr.shape[-1] // 8,
                device=device,
                dtype=lr.dtype,
            )

        self.noise_scheduler.set_timesteps(steps, device=device)

        for t in self.noise_scheduler.timesteps:
            t_tensor = torch.full((bsz,), t, device=device, dtype=torch.long)
            noise_pred = self.apply_model(x_t, t_tensor, lr, ref)
            x_t = self.noise_scheduler.step(noise_pred, t_tensor, x_t).prev_sample

        samples = self.decode_latent_eval(x_t)
        return samples

    @torch.no_grad()
    def log_images(self, batch, steps=50, sr_model=None):
        """
        借鉴 ControlLDM.log_images，返回可视化 dict。
        """
        lr, ref, hr = self.get_input(batch)
        samples = self.sample_log(lr, ref, steps=steps, sr_model=sr_model)
        log = {
            "lq": (lr + 1.0) / 2.0,
            "ref": (ref + 1.0) / 2.0,
            "hq": (hr + 1.0) / 2.0,
            "samples": (samples + 1.0) / 2.0,
        }
        return log

    @torch.no_grad()
    def generate_sr(self, lr, ref, steps=50, sr_model=None):
        """返回 pixel 空间 SR 图像 [0, 1]。"""
        samples = self.sample_log(lr, ref, steps=steps, sr_model=sr_model)
        return (samples + 1.0) / 2.0

    # ═══════════════════════════════════════════════════════
    #  Lightning 接口（单独训练时使用）
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
        params += list(self.adapter.parameters())
        params += list(self.unet.parameters())
        if self.use_semantic and self.sem_proj is not None:
            params += list(self.sem_proj.parameters())

        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
