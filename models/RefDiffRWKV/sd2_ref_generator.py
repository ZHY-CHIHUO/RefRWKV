"""
sd2_ref_generator.py — SD2 Ref-guided Generator (latent ε-prediction)

设计原则：
1. 只负责 diffusion 前向 / 采样 / 生成 pixel；
2. 使用 diffusers 标准 SD2 UNet + T2I-Adapter 风格注入
   (down_intrablock_additional_residuals)，无需自定义 UNet；
3. 借鉴 ControlLDM 的 get_input / p_losses / sample_log / log_images 接口。
4. UNet conv_in 扩展为 8 通道：前 4 为 noisy_latent，后 4 为 sr_latent（条件）。

本版变更（对接决策链）：
1. 新增 use_sr_condition：透传给 GlobalSemanticModule，启用 SR latent
   条件分支（语义金字塔同时看到 ref 与 SR 已重建的结构）。
2. apply_model 从 x_input 后 4 通道切出 sr_latent_cond 喂给语义模块，
   无需改签名，Phase1 forward / Phase2 _adapter_pred_x0 / 推理
   _denoise_step 三条路径自动覆盖。
3. 置信门控改用 scale2 的 cos_map：scale1 窗口大、cos 全图 ~0.92
   无区分度（已实测）；scale2 窗口 4×4、range 0.40~0.55，有真实
   匹配信息。门控用传播后置信（借来的区域要放行）。
4. adapter 输出解包兼容三元组 (feats, cos_maps, raw_cos_maps)：
   raw_cos_maps（传播前局部置信）供 gan_system 的 D_tex 加权使用，
   通过 _unpack_adapter_out 静态方法复用。
"""

import os
import logging
import numpy as np
from typing import Optional, List, Tuple, Dict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

logger = logging.getLogger(__name__)

try:
    from .sd2_ref_adapter import SD2_RefAdapter
    from .globalsemanticmodule import GlobalSemanticModule
except ImportError:
    from sd2_ref_adapter import SD2_RefAdapter

    try:
        from globalsemanticmodule import GlobalSemanticModule
    except ImportError:
        GlobalSemanticModule = None


def _half_resolution(h: int, w: int) -> Tuple[int, int]:
    """计算减半后的分辨率（ceil 模式）。"""
    return (h + 1) // 2, (w + 1) // 2


def _compute_latent_size(
    ref: torch.Tensor, hr: Optional[torch.Tensor] = None
) -> Tuple[int, int]:
    """根据目标像素尺寸计算 VAE latent 尺寸（8 倍下采样）。"""
    target = hr if hr is not None else ref
    h, w = target.shape[-2], target.shape[-1]
    assert h % 8 == 0 and w % 8 == 0, (
        f"图像尺寸 ({h}, {w}) 必须被 8 整除（VAE 下采样因子），"
        f"当前余数: ({h % 8}, {w % 8})"
    )
    return h // 8, w // 8


class SD2RefGenerator(LightningModule):
    """SD2 参考引导生成器，支持 8 通道 UNet 输入（噪声 + SR 条件）。"""

    CROSS_ATTN_CTX_LEN: int = 77  # SD2 cross-attention 固定上下文长度

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
        sem_base_dim: int = 768,
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
        normalize_input: bool = False,
        local_files_only: bool = True,
        sr_model: Optional[torch.nn.Module] = None,
        use_sr_latent_cond: bool = True,
        use_sr_condition: bool = False,
        use_confidence_gate: bool = False,
        confidence_alpha: float = 0.4,
        use_temporal_gate: bool = False,
        control_scale_min: float = 0.3,
        control_scale_max: float = 1.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["sr_model"])

        self.lr_key = lr_key
        self.ref_key = ref_key
        self.hr_key = hr_key
        self.t_min = t_min
        self.t_max = t_max
        assert t_min <= t_max, f"t_min({t_min}) 必须 <= t_max({t_max})"
        self.cfg_drop_prob = cfg_drop_prob
        self.control_scale = control_scale
        self.normalize_input = normalize_input
        self.sr_model = sr_model
        self.use_sr_latent_cond = use_sr_latent_cond
        self.use_sr_condition = use_sr_condition
        self.use_confidence_gate = use_confidence_gate
        # 注：adapter 内部 SelfSimTransfer 已做 conf 加权（决策链闭环），
        # 此处残差级门控为第二道，alpha 默认 0.4（原 0.2）以减弱二次压制
        self.confidence_alpha = confidence_alpha
        self.use_temporal_gate = use_temporal_gate
        self.control_scale_min = control_scale_min
        self.control_scale_max = control_scale_max

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
        #  Adapter
        # ═══════════════════════════════════════
        self.adapter = SD2_RefAdapter(strategy=strategy, rwkv_cfg=rwkv_cfg)

        # ═══════════════════════════════════════
        #  DINOv2 语义路径（DINOv2 冻结，proj/pyramid 可训练）
        # ═══════════════════════════════════════
        self.use_semantic = use_semantic and (GlobalSemanticModule is not None)
        self.global_semantic = (
            GlobalSemanticModule(
                dinov2_model_name=dinov2_model_name,
                use_sr_condition=use_sr_condition,
            )
            if self.use_semantic
            else None
        )
        if self.global_semantic is not None:
            # 只冻结 DINOv2 backbone（GlobalSemanticModule.__init__ 内部已处理）
            # proj 和 semantic_pyramid 保持可训练
            # 仅确保 DINOv2 保持 eval（关闭 Dropout）
            self.global_semantic.dinov2.eval()
            self.global_semantic.dinov2.requires_grad_(False)

        self.sem_proj: Optional[nn.Linear] = (
            nn.Linear(sem_base_dim, self.cross_attn_dim) if self.use_semantic else None
        )

        # ═══════════════════════════════════════
        #  Optimizer 配置
        # ═══════════════════════════════════════
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    # ═══════════════════════════════════════════════════════
    #  UNet conv_in 4→8 通道扩展
    # ═══════════════════════════════════════════════════════

    def _expand_conv_in_for_sr_latent(self) -> None:
        """将 UNet conv_in 从 4 通道扩展到 8 通道。

        前 4 通道保留 SD2 预训练权重，后 4 通道以缩小方差的正态分布初始化。
        """
        old_conv = self.unet.conv_in
        old_weight = old_conv.weight.data

        new_conv = nn.Conv2d(
            8,
            old_weight.shape[0],
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        )

        with torch.no_grad():
            new_conv.weight[:, :4] = old_weight
            # [FIX] xavier_uniform_ 对切片不生效，手动计算 std 后 normal_ 赋值
            fan_in = (
                old_weight.shape[1] * old_conv.kernel_size[0] * old_conv.kernel_size[1]
            )
            fan_out = (
                old_weight.shape[0] * old_conv.kernel_size[0] * old_conv.kernel_size[1]
            )
            std = 0.1 * (2.0 / (fan_in + fan_out)) ** 0.5
            new_conv.weight[:, 4:].normal_(0, std)
            new_conv.bias.copy_(old_conv.bias)

        self.unet.conv_in = new_conv

    # ═══════════════════════════════════════════════════════
    #  LoRA & Freeze
    # ═══════════════════════════════════════════════════════

    def _inject_lora(
        self, rank: int, target_modules: Optional[List[str]] = None
    ) -> None:
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

    def _freeze_unet_except_attn(self) -> None:
        for n, p in self.unet.named_parameters():
            if "attn" not in n and "lora" not in n:
                p.requires_grad = False

        # [FIX] 切片无法设置 requires_grad；整体开启后用 hook 屏蔽前 4 通道梯度
        self.unet.conv_in.weight.requires_grad = True
        self.unet.conv_in.bias.requires_grad = True

        def _mask_pretrained_grad(grad: torch.Tensor) -> torch.Tensor:
            grad[:, :4] = 0
            return grad

        self.unet.conv_in.weight.register_hook(_mask_pretrained_grad)

    # ═══════════════════════════════════════════════════════
    #  VAE helpers
    # ═══════════════════════════════════════════════════════

    @torch.no_grad()
    def encode_latent(self, img: torch.Tensor) -> torch.Tensor:
        """编码图像到 latent（无梯度，用于训练目标 / 推理）。"""
        return self.vae.encode(img).latent_dist.sample() * self.vae_scale_factor

    def encode_latent_with_grad(self, img: torch.Tensor) -> torch.Tensor:
        """编码图像到 latent（保留梯度，用于 sr_fixed=False 时 SR 模型微调）。

        注意：此方法故意不加 @torch.no_grad()，使梯度可从
        loss → latent → VAE decode/encode → sr_pixel → SR 模型 完整反传。
        """
        return self.vae.encode(img).latent_dist.sample() * self.vae_scale_factor

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """解码 latent 到像素（保留梯度，用于 Phase2 pred_x0 → pixel loss 反传）。

        注意：此方法故意不加 @torch.no_grad()。
        在 _adapter_pred_x0（无 no_grad）中调用时梯度可穿透到 UNet；
        在 _no_adapter_pred_x0（有 @torch.no_grad()）中调用时外层阻断梯度。
        """
        return self.vae.decode(z / self.vae_scale_factor).sample

    @torch.no_grad()
    def decode_latent_eval(self, z: torch.Tensor) -> torch.Tensor:
        """解码 latent 到像素（无梯度，专用于采样 / 推理）。"""
        return self.vae.decode(z / self.vae_scale_factor).sample

    # ═══════════════════════════════════════════════════════
    #  Adapter 输出解包（供 apply_model 与 gan_system 复用）
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _unpack_adapter_out(out):
        """解包 adapter 输出为 (feats, cos_maps, raw_cos_maps)。

        兼容两种返回约定：
          - (feats, cos_maps)                旧版 / 无 raw 分支
          - (feats, cos_maps, raw_cos_maps)  SelfSimTransfer 版
        非元组（仅 feats）时两个 map 均为 None。

        cos_maps:     传播后置信（借来的区域被放行），用于注入门控
        raw_cos_maps: 传播前置信（局部 LCA 匹配），用于 D_tex 加权；
                      旧版无此分支时回退为 cos_maps（语义等价）
        """
        if isinstance(out, tuple):
            if len(out) >= 3:
                return out[0], out[1], out[2]
            if len(out) == 2:
                return out[0], out[1], out[1]
        return out, None, None

    # ═══════════════════════════════════════════════════════
    #  语义 token 构建
    # ═══════════════════════════════════════════════════════

    def build_sem_tokens(
        self, sem_pyramid: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if sem_pyramid is None or self.sem_proj is None:
            return None

        layer_tokens = {}
        for key in ("e1", "e2", "e3", "latent"):
            if key in sem_pyramid:
                layer_tokens[key] = sem_pyramid[key].float()

        if not layer_tokens:
            return None

        projected = [
            self.sem_proj(layer_tokens[k])
            for k in ("e1", "e2", "e3", "latent")
            if k in layer_tokens
        ]
        return torch.cat(projected, dim=1)

    def build_context(
        self, bsz: int, sem_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        empty_ctx = torch.zeros(
            bsz,
            self.CROSS_ATTN_CTX_LEN,
            self.cross_attn_dim,
            device=self.device,
            dtype=torch.float32,
        )
        if sem_tokens is not None:
            return torch.cat([empty_ctx, sem_tokens], dim=1)
        return empty_ctx

    # ═══════════════════════════════════════════════════════
    #  Adapter 三尺度特征 → T2I-Adapter 风格 down 注入
    # ═══════════════════════════════════════════════════════

    def build_down_intrablock(
        self,
        ref_feats: List[torch.Tensor],
        latent_h: int,
        latent_w: int,
        t: Optional[torch.Tensor] = None,
        cos_maps: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        f320, f640, f1280 = ref_feats
        h0, w0 = latent_h, latent_w
        h1, w1 = _half_resolution(h0, w0)
        h2, w2 = _half_resolution(h1, w1)
        h3, w3 = _half_resolution(h2, w2)
        target_sizes = [(h0, w0), (h1, w1), (h2, w2), (h3, w3)]
        feats = [f320, f640, f1280, f1280]

        # ── 方案B：时序门控 ──
        if self.use_temporal_gate and t is not None:
            t_ratio = 1.0 - t.float().mean().item() / self.t_max
            scale = (
                self.control_scale_min
                + (self.control_scale_max - self.control_scale_min) * t_ratio
            )
        else:
            scale = self.control_scale

        # ── 方案A：置信门控（取 scale2 的 cos_map）──
        # scale1 (120×120, 窗口 8×8) cos 全图 ~0.92 无区分度（已实测）；
        # scale2 (60×60, 窗口 4×4) range 0.40~0.55，有真实匹配信息，
        # 且分辨率与 latent 最大尺度对齐，插值失真最小。
        conf = None
        if self.use_confidence_gate and cos_maps is not None and len(cos_maps) > 0:
            conf = cos_maps[1] if len(cos_maps) > 1 else cos_maps[0]
            if conf.dim() == 4 and conf.shape[1] > 1:
                conf = conf.mean(dim=1, keepdim=True)

        residuals = []
        for feat, (th, tw) in zip(feats, target_sizes):
            if feat.shape[-2] != th or feat.shape[-1] != tw:
                feat = F.interpolate(
                    feat, size=(th, tw), mode="bilinear", align_corners=False
                )
            gated = feat * scale
            if conf is not None:
                conf_resized = F.interpolate(
                    conf, size=(th, tw), mode="bilinear", align_corners=False
                )
                gate = (
                    self.confidence_alpha + (1.0 - self.confidence_alpha) * conf_resized
                )
                gated = gated * gate
            residuals.append(gated)
        return residuals

    # ═══════════════════════════════════════════════════════
    #  核心 UNet 前向
    # ═══════════════════════════════════════════════════════

    def apply_model(
        self,
        x_input: torch.Tensor,
        t: torch.Tensor,
        lr: torch.Tensor,
        ref: torch.Tensor,
        ref_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = x_input.shape[0]
        ref_input = ref if ref_input is None else ref_input

        # ── 方案A：获取置信图（传播后置信，用于注入门控）──
        if self.use_confidence_gate:
            adapter_out = self.adapter(lr, ref_input, return_cos_sim_map=True)
            ref_feats, cos_maps, _ = self._unpack_adapter_out(adapter_out)
            # raw_cos_maps（第三个返回值）在此丢弃：
            # 它由 gan_system._adapter_pred_x0 直接调 adapter 取用（D_tex 加权）
        else:
            ref_feats = self.adapter(lr, ref_input)
            cos_maps = None

        sem_tokens = None
        if self.use_semantic:
            # x_input = concat(x_t, sr_latent_cond)，后 4 通道即 SR 条件；
            # sr_latent_cond=None 时 concat_sr_latent 拼的是全零，
            # 经 SRLatentConditioner 后等价于固定偏置（无条件），行为可控
            sr_latent_cond = x_input[:, 4:] if x_input.shape[1] == 8 else None
            # DINOv2 前向在 GlobalSemanticModule.forward 内部已有 no_grad 保护
            # proj 和 semantic_pyramid 需要梯度，不能包裹 no_grad
            sem_pyramid = self.global_semantic(ref_input, sr_latent=sr_latent_cond)
            sem_tokens = self.build_sem_tokens(sem_pyramid)
        context = self.build_context(bsz, sem_tokens)
        _, _, latent_h, latent_w = x_input.shape
        down_intrablock = self.build_down_intrablock(
            ref_feats, latent_h, latent_w, t=t, cos_maps=cos_maps
        )
        return self.unet(
            x_input,
            t,
            encoder_hidden_states=context,
            down_intrablock_additional_residuals=list(down_intrablock),
        ).sample

    def get_input(self, batch, bs: Optional[int] = None, *args, **kwargs):
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
    #  反推 pred_x0
    # ═══════════════════════════════════════════════════════

    def predict_x0_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor
    ) -> torch.Tensor:
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(x_t.device)
        a_bar = alphas_cumprod[t].float().view(-1, 1, 1, 1)
        return (x_t - (1.0 - a_bar).sqrt() * noise_pred) / (a_bar.sqrt() + 1e-8)

    # ═══════════════════════════════════════════════════════
    #  SR latent 条件生成
    # ═══════════════════════════════════════════════════════

    def _compute_sr_prior(self, lr: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """统一的 SR prior 计算入口：SR 模型前向 + 数值安全处理。"""
        with torch.amp.autocast(self.device.type, enabled=False):
            # ★ 确保 SR 模型在正确设备上
            sr_pixel = self.sr_model(lr.float(), ref.float())
        sr_pixel = torch.nan_to_num(sr_pixel, nan=0.0, posinf=1.0, neginf=-1.0)
        return sr_pixel.clamp(-1.0, 1.0)

    def _get_sr_latent_cond(
        self, lr: torch.Tensor, ref: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self.sr_model is None or not self.use_sr_latent_cond:
            return None
        with torch.no_grad():
            sr_prior = self._compute_sr_prior(lr, ref)
            return self.encode_latent(sr_prior.to(self.vae.dtype))

    # ═══════════════════════════════════════════════════════
    #  拼接 sr_latent
    # ═══════════════════════════════════════════════════════

    def concat_sr_latent(
        self, x_t: torch.Tensor, sr_latent_cond: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if sr_latent_cond is not None:
            return torch.cat([x_t, sr_latent_cond], dim=1)
        bsz, _, h, w = x_t.shape
        return torch.cat(
            [x_t, torch.zeros(bsz, 4, h, w, device=x_t.device, dtype=x_t.dtype)], dim=1
        )

    # ═══════════════════════════════════════════════════════
    #  训练前向
    # ═══════════════════════════════════════════════════════

    def forward(
        self, lr: torch.Tensor, ref: torch.Tensor, hr: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        assert lr.shape[0] == ref.shape[0] == hr.shape[0], "Batch size mismatch"

        if torch.isnan(lr).any() or torch.isnan(ref).any() or torch.isnan(hr).any():
            raise ValueError("Input contains NaN")

        bsz = lr.shape[0]

        with torch.no_grad():
            hr_latent = self.encode_latent(hr)
            noise = torch.randn_like(hr_latent)
            t = torch.randint(
                self.t_min, self.t_max + 1, (bsz,), device=self.device, dtype=torch.long
            )
            x_t = self.noise_scheduler.add_noise(hr_latent, noise, t)

            sr_latent_cond = self._get_sr_latent_cond(lr, ref)

            ref_input = ref
            if self.cfg_drop_prob > 0:
                mask = torch.rand(bsz, device=self.device) < self.cfg_drop_prob
                if mask.any():
                    ref_input = ref_input.clone()
                    ref_input[mask] = 0.0

        x_t_input = self.concat_sr_latent(x_t, sr_latent_cond)
        noise_pred = self.apply_model(x_t_input, t, lr, ref, ref_input=ref_input)

        loss = F.mse_loss(noise_pred, noise)

        pred_x0_latent = self.predict_x0_from_eps(x_t, t, noise_pred)
        pred_x0_latent = torch.nan_to_num(
            pred_x0_latent, nan=0.0, posinf=20.0, neginf=-20.0
        ).clamp(-20.0, 20.0)

        return {
            "loss": loss,
            "noise_pred": noise_pred,
            "noise": noise,
            "x_t": x_t,
            "hr_latent": hr_latent,
            "pred_x0_latent": pred_x0_latent,
            "t": t,
        }

    def p_losses(self, lr: torch.Tensor, ref: torch.Tensor, hr: torch.Tensor):
        out = self.forward(lr, ref, hr)
        return out["loss"], {"train/loss_diff": out["loss"].detach()}

    # ═══════════════════════════════════════════════════════
    #  采样 / 推理
    # ═══════════════════════════════════════════════════════

    @torch.no_grad()
    def _prepare_sr_cond(self, lr, ref, actual_sr):
        if actual_sr is None:
            return None, None

        orig_sr = self.sr_model
        try:
            self.sr_model = actual_sr
            sr_prior = self._compute_sr_prior(lr, ref)
        finally:
            self.sr_model = orig_sr

        sr_latent = self.encode_latent(sr_prior.to(self.vae.dtype))
        return sr_latent, sr_prior

    @torch.no_grad()
    def _initialize_latent(
        self,
        bsz,
        device,
        dtype,
        latent_h,
        latent_w,
        actual_sr,
        sr_latent,
        sr_prior,
        ref,
        t_start,
    ):
        if (
            actual_sr is not None
            and t_start is not None
            and sr_latent is not None
            and sr_prior is not None
        ):
            # SR target 用于 DPS guidance（_denoise_step 中引导生成向 SR 结果靠近）
            sr_target = sr_latent.detach().clone()

            # 统一使用 t_start 作为噪声水平，保证与 scheduler 首步一致
            noise = torch.randn_like(sr_latent)
            t_noise_t = torch.full((bsz,), t_start, device=device, dtype=torch.long)
            x_t = self.noise_scheduler.add_noise(sr_latent, noise, t_noise_t)

            return x_t, sr_target

        return torch.randn(bsz, 4, latent_h, latent_w, device=device, dtype=dtype), None

    @torch.no_grad()
    def _denoise_step(
        self, x_t, t_int, lr, ref, sr_latent_cond, sr_target, guidance_scale, t_stop
    ):
        bsz = x_t.shape[0]
        t_tensor = torch.full((bsz,), t_int, device=x_t.device, dtype=torch.long)

        if sr_target is not None and guidance_scale > 0 and t_int > t_stop:
            # DPS 近似（Chung et al. 2023）：noise_pred 对应梯度修改前的 x_t，
            # 修改后的 x_t 与原始 noise_pred 配合送入 scheduler.step
            with torch.enable_grad():
                x_t.requires_grad_(True)
                x_t_input = self.concat_sr_latent(x_t, sr_latent_cond)
                noise_pred = self.apply_model(x_t_input, t_tensor, lr, ref)
                pred_x0 = self.predict_x0_from_eps(x_t, t_tensor, noise_pred)
                grad = torch.autograd.grad(F.mse_loss(pred_x0, sr_target), x_t)[0]
                x_t = x_t.detach() - guidance_scale * grad
                noise_pred = noise_pred.detach()
        else:
            x_t_input = self.concat_sr_latent(x_t, sr_latent_cond)
            noise_pred = self.apply_model(x_t_input, t_tensor, lr, ref)

        return self.noise_scheduler.step(noise_pred, t_int, x_t).prev_sample

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
        val_seed: int = None,
    ):
        if torch.isnan(lr).any() or torch.isnan(ref).any():
            raise ValueError("sample_log: Input contains NaN")

        bsz, device, actual_sr = lr.shape[0], self.device, sr_model or self.sr_model

        sr_latent_cond, sr_prior = self._prepare_sr_cond(lr, ref, actual_sr)
        latent_h, latent_w = _compute_latent_size(ref, hr)

        # 验证时固定噪声，保证跨 epoch 可比较
        if val_seed is not None:
            rng_state = torch.random.get_rng_state()
            cuda_rng_state = (
                torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            )
            torch.manual_seed(val_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(val_seed)

        x_t, sr_target = self._initialize_latent(
            bsz,
            device,
            lr.dtype,
            latent_h,
            latent_w,
            actual_sr,
            sr_latent_cond,
            sr_prior,
            ref,
            t_start,
        )

        if val_seed is not None:
            torch.random.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)

        self.noise_scheduler.set_timesteps(steps, device=device)
        timesteps = self.noise_scheduler.timesteps
        if t_start is not None:
            timesteps = [t for t in timesteps if t <= t_start]

        for t in timesteps:
            x_t = self._denoise_step(
                x_t, int(t), lr, ref, sr_latent_cond, sr_target, guidance_scale, t_stop
            )

        return self.decode_latent_eval(x_t)

    @torch.no_grad()
    def visual_steps(
        self, lr, ref, steps=50, sr_model=None, hr=None
    ) -> List[torch.Tensor]:
        bsz, device, actual_sr = lr.shape[0], self.device, sr_model or self.sr_model

        if actual_sr is not None:
            sr_latent_cond, sr_prior = self._prepare_sr_cond(lr, ref, actual_sr)
            x_t = sr_latent_cond  # 以 SR latent 作为初始噪声
        else:
            latent_h, latent_w = _compute_latent_size(ref, hr)
            x_t = torch.randn(bsz, 4, latent_h, latent_w, device=device, dtype=lr.dtype)
            sr_latent_cond = self._get_sr_latent_cond(lr, ref)

        self.noise_scheduler.set_timesteps(steps, device=device)

        pixel_each_step = []
        for t in self.noise_scheduler.timesteps:
            t_tensor = torch.full((bsz,), int(t), device=device, dtype=torch.long)
            x_t_input = self.concat_sr_latent(x_t, sr_latent_cond)
            noise_pred = self.apply_model(x_t_input, t_tensor, lr, ref)
            x_t = self.noise_scheduler.step(noise_pred, t, x_t).prev_sample

            x_t_clamped = x_t.clamp(
                -self.vae_scale_factor * 5, self.vae_scale_factor * 5
            )
            pixel_each_step.append(
                torch.clamp(
                    (self.decode_latent_eval(x_t_clamped) + 1.0) / 2.0, 0.0, 1.0
                )
            )

        return pixel_each_step

    @torch.no_grad()
    def log_images(
        self,
        batch,
        steps=50,
        sr_model=None,
        t_start=None,
        guidance_scale=0.0,
        t_stop=200,
        val_seed: int = 42,
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
            val_seed=val_seed,
        )
        result = {
            "lq": (lr + 1.0) / 2.0,
            "ref": (ref + 1.0) / 2.0,
            "hq": (hr + 1.0) / 2.0,
            "samples": (samples + 1.0) / 2.0,
        }
        return {k: v.clamp(0, 1) for k, v in result.items()}

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

    @torch.no_grad()
    def validation_inference(self, batch, save_dir, steps=50, sr_model=None):
        # [FIX] 添加 CUDA 可用性检查，避免 CPU 环境报错
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        lr, ref, hr = self.get_input(batch)

        use_cuda_timer = torch.cuda.is_available()
        if use_cuda_timer:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()

        val_results = self.log_images(batch, steps=steps, sr_model=sr_model)

        if use_cuda_timer:
            ender.record()
            torch.cuda.synchronize()
            logger.info(
                "Inference Time: %.2f ms, Memory: %.2f MB",
                starter.elapsed_time(ender),
                torch.cuda.memory_allocated() / 1024**2,
            )

        os.makedirs(save_dir, exist_ok=True)
        for image_key in val_results:
            image = val_results[image_key].detach().cpu()
            for i in range(len(image)):
                img = (
                    (image[i].permute(1, 2, 0).numpy() * 255)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                Image.fromarray(img).save(
                    os.path.join(save_dir, f"{i}_{image_key}.png")
                )

    # ═══════════════════════════════════════════════════════
    #  Lightning 接口
    # ═══════════════════════════════════════════════════════

    def training_step(self, batch, batch_idx):
        try:
            lr, ref, hr = self.get_input(batch)
            loss, log_dict = self.p_losses(lr, ref, hr)
            self.log_dict(log_dict, on_step=True, prog_bar=True)
            return loss
        except ValueError as e:
            logger.warning("Generator training_step 异常: %s", e)
            torch.cuda.empty_cache()
            return None
        except RuntimeError as e:
            logger.error("Generator training_step RuntimeError: %s", e)
            torch.cuda.empty_cache()
            return None

    def validation_step(self, batch, batch_idx):
        try:
            lr, ref, hr = self.get_input(batch)
            loss, _ = self.p_losses(lr, ref, hr)
            self.log("val/loss_diff", loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss
        except (ValueError, RuntimeError) as e:
            logger.warning("Generator validation_step 异常: %s", e)
            torch.cuda.empty_cache()
            return None

    def configure_optimizers(self):
        params = []
        params.extend(p for p in self.adapter.parameters() if p.requires_grad)
        params.extend(p for p in self.unet.parameters() if p.requires_grad)
        if self.use_semantic and self.sem_proj is not None:
            params.extend(self.sem_proj.parameters())

        # [FIX] 空参数保护，避免 AdamW 收到空列表报错
        if not params:
            logger.warning("Generator 无可训练参数")
            return None

        return torch.optim.AdamW(
            params, lr=self.learning_rate, weight_decay=self.weight_decay
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sd_path", type=str, default="sd2-community/stable-diffusion-2-1-base"
    )
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--hr_size", type=int, default=480)
    parser.add_argument("--lr_size", type=int, default=48)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_logger = logging.getLogger("test_sd2_gen")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    assert model.unet.conv_in.in_channels == 8
    test_logger.info("✅ conv_in 8 通道验证通过")

    assert model.unet.conv_in.weight[:, :4].abs().sum() > 0
    test_logger.info("✅ 前 4 通道预训练权重保留")

    assert model.unet.conv_in.weight[:, 4:].abs().sum() > 0
    test_logger.info("✅ 后 4 通道初始化（非零）")

    B, hr_sz, lr_sz = 1, args.hr_size, args.lr_size
    hr = torch.rand(B, 3, hr_sz, hr_sz, device=device) * 2 - 1
    ref = torch.rand(B, 3, hr_sz, hr_sz, device=device) * 2 - 1
    lr = torch.rand(B, 3, lr_sz, lr_sz, device=device) * 2 - 1

    out = model.forward(lr, ref, hr)
    test_logger.info(
        "forward loss = %.4f, noise_pred = %s",
        out["loss"].item(),
        list(out["noise_pred"].shape),
    )
    assert out["noise_pred"].shape == out["hr_latent"].shape

    # [FIX] 验证梯度流
    out["loss"].backward()
    has_grad = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.unet.parameters()
        if p.requires_grad
    )
    test_logger.info("梯度流验证: %s", "✅ 通过" if has_grad else "❌ 失败")

    # 验证 conv_in 后 4 通道有梯度、前 4 通道梯度被 hook 屏蔽
    conv_in_grad = model.unet.conv_in.weight.grad
    if conv_in_grad is not None:
        assert conv_in_grad[:, :4].abs().sum() == 0, "前 4 通道梯度应被 hook 屏蔽"
        assert conv_in_grad[:, 4:].abs().sum() > 0, "后 4 通道应有梯度"
        test_logger.info("✅ conv_in 梯度屏蔽验证通过")

    samples = model.sample_log(lr, ref, steps=2, hr=hr)
    assert samples.shape[-1] == hr_sz
    test_logger.info("sample_log passed")

    test_logger.info(
        "Trainable params: %d",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    test_logger.info("✅ 所有测试通过")
