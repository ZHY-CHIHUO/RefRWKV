# Copyright (c) Shanghai AI Lab. All rights reserved.
"""
RefSRWKV: Reference-based Super-Resolution with RWKV Backbone.

架构：2 路空间扫描（H→W 与 W→H 各一次双向 WKV）
训练：EMA + ReduceLROnPlateau + 梯度裁剪 + warmup
"""

import torch

torch.set_float32_matmul_precision("high")
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pytorch_lightning as pl
from torch.utils.cpp_extension import load
import os

_cuda_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda")

# ═══════════════════════════════════════════════════════════════
# CUDA WKV
# ═══════════════════════════════════════════════════════════════
_wkv_cuda = None
_wkv_load_error = None


def _get_wkv_cuda():
    """按需编译并缓存 Bi-WKV CUDA 扩展（首次真正使用时才编译）。

    原实现于模块 import 时即查询 GPU capability 并触发 JIT 编译，导致：
      1. 无 CUDA 环境下 import models.RefSRWKV 直接失败；
      2. 每个 DataLoader worker 都会争抢首次编译。
    现在改为延迟加载：仅在 WKV 真正在 CUDA 上运行时才编译一次。
    """
    global _wkv_cuda, _wkv_load_error
    if _wkv_cuda is not None:
        return _wkv_cuda
    if _wkv_load_error is not None:
        raise RuntimeError("Bi-WKV CUDA 扩展此前加载失败") from _wkv_load_error
    if not torch.cuda.is_available():
        raise RuntimeError("Bi-WKV 需要 CUDA；当前 torch.cuda.is_available()=False")

    cap = torch.cuda.get_device_capability()
    arch = f"compute_{cap[0]}{cap[1]}"
    _wkv_cuda = load(
        name="bi_wkv",
        sources=[
            os.path.join(_cuda_dir, "bi_wkv.cpp"),
            os.path.join(_cuda_dir, "bi_wkv_kernel.cu"),
        ],
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--maxrregcount 60",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            f"-gencode arch={arch},code={arch}",
        ],
    )
    return _wkv_cuda

try:
    _compiler_disable = torch.compiler.disable
except AttributeError:

    def _compiler_disable(fn=None, **kwargs):
        return fn if fn is not None else (lambda f: f)


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):
        half_mode = w.dtype == torch.half
        bf_mode = w.dtype == torch.bfloat16
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = _get_wkv_cuda().bi_wkv_forward(w, u, k, v)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        half_mode = w.dtype == torch.half
        bf_mode = w.dtype == torch.bfloat16
        gw, gu, gk, gv = _get_wkv_cuda().bi_wkv_backward(
            w.float().contiguous(),
            u.float().contiguous(),
            k.float().contiguous(),
            v.float().contiguous(),
            gy.float().contiguous(),
        )
        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            return (gw, gu, gk, gv)


@_compiler_disable()
def RUN_CUDA(w, u, k, v):
    w = w.float()
    u = u.float()
    k = k.float()
    v = v.float()
    return WKV.apply(w, u, k, v)


# ═══════════════════════════════════════════════════════════════
# DropPath
# ═══════════════════════════════════════════════════════════════
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


# ═══════════════════════════════════════════════════════════════
# OmniShift（4 分支：Identity + 1×1 + 3×3 + 5×5）
# ═══════════════════════════════════════════════════════════════
class OmniShift(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv1x1 = nn.Conv2d(dim, dim, 1, groups=dim, bias=False)
        self.conv3x3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.conv5x5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.alpha = nn.Parameter(torch.ones(4) * 0.25)
        self.register_buffer("conv5x5_reparam_weight", torch.zeros(dim, 1, 5, 5))
        self._reparam_done = False

    def forward_train(self, x):
        alpha = torch.softmax(self.alpha, dim=0)
        return (
            alpha[0] * x
            + alpha[1] * self.conv1x1(x)
            + alpha[2] * self.conv3x3(x)
            + alpha[3] * self.conv5x5(x)
        )

    def reparam_5x5(self):
        if self._reparam_done:
            return
        alpha = torch.softmax(self.alpha, dim=0)
        identity = torch.zeros(self.dim, 1, 5, 5, device=self.conv1x1.weight.device)
        identity[:, :, 2, 2] = 1.0
        w1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        w3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))
        w5 = self.conv5x5.weight
        combined = alpha[0] * identity + alpha[1] * w1 + alpha[2] * w3 + alpha[3] * w5
        self.conv5x5_reparam_weight.copy_(combined)
        self._reparam_done = True

    def forward(self, x):
        if self.training:
            self._reparam_done = False
            return self.forward_train(x)
        else:
            if not self._reparam_done:
                self.reparam_5x5()
            return F.conv2d(x, self.conv5x5_reparam_weight, padding=2, groups=self.dim)


# ═══════════════════════════════════════════════════════════════
# VRWKV Blocks（2 路 H→W 串联）
# ═══════════════════════════════════════════════════════════════
class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, head_dim=64):
        super().__init__()
        self.n_embd = n_embd
        self.recurrence = 2
        attn_sz = n_embd
        self.omni_shift = OmniShift(dim=n_embd)
        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        self.output = nn.Linear(attn_sz, n_embd, bias=False)
        self.register_buffer("scale", torch.tensor(n_embd**0.5))
        with torch.no_grad():
            self.spatial_decay = nn.Parameter(torch.zeros(self.recurrence, self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.recurrence, self.n_embd))

    def jit_func(self, x, resolution):
        h, w = resolution
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)
        return sr, k, v

    def forward(self, x, resolution):
        B, T, C = x.size()
        sr, k, v = self.jit_func(x, resolution)
        s = self.scale
        for j in range(self.recurrence):
            if j % 2 == 0:
                v = RUN_CUDA(self.spatial_decay[j] / s, self.spatial_first[j] / s, k, v)
            else:
                h, w = resolution
                k = rearrange(k, "b (h w) c -> b (w h) c", h=h, w=w)
                v = rearrange(v, "b (h w) c -> b (w h) c", h=h, w=w)
                v = RUN_CUDA(self.spatial_decay[j] / s, self.spatial_first[j] / s, k, v)
                k = rearrange(k, "b (w h) c -> b (h w) c", h=h, w=w)
                v = rearrange(v, "b (w h) c -> b (h w) c", h=h, w=w)
        x = sr * v
        x = self.output(x)
        return x


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, hidden_rate=4):
        super().__init__()
        self.n_embd = n_embd
        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.omni_shift = OmniShift(dim=n_embd)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x, resolution):
        h, w = resolution
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        k = self.key(x)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv
        return x


class Block(nn.Module):
    def __init__(self, n_embd, hidden_rate=4, drop_path=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.att = VRWKV_SpatialMix(n_embd)
        self.ffn = VRWKV_ChannelMix(n_embd, hidden_rate)
        self.gamma1 = nn.Parameter(torch.ones(n_embd), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones(n_embd), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        resolution = (h, w)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), resolution))
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), resolution))
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x


# ═══════════════════════════════════════════════════════════════
# Resizing
# ═══════════════════════════════════════════════════════════════
class Downsample(nn.Module):
    def __init__(self, n_feat, channel_scale=2):
        super().__init__()
        mid_channels = n_feat * channel_scale // 4
        assert mid_channels > 0
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, mid_channels, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat, channel_scale=0.5):
        super().__init__()
        mid_channels = int(n_feat * channel_scale * 4)
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, mid_channels, 3, 1, 1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


# ═══════════════════════════════════════════════════════════════
# GatedFusion
# ═══════════════════════════════════════════════════════════════
def _gn_groups(num_channels: int, max_groups: int = 32) -> int:
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class GatedFusion(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fuse_conv = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.norm = nn.GroupNorm(_gn_groups(dim), dim)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // reduction, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(dim // reduction, 8), dim, 1),
            nn.Sigmoid(),
        )
        nn.init.trunc_normal_(self.fuse_conv.weight, std=0.02)
        nn.init.constant_(self.gate[-2].bias, 0.0)

    def forward(self, lr_feat, ref_feat):
        fused = self.fuse_conv(torch.cat([lr_feat, ref_feat], dim=1))
        fused = self.norm(fused)
        gate = self.gate(fused)
        return lr_feat + gate * fused


# ═══════════════════════════════════════════════════════════════
# RefSRWKV
# ═══════════════════════════════════════════════════════════════
class RefSRWKV(nn.Module):
    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: tuple = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        scale: int = 10,
        drop_path_rate: float = 0.1,
        hidden_rate: int = 4,
    ):
        super().__init__()
        self.scale = scale
        self.dim = dim
        self.out_channels = out_channels

        # ── LR 编码器 ──
        self.lr_up = nn.Sequential(
            # nn.Upsample(scale_factor=2.5, mode="bilinear", align_corners=False),
            nn.Conv2d(inp_channels, dim, 3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(dim), dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
        )

        # ── Ref 编码器 ──
        self.ref_to_level1 = nn.Sequential(
            nn.PixelUnshuffle(4),
            nn.Conv2d(out_channels * 16, dim, 1, bias=False),
            nn.GroupNorm(_gn_groups(dim), dim),
        )
        self.ref_down2 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(dim * 2), dim * 2),
        )
        self.ref_down3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(dim * 4), dim * 4),
        )
        self.ref_down4 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 8, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(dim * 8), dim * 8),
        )

        # ── 门控融合 ──
        self.fuse1 = GatedFusion(dim)
        self.fuse2 = GatedFusion(dim * 2)
        self.fuse3 = GatedFusion(dim * 4)
        self.fuse4 = GatedFusion(dim * 8)
        nn.init.constant_(self.fuse1.gate[-2].bias, 1.5)
        nn.init.constant_(self.fuse2.gate[-2].bias, 0.5)
        nn.init.constant_(self.fuse3.gate[-2].bias, 0.0)
        nn.init.constant_(self.fuse4.gate[-2].bias, -0.5)

        # ── DropPath ──
        dp_rates = [
            drop_path_rate * i / (sum(num_blocks) - 1) for i in range(sum(num_blocks))
        ]
        dp_idx = 0

        # ── 编码器 ──
        self.encoder_level1 = nn.Sequential(
            *[
                Block(dim, hidden_rate, dp_rates[dp_idx + i])
                for i in range(num_blocks[0])
            ]
        )
        dp_idx += num_blocks[0]
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(
            *[
                Block(dim * 2, hidden_rate, dp_rates[dp_idx + i])
                for i in range(num_blocks[1])
            ]
        )
        dp_idx += num_blocks[1]
        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = nn.Sequential(
            *[
                Block(dim * 4, hidden_rate, dp_rates[dp_idx + i])
                for i in range(num_blocks[2])
            ]
        )
        dp_idx += num_blocks[2]
        self.down3_4 = Downsample(dim * 4)

        self.latent = nn.Sequential(
            *[
                Block(dim * 8, hidden_rate, dp_rates[dp_idx + i])
                for i in range(num_blocks[3])
            ]
        )

        # ── 解码器 ──
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Sequential(
            nn.Conv2d(dim * 8, dim * 4, 1, bias=False),
            nn.GroupNorm(_gn_groups(dim * 4), dim * 4),
        )
        self.decoder_level3 = nn.Sequential(
            *[Block(dim * 4, hidden_rate) for _ in range(num_blocks[2])]
        )

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 2, 1, bias=False),
            nn.GroupNorm(_gn_groups(dim * 2), dim * 2),
        )
        self.decoder_level2 = nn.Sequential(
            *[Block(dim * 2, hidden_rate) for _ in range(num_blocks[1])]
        )

        self.up2_1 = Upsample(dim * 2)
        self.reduce_chan_level1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.GroupNorm(_gn_groups(dim), dim),
        )
        self.decoder_level1 = nn.Sequential(
            *[Block(dim, hidden_rate) for _ in range(num_blocks[0])]
        )

        # ── 精修 ──
        self.refinement = nn.Sequential(
            *[Block(dim, hidden_rate) for _ in range(num_refinement_blocks)]
        )

        # ── 上采样输出 ──
        self.up_final = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(dim, dim * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )
        self.output_conv = nn.Conv2d(dim, out_channels, 3, padding=1, bias=True)

        # ── Ref 引导残差 ──
        self.ref_guided_refine = nn.Conv2d(
            out_channels * 2, out_channels, 3, padding=1, bias=False
        )

        # ★ 先全局初始化，再零初始化（顺序不能反）
        self.apply(self._init_weights)
        nn.init.zeros_(self.ref_guided_refine.weight)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def _extract_ref_pyramid(self, ref):
        ref_1 = self.ref_to_level1(ref)
        ref_2 = self.ref_down2(ref_1)
        ref_3 = self.ref_down3(ref_2)
        ref_4 = self.ref_down4(ref_3)
        return ref_1, ref_2, ref_3, ref_4

    def forward(self, lr, ref):
        # fea = self.lr_up(lr)
        # ★ 动态计算 Level1 目标尺寸（= Ref 经过 PixelUnshuffle(4) 后的尺寸）
        target_h = ref.shape[2] // 4
        target_w = ref.shape[3] // 4

        # ★ 将 LR 插值到 Level1 分辨率（自动适配任意 scale）
        if lr.shape[2] != target_h or lr.shape[3] != target_w:
            fea = F.interpolate(
                lr, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
        else:
            fea = lr  # scale=4 时 LR 和 Level1 同尺寸，跳过插值

        fea = self.lr_up(fea)  # 只做卷积特征提取，不做尺寸变换
        ref_1, ref_2, ref_3, ref_4 = self._extract_ref_pyramid(ref)

        e1 = self.encoder_level1(self.fuse1(fea, ref_1))
        e2 = self.encoder_level2(self.fuse2(self.down1_2(e1), ref_2))
        e3 = self.encoder_level3(self.fuse3(self.down2_3(e2), ref_3))
        latent = self.latent(self.fuse4(self.down3_4(e3), ref_4))

        d3 = self.decoder_level3(
            self.reduce_chan_level3(torch.cat([self.up4_3(latent), e3], dim=1))
        )
        d2 = self.decoder_level2(
            self.reduce_chan_level2(torch.cat([self.up3_2(d3), e2], dim=1))
        )
        d1 = self.decoder_level1(
            self.reduce_chan_level1(torch.cat([self.up2_1(d2), e1], dim=1))
        )

        d1 = self.refinement(d1)
        hr_feat = self.up_final(d1)
        # --- 临时 Debug 保险丝 ---
        if torch.isnan(hr_feat).any() or torch.isinf(hr_feat).any():
            print(f"[WARNING] hr_feat contains NaN/Inf! Clamping...")
            hr_feat = torch.nan_to_num(hr_feat, nan=0.0, posinf=1.0, neginf=-1.0)
        # -------------------------
        out = self.output_conv(hr_feat)

        residual = self.ref_guided_refine(torch.cat([out, ref], dim=1))
        out = torch.clamp(out + residual, -1.0, 1.0)
        return out

    def prepare_for_inference(self):
        self.eval()
        for m in self.modules():
            if isinstance(m, OmniShift):
                m.reparam_5x5()
        print("✓ RefSRWKV: OmniShift reparameterized.")
        return self


# ═══════════════════════════════════════════════════════════════
# EMA
# ═══════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        self._initialized = False

    def _lazy_init(self, model: nn.Module):
        if self._initialized:
            return
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self._initialized = True

    @torch.no_grad()
    def update(self, model: nn.Module):
        self._lazy_init(model)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": self.shadow,
            "initialized": self._initialized,
        }

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        self.shadow = sd["shadow"]
        self._initialized = sd["initialized"]


# ═══════════════════════════════════════════════════════════════
# LitRefSRWKV
# ═══════════════════════════════════════════════════════════════
class LitRefSRWKV(pl.LightningModule):
    def __init__(
        self,
        model_sr: RefSRWKV,
        learning_rate: float = 1e-4,
        warmup_steps: int = 500,
        grad_clip_norm: float = 1.0,
        ema_decay: float = 0.999,
        use_ema: bool = True,
        loss_fn=None,
        lr_key: str = "lr",
        hr_key: str = "hr",
        ref_key: str = "ref",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_sr", "loss_fn"])
        self.model_sr = model_sr
        self.criterion = loss_fn or nn.L1Loss()
        self.lr_key = lr_key
        self.hr_key = hr_key
        self.ref_key = ref_key
        self.ema = EMA(decay=ema_decay) if use_ema else None
        self.plateau_scheduler = None
        self._pending_plateau_state = None

    def _unpack_batch(self, batch):
        if isinstance(batch, dict):
            return batch[self.lr_key], batch[self.hr_key], batch[self.ref_key]
        return batch[0], batch[1], batch[2]

    def forward(self, lr, ref):
        return self.model_sr(lr, ref)

    # ── 训练 ──
    def training_step(self, batch, batch_idx):
        lr, hr, ref = self._unpack_batch(batch)
        output = self(lr, ref)
        loss = self.criterion(output, hr)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        if self.global_step < self.hparams.warmup_steps:
            progress = (self.global_step + 1) / self.hparams.warmup_steps
            lr_scale = 1e-3 + (1.0 - 1e-3) * progress
            for pg in self.optimizers().param_groups:
                pg["lr"] = self.hparams.learning_rate * lr_scale

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema is not None:
            self.ema.update(self.model_sr)

    # ── 验证 ──
    def on_validation_epoch_start(self):
        if self.ema is not None:
            self.ema.apply_shadow(self.model_sr)

    def validation_step(self, batch, batch_idx):
        lr, hr, ref = self._unpack_batch(batch)
        output = self(lr, ref)
        loss = self.criterion(output, hr)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if self.ema is not None:
            self.ema.restore(self.model_sr)
        if self.plateau_scheduler is not None:
            val_loss = self.trainer.callback_metrics.get("val_loss")
            if val_loss is not None:
                self.plateau_scheduler.step(val_loss)
                current_lr = self.plateau_scheduler.optimizer.param_groups[0]["lr"]
                self.log("lr", current_lr, prog_bar=True, logger=True)

    # ── 测试 ──
    def on_test_epoch_start(self):
        if self.ema is not None:
            self.ema.apply_shadow(self.model_sr)

    def test_step(self, batch, batch_idx):
        lr, hr, ref = self._unpack_batch(batch)
        output = self(lr, ref)
        loss = self.criterion(output, hr)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return output, hr

    def on_test_epoch_end(self):
        if self.ema is not None:
            self.ema.restore(self.model_sr)

    # ── 优化器 ──
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return optimizer

    # ── 梯度裁剪 ──
    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        clip_val = gradient_clip_val or self.hparams.grad_clip_norm
        if clip_val and clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )

    # ── Checkpoint ──
    def on_save_checkpoint(self, checkpoint):
        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()
        if self.plateau_scheduler is not None:
            checkpoint["plateau_scheduler"] = self.plateau_scheduler.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
        self._pending_plateau_state = checkpoint.get("plateau_scheduler")

    def on_train_start(self):
        if self._pending_plateau_state and self.plateau_scheduler:
            self.plateau_scheduler.load_state_dict(self._pending_plateau_state)
            self._pending_plateau_state = None
        total = sum(p.numel() for p in self.parameters())
        ema_info = f" | EMA decay={self.ema.decay}" if self.ema else " | EMA=off"
        print(
            f"✅ LitRefSRWKV 训练开始 | 参数量: {total / 1e6:.2f}M"
            f" | grad_clip={self.hparams.grad_clip_norm}{ema_info}"
        )
