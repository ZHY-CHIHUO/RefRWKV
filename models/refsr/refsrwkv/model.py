# Copyright (c) Shanghai AI Lab. All rights reserved.
"""
RefSRWKV: Reference-based Super-Resolution with RWKV Backbone.
"""

import torch

try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

from kernels.wkv import OmniShift, RUN_CUDA


# Window stages are named after the U-Net path.  Keeping the names here makes
# the YAML schema and checkpoint signature independent of module attributes.
_WINDOW_STAGE_NAMES = (
    "enc1",
    "enc2",
    "enc3",
    "latent",
    "dec3",
    "dec2",
    "dec1",
    "refine",
)
_DEFAULT_WINDOW_SPECS = {
    "enc1": {"size": 8, "offsets": (0, 4)},
    "enc2": {"size": 8, "offsets": (0, 4)},
    "enc3": {"size": 4, "offsets": (0, 2)},
    "latent": {"size": 3, "offsets": (0, 1)},
    "dec3": {"size": 4, "offsets": (0, 2)},
    "dec2": {"size": 8, "offsets": (0, 4)},
    "dec1": {"size": 8, "offsets": (0, 4)},
    "refine": {"size": 8, "offsets": (0, 4)},
}


def _pixel_shuffle_factors(factor):
    """Factor an integer reconstruction ratio into practical shuffle stages."""
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError(f"重建倍率必须为正整数，得到 {factor!r}")
    factors = []
    remainder = factor
    for prime in (2, 3):
        while remainder % prime == 0:
            factors.append(prime)
            remainder //= prime
    if remainder > 1:
        factors.append(remainder)
    return tuple(factors)


def _window_positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 必须是正整数，得到 {value!r}")
    return int(value)


def _window_offsets(value, size, name):
    if isinstance(value, int) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} 必须是非空整数列表")
    offsets = []
    for index, offset in enumerate(value):
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise ValueError(f"{name}[{index}] 必须是整数")
        if offset < 0 or offset >= size:
            raise ValueError(
                f"{name}[{index}]={offset} 必须满足 0 <= offset < window_size({size})"
            )
        offsets.append(int(offset))
    return tuple(offsets)


def _parse_window_spec(raw, stage_name):
    """Validate one explicit stage window specification."""
    if not isinstance(raw, dict):
        raise ValueError(f"windows.{stage_name} 必须是包含 size 和 offsets 的 mapping")
    unknown = set(raw).difference({"size", "offsets"})
    if unknown:
        raise ValueError(
            f"windows.{stage_name} 包含未知字段: {', '.join(sorted(unknown))}"
        )
    missing = {"size", "offsets"}.difference(raw)
    if missing:
        raise ValueError(
            f"windows.{stage_name} 缺少字段: {', '.join(sorted(missing))}"
        )
    size = _window_positive_int(raw["size"], f"windows.{stage_name}.size")
    return {
        "size": size,
        "offsets": _window_offsets(
            raw["offsets"], size, f"windows.{stage_name}.offsets"
        ),
    }


def normalize_window_config(windows=None):
    """Return the canonical explicit per-stage window configuration."""
    if windows is None:
        windows = {"phase_mode": "local", "stages": _DEFAULT_WINDOW_SPECS}
    if not isinstance(windows, dict):
        raise ValueError("windows 必须是分层 mapping")

    if "stages" in windows:
        unknown = set(windows).difference({"phase_mode", "stages"})
        if unknown:
            raise ValueError(f"windows 包含未知字段: {', '.join(sorted(unknown))}")
        stage_map = windows["stages"]
    else:
        stage_map = {key: value for key, value in windows.items() if key != "phase_mode"}
    if not isinstance(stage_map, dict):
        raise ValueError("windows.stages 必须是 mapping")

    missing = set(_WINDOW_STAGE_NAMES).difference(stage_map)
    unknown = set(stage_map).difference(_WINDOW_STAGE_NAMES)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"缺少: {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"未知: {', '.join(sorted(unknown))}")
        raise ValueError("windows 阶段配置无效（" + "；".join(details) + "）")

    mode = str(windows.get("phase_mode", "local")).lower()
    if mode not in {"local", "global"}:
        raise ValueError("windows.phase_mode 只能是 local 或 global")
    return {
        "phase_mode": mode,
        "stages": {
            stage_name: _parse_window_spec(stage_map[stage_name], stage_name)
            for stage_name in _WINDOW_STAGE_NAMES
        },
    }
# ═══════════════════════════════════════════════════════════
# 基础组件
# ═══════════════════════════════════════════════════════════
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob, self.scale_by_keep = drop_prob, scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        random_tensor = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(
            keep_prob
        )
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


# ═══════════════════════════════════════════════════════════
# RWKV 空间与通道混合模块
# ═══════════════════════════════════════════════════════════
class VRWKV_SpatialMix(nn.Module):
    def __init__(
        self,
        n_embd,
        window_size=8,
        num_groups=None,
        shift_offsets=None,
    ):
        super().__init__()
        if num_groups is None:
            num_groups = max(1, n_embd // 16)
        if not isinstance(num_groups, int) or num_groups < 1:
            raise ValueError("num_groups 必须为正整数")
        if not isinstance(window_size, int):
            raise ValueError("window_size 必须为整数")
        if n_embd < 16 or n_embd % num_groups != 0 or n_embd % 16 != 0:
            raise ValueError("n_embd 必须被 num_groups 整除，且至少为 16 和 16 的倍数")
        if window_size < 1:
            raise ValueError("window_size 必须为正数")
        if shift_offsets is None:
            raise ValueError("shift_offsets 必须由所属阶段的窗口配置提供")
        offsets = _window_offsets(shift_offsets, window_size, "shift_offsets")
        group_dim = n_embd // num_groups
        self.n_embd, self.window_size = n_embd, window_size
        self.shift_offsets = offsets
        self.recurrence = 2
        self.omni_shift = OmniShift(dim=n_embd)
        self.key, self.value, self.receptance, self.output = [
            nn.Linear(n_embd, n_embd, bias=False) for _ in range(4)
        ]
        self.register_buffer("scale", torch.tensor(n_embd**0.5))
        with torch.no_grad():
            decay_init = torch.zeros(self.recurrence, n_embd)
            # The CUDA kernel expects a positive distance-decay coefficient.
            for g in range(num_groups):
                target_decay = 0.5 * (g + 1)
                decay_init[:, g * group_dim : (g + 1) * group_dim] = math.log(
                    math.expm1(target_decay)
                )
            self.spatial_decay, self.spatial_first = nn.Parameter(
                decay_init
            ), nn.Parameter(torch.zeros(self.recurrence, n_embd))
        mid_ch = max(n_embd // 4, 8)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(n_embd, mid_ch, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, n_embd, 1, bias=True),
            nn.Sigmoid(),
        )

    def jit_func(self, x, resolution):
        h, w = resolution
        x = self.omni_shift(rearrange(x, "b (h w) c -> b c h w", h=h, w=w))
        x = rearrange(x, "b c h w -> b (h w) c")
        return self.key(x), self.value(x), torch.sigmoid(self.receptance(x))

    def _window_wkv(self, k, v, sr):
        s, ws = self.scale, self.window_size
        for j in range(self.recurrence):
            dj, fj = F.softplus(self.spatial_decay[j]) / s, self.spatial_first[j] / s
            if j % 2 == 0:
                v = RUN_CUDA(dj, fj, k, v)
            else:
                kt, vt = rearrange(k, "b (h w) c -> b (w h) c", h=ws, w=ws), rearrange(
                    v, "b (h w) c -> b (w h) c", h=ws, w=ws
                )
                v = rearrange(
                    RUN_CUDA(dj, fj, kt, vt), "b (w h) c -> b (h w) c", h=ws, w=ws
                )
        return sr * v

    def forward(self, x, resolution, layer_idx=0):
        B, T, C = x.size()
        h, w, ws = resolution[0], resolution[1], self.window_size
        if T != h * w or C != self.n_embd:
            raise ValueError(
                f"SpatialMix 输入形状与 resolution 不一致: x={tuple(x.shape)}, resolution={resolution}"
            )
        sr, k, v = self.jit_func(x, resolution)
        if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
            raise ValueError(f"layer_idx 必须是整数，得到 {layer_idx!r}")
        shift_amt = self.shift_offsets[layer_idx % len(self.shift_offsets)]
        k, v, sr = [
            rearrange(t, "b (hh ww) c -> b hh ww c", hh=h, ww=w) for t in (k, v, sr)
        ]

        # Shift the window origin without circular wrap-around.  Content is
        # padded on the top/left, then the bottom/right is padded to a window
        # multiple; the output is cropped back to the original grid.
        pad_bottom = (ws - (h + shift_amt) % ws) % ws
        pad_right = (ws - (w + shift_amt) % ws) % ws
        if shift_amt or pad_bottom or pad_right:
            pad = (0, 0, shift_amt, pad_right, shift_amt, pad_bottom)
            k, v, sr = [F.pad(t, pad) for t in (k, v, sr)]
        Hp, Wp = h + shift_amt + pad_bottom, w + shift_amt + pad_right
        if Hp % ws or Wp % ws:
            raise RuntimeError(f"窗口 padding 失败: ({Hp}, {Wp}) 不能被窗口 {ws} 整除")

        k, v, sr = [
            rearrange(t, "b (nh w1) (nw w2) c -> (b nh nw) (w1 w2) c", w1=ws, w2=ws)
            for t in (k, v, sr)
        ]
        out = self._window_wkv(k, v, sr)
        out = rearrange(
            out,
            "(b nh nw) (w1 w2) c -> b (nh w1) (nw w2) c",
            nh=Hp // ws,
            nw=Wp // ws,
            w1=ws,
            w2=ws,
        )
        out = out[:, shift_amt : shift_amt + h, shift_amt : shift_amt + w, :]

        out = out * self.channel_gate(out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.output(rearrange(out, "b hh ww c -> b (hh ww) c"))


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, hidden_rate=4):
        super().__init__()
        hidden_sz = int(hidden_rate * n_embd)
        self.key, self.omni_shift, self.receptance, self.value = (
            nn.Linear(n_embd, hidden_sz, bias=False),
            OmniShift(dim=n_embd),
            nn.Linear(n_embd, n_embd, bias=False),
            nn.Linear(hidden_sz, n_embd, bias=False),
        )

    def forward(self, x, resolution):
        h, w = resolution
        x = self.omni_shift(rearrange(x, "b (h w) c -> b c h w", h=h, w=w))
        x = rearrange(x, "b c h w -> b (h w) c")
        return torch.sigmoid(self.receptance(x)) * self.value(
            torch.square(torch.relu(self.key(x)))
        )


class Block(nn.Module):
    def __init__(
        self,
        n_embd,
        hidden_rate=4,
        drop_path=0.0,
        layer_idx=0,
        window_size=8,
        shift_offsets=None,
    ):
        super().__init__()
        if (
            isinstance(layer_idx, bool)
            or not isinstance(layer_idx, int)
            or layer_idx < 0
        ):
            raise ValueError("layer_idx 必须是非负整数")
        if shift_offsets is None:
            # Standalone users of Block receive a valid explicit phase list.
            shift_offsets = (0, 3, 6) if window_size == 8 else (0, window_size // 2)
        self.layer_idx, self.ln1, self.ln2 = (
            layer_idx,
            nn.LayerNorm(n_embd),
            nn.LayerNorm(n_embd),
        )
        self.att = VRWKV_SpatialMix(
            n_embd,
            window_size=window_size,
            shift_offsets=shift_offsets,
        )
        self.ffn = VRWKV_ChannelMix(n_embd, hidden_rate)
        self.gamma1, self.gamma2 = nn.Parameter(torch.ones(n_embd)), nn.Parameter(
            torch.ones(n_embd)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        resolution = (h, w)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.drop_path(
            self.gamma1 * self.att(self.ln1(x), resolution, self.layer_idx)
        )
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), resolution))
        return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


# ═══════════════════════════════════════════════════════════
# 空间缩放与特征融合
# ═══════════════════════════════════════════════════════════
class Downsample(nn.Module):
    def __init__(self, n_feat, channel_scale=2):
        super().__init__()
        mid_channels = n_feat * channel_scale // 4
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, mid_channels, 3, 1, 1, bias=False), nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat, channel_scale=0.5):
        super().__init__()
        mid_channels = int(n_feat * channel_scale * 4)
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, mid_channels, 3, 1, 1, bias=False), nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class RMSNorm2d(nn.Module):
    """Per-pixel RMS normalization over channels only.

    Unlike GroupNorm, this has no spatial reduction, so a model trained on
    48x48 LR crops sees the same normalization rule on an arbitrary full image.
    RMS normalization deliberately does not subtract the channel mean because
    low-frequency/DC colour information matters for super-resolution.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        if not isinstance(num_channels, int) or num_channels < 1:
            raise ValueError("RMSNorm2d 的 num_channels 必须为正整数")
        if not math.isfinite(float(eps)) or float(eps) <= 0:
            raise ValueError("RMSNorm2d 的 eps 必须为正有限数")
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.eps = float(eps)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.weight.numel():
            raise ValueError(
                "RMSNorm2d 输入必须是匹配通道数的 NCHW 张量，"
                f"得到 {tuple(x.shape)}"
            )
        input_dtype = x.dtype
        x_float = x.float()
        rms = x_float.square().mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        normalized = x_float * rms * self.weight.float().view(1, -1, 1, 1)
        return normalized.to(input_dtype)


class GatedFusion(nn.Module):
    """Match reference features in a local neighborhood before fusion."""

    def __init__(self, dim, reduction=4, window_size=7):
        super().__init__()
        if not isinstance(window_size, int) or window_size < 1 or window_size % 2 == 0:
            raise ValueError("GatedFusion window_size must be a positive odd integer")
        self.dim = dim
        self.window_size = window_size
        self.radius = window_size // 2
        self.kernel = window_size * window_size
        self.query = nn.Conv2d(dim, dim, 1, bias=False)
        self.key = nn.Conv2d(dim, dim, 1, bias=False)
        self.value = nn.Conv2d(dim, dim, 1, bias=False)
        self.message = nn.Conv2d(dim, dim, 1, bias=False)
        self.fuse_conv = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.norm = RMSNorm2d(dim)
        self.message_norm = RMSNorm2d(dim)
        gate_hidden = max(dim // reduction, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 3 + 1, gate_hidden, 1),
            nn.GELU(),
            nn.Conv2d(gate_hidden, dim, 1),
            nn.Sigmoid(),
        )
        self.quality = nn.Sequential(
            nn.Conv2d(dim * 3 + 1, gate_hidden, 1),
            nn.GELU(),
            nn.Conv2d(gate_hidden, 1, 1),
            nn.Sigmoid(),
        )
        self.relative_bias = nn.Parameter(torch.zeros(self.kernel))
        # A learnable temperature makes cosine matching selective enough for
        # repeated textures while remaining adaptable to ambiguous references.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        nn.init.trunc_normal_(self.fuse_conv.weight, std=0.02)
        nn.init.constant_(self.gate[2].bias, -1.0)
        nn.init.constant_(self.quality[2].bias, -1.0)

    def forward(self, lr_feat, ref_feat):
        b, _, h, w = lr_feat.shape
        if ref_feat.shape != lr_feat.shape:
            raise ValueError(
                f"GatedFusion expects equal feature shapes, got "
                f"{tuple(lr_feat.shape)} and {tuple(ref_feat.shape)}"
            )

        q = F.normalize(self.query(lr_feat), dim=1, eps=1e-6)
        k = F.normalize(self.key(ref_feat), dim=1, eps=1e-6)
        v = self.value(ref_feat)
        kernel = self.kernel
        k = F.unfold(k, self.window_size, padding=self.radius)
        v = F.unfold(v, self.window_size, padding=self.radius)
        k = k.view(b, self.dim, kernel, h, w).permute(0, 3, 4, 2, 1)
        v = v.view(b, self.dim, kernel, h, w).permute(0, 3, 4, 2, 1)
        q = q.permute(0, 2, 3, 1).unsqueeze(-2)
        scale = self.logit_scale.float().exp().clamp(1.0, 100.0).to(q.dtype)
        logits = (q * k).sum(dim=-1) * scale
        logits = logits + self.relative_bias.view(1, 1, 1, kernel)
        valid = F.unfold(
            lr_feat.new_ones((b, 1, h, w)), self.window_size, padding=self.radius
        )
        valid = valid.view(b, kernel, h, w).permute(0, 2, 3, 1)
        logits = logits.masked_fill(valid < 0.5, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        matched = (attention.unsqueeze(-1) * v).sum(dim=-2)
        matched = self.message_norm(self.message(matched.permute(0, 3, 1, 2)))

        if kernel == 1:
            match_conf = attention.new_ones((b, h, w))
        else:
            entropy = -(attention.clamp_min(1e-6) * attention.clamp_min(1e-6).log()).sum(-1)
            match_conf = (1.0 - entropy / math.log(kernel)).clamp(0.0, 1.0)
        match_conf = match_conf.unsqueeze(1)
        context = torch.cat([lr_feat, ref_feat, matched, match_conf], dim=1)
        direct = self.norm(self.fuse_conv(torch.cat([lr_feat, ref_feat], dim=1)))
        # Direct fusion remains useful for aligned references and for SISR
        # (where Ref is bicubic LR-up); confidence gates only the retrieved
        # message, whose quality depends on a reliable local match.
        return lr_feat + self.gate(context) * (
            direct + self.quality(context) * match_conf * matched
        )


class GlobalLatentBlock(nn.Module):
    """Full-image context block used after the U-Net bottleneck."""

    def __init__(self, channels, num_heads=8, hidden_rate=2):
        super().__init__()
        if channels % num_heads:
            raise ValueError("GlobalLatentBlock channels must be divisible by num_heads")
        hidden = max(channels * hidden_rate, channels)
        self.norm1 = nn.LayerNorm(channels)
        self.positional = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.pos_scale = nn.Parameter(torch.full((channels,), 0.1))
        self.attn = nn.MultiheadAttention(
            channels, num_heads, dropout=0.0, batch_first=True
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        self.gamma1 = nn.Parameter(torch.ones(channels))
        self.gamma2 = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.pos_scale.view(1, c, 1, 1) * self.positional(x)
        tokens = rearrange(x, "b c h w -> b (h w) c")
        normalized = self.norm1(tokens)
        attended = self.attn(normalized, normalized, normalized, need_weights=False)[0]
        tokens = tokens + self.gamma1 * attended
        tokens = tokens + self.gamma2 * self.ffn(self.norm2(tokens))
        return rearrange(tokens, "b (h w) c -> b c h w", h=h, w=w)

class RefSRWKV(nn.Module):
    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: tuple = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        scale: int = 4,
        upsampler: str = "progressive",
        color_match: str = "global",
        drop_path_rate: float = 0.1,
        hidden_rate: int = 4,
        ref_channels: int = None,
        windows=None,
    ):
        super().__init__()
        if not isinstance(num_blocks, (tuple, list)) or len(num_blocks) != 4:
            raise ValueError("num_blocks 必须包含四个编码器层的 block 数")
        if any(isinstance(n, bool) or not isinstance(n, int) for n in num_blocks):
            raise ValueError("num_blocks 的每一项必须为整数")
        num_blocks = tuple(num_blocks)
        if any(n < 1 for n in num_blocks):
            raise ValueError("num_blocks 的每一项必须为正整数")
        if not isinstance(num_refinement_blocks, int) or num_refinement_blocks < 0:
            raise ValueError("num_refinement_blocks 必须为非负整数")
        if not isinstance(dim, int) or dim < 16 or dim % 16 != 0:
            raise ValueError("dim 必须是至少 16 的 16 倍数，以适配 CUDA WKV")
        if not isinstance(scale, int) or scale < 1:
            raise ValueError("scale 必须为正整数")
        upsampler = str(upsampler).lower()
        if upsampler not in {"progressive", "direct"}:
            raise ValueError("upsampler 只能是 progressive 或 direct")
        color_match = str(color_match).lower()
        if color_match not in {"global", "none"}:
            raise ValueError("color_match 只能是 global 或 none")
        if (
            not math.isfinite(float(drop_path_rate))
            or not 0.0 <= float(drop_path_rate) < 1.0
        ):
            raise ValueError("drop_path_rate 必须位于 [0, 1)")
        if (
            not isinstance(hidden_rate, (int, float))
            or isinstance(hidden_rate, bool)
            or not math.isfinite(float(hidden_rate))
            or hidden_rate <= 0
            or int(hidden_rate * dim) < 1
        ):
            raise ValueError("hidden_rate 必须为正数")
        if ref_channels is None:
            # Reference colour statistics are matched against the LR image,
            # so the reference stream follows the input channel count.
            ref_channels = inp_channels
        if not isinstance(inp_channels, int) or inp_channels < 1:
            raise ValueError("inp_channels 必须为正整数")
        if not isinstance(out_channels, int) or out_channels < 1:
            raise ValueError("out_channels 必须为正整数")
        if not isinstance(ref_channels, int) or ref_channels < 1:
            raise ValueError("ref_channels 必须为正整数")
        if ref_channels != inp_channels:
            raise ValueError("当前颜色对齐路径要求 ref_channels == inp_channels")

        # The U-Net lives on the incoming LR grid. ``scale`` controls both Ref
        # folding and residual reconstruction.
        shuffle_factors = _pixel_shuffle_factors(scale)
        self.scale = scale
        self.inp_channels, self.ref_channels = inp_channels, ref_channels
        self.upsampler, self.color_match = upsampler, color_match
        self.window_config = normalize_window_config(windows)
        if inp_channels == out_channels:
            self.skip_proj = nn.Identity()
        else:
            self.skip_proj = nn.Conv2d(inp_channels, out_channels, 1, bias=False)

        self.lr_up = nn.Sequential(
            nn.Conv2d(inp_channels, dim, 3, padding=1, bias=False),
            RMSNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            RMSNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=2, dilation=2, bias=False),
            RMSNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        self.ref_to_level1 = nn.Sequential(
            nn.Conv2d(ref_channels, ref_channels, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(ref_channels, ref_channels, 3, padding=1, bias=False),
            nn.PixelUnshuffle(scale),
            nn.Conv2d(
                ref_channels * (scale**2), dim, 1, bias=False
            ),
            RMSNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.ref_down2 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1, bias=False),
            RMSNorm2d(dim * 2),
        )
        self.ref_down3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, 3, stride=2, padding=1, bias=False),
            RMSNorm2d(dim * 4),
        )
        self.ref_down4 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 8, 3, stride=2, padding=1, bias=False),
            RMSNorm2d(dim * 8),
        )

        self.fuse1, self.fuse2, self.fuse3, self.fuse4 = (
            GatedFusion(dim, window_size=7),
            GatedFusion(dim * 2, window_size=5),
            GatedFusion(dim * 4, window_size=5),
            GatedFusion(dim * 8, window_size=3),
        )

        total_blocks = sum(num_blocks) * 2 + num_refinement_blocks
        dp_rates = [
            drop_path_rate * i / max(1, total_blocks - 1) for i in range(total_blocks)
        ]
        dp_idx, global_layer_idx = 0, 0

        def make_stage(stage_name, channels, count, *, drop_rates=None):
            nonlocal global_layer_idx
            spec = self.window_config["stages"][stage_name]
            phase_base = (
                global_layer_idx if self.window_config["phase_mode"] == "global" else 0
            )
            blocks = []
            for local_idx in range(count):
                drop_path = 0.0 if drop_rates is None else drop_rates[local_idx]
                blocks.append(
                    Block(
                        channels,
                        hidden_rate,
                        drop_path,
                        layer_idx=phase_base + local_idx,
                        window_size=spec["size"],
                        shift_offsets=spec["offsets"],
                    )
                )
            global_layer_idx += count
            return nn.Sequential(*blocks)

        self.encoder_level1 = make_stage(
            "enc1",
            dim,
            num_blocks[0],
            drop_rates=dp_rates[dp_idx : dp_idx + num_blocks[0]],
        )
        dp_idx += num_blocks[0]
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = make_stage(
            "enc2",
            dim * 2,
            num_blocks[1],
            drop_rates=dp_rates[dp_idx : dp_idx + num_blocks[1]],
        )
        dp_idx += num_blocks[1]
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = make_stage(
            "enc3",
            dim * 4,
            num_blocks[2],
            drop_rates=dp_rates[dp_idx : dp_idx + num_blocks[2]],
        )
        dp_idx += num_blocks[2]
        self.down3_4 = Downsample(dim * 4)
        self.latent = make_stage(
            "latent",
            dim * 8,
            num_blocks[3],
            drop_rates=dp_rates[dp_idx : dp_idx + num_blocks[3]],
        )
        dp_idx += num_blocks[3]
        self.global_latent = nn.Sequential(
            GlobalLatentBlock(dim * 8, num_heads=8, hidden_rate=2),
            GlobalLatentBlock(dim * 8, num_heads=8, hidden_rate=2),
        )
        self.decoder_fuse3 = GatedFusion(dim * 4, window_size=5)
        self.decoder_fuse2 = GatedFusion(dim * 2, window_size=5)
        self.decoder_fuse1 = GatedFusion(dim, window_size=7)

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Sequential(
            nn.Conv2d(dim * 8, dim * 4, 1, bias=False),
            RMSNorm2d(dim * 4),
        )
        self.decoder_level3 = make_stage("dec3", dim * 4, num_blocks[2], drop_rates=dp_rates[dp_idx : dp_idx + num_blocks[2]])
        dp_idx += num_blocks[2]
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 2, 1, bias=False),
            RMSNorm2d(dim * 2),
        )
        self.decoder_level2 = make_stage("dec2", dim * 2, num_blocks[1], drop_rates=dp_rates[dp_idx : dp_idx + num_blocks[1]])
        dp_idx += num_blocks[1]
        self.up2_1 = Upsample(dim * 2)
        self.reduce_chan_level1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False), RMSNorm2d(dim)
        )
        self.decoder_level1 = make_stage("dec1", dim, num_blocks[0], drop_rates=dp_rates[dp_idx : dp_idx + num_blocks[0]])
        dp_idx += num_blocks[0]

        refine_rates = [
            drop_path_rate * i / max(1, num_refinement_blocks - 1)
            for i in range(num_refinement_blocks)
        ]
        self.refinement = make_stage("refine", dim, num_refinement_blocks, drop_rates=refine_rates)

        # The progressive head preserves the conventional x4 x2+x2 path.  The
        # direct head keeps all expensive activations on the LR grid and only
        # expands channels in the final convolution, which is preferable for
        # high factors such as x10.
        if self.upsampler == "progressive":
            up_layers = []
            for shuffle_factor in shuffle_factors:
                up_layers.extend(
                    [
                        nn.Conv2d(
                            dim,
                            dim * (shuffle_factor**2),
                            3,
                            padding=1,
                            bias=False,
                        ),
                        nn.PixelShuffle(shuffle_factor),
                        nn.GELU(),
                    ]
                )
            self.up_final = (
                nn.Sequential(*up_layers) if up_layers else nn.Identity()
            )
            self.output_conv = nn.Conv2d(
                dim, out_channels, 3, padding=1, bias=False
            )
            self.output_shuffle = nn.Identity()
        else:
            self.up_final = nn.Identity()
            self.output_conv = nn.Conv2d(
                dim, out_channels * (scale**2), 3, padding=1, bias=False
            )
            self.output_shuffle = nn.PixelShuffle(scale)
        self.apply(self._init_weights)
        if isinstance(self.skip_proj, nn.Conv2d):
            # Preserve the available channels at initialization; the learned
            # residual then starts from a predictable bicubic baseline.
            nn.init.zeros_(self.skip_proj.weight)
            with torch.no_grad():
                for channel in range(min(inp_channels, out_channels)):
                    self.skip_proj.weight[channel, channel, 0, 0] = 1.0
        nn.init.zeros_(self.output_conv.weight)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, RMSNorm2d):
            nn.init.ones_(m.weight)

    def _match_color(self, ref, target):
        if self.color_match == "none":
            return ref
        # Keep statistics in float32: bf16/half spatial reductions can lose
        # enough precision to create visible colour shifts on flat patches.
        input_dtype = ref.dtype
        ref_f, target_f = ref.float(), target.float()
        ref_mean = ref_f.mean(dim=(2, 3), keepdim=True)
        ref_std = ref_f.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
        tgt_mean = target_f.mean(dim=(2, 3), keepdim=True)
        tgt_std = target_f.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
        matched = (ref_f - ref_mean) / ref_std * tgt_std + tgt_mean
        return matched.to(input_dtype)

    def _extract_ref_pyramid(self, ref):
        ref_1 = self.ref_to_level1(ref)
        ref_2 = self.ref_down2(ref_1)
        ref_3 = self.ref_down3(ref_2)
        ref_4 = self.ref_down4(ref_3)
        return ref_1, ref_2, ref_3, ref_4

    def forward(self, lr, ref):
        if lr.ndim != 4 or ref.ndim != 4:
            raise ValueError(
                f"lr/ref 必须是 4D NCHW 张量，得到 {lr.shape} 和 {ref.shape}"
            )
        if lr.shape[0] != ref.shape[0]:
            raise ValueError(f"lr/ref batch 不一致: {lr.shape[0]} vs {ref.shape[0]}")
        if lr.shape[1] != self.inp_channels:
            raise ValueError(f"lr 通道数应为 {self.inp_channels}，得到 {lr.shape[1]}")
        if ref.shape[1] != self.ref_channels:
            raise ValueError(f"ref 通道数应为 {self.ref_channels}，得到 {ref.shape[1]}")
        if lr.shape[2] < 1 or lr.shape[3] < 1 or ref.shape[2] < 1 or ref.shape[3] < 1:
            raise ValueError("lr/ref 的空间尺寸必须为正数")
        target_hr_h, target_hr_w = lr.shape[2] * self.scale, lr.shape[3] * self.scale
        if ref.shape[2:] != (target_hr_h, target_hr_w):
            raise ValueError(
                "Ref 尺寸必须严格等于 LR x scale；"
                f"得到 LR={tuple(lr.shape[2:])}, scale=x{self.scale}, "
                f"Ref={tuple(ref.shape[2:])}"
            )
        lr_hr_input = F.interpolate(
            lr, size=(target_hr_h, target_hr_w), mode="bicubic", align_corners=False
        )
        ref_aligned = self._match_color(ref, lr_hr_input)

        # Three PixelUnshuffle downsampling stages require an LR multiple of
        # eight.  Pad only the bottom/right edge, and pad Ref by exactly the
        # physical scale so PixelUnshuffle(scale) remains phase-aligned.
        lr_h, lr_w = lr.shape[2:]
        pad_h, pad_w = (-lr_h) % 8, (-lr_w) % 8
        if pad_h or pad_w:
            lr_internal = F.pad(lr, (0, pad_w, 0, pad_h), mode="replicate")
            ref_internal = F.pad(
                ref_aligned,
                (0, pad_w * self.scale, 0, pad_h * self.scale),
                mode="replicate",
            )
        else:
            lr_internal, ref_internal = lr, ref_aligned
        padded_hr_size = (
            lr_internal.shape[2] * self.scale,
            lr_internal.shape[3] * self.scale,
        )
        lr_hr = self.skip_proj(
            F.interpolate(
                lr_internal,
                size=padded_hr_size,
                mode="bicubic",
                align_corners=False,
            )
        )

        fea = self.lr_up(lr_internal)
        ref_1, ref_2, ref_3, ref_4 = self._extract_ref_pyramid(ref_internal)

        e1 = self.encoder_level1(self.fuse1(fea, ref_1))
        e2 = self.encoder_level2(self.fuse2(self.down1_2(e1), ref_2))
        e3 = self.encoder_level3(self.fuse3(self.down2_3(e2), ref_3))
        latent = self.latent(self.fuse4(self.down3_4(e3), ref_4))
        latent = self.global_latent(latent)

        d3 = self.decoder_level3(
            self.decoder_fuse3(
                self.reduce_chan_level3(torch.cat([self.up4_3(latent), e3], dim=1)),
                ref_3,
            )
        )
        d2 = self.decoder_level2(
            self.decoder_fuse2(
                self.reduce_chan_level2(torch.cat([self.up3_2(d3), e2], dim=1)),
                ref_2,
            )
        )
        d1 = self.decoder_level1(
            self.decoder_fuse1(
                self.reduce_chan_level1(torch.cat([self.up2_1(d2), e1], dim=1)),
                ref_1,
            )
        )
        d1 = self.refinement(d1)

        out_feat = self.output_shuffle(self.output_conv(self.up_final(d1)))
        if out_feat.shape[2:] != padded_hr_size:
            raise RuntimeError(
                "输出头没有按 scale 重建到 LR x scale: "
                f"{tuple(out_feat.shape[2:])} vs {padded_hr_size}"
            )
        output = lr_hr + out_feat
        output = output[:, :, :target_hr_h, :target_hr_w]
        return torch.clamp(output, min=-1.0, max=1.0)

    def prepare_for_inference(self):
        self.eval()
        for m in self.modules():
            if isinstance(m, OmniShift):
                m.reparam_5x5()
        return self


# ═══════════════════════════════════════════════════════════
