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
import pytorch_lightning as pl
from torch.utils.cpp_extension import load
import os
import math

_cuda_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda")


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
# CUDA WKV 算子封装
# ═══════════════════════════════════════════════════════════
_wkv_cuda = None
_wkv_load_error = None


def _get_wkv_cuda():
    global _wkv_cuda, _wkv_load_error
    if _wkv_cuda is not None:
        return _wkv_cuda
    if _wkv_load_error is not None:
        raise RuntimeError("Bi-WKV CUDA 扩展加载失败") from _wkv_load_error
    if not torch.cuda.is_available():
        raise RuntimeError("Bi-WKV 需要 CUDA 环境")
    if not os.path.isfile(os.path.join(_cuda_dir, "bi_wkv.cpp")) or not os.path.isfile(
        os.path.join(_cuda_dir, "bi_wkv_kernel.cu")
    ):
        raise FileNotFoundError(f"Bi-WKV CUDA 源文件不存在: {_cuda_dir}")
    cap = torch.cuda.get_device_capability()
    arch, sm = f"compute_{cap[0]}{cap[1]}", f"sm_{cap[0]}{cap[1]}"
    try:
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
                f"-gencode arch={arch},code={sm}",
                f"-gencode arch={arch},code={arch}",
            ],
        )
    except Exception as exc:
        _wkv_load_error = exc
        raise RuntimeError(
            f"Bi-WKV CUDA 扩展编译/加载失败 (sm_{cap[0]}{cap[1]})"
        ) from exc
    return _wkv_cuda


try:
    _compiler_disable = torch.compiler.disable
except AttributeError:

    def _compiler_disable(fn=None, **kwargs):
        return fn if fn is not None else (lambda f: f)


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):
        half_mode, bf_mode = w.dtype == torch.half, w.dtype == torch.bfloat16
        ctx.save_for_backward(w, u, k, v)
        y = _get_wkv_cuda().bi_wkv_forward(
            w.float().contiguous(),
            u.float().contiguous(),
            k.float().contiguous(),
            v.float().contiguous(),
        )
        return y.half() if half_mode else (y.bfloat16() if bf_mode else y)

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        half_mode, bf_mode = w.dtype == torch.half, w.dtype == torch.bfloat16
        gw, gu, gk, gv = _get_wkv_cuda().bi_wkv_backward(
            w.float().contiguous(),
            u.float().contiguous(),
            k.float().contiguous(),
            v.float().contiguous(),
            gy.float().contiguous(),
        )
        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        if bf_mode:
            return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        return (gw, gu, gk, gv)


@_compiler_disable()
def RUN_CUDA(w, u, k, v):
    return WKV.apply(w.float(), u.float(), k.float(), v.float())


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


class OmniShift(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv1x1 = nn.Conv2d(dim, dim, 1, groups=dim, bias=False)
        self.conv3x3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.conv5x5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.alpha = nn.Parameter(torch.ones(4) * 0.25)
        self.gate = nn.Parameter(torch.zeros(1))
        self.register_buffer("conv5x5_reparam_weight", torch.zeros(dim, 1, 5, 5))
        self._reparam_done = False

    def forward_train(self, x):
        alpha = torch.softmax(self.alpha, dim=0)
        shifted = (
            alpha[0] * x
            + alpha[1] * self.conv1x1(x)
            + alpha[2] * self.conv3x3(x)
            + alpha[3] * self.conv5x5(x)
        )
        return x + torch.tanh(self.gate) * (shifted - x)

    def reparam_5x5(self):
        if self._reparam_done:
            return
        with torch.no_grad():
            g, alpha = torch.tanh(self.gate), torch.softmax(self.alpha, dim=0)
            weight = self.conv1x1.weight
            identity = torch.zeros(
                self.dim, 1, 5, 5, device=weight.device, dtype=weight.dtype
            )
            identity[:, :, 2, 2] = 1.0
            w1, w3, w5 = (
                F.pad(weight, (2, 2, 2, 2)),
                F.pad(self.conv3x3.weight, (1, 1, 1, 1)),
                self.conv5x5.weight,
            )
            self.conv5x5_reparam_weight.copy_(
                (1.0 - g) * identity
                + g
                * (alpha[0] * identity + alpha[1] * w1 + alpha[2] * w3 + alpha[3] * w5)
            )
        self._reparam_done = True

    def forward(self, x):
        if self.training:
            self._reparam_done = False
            return self.forward_train(x)
        if not self._reparam_done:
            self.reparam_5x5()
        return F.conv2d(x, self.conv5x5_reparam_weight, padding=2, groups=self.dim)


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
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fuse_conv = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.norm = RMSNorm2d(dim)
        gate_hidden = max(dim // reduction, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, gate_hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, dim, 1),
            nn.Sigmoid(),
        )
        nn.init.trunc_normal_(self.fuse_conv.weight, std=0.02)
        nn.init.constant_(self.gate[2].bias, 0.0)

    def forward(self, lr_feat, ref_feat):
        sim = F.cosine_similarity(lr_feat, ref_feat, dim=1).unsqueeze(1)
        # ★ 修复：将 [-1, 1] 线性映射到 [0, 1]，避免 sigmoid 导致的过度抑制
        conf = (sim + 1.0) / 2.0
        fused = self.norm(self.fuse_conv(torch.cat([lr_feat, ref_feat], dim=1)))
        return lr_feat + self.gate(fused) * conf * fused


# ═══════════════════════════════════════════════════════════
# 核心超分网络
# ═══════════════════════════════════════════════════════════
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
            GatedFusion(dim),
            GatedFusion(dim * 2),
            GatedFusion(dim * 4),
            GatedFusion(dim * 8),
        )

        total_blocks = sum(num_blocks)
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

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Sequential(
            nn.Conv2d(dim * 8, dim * 4, 1, bias=False),
            RMSNorm2d(dim * 4),
        )
        self.decoder_level3 = make_stage("dec3", dim * 4, num_blocks[2])
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 2, 1, bias=False),
            RMSNorm2d(dim * 2),
        )
        self.decoder_level2 = make_stage("dec2", dim * 2, num_blocks[1])
        self.up2_1 = Upsample(dim * 2)
        self.reduce_chan_level1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False), RMSNorm2d(dim)
        )
        self.decoder_level1 = make_stage("dec1", dim, num_blocks[0])

        self.refinement = make_stage("refine", dim, num_refinement_blocks)

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
                        nn.ReLU(inplace=True),
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
# EMA 与 Lightning 训练封装
# ═══════════════════════════════════════════════════════════
class EMA:
    def __init__(self, decay: float = 0.999):
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("EMA decay 必须位于 [0, 1)")
        self.decay = float(decay)
        self.shadow, self.backup = {}, {}
        self._initialized, self._applied = False, False

    def _lazy_init(self, model: nn.Module):
        # Shadows stay in float32 even when the model is trained with AMP.
        for name, param in model.named_parameters():
            if param.requires_grad and name not in self.shadow:
                self.shadow[name] = param.detach().float().clone()
        self._initialized = True

    @torch.no_grad()
    def update(self, model: nn.Module):
        self._lazy_init(model)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                p_data = param.detach().float()
                self.shadow[name].mul_(self.decay).add_(p_data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        if self._applied:
            return
        self._lazy_init(model)
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                self.backup[name] = param.detach().clone()
                param.data.copy_(
                    self.shadow[name].to(device=param.device, dtype=param.dtype)
                )
        self._applied = True

    def restore(self, model: nn.Module):
        if not self._applied:
            return
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name].to(param.device))
        self.backup = {}
        self._applied = False

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": {
                name: value.detach().clone() for name, value in self.shadow.items()
            },
            "initialized": self._initialized,
        }

    def load_state_dict(self, sd):
        if not isinstance(sd, dict):
            raise TypeError("EMA state 必须是字典")
        self.decay = float(sd.get("decay", self.decay))
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("checkpoint 中的 EMA decay 不在 [0, 1) 范围")
        shadow = sd.get("shadow", {})
        self.shadow = {
            name: value.detach().float().clone()
            for name, value in shadow.items()
            if isinstance(name, str) and torch.is_tensor(value)
        }
        self._initialized = bool(sd.get("initialized", bool(self.shadow)))
        self.backup, self._applied = {}, False


class LitRefSRWKV(pl.LightningModule):
    def __init__(
        self,
        model_sr: RefSRWKV,
        learning_rate: float = 1e-4,
        lr_scheduler: str = "plateau",
        lr_patience: int = 2,
        lr_factor: float = 0.5,
        lr_min: float = 1e-6,
        lr_threshold: float = 1e-4,
        warmup_steps: int = 0,
        grad_clip_norm: float = 1.0,
        ema_decay: float = 0.999,
        use_ema: bool = True,
        adam_betas=(0.9, 0.999),
        weight_decay: float = 0.0,
        ssim_weight: float = 0.0,
        fft_weight: float = 0.0,
        ref_drop_prob: float = 0.0,
        reference_mode: str = "paired",
        loss_fn=None,
        lr_key: str = "lr",
        hr_key: str = "hr",
        ref_key: str = "ref",
    ):
        super().__init__()
        if not isinstance(model_sr, nn.Module):
            raise TypeError("model_sr 必须是 torch.nn.Module 实例")
        if not float(learning_rate) > 0:
            raise ValueError("learning_rate 必须为正数")
        lr_scheduler = str(lr_scheduler).lower()
        if lr_scheduler not in {"plateau", "cosine"}:
            raise ValueError("lr_scheduler 只能是 plateau 或 cosine")
        if not isinstance(lr_patience, int) or lr_patience < 0:
            raise ValueError("lr_patience 必须为非负整数")
        if not 0.0 < float(lr_factor) < 1.0:
            raise ValueError("lr_factor 必须位于 (0, 1)")
        if not 0.0 <= float(lr_min) <= float(learning_rate):
            raise ValueError("lr_min 必须位于 [0, learning_rate]")
        if not math.isfinite(float(lr_threshold)) or float(lr_threshold) < 0.0:
            raise ValueError("lr_threshold 必须是非负有限数值")
        if not isinstance(warmup_steps, int) or warmup_steps < 0:
            raise ValueError("warmup_steps 必须为非负整数")
        if grad_clip_norm is not None and float(grad_clip_norm) < 0:
            raise ValueError("grad_clip_norm 不能为负数")
        if (
            not isinstance(adam_betas, (list, tuple))
            or len(adam_betas) != 2
            or any(
                isinstance(beta, bool)
                or not isinstance(beta, (int, float))
                or not math.isfinite(float(beta))
                or not 0.0 <= float(beta) < 1.0
                for beta in adam_betas
            )
        ):
            raise ValueError("adam_betas 必须是两个位于 [0, 1) 的有限数值")
        if not math.isfinite(float(weight_decay)) or float(weight_decay) < 0.0:
            raise ValueError("weight_decay 必须是非负有限数值")
        if not 0.0 <= float(ssim_weight) or not 0.0 <= float(fft_weight):
            raise ValueError("ssim_weight 和 fft_weight 不能为负数")
        if not 0.0 <= float(ref_drop_prob) <= 1.0:
            raise ValueError("ref_drop_prob 必须位于 [0, 1]")
        reference_mode = str(reference_mode).lower()
        if reference_mode in {"sisr", "lr", "lr_up", "bicubic_lr"}:
            reference_mode = "lr_up"
        if reference_mode not in {"paired", "lr_up"}:
            raise ValueError("reference_mode 只能是 paired 或 lr_up")
        for key in (lr_key, hr_key, ref_key):
            if not isinstance(key, str) or not key:
                raise ValueError("batch key 必须是非空字符串")
        self.save_hyperparameters(ignore=["model_sr", "loss_fn"])
        self.model_sr = model_sr
        self.adam_betas = tuple(float(beta) for beta in adam_betas)
        self.weight_decay = float(weight_decay)
        self.ssim_weight, self.fft_weight = float(ssim_weight), float(fft_weight)
        self.reference_mode = reference_mode
        self.criterion = loss_fn or nn.L1Loss()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss_fn, self._ssim_backend = None, "manual"
        if self.ssim_weight > 0:
            try:
                from pyiqa import create_metric as _create_pyiqa_metric

                # `as_loss` enables gradients through the metric; it is the
                # supported pyiqa API (the old `loss_mode` argument is not).
                metric = _create_pyiqa_metric("ssim", as_loss=True, device="cpu")
                metric.eval()
                for parameter in metric.parameters():
                    parameter.requires_grad_(False)
                # Keep the optional metric out of Lightning's model
                # state_dict/optimizer.  It is a fixed loss helper, not part
                # of the SR checkpoint and can otherwise make resume checks
                # depend on pyiqa internals.
                object.__setattr__(self, "ssim_loss_fn", metric)
                self._ssim_backend = "pyiqa"
            except Exception:
                self.ssim_loss_fn, self._ssim_backend = None, "manual"
        self.lr_key, self.hr_key, self.ref_key = lr_key, hr_key, ref_key
        self.ema = EMA(decay=float(ema_decay)) if use_ema else None
        self._ema_last_step = -1
        self._plateau_scheduler = None

    def _unpack_batch(self, batch):
        if isinstance(batch, dict):
            missing = [
                key
                for key in (self.lr_key, self.hr_key, self.ref_key)
                if key not in batch
            ]
            if missing:
                raise KeyError(f"batch 缺少字段: {missing}")
            return batch[self.lr_key], batch[self.hr_key], batch[self.ref_key]
        if not isinstance(batch, (tuple, list)) or len(batch) < 3:
            raise ValueError("batch 必须是包含 lr/hr/ref 的字典或序列")
        return batch[0], batch[1], batch[2]

    def _apply_ref_dropout(self, ref, lr=None):
        p = self.hparams.ref_drop_prob
        if p <= 0 or not self.training:
            return ref
        batch_size = ref.size(0)
        if lr is not None:
            # Dropped references follow the single-image SR path for every
            # batch size: use the current sample's bicubic-upsampled LR.
            replacement = F.interpolate(
                lr, size=ref.shape[-2:], mode="bicubic", align_corners=False
            )
            if replacement.shape[1] != ref.shape[1]:
                replacement = torch.zeros_like(ref)
        else:
            replacement = torch.zeros_like(ref)
        drop = torch.rand(batch_size, 1, 1, 1, device=ref.device) < p
        return torch.where(drop, replacement, ref)

    def _reference_input(self, ref, lr):
        if self.reference_mode != "lr_up":
            return ref
        replacement = F.interpolate(
            lr, size=ref.shape[-2:], mode="bicubic", align_corners=False
        )
        if replacement.shape[1] != ref.shape[1]:
            raise ValueError(
                f"lr_up reference 的通道数与 ref 不一致: {replacement.shape[1]} vs {ref.shape[1]}"
            )
        return replacement

    def forward(self, lr, ref):
        return self.model_sr(lr, ref)

    @staticmethod
    def _fft_loss(pred, target):
        pred_f, target_f = pred.float(), target.float()
        return (
            (
                torch.fft.rfft2(pred_f, norm="ortho")
                - torch.fft.rfft2(target_f, norm="ortho")
            )
            .abs()
            .mean()
        )

    @staticmethod
    def _manual_ssim_loss(pred, target):
        if pred.ndim != 4 or target.shape != pred.shape:
            raise ValueError(
                f"SSIM 输入形状不一致: {tuple(pred.shape)} vs {tuple(target.shape)}"
            )
        channels, window_size, sigma = pred.shape[1], 11, 1.5
        pred_f, target_f = pred.float(), target.float()
        coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
        gaussian = torch.exp(-((coords - window_size // 2) ** 2) / (2.0 * sigma**2))
        window_2d = torch.outer(gaussian, gaussian)
        window_2d = window_2d / window_2d.sum()
        window = (
            window_2d.view(1, 1, window_size, window_size)
            .expand(channels, 1, -1, -1)
            .contiguous()
        )
        pad = window_size // 2
        mu_p = F.conv2d(pred_f, window, padding=pad, groups=channels)
        mu_t = F.conv2d(target_f, window, padding=pad, groups=channels)
        sigma_p_sq = (
            F.conv2d(pred_f.square(), window, padding=pad, groups=channels)
            - mu_p.square()
        ).clamp_min(0.0)
        sigma_t_sq = (
            F.conv2d(target_f.square(), window, padding=pad, groups=channels)
            - mu_t.square()
        ).clamp_min(0.0)
        sigma_pt = (
            F.conv2d(pred_f * target_f, window, padding=pad, groups=channels)
            - mu_p * mu_t
        )
        c1, c2 = (0.01 * 2.0) ** 2, (0.03 * 2.0) ** 2
        numerator = (2.0 * mu_p * mu_t + c1) * (2.0 * sigma_pt + c2)
        denominator = (mu_p.square() + mu_t.square() + c1) * (
            sigma_p_sq + sigma_t_sq + c2
        )
        score = (numerator / denominator.clamp_min(1e-12)).clamp(-1.0, 1.0).mean()
        return 1.0 - score

    def _ssim_loss(self, pred, target):
        if self._ssim_backend == "pyiqa" and self.ssim_loss_fn is not None:
            metric = self.ssim_loss_fn
            metric_device = next(metric.parameters(), None)
            metric_device = (
                metric_device.device if metric_device is not None else pred.device
            )
            if metric_device != pred.device:
                metric.to(pred.device)
            metric.eval()
            pred_01 = ((pred.float() + 1.0) * 0.5).clamp(0.0, 1.0)
            target_01 = ((target.float() + 1.0) * 0.5).clamp(0.0, 1.0)
            score = metric(pred_01, target_01)
            return 1.0 - score.float().mean()
        return self._manual_ssim_loss(pred, target)

    def _compute_loss(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError(
                f"模型输出与 HR 形状不一致: {tuple(pred.shape)} vs {tuple(target.shape)}"
            )
        base_loss = self.criterion(pred, target)
        if base_loss.ndim:
            base_loss = base_loss.mean()
        total = base_loss
        metrics = {"l1": self.l1_loss(pred, target)}
        if self.ssim_weight > 0:
            metrics["ssim_loss"] = self._ssim_loss(pred, target)
            total = total + self.ssim_weight * metrics["ssim_loss"]
        if self.fft_weight > 0:
            metrics["fft_loss"] = self._fft_loss(pred, target)
            total = total + self.fft_weight * metrics["fft_loss"]
        return total, metrics

    def training_step(self, batch, batch_idx):
        lr, hr, ref = self._unpack_batch(batch)
        reference = self._reference_input(ref, lr)
        output = self(lr, self._apply_ref_dropout(reference, lr=lr))
        loss, metrics = self._compute_loss(output, hr)
        batch_size = hr.size(0)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train_l1",
            metrics["l1"],
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        if "ssim_loss" in metrics:
            self.log(
                "train_ssim_loss",
                metrics["ssim_loss"],
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )
        if "fft_loss" in metrics:
            self.log(
                "train_fft_loss",
                metrics["fft_loss"],
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        lr, hr, ref = self._unpack_batch(batch)
        output = self(lr, self._reference_input(ref, lr))
        loss, metrics = self._compute_loss(output, hr)
        batch_size = hr.size(0)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "val_l1", metrics["l1"], on_step=False, on_epoch=True, batch_size=batch_size
        )
        if "ssim_loss" in metrics:
            self.log(
                "val_ssim_loss",
                metrics["ssim_loss"],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        if "fft_loss" in metrics:
            self.log(
                "val_fft_loss",
                metrics["fft_loss"],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        mse = F.mse_loss(output.float(), hr.float())
        self.log(
            "val/psnr",
            10.0 * torch.log10(4.0 / mse.clamp_min(1e-8)),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def test_step(self, batch, batch_idx):
        lr, hr, ref = self._unpack_batch(batch)
        output = self(lr, self._reference_input(ref, lr))
        loss, metrics = self._compute_loss(output, hr)
        batch_size = hr.size(0)
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(
            "test_l1",
            metrics["l1"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def _ema_apply(self):
        if self.ema:
            self.ema.apply_shadow(self.model_sr)

    def _ema_restore(self):
        if self.ema:
            self.ema.restore(self.model_sr)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Lightning increments global_step only after a real optimizer step.
        # Tracking it avoids updating EMA on accumulated micro-batches and also
        # handles the final step of an epoch without a one-step lag.
        if self.ema:
            current_step = int(self.global_step)
            if current_step > self._ema_last_step:
                self.ema.update(self.model_sr)
                self._ema_last_step = current_step

    # Lightning 2.x model-state hooks, with start/end fallbacks for older
    # releases.  EMA.apply/restore are idempotent so both hook families can
    # safely coexist.
    def on_validation_model_eval(self):
        self._ema_apply()

    def on_validation_model_train(self):
        self._ema_restore()

    def on_validation_start(self):
        self._ema_apply()

    def on_validation_epoch_end(self):
        # Validation metrics are aggregated by Lightning before this hook.
        # Step here so monitoring callbacks see the updated optimizer and
        # scheduler state when they save a checkpoint for this validation run.
        self._step_plateau_scheduler()

    def on_validation_end(self):
        self._ema_restore()

    def on_test_model_eval(self):
        self._ema_apply()

    def on_test_model_train(self):
        self._ema_restore()

    def on_test_start(self):
        self._ema_apply()

    def on_test_end(self):
        self._ema_restore()

    def _step_plateau_scheduler(self):
        """Update ReduceLROnPlateau exactly once after each validation run.

        Lightning's automatic epoch update only sees the last validation of an
        epoch, while its step update also runs on batches without validation.
        Calling the scheduler here preserves every validation result without
        accidentally letting Lightning count only the final epoch metric.
        """
        if str(self.hparams.lr_scheduler).lower() != "plateau":
            return
        try:
            trainer = self.trainer
        except RuntimeError:
            return
        if trainer is None or getattr(trainer, "sanity_checking", False):
            return
        # During a validation loop nested inside ``fit``, Lightning switches
        # the running stage to VALIDATING, while ``state.fn`` remains FITTING.
        # Standalone ``validate()`` must not mutate the optimizer scheduler.
        run_fn = getattr(getattr(trainer, "state", None), "fn", None)
        run_fn = getattr(run_fn, "value", run_fn)
        if run_fn is not None:
            if str(run_fn).lower() not in {"fit", "fitting"}:
                return

        scheduler = self._plateau_scheduler
        if scheduler is None:
            for config in getattr(trainer, "lr_scheduler_configs", ()):
                candidate = getattr(config, "scheduler", None)
                if isinstance(candidate, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler = candidate
                    self._plateau_scheduler = candidate
                    break
        if scheduler is None:
            return

        callback_metrics = getattr(trainer, "callback_metrics", {}) or {}
        metric = callback_metrics.get("val_loss")
        if metric is None:
            raise RuntimeError(
                "ReduceLROnPlateau 需要验证指标 val_loss，但当前验证未记录该指标"
            )
        if torch.is_tensor(metric):
            if metric.numel() != 1 or not torch.isfinite(metric.detach()).item():
                raise RuntimeError(f"验证指标 val_loss 必须是有限标量，得到 {metric}")
            metric = metric.detach().float().item()
        else:
            metric = float(metric)
        if not math.isfinite(metric):
            raise RuntimeError(f"验证指标 val_loss 必须是有限标量，得到 {metric}")
        scheduler.step(metric)

    def lr_scheduler_step(self, scheduler, metric):
        # Plateau is stepped manually in on_validation_epoch_end so each
        # validation counts exactly once. Other scheduler types retain
        # Lightning's normal step behaviour.
        if str(self.hparams.lr_scheduler).lower() == "plateau" and isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            return
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def configure_optimizers(self):
        parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("没有可训练参数")
        learning_rate = float(self.hparams.learning_rate)
        optimizer = torch.optim.Adam(
            parameters,
            lr=learning_rate,
            betas=tuple(self.hparams.adam_betas),
            weight_decay=float(self.hparams.weight_decay),
        )
        scheduler_name = str(self.hparams.lr_scheduler).lower()
        if scheduler_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(self.hparams.lr_factor),
                patience=int(self.hparams.lr_patience),
                min_lr=float(self.hparams.lr_min),
                threshold=float(self.hparams.lr_threshold),
                threshold_mode="abs",
            )
            self._plateau_scheduler = scheduler
            return [optimizer], [
                {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": True,
                }
            ]

        warmup_steps = int(self.hparams.warmup_steps)
        try:
            max_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0))
        except (RuntimeError, TypeError, ValueError):
            max_steps = 0
        if max_steps <= 0:
            max_steps = 100000
        eta_min = min(1e-6, learning_rate * 0.01)

        if warmup_steps <= 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, max_steps), eta_min=eta_min
            )
        elif max_steps <= warmup_steps:
            # There is no post-warmup phase in a very short run.
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=max(1, max_steps),
            )
        else:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, max_steps - warmup_steps), eta_min=eta_min
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        clip_val = (
            gradient_clip_val
            if gradient_clip_val is not None
            else self.hparams.grad_clip_norm
        )
        if clip_val is not None and float(clip_val) > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=float(clip_val),
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint["ema_state_dict"] = self.ema.state_dict()
        if str(self.hparams.lr_scheduler).lower() == "plateau":
            checkpoint["plateau_step_unit"] = "validation"
        signature = getattr(self, "_experiment_signature", None)
        if isinstance(signature, dict):
            checkpoint["refsrwkv_experiment_signature"] = dict(signature)

    def on_load_checkpoint(self, checkpoint):
        if self.ema and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

    def on_train_start(self):
        if self.ema:
            self.ema._lazy_init(self.model_sr)
            self._ema_last_step = int(self.global_step)
        scheduler_name = str(self.hparams.lr_scheduler).lower()
        scheduler_text = (
            f"plateau(each validation, patience={self.hparams.lr_patience}, threshold={self.hparams.lr_threshold}, factor={self.hparams.lr_factor}, min={self.hparams.lr_min})"
            if scheduler_name == "plateau"
            else f"cosine(warmup={self.hparams.warmup_steps})"
        )
        print(
            f"✅ LitRefSRWKV 训练开始 | 参数量: {sum(p.numel() for p in self.model_sr.parameters()) / 1e6:.2f}M | lr_scheduler={scheduler_text} | grad_clip={self.hparams.grad_clip_norm} | EMA={'on' if self.ema else 'off'} | SSIM={self.ssim_weight} | FFT={self.fft_weight}"
        )
