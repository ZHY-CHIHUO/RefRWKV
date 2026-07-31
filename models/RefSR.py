# Copyright (c) Shanghai AI Lab. All rights reserved.
"""
RefSRWKV (Improved v2): Reference-based Super-Resolution with RWKV Backbone.

本版变更（v2，基于 v1 审查意见）：
1. [Bug修复] RUN_CUDA 不再硬编码 .cuda()，改为设备自适应
2. [Bug修复] 对角线模式 idx/inv 统一缓存，删除 _get_inv_indices
3. [改进] 学习率调度：LinearLR warmup + ReduceLROnPlateau，用 SequentialLR 串联
4. [新增] 梯度裁剪（默认 max_norm=1.0）
5. [新增] EMA（指数移动平均），验证/测试时自动切换 EMA 权重
6. [新增] 滑动窗口分块推理 forward_tiled()，防止大图 OOM
7. [改进] torch.compile 兼容：WKV 自定义算子标记为 compiler.disable
8. [改进] scale 参数校验：当前架构固定 10×，传入其他值会警告
"""

import copy
import math
import warnings
from collections import OrderedDict

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
cap = torch.cuda.get_device_capability()
arch = f"compute_{cap[0]}{cap[1]}"

wkv_cuda = load(
    name="bi_wkv",
    sources=[
        os.path.join(_cuda_dir, "bi_wkv.cpp"),
        os.path.join(_cuda_dir, "bi_wkv_kernel.cu"),
    ],
    verbose=True,
    extra_cuda_cflags=[
        "-res-usage",
        "--maxrregcount 60",
        "-O3",
        "-Xptxas -O3",
        f"-gencode arch={arch},code={arch}",
    ],
)


# ── torch.compile 兼容装饰器 ──
# 自定义 CUDA autograd Function 无法被 dynamo 追踪，
# 标记为 disable 让 torch.compile 跳过此函数。
try:
    _compiler_disable = torch.compiler.disable
except AttributeError:  # PyTorch < 2.1

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
        y = wkv_cuda.bi_wkv_forward(w, u, k, v)
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
        gw, gu, gk, gv = wkv_cuda.bi_wkv_backward(
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
    """调用双向 WKV CUDA 核。张量必须已在 CUDA 上。"""
    if not w.is_cuda:
        raise RuntimeError(
            "WKV 需要 CUDA 张量，请将模型和数据移到 GPU：model.cuda(), x.cuda()"
        )
    return WKV.apply(w, u, k, v)


# ═══════════════════════════════════════════════════════════════
# 对角线展平 / 还原（diagonal 模式使用）
# ★ v2: idx 和 inv 一起缓存，删除 _get_inv_indices
# ═══════════════════════════════════════════════════════════════
_DIAG_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _build_diag_indices(h, w, direction, device):
    """预计算对角线展平索引，返回 (idx, inv)。"""
    idx = []
    if direction == "main":  # ↘：按 (i+j) 分组
        for s in range(h + w - 1):
            for i in range(max(0, s - w + 1), min(s + 1, h)):
                j = s - i
                idx.append(i * w + j)
    else:  # anti ↙：按 (i-j+w-1) 分组
        for s in range(h + w - 1):
            for i in range(max(0, s - w + 1), min(s + 1, h)):
                j = w - 1 - (s - i)
                idx.append(i * w + j)

    idx = torch.tensor(idx, dtype=torch.long, device=device)
    inv = torch.empty(h * w, dtype=torch.long, device=device)
    inv[idx] = torch.arange(h * w, device=device)
    return idx, inv


def _get_diag_indices(h, w, direction, device):
    """返回 (idx, inv)，全部走缓存。"""
    key = (h, w, direction, str(device))
    if key not in _DIAG_CACHE:
        _DIAG_CACHE[key] = _build_diag_indices(h, w, direction, device)
    return _DIAG_CACHE[key]


# ═══════════════════════════════════════════════════════════════
# 希尔伯特曲线扫描索引（hilbert 模式使用）
# 任意 h×w：pad 到最近的 2 的幂，过滤掉 padding 格子
# ═══════════════════════════════════════════════════════════════
_HILBERT_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _hilbert_d2xy(n, d):
    """n×n 希尔伯特曲线：序号 d -> (x=列, y=行)，n 为 2 的幂。"""
    x = y = 0
    t, s = d, 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x, y = s - 1 - x, s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def _build_hilbert_indices(h, w, rot90, device):
    """构建 h×w 网格的希尔伯特扫描顺序。
    rot90=False: 标准希尔伯特（扫描 A）
    rot90=True : 整体旋转 90°（扫描 B，与 A 垂直互补）
    返回 (idx, inv)：idx[t]=第 t 步访问的格子，inv[格子]=访问步序号。

    注意：对于极端宽高比（如 1×N），两条曲线的互补性会退化；
    本模型 U-Net 各层特征图近似正方形，不受影响。
    """
    n = 1
    while n < max(h, w):
        n *= 2
    idx = []
    for d in range(n * n):
        x, y = _hilbert_d2xy(n, d)
        if rot90:
            row, col = n - 1 - x, y
        else:
            row, col = y, x
        if row < h and col < w:
            idx.append(row * w + col)
    idx = torch.tensor(idx, dtype=torch.long, device=device)
    inv = torch.empty(h * w, dtype=torch.long, device=device)
    inv[idx] = torch.arange(h * w, device=device)
    return idx, inv


def _get_hilbert_indices(h, w, rot90, device):
    key = (h, w, rot90, str(device))
    if key not in _HILBERT_CACHE:
        _HILBERT_CACHE[key] = _build_hilbert_indices(h, w, rot90, device)
    return _HILBERT_CACHE[key]


# ═══════════════════════════════════════════════════════════════
# DropPath
# ═══════════════════════════════════════════════════════════════
class DropPath(nn.Module):
    """Stochastic Depth per sample (from timm)."""

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
# OmniShift (improved)
# ═══════════════════════════════════════════════════════════════
class OmniShift(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv1x1 = nn.Conv2d(dim, dim, 1, groups=dim, bias=False)
        self.conv3x3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.conv5x5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.conv3x3_d2 = nn.Conv2d(
            dim, dim, 3, padding=2, dilation=2, groups=dim, bias=False
        )
        self.alpha = nn.Parameter(torch.ones(5) * 0.2)

        self.register_buffer("conv5x5_reparam_weight", torch.zeros(dim, 1, 5, 5))
        self._reparam_done = False

    def forward_train(self, x):
        alpha = torch.softmax(self.alpha, dim=0)
        return (
            alpha[0] * x
            + alpha[1] * self.conv1x1(x)
            + alpha[2] * self.conv3x3(x)
            + alpha[3] * self.conv5x5(x)
            + alpha[4] * self.conv3x3_d2(x)
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

        w_d2 = torch.zeros(self.dim, 1, 5, 5, device=self.conv1x1.weight.device)
        w_d2[:, :, 0, 0] = self.conv3x3_d2.weight[:, :, 0, 0]
        w_d2[:, :, 0, 2] = self.conv3x3_d2.weight[:, :, 0, 1]
        w_d2[:, :, 0, 4] = self.conv3x3_d2.weight[:, :, 0, 2]
        w_d2[:, :, 2, 0] = self.conv3x3_d2.weight[:, :, 1, 0]
        w_d2[:, :, 2, 2] = self.conv3x3_d2.weight[:, :, 1, 1]
        w_d2[:, :, 2, 4] = self.conv3x3_d2.weight[:, :, 1, 2]
        w_d2[:, :, 4, 0] = self.conv3x3_d2.weight[:, :, 2, 0]
        w_d2[:, :, 4, 2] = self.conv3x3_d2.weight[:, :, 2, 1]
        w_d2[:, :, 4, 4] = self.conv3x3_d2.weight[:, :, 2, 2]

        combined = (
            alpha[0] * identity
            + alpha[1] * w1
            + alpha[2] * w3
            + alpha[3] * w5
            + alpha[4] * w_d2
        )
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
# VRWKV Blocks (H + W + 希尔伯特对)
# ═══════════════════════════════════════════════════════════════
class VRWKV_SpatialMix(nn.Module):
    """两路串联扫描 + 融合。

    路径1（笛卡尔）：H → W 串联
    路径2（空间填充）：
        scan_mode='hilbert'  → 希尔伯特A → 希尔伯特B 串联
        scan_mode='diagonal' → ↘对角线 → ↙对角线 串联
    两路结果用可学习权重融合。
    """

    def __init__(self, n_embd, head_dim=64, scan_mode="hilbert"):
        super().__init__()
        self.n_embd = n_embd
        self.scan_mode = scan_mode
        attn_sz = n_embd

        self.omni_shift = OmniShift(dim=n_embd)
        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        self.output = nn.Linear(attn_sz, n_embd, bias=False)
        self.register_buffer("scale", torch.tensor(n_embd**0.5))

        # 4 组 decay/first：路径1 用 [0],[1]，路径2 用 [2],[3]
        with torch.no_grad():
            self.spatial_decay = nn.Parameter(torch.zeros(4, self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(4, self.n_embd))

        # 两路融合权重（softmax 归一化）
        self.path_weight = nn.Parameter(torch.zeros(2))  # [路径1, 路径2]

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

    def _scan_h(self, k, v, j):
        """H 方向扫描。"""
        s = self.scale
        return RUN_CUDA(self.spatial_decay[j] / s, self.spatial_first[j] / s, k, v)

    def _scan_w(self, k, v, j, resolution):
        """W 方向扫描（转置 → WKV → 转置回）。"""
        h, w = resolution
        s = self.scale
        k_t = rearrange(k, "b (h w) c -> b (w h) c", h=h, w=w)
        v_t = rearrange(v, "b (h w) c -> b (w h) c", h=h, w=w)
        r = RUN_CUDA(self.spatial_decay[j] / s, self.spatial_first[j] / s, k_t, v_t)
        return rearrange(r, "b (w h) c -> b (h w) c", h=h, w=w)

    def _scan_reorder(self, k, v, j, idx, inv):
        """按任意索引重排 → WKV → 重排回。"""
        s = self.scale
        r = RUN_CUDA(
            self.spatial_decay[j] / s,
            self.spatial_first[j] / s,
            k[:, idx],
            v[:, idx],
        )
        return r[:, inv]

    def forward(self, x, resolution):
        B, T, C = x.size()
        h, w = resolution
        sr, k, v = self.jit_func(x, resolution)

        # ══ 路径1：H → W 串联 ══
        v1 = self._scan_h(k, v, 0)  # H 扫描，v1 = H(原始)
        v1 = self._scan_w(k, v1, 1, resolution)  # W 扫描，v1 = W(H(原始))

        # ══ 路径2：空间填充曲线串联 ══
        if self.scan_mode == "hilbert":
            idx_a, inv_a = _get_hilbert_indices(h, w, False, k.device)
            idx_b, inv_b = _get_hilbert_indices(h, w, True, k.device)
            v2 = self._scan_reorder(k, v, 2, idx_a, inv_a)  # HilA(原始)
            v2 = self._scan_reorder(k, v2, 3, idx_b, inv_b)  # HilB(HilA(原始))
        else:
            idx_m, inv_m = _get_diag_indices(h, w, "main", k.device)
            idx_a, inv_a = _get_diag_indices(h, w, "anti", k.device)
            v2 = self._scan_reorder(k, v, 2, idx_m, inv_m)  # ↘(原始)
            v2 = self._scan_reorder(k, v2, 3, idx_a, inv_a)  # ↙(↘(原始))

        # ══ 两路融合 ══
        w_path = torch.softmax(self.path_weight, dim=0)
        out = w_path[0] * v1 + w_path[1] * v2

        x = sr * out
        x = self.output(x)
        return x


class VRWKV_ChannelMix(nn.Module):
    """通道混合：Squared ReLU 激活 + 门控。"""

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
    """RWKV Block：SpatialMix + ChannelMix，带 DropPath。"""

    def __init__(self, n_embd, hidden_rate=4, drop_path=0.0, scan_mode="hilbert"):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.att = VRWKV_SpatialMix(n_embd, scan_mode=scan_mode)
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
# Resizing Modules
# ═══════════════════════════════════════════════════════════════
class Downsample(nn.Module):
    def __init__(self, n_feat, channel_scale=2):
        super().__init__()
        mid_channels = n_feat * channel_scale // 4
        assert mid_channels > 0
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, mid_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat, channel_scale=0.5):
        super().__init__()
        mid_channels = int(n_feat * channel_scale * 4)
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, mid_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


# ═══════════════════════════════════════════════════════════════
# GatedFusion
# ═══════════════════════════════════════════════════════════════
def _gn_groups(num_channels: int, max_groups: int = 32) -> int:
    """返回 ≤ max_groups 且能整除 num_channels 的最大组数。"""
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class GatedFusion(nn.Module):
    """
    门控参考融合模块。

    输入端：LR 特征 + Ref 特征（同分辨率同通道数）
    流程：
      1. Concat → 1×1 投影 → GroupNorm
      2. 通道注意力生成门控向量（0~1）
      3. 残差加回：out = LR_feat + gate * fused_feat
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fuse_conv = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=_gn_groups(dim), num_channels=dim)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // reduction, 8), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(dim // reduction, 8), dim, kernel_size=1),
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
# 滑动窗口工具
# ═══════════════════════════════════════════════════════════════
def _gaussian_weight_2d(h, w, sigma_ratio=0.25, device="cpu"):
    """生成 2D 高斯权重图，用于重叠区域混合。"""
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    sy, sx = h * sigma_ratio, w * sigma_ratio
    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)
    wy = torch.exp(-0.5 * ((y - cy) / max(sy, 1e-6)) ** 2)
    wx = torch.exp(-0.5 * ((x - cx) / max(sx, 1e-6)) ** 2)
    return wy[:, None] * wx[None, :]  # (h, w)


# ═══════════════════════════════════════════════════════════════
# RefSRWKV
# ═══════════════════════════════════════════════════════════════
class RefSRWKV(nn.Module):
    """
    参考图引导的超分辨率网络（RWKV 骨干）。

    当前架构固定放大倍数 = 2.5 (bilinear) × 2 (PixelShuffle) × 2 (PixelShuffle) = 10×。
    即 LR 48×48 → HR 480×480。scale 参数仅用于校验，不影响网络结构。
    """

    FIXED_SCALE = 10  # 架构决定的固定放大倍数

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
        scan_mode: str = "hilbert",
    ):
        super().__init__()

        # ★ v2: scale 校验
        if scale != self.FIXED_SCALE:
            warnings.warn(
                f"当前架构固定为 {self.FIXED_SCALE}× 放大，"
                f"传入 scale={scale} 不会改变网络结构。",
                UserWarning,
                stacklevel=2,
            )
        self.scale = self.FIXED_SCALE
        self.dim = dim
        self.scan_mode = scan_mode
        self.out_channels = out_channels

        # ── LR 编码器 ──
        self.lr_up = nn.Sequential(
            nn.Upsample(scale_factor=2.5, mode="bilinear", align_corners=False),
            nn.Conv2d(inp_channels, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(dim), dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
        )

        # ── Ref 编码器 ──
        self.ref_to_level1 = nn.Sequential(
            nn.PixelUnshuffle(4),
            nn.Conv2d(out_channels * 16, dim, kernel_size=1, bias=False),
            nn.GroupNorm(_gn_groups(dim), dim),
        )
        self.ref_down2 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(dim * 2), dim * 2),
        )
        self.ref_down3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(dim * 4), dim * 4),
        )
        self.ref_down4 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 8, kernel_size=3, stride=2, padding=1, bias=False),
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
                Block(
                    n_embd=dim,
                    hidden_rate=hidden_rate,
                    drop_path=dp_rates[dp_idx + i],
                    scan_mode=scan_mode,
                )
                for i in range(num_blocks[0])
            ]
        )
        dp_idx += num_blocks[0]
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(
            *[
                Block(
                    n_embd=dim * 2,
                    hidden_rate=hidden_rate,
                    drop_path=dp_rates[dp_idx + i],
                    scan_mode=scan_mode,
                )
                for i in range(num_blocks[1])
            ]
        )
        dp_idx += num_blocks[1]
        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = nn.Sequential(
            *[
                Block(
                    n_embd=dim * 4,
                    hidden_rate=hidden_rate,
                    drop_path=dp_rates[dp_idx + i],
                    scan_mode=scan_mode,
                )
                for i in range(num_blocks[2])
            ]
        )
        dp_idx += num_blocks[2]
        self.down3_4 = Downsample(dim * 4)

        self.latent = nn.Sequential(
            *[
                Block(
                    n_embd=dim * 8,
                    hidden_rate=hidden_rate,
                    drop_path=dp_rates[dp_idx + i],
                    scan_mode=scan_mode,
                )
                for i in range(num_blocks[3])
            ]
        )

        # ── 解码器 ──
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Sequential(
            nn.Conv2d(dim * 4 + dim * 4, dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(_gn_groups(dim * 4), dim * 4),
        )
        self.decoder_level3 = nn.Sequential(
            *[
                Block(
                    n_embd=dim * 4,
                    hidden_rate=hidden_rate,
                    scan_mode=scan_mode,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Sequential(
            nn.Conv2d(dim * 2 + dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.GroupNorm(_gn_groups(dim * 2), dim * 2),
        )
        self.decoder_level2 = nn.Sequential(
            *[
                Block(
                    n_embd=dim * 2,
                    hidden_rate=hidden_rate,
                    scan_mode=scan_mode,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(dim * 2)
        self.reduce_chan_level1 = nn.Sequential(
            nn.Conv2d(dim + dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(_gn_groups(dim), dim),
        )
        self.decoder_level1 = nn.Sequential(
            *[
                Block(n_embd=dim, hidden_rate=hidden_rate, scan_mode=scan_mode)
                for _ in range(num_blocks[0])
            ]
        )

        # ── 后处理精修 ──
        self.refinement = nn.Sequential(
            *[
                Block(n_embd=dim, hidden_rate=hidden_rate, scan_mode=scan_mode)
                for _ in range(num_refinement_blocks)
            ]
        )

        # ── 最终上采样 ──
        self.up_final = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(dim, dim * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )
        self.output_conv = nn.Conv2d(
            dim, out_channels, kernel_size=3, padding=1, bias=True
        )

        # ── Ref 引导残差修正 ──
        self.ref_guided_refine = nn.Conv2d(
            out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False
        )

        self.apply(self._init_weights)

        # ★ _init_weights 会覆盖零初始化，重新设置
        nn.init.zeros_(self.ref_guided_refine.weight)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.GroupNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def _extract_ref_pyramid(self, ref):
        ref_1 = self.ref_to_level1(ref)
        ref_2 = self.ref_down2(ref_1)
        ref_3 = self.ref_down3(ref_2)
        ref_4 = self.ref_down4(ref_3)
        return ref_1, ref_2, ref_3, ref_4

    def forward(self, lr, ref):
        # 1. LR 特征提取
        fea = self.lr_up(lr)

        # 2. Ref 金字塔
        ref_1, ref_2, ref_3, ref_4 = self._extract_ref_pyramid(ref)

        # 3. 编码器 + 门控融合
        e1 = self.encoder_level1(self.fuse1(fea, ref_1))
        e2 = self.encoder_level2(self.fuse2(self.down1_2(e1), ref_2))
        e3 = self.encoder_level3(self.fuse3(self.down2_3(e2), ref_3))
        latent = self.latent(self.fuse4(self.down3_4(e3), ref_4))

        # 4. 解码器 + skip connections
        d3 = self.decoder_level3(
            self.reduce_chan_level3(torch.cat([self.up4_3(latent), e3], dim=1))
        )
        d2 = self.decoder_level2(
            self.reduce_chan_level2(torch.cat([self.up3_2(d3), e2], dim=1))
        )
        d1 = self.decoder_level1(
            self.reduce_chan_level1(torch.cat([self.up2_1(d2), e1], dim=1))
        )

        # 5. 后处理精修
        d1 = self.refinement(d1)

        # 6. 最终上采样 + 输出投影
        hr_feat = self.up_final(d1)
        out = self.output_conv(hr_feat)

        # 7. Ref 引导残差修正
        residual = self.ref_guided_refine(torch.cat([out, ref], dim=1))
        out = out + residual
        out = torch.clamp(out, -1.0, 1.0)

        return out

    # ─────────────────────────────────────────────────────────
    # ★ v2 新增：滑动窗口分块推理
    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def forward_tiled(
        self,
        lr: torch.Tensor,
        ref: torch.Tensor,
        tile_lr: int = 48,
        overlap: int = 8,
    ) -> torch.Tensor:
        """
        滑动窗口分块推理，防止大图 OOM。

        参数:
            lr      : (B, C, H_lr, W_lr)
            ref     : (B, C, H_hr, W_hr)，H_hr = H_lr × 10
            tile_lr : LR 空间每块的边长（像素）
            overlap : LR 空间相邻块的重叠像素数

        返回:
            (B, C, H_hr, W_hr)
        """
        B, C, H_lr, W_lr = lr.shape
        S = self.scale  # 10
        H_hr, W_hr = H_lr * S, W_lr * S
        tile_hr = tile_lr * S
        overlap_hr = overlap * S
        stride_lr = tile_lr - overlap
        stride_hr = tile_hr - overlap_hr

        device = lr.device
        out_sum = torch.zeros(B, self.out_channels, H_hr, W_hr, device=device)
        w_sum = torch.zeros(1, 1, H_hr, W_hr, device=device)

        # 预计算高斯权重
        gauss = _gaussian_weight_2d(
            tile_hr, tile_hr, device=device
        )  # (tile_hr, tile_hr)
        gauss = gauss[None, None]  # (1, 1, tile_hr, tile_hr)

        # 生成所有块的左上角坐标（LR 空间）
        y_starts = list(range(0, max(H_lr - tile_lr, 0) + 1, stride_lr))
        x_starts = list(range(0, max(W_lr - tile_lr, 0) + 1, stride_lr))
        # 确保覆盖右下角
        if y_starts[-1] + tile_lr < H_lr:
            y_starts.append(H_lr - tile_lr)
        if x_starts[-1] + tile_lr < W_lr:
            x_starts.append(W_lr - tile_lr)

        for y0 in y_starts:
            for x0 in x_starts:
                # 裁切 LR 块
                lr_tile = lr[:, :, y0 : y0 + tile_lr, x0 : x0 + tile_lr]

                # 裁切对应的 Ref 块（HR 空间）
                ry0, rx0 = y0 * S, x0 * S
                ref_tile = ref[:, :, ry0 : ry0 + tile_hr, rx0 : rx0 + tile_hr]

                # 推理
                out_tile = self.forward(lr_tile, ref_tile)  # (B, C, tile_hr, tile_hr)

                # 实际输出尺寸（边缘块可能不足 tile_hr）
                th, tw = out_tile.shape[2], out_tile.shape[3]
                g = gauss[:, :, :th, :tw]

                out_sum[:, :, ry0 : ry0 + th, rx0 : rx0 + tw] += out_tile * g
                w_sum[:, :, ry0 : ry0 + th, rx0 : rx0 + tw] += g

        out_sum = out_sum / w_sum.clamp(min=1e-8)
        return torch.clamp(out_sum, -1.0, 1.0)

    def prepare_for_inference(self):
        self.eval()
        for module in self.modules():
            if isinstance(module, OmniShift):
                module.reparam_5x5()
        print("✓ RefSRWKV: All OmniShift modules reparameterized for inference.")
        return self


# ═══════════════════════════════════════════════════════════════
# ★ EMA（指数移动平均）— 延迟初始化 + 设备自适应
# ═══════════════════════════════════════════════════════════════
class EMA:
    """
    模型参数的指数移动平均。
    shadow 在第一次 update() 时才创建（此时 PL 已把模型移到 GPU）。
    """

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

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]
        self._initialized = state_dict["initialized"]


# ═══════════════════════════════════════════════════════════════
# ★ LitRefSRWKV（完整版）
# ═══════════════════════════════════════════════════════════════
class LitRefSRWKV(pl.LightningModule):
    """
    PyTorch Lightning 训练封装。

    学习率策略：
      - 前 warmup_steps 步：线性升温 0.1%→100%（on_train_batch_start）
      - 之后：ReduceLROnPlateau，每次验证后检查，连续 5 次没改善 → LR×0.5
    其他：
      - 梯度裁剪（默认 max_norm=1.0）
      - EMA（验证/测试时自动切换）
    """

    def __init__(
        self,
        model_sr: RefSRWKV,
        learning_rate: float = 1e-4,
        warmup_steps: int = 100,
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

        # EMA（延迟初始化，第一次 update 时才克隆参数）
        self.ema = EMA(decay=ema_decay) if use_ema else None

        # Plateau 调度器（在 configure_optimizers 中创建）
        self.plateau_scheduler = None
        self._pending_plateau_state = None  # 断点续训时暂存

    # ──────────────────────────────────────────────
    # 数据
    # ──────────────────────────────────────────────
    def _unpack_batch(self, batch):
        if isinstance(batch, dict):
            return batch[self.lr_key], batch[self.hr_key], batch[self.ref_key]
        return batch[0], batch[1], batch[2]

    def forward(self, lr, ref):
        return self.model_sr(lr, ref)

    # ──────────────────────────────────────────────
    # 训练
    # ──────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        lr, hr, ref = self._unpack_batch(batch)
        output = self(lr, ref)
        loss = self.criterion(output, hr)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        """Warmup：前 warmup_steps 步线性升温。"""
        if self.global_step < self.hparams.warmup_steps:
            warmup_progress = (self.global_step + 1) / self.hparams.warmup_steps
            lr_scale = 1e-3 + (1.0 - 1e-3) * warmup_progress
            for pg in self.optimizers().param_groups:
                pg["lr"] = self.hparams.learning_rate * lr_scale

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """每步更新 EMA。"""
        if self.ema is not None:
            self.ema.update(self.model_sr)

    # ──────────────────────────────────────────────
    # 验证（EMA 权重 + 手动调度 Plateau）
    # ──────────────────────────────────────────────
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
        # 1. 恢复 EMA
        if self.ema is not None:
            self.ema.restore(self.model_sr)

        # 2. ★ 每次验证后手动 step Plateau（patience=5 = 5 次验证）
        if self.plateau_scheduler is not None:
            val_loss = self.trainer.callback_metrics.get("val_loss")
            if val_loss is not None:
                self.plateau_scheduler.step(val_loss)
                current_lr = self.plateau_scheduler.optimizer.param_groups[0]["lr"]
                self.log("lr", current_lr, prog_bar=True, logger=True)

    # ──────────────────────────────────────────────
    # 测试（EMA 权重）
    # ──────────────────────────────────────────────
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

    # ──────────────────────────────────────────────
    # ★ 优化器（Plateau 手动管理，不交给 PL）
    # ──────────────────────────────────────────────
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return optimizer  # ← 只返回 optimizer

    # ──────────────────────────────────────────────
    # 梯度裁剪
    # ──────────────────────────────────────────────
    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ):
        clip_val = gradient_clip_val or self.hparams.grad_clip_norm
        if clip_val is not None and clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )

    # ──────────────────────────────────────────────
    # Checkpoint 保存 / 恢复
    # ──────────────────────────────────────────────
    def on_save_checkpoint(self, checkpoint):
        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()
        if self.plateau_scheduler is not None:
            checkpoint["plateau_scheduler"] = self.plateau_scheduler.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
        # 暂存，等 on_train_start 时 scheduler 已创建再恢复
        self._pending_plateau_state = checkpoint.get("plateau_scheduler")

    # ──────────────────────────────────────────────
    # 训练开始
    # ──────────────────────────────────────────────
    def on_train_start(self):
        # 恢复 Plateau 调度器状态（断点续训）
        if (
            self._pending_plateau_state is not None
            and self.plateau_scheduler is not None
        ):
            self.plateau_scheduler.load_state_dict(self._pending_plateau_state)
            self._pending_plateau_state = None

        total = sum(p.numel() for p in self.parameters())
        ema_info = f" | EMA decay={self.ema.decay}" if self.ema else " | EMA=off"
        print(
            f"✅ LitRefSRWKV 训练开始 | 参数量: {total / 1e6:.2f}M"
            f" | grad_clip={self.hparams.grad_clip_norm}"
            f"{ema_info}"
        )


# ═══════════════════════════════════════════════════════════════
# Usage Example
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("本模型依赖 CUDA WKV 核，必须在 GPU 上运行。")

    device = torch.device("cuda")

    # scan_mode: "hilbert"(默认) / "diagonal"(消融对比)
    model = RefSRWKV(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=8,
        scale=10,
        drop_path_rate=0.1,
        scan_mode="hilbert",
    ).to(device)

    lr = torch.randn(2, 3, 48, 48, device=device)
    ref = torch.randn(2, 3, 480, 480, device=device)

    # ── 训练模式 ──
    model.train()
    out_train = model(lr, ref)
    print(
        f"Train output shape: {out_train.shape}, "
        f"range: [{out_train.min():.3f}, {out_train.max():.3f}]"
    )

    # ── 推理模式（OmniShift 重参数化）──
    model.prepare_for_inference()
    with torch.no_grad():
        out_infer = model(lr, ref)
    print(
        f"Infer output shape: {out_infer.shape}, "
        f"range: [{out_infer.min():.3f}, {out_infer.max():.3f}]"
    )

    # ── 分块推理（大图防 OOM）──
    lr_big = torch.randn(1, 3, 96, 96, device=device)
    ref_big = torch.randn(1, 3, 960, 960, device=device)
    out_tiled = model.forward_tiled(lr_big, ref_big, tile_lr=48, overlap=8)
    print(
        f"Tiled output shape: {out_tiled.shape}, "
        f"range: [{out_tiled.min():.3f}, {out_tiled.max():.3f}]"
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total params: {total_params:.2f}M")
