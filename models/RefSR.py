# Copyright (c) Shanghai AI Lab. All rights reserved.
"""
RefSRWKV (Improved): Reference-based Super-Resolution with RWKV Backbone.

本版变更：
1. VRWKV_SpatialMix 扩展为 8 方向扫描（H + W + ↘ + ↙，每个方向 bi_wkv 内部双向）
2. 新增对角线展平/还原工具
3. 修复 ref_guided_refine 零初始化被 _init_weights 覆盖的问题
"""

import torch
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


def RUN_CUDA(w, u, k, v):
    return WKV.apply(w.cuda(), u.cuda(), k.cuda(), v.cuda())


# ═══════════════════════════════════════════════════════════════
# 对角线展平 / 还原
# ═══════════════════════════════════════════════════════════════

_DIAG_CACHE = {}


def _build_diag_indices(h, w, direction, device):
    """预计算对角线展平索引。"""
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
    return torch.tensor(idx, dtype=torch.long, device=device)


def _get_diag_indices(h, w, direction, device):
    key = (h, w, direction, str(device))
    if key not in _DIAG_CACHE:
        _DIAG_CACHE[key] = _build_diag_indices(h, w, direction, device)
    return _DIAG_CACHE[key]


def _get_inv_indices(idx, T, device):
    """构建还原索引：inv[idx[i]] = i。"""
    inv = torch.empty(T, dtype=torch.long, device=device)
    inv[idx] = torch.arange(T, device=device)
    return inv


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
        # ★ 新增：3×3 空洞卷积（dilation=2，感受野 5×5 但覆盖不同位置）
        self.conv3x3_d2 = nn.Conv2d(
            dim, dim, 3, padding=2, dilation=2, groups=dim, bias=False
        )
        # ★ 5 个分支（原来 4 个）
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
            + alpha[4] * self.conv3x3_d2(x)  # ★ 新增
        )

    def reparam_5x5(self):
        if self._reparam_done:
            return
        alpha = torch.softmax(self.alpha, dim=0)

        # Identity → 5×5
        identity = torch.zeros(self.dim, 1, 5, 5, device=self.conv1x1.weight.device)
        identity[:, :, 2, 2] = 1.0

        # 1×1 → pad 到 5×5
        w1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        # 3×3 → pad 到 5×5
        w3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))
        # 5×5 不变
        w5 = self.conv5x5.weight

        # ★ 3×3 dilation=2 → 填充到 5×5（间隔填零）
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
            + alpha[4] * w_d2  # ★ 新增
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
# VRWKV Blocks (8-directional)
# ═══════════════════════════════════════════════════════════════
class VRWKV_SpatialMix(nn.Module):
    """8 方向空间混合：H + W + ↘ + ↙，每个方向 bi_wkv 内部双向。

    扫描方向：
        0: H 正向 + H 反向（bi_wkv 内部）
        1: W 正向 + W 反向（bi_wkv 内部）
        2: ↘ 正向 + ↖ 反向（bi_wkv 内部）
        3: ↙ 正向 + ↗ 反向（bi_wkv 内部）
    共 8 方向，加权融合。
    """

    def __init__(self, n_embd, head_dim=64):
        super().__init__()
        self.n_embd = n_embd
        self.num_scans = 4  # H, W, ↘, ↙
        attn_sz = n_embd

        self.omni_shift = OmniShift(dim=n_embd)

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.register_buffer("scale", torch.tensor(n_embd**0.5))

        with torch.no_grad():
            self.spatial_decay = nn.Parameter(torch.zeros(self.num_scans, self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.num_scans, self.n_embd))

        # 4 方向融合权重（可学习）
        self.dir_weight = nn.Parameter(torch.ones(self.num_scans) / self.num_scans)

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
        h, w = resolution
        sr, k, v = self.jit_func(x, resolution)
        s = self.scale

        results = []

        # ── 扫描 0: H 方向（bi_wkv 内部 = H正 + H反）──
        results.append(
            RUN_CUDA(self.spatial_decay[0] / s, self.spatial_first[0] / s, k, v)
        )

        # ── 扫描 1: W 方向（bi_wkv 内部 = W正 + W反）──
        k_t = rearrange(k, "b (h w) c -> b (w h) c", h=h, w=w)
        v_t = rearrange(v, "b (h w) c -> b (w h) c", h=h, w=w)
        r_w = RUN_CUDA(self.spatial_decay[1] / s, self.spatial_first[1] / s, k_t, v_t)
        results.append(rearrange(r_w, "b (w h) c -> b (h w) c", h=h, w=w))

        # ── 扫描 2: ↘ 对角线（bi_wkv 内部 = ↘ + ↖）──
        idx_main = _get_diag_indices(h, w, "main", k.device)
        inv_main = _get_inv_indices(idx_main, T, k.device)
        r_d = RUN_CUDA(
            self.spatial_decay[2] / s,
            self.spatial_first[2] / s,
            k[:, idx_main],
            v[:, idx_main],
        )
        results.append(r_d[:, inv_main])

        # ── 扫描 3: ↙ 对角线（bi_wkv 内部 = ↙ + ↗）──
        idx_anti = _get_diag_indices(h, w, "anti", k.device)
        inv_anti = _get_inv_indices(idx_anti, T, k.device)
        r_a = RUN_CUDA(
            self.spatial_decay[3] / s,
            self.spatial_first[3] / s,
            k[:, idx_anti],
            v[:, idx_anti],
        )
        results.append(r_a[:, inv_anti])

        # ── 加权融合 ──
        w_dir = torch.softmax(self.dir_weight, dim=0)
        out = sum(w_dir[i] * results[i] for i in range(self.num_scans))

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
                    n_embd=dim, hidden_rate=hidden_rate, drop_path=dp_rates[dp_idx + i]
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
                Block(n_embd=dim * 4, hidden_rate=hidden_rate)
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
                Block(n_embd=dim * 2, hidden_rate=hidden_rate)
                for _ in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(dim * 2)
        self.reduce_chan_level1 = nn.Sequential(
            nn.Conv2d(dim + dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(_gn_groups(dim), dim),
        )
        self.decoder_level1 = nn.Sequential(
            *[Block(n_embd=dim, hidden_rate=hidden_rate) for _ in range(num_blocks[0])]
        )

        # ── 后处理精修 ──
        self.refinement = nn.Sequential(
            *[
                Block(n_embd=dim, hidden_rate=hidden_rate)
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

    def prepare_for_inference(self):
        self.eval()
        for module in self.modules():
            if isinstance(module, OmniShift):
                module.reparam_5x5()
        print("✓ RefSRWKV: All OmniShift modules reparameterized for inference.")
        return self


# ═══════════════════════════════════════════════════════════════
# LitRefSRWKV
# ═══════════════════════════════════════════════════════════════
class LitRefSRWKV(pl.LightningModule):
    def __init__(
        self,
        model_sr,
        learning_rate=1e-4,
        warmup_steps=100,
        loss_fn=None,
        lr_key="lr",
        hr_key="hr",
        ref_key="ref",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_sr", "loss_fn"])
        self.model_sr = model_sr
        self.criterion = loss_fn or nn.L1Loss()
        self._step_count = 0
        self.lr_key = lr_key
        self.hr_key = hr_key
        self.ref_key = ref_key

    def _unpack_batch(self, batch):
        if isinstance(batch, dict):
            return batch[self.lr_key], batch[self.hr_key], batch[self.ref_key]
        return batch[0], batch[1], batch[2]

    def forward(self, lr, ref):
        return self.model_sr(lr, ref)

    def training_step(self, batch, batch_idx):
        lr, hr, ref = self._unpack_batch(batch)
        output = self(lr, ref)
        loss = self.criterion(output, hr)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lr, hr, ref = self._unpack_batch(batch)
        output = self(lr, ref)
        loss = self.criterion(output, hr)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        lr, hr, ref = self._unpack_batch(batch)
        output = self(lr, ref)
        loss = self.criterion(output, hr)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return output, hr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        if self._step_count < self.hparams.warmup_steps:
            lr_scale = min(1.0, (self._step_count + 1) / self.hparams.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = self.hparams.learning_rate * lr_scale
        if optimizer_closure is not None:
            optimizer.step(closure=optimizer_closure)
        else:
            optimizer.step()
        self._step_count += 1

    def on_train_start(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"✅ LitRefSRWKV 训练开始 | 参数量: {total / 1e6:.2f}M")


# ═══════════════════════════════════════════════════════════════
# Usage Example
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RefSRWKV(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=8,
        scale=10,
        drop_path_rate=0.1,
    ).to(device)

    lr = torch.randn(2, 3, 48, 48).to(device)
    ref = torch.randn(2, 3, 480, 480).to(device)

    model.train()
    out_train = model(lr, ref)
    print(
        f"Train output shape: {out_train.shape}, range: [{out_train.min():.3f}, {out_train.max():.3f}]"
    )

    model.prepare_for_inference()
    with torch.no_grad():
        out_infer = model(lr, ref)
    print(
        f"Infer output shape: {out_infer.shape}, range: [{out_infer.min():.3f}, {out_infer.max():.3f}]"
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total params: {total_params:.2f}M")
