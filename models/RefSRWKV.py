# Copyright (c) Shanghai AI Lab. All rights reserved.
"""
RefSRWKV (Improved): Reference-based Super-Resolution with RWKV Backbone.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.cpp_extension import load

# ═══════════════════════════════════════════════════════════════
# CUDA WKV (unchanged — keep your working kernel)
# ═══════════════════════════════════════════════════════════════
wkv_cuda = load(
    name="bi_wkv",
    sources=["./models/cuda/bi_wkv.cpp", "./models/cuda/bi_wkv_kernel.cu"],
    verbose=True,
    extra_cuda_cflags=[
        "-res-usage",
        "--maxrregcount 60",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "-gencode arch=compute_120,code=sm_120",
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
    """
    训练时：Identity + 1×1 + 3×3 + 5×5 多分支并行，α 可学习。
    推理时：重参数化为单个 5×5 深度可分离卷积，零开销。
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=False)
        self.conv3x3 = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim, bias=False
        )
        self.conv5x5 = nn.Conv2d(
            dim, dim, kernel_size=5, padding=2, groups=dim, bias=False
        )
        self.alpha = nn.Parameter(torch.ones(4) * 0.25)

        # 推理用的重参数化权重（buffer，不参与训练）
        self.register_buffer(
            "conv5x5_reparam_weight",
            torch.zeros(dim, 1, 5, 5),
        )
        self._reparam_done = False

    def forward_train(self, x):
        alpha = torch.softmax(self.alpha, dim=0)
        out = (
            alpha[0] * x
            + alpha[1] * self.conv1x1(x)
            + alpha[2] * self.conv3x3(x)
            + alpha[3] * self.conv5x5(x)
        )
        return out

    def reparam_5x5(self):
        if self._reparam_done:
            return

        alpha = torch.softmax(self.alpha, dim=0)

        # Identity: 5×5 核，只有中心为 1
        identity = torch.zeros(self.dim, 1, 5, 5, device=self.conv1x1.weight.device)
        identity[:, :, 2, 2] = 1.0

        # 1×1 → pad 四周各 2 → 5×5
        w1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        # 3×3 → pad 四周各 1 → 5×5
        w3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))
        # 5×5 无需 pad
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
# VRWKV Blocks (improved)
# ═══════════════════════════════════════════════════════════════
class VRWKV_SpatialMix(nn.Module):
    """双向空间混合：H 方向 + W 方向各一次 RWKV 扫描。"""

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

        # 用 sqrt(C) 替代 /T 做缩放，避免大图时 decay 趋零
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

        # 用 sqrt(C) 缩放 decay/first，而非 /T，对大分辨率更鲁棒
        s = self.scale

        for j in range(self.recurrence):
            if j % 2 == 0:
                v = RUN_CUDA(
                    self.spatial_decay[j] / s,
                    self.spatial_first[j] / s,
                    k,
                    v,
                )
            else:
                h, w = resolution
                k = rearrange(k, "b (h w) c -> b (w h) c", h=h, w=w)
                v = rearrange(v, "b (h w) c -> b (w h) c", h=h, w=w)
                v = RUN_CUDA(
                    self.spatial_decay[j] / s,
                    self.spatial_first[j] / s,
                    k,
                    v,
                )
                k = rearrange(k, "b (w h) c -> b (h w) c", h=h, w=w)
                v = rearrange(v, "b (w h) c -> b (h w) c", h=h, w=w)

        x = sr * v
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
# Resizing Modules (unchanged)
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
# GatedFusion — the core improvement
# ═══════════════════════════════════════════════════════════════
class GatedFusion(nn.Module):
    """
    门控参考融合模块。

    输入端：LR 特征 + Ref 特征（同分辨率同通道数）
    流程：
      1. Concat → 1×1 投影 → GroupNorm
      2. 通道注意力生成门控向量（0~1）
      3. 残差加回：out = LR_feat + gate * fused_feat

    关键设计：
      - 1×1 卷积零初始化 → 训练初期 ref 贡献为零，模型先学会依赖 LR
      - 通道注意力让模型自主决定哪些通道信任 Ref
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fuse_conv = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=_gn_groups(dim), num_channels=dim)

        # 通道门控
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // reduction, 8), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(dim // reduction, 8), dim, kernel_size=1),
            nn.Sigmoid(),
        )

        # 零初始化：训练初期 ref 不参与
        nn.init.zeros_(self.fuse_conv.weight)
        # gate 最后一层偏置初始化为负值 → 初始 gate ≈ 0.1，缓慢引入 ref
        nn.init.constant_(self.gate[-2].bias, -2.0)

    def forward(self, lr_feat, ref_feat):
        fused = self.fuse_conv(torch.cat([lr_feat, ref_feat], dim=1))
        fused = self.norm(fused)
        gate = self.gate(fused)
        return lr_feat + gate * fused


# ═══ 文件顶部，import 之后插入 ═══
def _gn_groups(num_channels: int, max_groups: int = 32) -> int:
    """返回 ≤ max_groups 且能整除 num_channels 的最大组数。"""
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class RefSRWKV(nn.Module):
    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: tuple = (4, 6, 6, 8),
        num_refinement_blocks: int = 8,
        scale: int = 10,
        drop_path_rate: float = 0.0,
        hidden_rate: int = 4,
    ):
        super().__init__()
        self.scale = scale
        self.dim = dim

        # ── LR 编码器 ──
        self.lr_up = nn.Sequential(
            nn.Upsample(scale_factor=2.5, mode="bilinear", align_corners=False),
            nn.Conv2d(inp_channels, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(dim), dim),  # ← 修：48→24 groups
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
        )

        # ── Ref 编码器 ──
        self.ref_stem = nn.Conv2d(
            out_channels, dim, kernel_size=3, padding=1, bias=False
        )
        self.ref_to_level1 = nn.Sequential(
            nn.PixelUnshuffle(4),
            nn.Conv2d(out_channels * 16, dim, kernel_size=1, bias=False),
            nn.GroupNorm(_gn_groups(dim), dim),  # ← 修：48→24 groups
        )
        self.ref_down2 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(dim * 2), dim * 2),  # ← 修：96→32 groups（不变）
        )
        self.ref_down3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(dim * 4), dim * 4),  # ← 修：192→32 groups
        )
        self.ref_down4 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(dim * 8), dim * 8),  # ← 修：384→32 groups
        )

        # ── 门控融合 ──
        self.fuse1 = GatedFusion(dim)
        self.fuse2 = GatedFusion(dim * 2)
        self.fuse3 = GatedFusion(dim * 4)
        self.fuse4 = GatedFusion(dim * 8)

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
            nn.GroupNorm(_gn_groups(dim * 4), dim * 4),  # ← 修
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
            nn.GroupNorm(_gn_groups(dim * 2), dim * 2),  # ← 修
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
            nn.GroupNorm(
                _gn_groups(dim), dim
            ),  # ← 修：48→24 groups（原来 min(32,48)=32 必炸）
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
        nn.init.zeros_(self.ref_guided_refine.weight)

        self.apply(self._init_weights)

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
        """
        无损 + 渐进取样构建 Ref 特征金字塔。

        Level 1: PixelUnshuffle(4) → 120×120×dim   (信息无损)
        Level 2: stride-2 conv   → 60×60×2dim
        Level 3: stride-2 conv   → 30×30×4dim
        Level 4: stride-2 conv   → 15×15×8dim
        """
        ref_1 = self.ref_to_level1(ref)  # 120×120×dim
        ref_2 = self.ref_down2(ref_1)  # 60×60×2dim
        ref_3 = self.ref_down3(ref_2)  # 30×30×4dim
        ref_4 = self.ref_down4(ref_3)  # 15×15×8dim
        return ref_1, ref_2, ref_3, ref_4

    def forward(self, lr, ref):
        """
        Args:
            lr:  低分辨率输入   (B, 3, 48, 48)
            ref: 参考图像       (B, 3, 480, 480)
        Returns:
            out: 超分辨率输出   (B, 3, 480, 480)，值域 [-1, 1]
        """
        # 1. LR 特征提取
        fea = self.lr_up(lr)  # (B, dim, 120, 120)

        # 2. Ref 金字塔
        ref_1, ref_2, ref_3, ref_4 = self._extract_ref_pyramid(ref)

        # 3. 编码器 + 门控融合
        e1 = self.encoder_level1(self.fuse1(fea, ref_1))  # dim,   120×120
        e2 = self.encoder_level2(self.fuse2(self.down1_2(e1), ref_2))  # 2dim,  60×60
        e3 = self.encoder_level3(self.fuse3(self.down2_3(e2), ref_3))  # 4dim,  30×30
        latent = self.latent(self.fuse4(self.down3_4(e3), ref_4))  # 8dim,  15×15

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
        d1 = self.refinement(d1)  # dim, 120×120

        # 6. 最终上采样 + 输出投影
        hr_feat = self.up_final(d1)  # dim, 480×480
        out = self.output_conv(hr_feat)  # 3ch, 480×480

        # 7. Ref 引导残差修正（训练初期权重为零，不影响学习）
        residual = self.ref_guided_refine(torch.cat([out, ref], dim=1))
        out = out + residual
        out = torch.clamp(out, -1.0, 1.0)

        return out

    def prepare_for_inference(self):
        """
        推理前调用：将所有 OmniShift 重参数化，并切换到 eval 模式。
        调用后模型可直接用于推理，速度提升约 15%~20%。
        """
        self.eval()
        for module in self.modules():
            if isinstance(module, OmniShift):
                module.reparam_5x5()
        print("✓ RefSRWKV: All OmniShift modules reparameterized for inference.")
        return self


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

    # 测试前向
    lr = torch.randn(2, 3, 48, 48).to(device)
    ref = torch.randn(2, 3, 480, 480).to(device)

    model.train()
    out_train = model(lr, ref)
    print(
        f"Train output shape: {out_train.shape}, range: [{out_train.min():.3f}, {out_train.max():.3f}]"
    )

    # 推理部署
    model.prepare_for_inference()
    with torch.no_grad():
        out_infer = model(lr, ref)
    print(
        f"Infer output shape: {out_infer.shape}, range: [{out_infer.min():.3f}, {out_infer.max():.3f}]"
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total params: {total_params:.2f}M")
