"""
RefDiffRWKV.py — Ref 特征提取管线

用法 (Usage):
    model = RefDiffRWKV(patch_size=4, embed_dim=384)
    rf1, rf2, rf3 = model.extract_ref_features(
        x_t=noisy_image_pixel,   # VAE 解码后的 noisy latent（像素空间）
        LR=lr_image,             # 低分辨率输入
        Ref=ref_image            # 参考图像
    )
    # rf1: (B, 384, H/4, W/4)   → SD2 down_blocks[0]
    # rf2: (B, 768, H/8, W/8)   → SD2 down_blocks[1]
    # rf3: (B, 1536, H/16, W/16) → SD2 down_blocks[2]

Copyright (c) Shanghai AI Lab. All rights reserved.
RefDiffRWKV 原版版权归上海人工智能实验室所有。
本文件为精简改编版，仅保留 ref 特征提取管线。
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import sys
from pathlib import Path

# 添加项目根目录到 sys.path，确保能导入 RefSRWKV 中的 CUDA 算子
# Add project root to sys.path for importing CUDA ops from RefSRWKV
root_dir = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, root_dir)
from models.RefSRWKV import RUN_CUDA, OmniShift

# ======================================================================
#  1. 正弦余弦 2D 位置编码
#     Sinusoidal 2D Position Encoding
#     (支持任意分辨率，动态生成，无需插值)
# ======================================================================


def get_2d_sincos_pos_embed(embed_dim, h, w, cls_token=False, extra_tokens=0):
    """
    为任意 (h, w) 网格生成 2D 正弦余弦位置编码。
    Generate 2D sinusoidal position embeddings for arbitrary grid sizes.

    工作原理 (How it works):
        - 对 H 和 W 维度分别计算 1D 正弦余弦编码
        - 拼接得到 (H*W, embed_dim) 的位置编码矩阵
        - embed_dim 必须为偶数（H/W 各占一半维度）

    Args:
        embed_dim:   位置编码输出维度（必须为偶数）
        h, w:        特征图的高度和宽度（patch 数量）
        cls_token:   是否在开头预留 CLS token 位置
        extra_tokens:额外预留的 token 数量

    Returns:
        pos_embed: np.ndarray of shape (extra_tokens + h*w, embed_dim)
    """
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # 注意：先 w 后 h，产生 (2, h, w)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, h, w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    从 2D 网格坐标生成位置编码。
    embed_dim 的一半用于编码 H 坐标，另一半用于编码 W 坐标。
    """
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    1D 正弦余弦位置编码（标准 Transformer 风格）。
    Standard 1D sinusoidal position encoding.

    公式: PE(pos, 2i)   = sin(pos / 10000^(2i/d))
          PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Args:
        embed_dim: 输出维度（必须为偶数）
        pos:       位置坐标数组 shape (M,)

    Returns:
        emb: shape (M, embed_dim)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), 外积

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# ======================================================================
#  2. Patch Embedding — 图像分块嵌入
#     Image → Patch Tokens via strided convolution
# ======================================================================


class PatchEmbed(nn.Module):
    """
    将图像分割为不重叠的 patch，并通过步长卷积投影到嵌入空间。
    Image to Patch Embedding via strided convolution.

    输入: (B, in_chans, H, W)
    输出: (B, num_patches, embed_dim)  where num_patches = (H/patch_size) * (W/patch_size)

    Args:
        patch_size: patch 边长（卷积核大小 = 步长）
        in_chans:   输入图像通道数
        embed_dim:  输出嵌入维度
    """

    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        # 步长 = patch_size 的卷积等价于不重叠分块 + 线性投影
        # Conv with stride=patch_size is equivalent to non-overlapping patchify + linear proj
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H % self.patch_size == 0 and W % self.patch_size == 0
        ), f"Input size {H}x{W} must be divisible by patch_size {self.patch_size}"
        x = self.proj(x)  # (B, embed_dim, H/p, W/p)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


# ======================================================================
#  3. VRWKV 空间混合 & 通道混合
#     VRWKV Spatial Mix & Channel Mix (RWKV-7 style, bidirectional)
#     CrossFusion 依赖 VRWKV_SpatialMix 做双向 token 交互
# ======================================================================


class VRWKV_SpatialMix(nn.Module):
    """
    双向 WKV 空间混合模块（用于图像 token 序列）。
    Bidirectional WKV spatial mixing for image token sequences.

    核心改进 (vs. 原始 RWKV):
        - 用 OmniShift（多尺度深度卷积）替代 q_shift，提供局部邻域信息
        - 使用两次 WKV 扫描（正向 + 转置方向），实现近似全局双向交互
        - 移除与层数相关的初始化，参数更干净

    Args:
        n_embd: token 嵌入维度
    """

    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = n_embd
        self.recurrence = 2  # 双向扫描次数: 正向 + 转置方向

        # 局部空间增强（替代原版 q_shift）
        # Local enhancement via OmniShift (replaces q_shift)
        self.omni_shift = OmniShift(dim=n_embd)

        # 标准 RWKV 投影: Key / Value / Receptance / Output
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

        # 双向衰减 & 首位置偏置参数，形状 (2, n_embd)
        # Bidirectional decay and first-position bias, shape (2, n_embd)
        with torch.no_grad():
            self.spatial_decay = nn.Parameter(
                torch.randn(self.recurrence, n_embd) * 0.1
            )
            self.spatial_first = nn.Parameter(
                torch.randn(self.recurrence, n_embd) * 0.1
            )

    def jit_func(self, x: torch.Tensor, resolution: tuple):
        """
        生成 Key / Value / Receptance（门控信号 sr）。
        Generate K, V, and gate signal sr.

        Args:
            x:          (B, N, C) token 序列
            resolution: (H, W) 2D 网格尺寸

        Returns:
            sr: sigmoid(receptance)，门控信号 ∈ (0, 1)
            k:  Key 投影
            v:  Value 投影
        """
        h, w = resolution
        # 转为 2D → OmniShift 局部增强 → 展平回序列
        # Reshape to 2D → local enhancement → flatten back to sequence
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)  # 最终门控 ∈ (0, 1)
        return sr, k, v

    def forward(self, x: torch.Tensor, resolution: tuple):
        """
        双向 WKV 扫描。

        recurrence=2 的两次扫描:
            j=0: 标准正向扫描（行优先）
            j=1: 转置方向扫描（列优先 → WKV → 转回）
                 等效于在另一个方向做双向感受野
        """
        B, T, C = x.shape
        sr, k, v = self.jit_func(x, resolution)
        s = C**0.5  # sqrt(C) 缩放，稳定训练

        for j in range(self.recurrence):
            if j % 2 == 0:
                # 正向扫描: 行优先 → 每个 token 看到左侧上下文
                # Forward scan: row-major, each token attends to left context
                v = RUN_CUDA(
                    self.spatial_decay[j] / s,
                    self.spatial_first[j] / s,
                    k,
                    v,
                )
            else:
                # 转置扫描: 列优先 → 等效于从上到下看上下文
                # Transposed scan: col-major, equivalent to top-down context
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

        x = sr * v  # 门控
        x = self.output(x)  # 输出投影
        return x


class VRWKV_ChannelMix(nn.Module):
    """
    RWKV 通道混合模块（FFN），融入 OmniShift 局部增强。
    RWKV Channel Mixing (FFN) with OmniShift local enhancement.

    与 Diffusion-RWKV 原版的区别:
        - 移除 q_shift + spatial_mix_k/r 的插值混合
        - 改用 OmniShift 提供局部空间信息
        - 保留 RWKV 的平方激活门控: k = relu(k)^2

    注: 本模块在当前精简版中未被直接调用，保留作为备用的通道混合工具。
    Note: Not directly used in this stripped-down version; kept as a utility.

    Args:
        n_embd:      输入/输出维度
        hidden_rate: FFN 隐藏层倍率（默认 4x）
    """

    def __init__(self, n_embd: int, hidden_rate: int = 4):
        super().__init__()
        self.n_embd = n_embd
        hidden_sz = int(hidden_rate * n_embd)

        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)
        self.omni_shift = OmniShift(dim=n_embd)

    def forward(self, x: torch.Tensor, resolution: tuple):
        """
        Args:
            x:          (B, N, C) token 序列
            resolution: (H, W) 2D 网格尺寸
        Returns:
            (B, N, C) 通道混合后的 token
        """
        h, w = resolution
        # 2D → OmniShift → 展平
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        k = self.key(x)
        k = torch.square(torch.relu(k))  # 平方激活门控
        kv = self.value(k)
        r = torch.sigmoid(self.receptance(x))
        return r * kv


# ======================================================================
#  4. 上下采样模块
#     Downsample / Upsample (用于 RefMultiScaleProcessor 的多尺度构建)
# ======================================================================


class Downsample(nn.Module):
    """
    2× 下采样: Conv3×3 → PixelUnshuffle(2)
    通道数翻倍，空间分辨率减半。

    示例: (B, C, H, W) → (B, C*2, H/2, W/2)
    """

    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """
    2× 上采样: Conv3×3 → PixelShuffle(2)
    通道数减半，空间分辨率翻倍。

    示例: (B, C, H, W) → (B, C/2, H*2, W*2)

    注: 在当前精简版中保留作为备用工具。
    """

    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


# ======================================================================
#  5. LR 上采样器
#     LR Upsamplers — 将低分辨率输入升采样到目标分辨率
# ======================================================================


def lr_upsample_bilinear(lr: torch.Tensor, scale: int = 10):
    """
    双线性插值上采样（最简单、最快、无参数）。
    Bilinear upsampling — simplest, fastest, parameter-free.

    Args:
        lr:    (B, C, H,   W)   低分辨率图像
        scale: 上采样倍率（默认 10×，如 48→480）
    Returns:
        (B, C, H*scale, W*scale)
    """
    _, _, h, w = lr.shape
    return F.interpolate(
        lr, size=(h * scale, w * scale), mode="bilinear", align_corners=False
    )


class LRUpsamplerCNN(nn.Module):
    """
    CNN 上采样器: Conv → bilinear upsample → Conv。
    比纯 bilinear 多一些可学习参数，但保持轻量。
    """

    def __init__(self, in_ch=3, out_ch=3, scale_factor=10, hidden_ch=64):
        super().__init__()
        self.scale_factor = scale_factor
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(
                scale_factor=scale_factor, mode="bilinear", align_corners=False
            ),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.body(x)


class LRUpsamplerPixelShuffle(nn.Module):
    """
    PixelShuffle 上采样器: 两阶段 PixelShuffle（2× → 5× = 10×）。
    比 bilinear 更锐利，适合保留高频细节。
    """

    def __init__(self, in_ch=3, out_ch=3, hidden_ch=64):
        super().__init__()
        # 阶段1: 2× 上采样 (48 → 96)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )
        # 阶段2: 5× 上采样 (96 → 480)
        self.stage2 = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch * 25, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(5),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.stage1(x)  # (B, hidden_ch, 96, 96)
        x = self.stage2(x)  # (B, out_ch, 480, 480)
        return x


# ======================================================================
#  6. Ref 多尺度处理器
#     RefMultiScaleProcessor — 将 ref token 转为多尺度 2D 特征图
# ======================================================================


class RefMultiScaleProcessor(nn.Module):
    """
    将 CrossFusion 交互后的 Ref token 转换为多尺度 2D 特征图，
    分别对应 U-Net 编码器 e1, e2, e3 三个尺度。

    数据流 (Data Flow):
        ref_tokens (B, N, embed_dim)
            │
            ▼ transpose + reshape
        (B, embed_dim, H, W)
            │
            ├─► proj1 → ChannelMix(f1) ─────────────► f1  (B, d1, H,   W)
            │
            ├─► downsample → adapt1 → ChannelMix(f2) ► f2  (B, d2, H/2, W/2)
            │
            └─► downsample → adapt2 → ChannelMix(f3) ► f3  (B, d3, H/4, W/4)


    Args:
        embed_dim: Ref token 的嵌入维度
        dims:      (d1, d2, d3) 三个尺度的输出通道数
    """

    def __init__(self, embed_dim, dims):
        super().__init__()
        d1, d2, d3 = dims

        # ── 第 1 层：embed_dim → d1，保持分辨率 ──
        self.proj1 = nn.Conv2d(embed_dim, d1, 1)
        self.channel_mix1 = VRWKV_ChannelMix(d1)

        # ── 第 2 层：d1 → downsample → d1*2 → d2 ──
        self.down1 = Downsample(d1)  # d1 → d1*2, H→H/2
        self.adapt1 = nn.Conv2d(d1 * 2, d2, 1)  # d1*2 → d2
        self.channel_mix2 = VRWKV_ChannelMix(d2)

        # ── 第 3 层：d2 → downsample → d2*2 → d3 ──
        self.down2 = Downsample(d2)  # d2 → d2*2
        self.adapt2 = nn.Conv2d(d2 * 2, d3, 1)  # d2*2 → d3
        self.channel_mix3 = VRWKV_ChannelMix(d3)

    def forward(self, ref_tokens, H, W):
        """
        Args:
            ref_tokens: (B, N, embed_dim)
            H, W:       特征图的空间尺寸（patch 数量）

        Returns:
            f1: (B, d1, H,   W)
            f2: (B, d2, H/2, W/2)
            f3: (B, d3, H/4, W/4)
        """
        B, _, C = ref_tokens.shape
        x = ref_tokens.transpose(1, 2).reshape(B, C, H, W)

        # Level 1
        f1 = self.proj1(x)  # (B, d1, H, W)
        f1 = self._apply_channel_mix(f1, self.channel_mix1)

        # Level 2
        f2 = self.adapt1(self.down1(f1))  # (B, d2, H/2, W/2)
        f2 = self._apply_channel_mix(f2, self.channel_mix2)

        # Level 3
        f3 = self.adapt2(self.down2(f2))  # (B, d3, H/4, W/4)
        f3 = self._apply_channel_mix(f3, self.channel_mix3)

        return f1, f2, f3

    @staticmethod
    def _apply_channel_mix(x: torch.Tensor, mix: VRWKV_ChannelMix) -> torch.Tensor:
        """
        将 2D 特征图送入 VRWKV_ChannelMix 做通道间非线性交互。

        VRWKV_ChannelMix 的接口要求 (B, N, C) 的 token 序列，
        因此需要先 flatten 成序列 → 调用 → 再 reshape 回 2D。

        Args:
            x:   (B, C, H, W) 2D 特征图
            mix: VRWKV_ChannelMix 实例

        Returns:
            (B, C, H, W) 通道增强后的特征图
        """
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = mix(x, (H, W))  # 传入分辨率供 OmniShift 用
        x = x.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
        return x


# ======================================================================
#  7. 跨图融合模块
#     CrossFusion — main 与 ref token 的门控交互融合
# ======================================================================


class CrossFusion(nn.Module):
    """
    跨图融合模块: main 和 ref 各自经过双向 WKV 扫描后，
    通过门控机制融合: main_out = main + gate * fused。

    设计动机 (Motivation):
        - main token 携带当前 noisy image 的状态信息
        - ref  token 携带参考图像的纹理/结构信息
        - 门控融合允许模型学习「何时引入 ref 信息、引入多少」

    零初始化策略 (Zero-Init):
        fuse_proj.weight = 0, gate bias = -2 (sigmoid(-2) ≈ 0.12)
        → 训练初期 ref 贡献几乎为零，逐步释放。

    Args:
        embed_dim: token 嵌入维度
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.n_embd = embed_dim

        # main 和 ref 各自独立的双向 WKV 扫描
        # Independent bidirectional WKV scans for main and ref
        self.wkv_main = VRWKV_SpatialMix(embed_dim)
        self.wkv_ref = VRWKV_SpatialMix(embed_dim)

        # 融合层: 拼接 [main, ref] → 投影 → 门控 → 残差
        # Fusion: concat → project → gate → residual
        self.fuse_proj = nn.Linear(embed_dim * 2, embed_dim, bias=False)
        self.fuse_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.fuse_norm = nn.LayerNorm(embed_dim)

        # 零初始化: 训练初期 ref 贡献 ≈ 0
        nn.init.zeros_(self.fuse_proj.weight)
        nn.init.constant_(self.fuse_gate[0].bias, -2.0)  # sigmoid(-2) ≈ 0.12

    def forward(self, main_tokens, ref_tokens, resolution):
        """
        Args:
            main_tokens: (B, N, C) 主分支 token（noisy image + LR_up 的 patch embed）
            ref_tokens:  (B, N, C) 参考分支 token（Ref 的 patch embed）
            resolution:  (H, W)    2D 网格尺寸

        Returns:
            main_out: (B, N, C) 融合后的 main token（残差连接：main + gate * fused）
            ref_out:  (B, N, C) 处理后的 ref token（供后续 RefMultiScaleProcessor 提取多尺度特征）
        """
        # 各自独立做双向 WKV 扫描
        main_out = self.wkv_main(main_tokens, resolution)
        ref_out = self.wkv_ref(ref_tokens, resolution)

        # 门控融合: Concat → Project + Gate → Residual
        concat = torch.cat([main_out, ref_out], dim=-1)  # (B, N, 2C)
        fused = self.fuse_proj(concat)  # (B, N, C)
        gate = self.fuse_gate(concat)  # (B, N, C), ∈ (0, 1)
        main_out = main_tokens + gate * self.fuse_norm(fused)  # 残差连接

        return main_out, ref_out


# ======================================================================
#  8. RefDiffRWKV（精简版）— 纯 Ref 特征提取器
#     Stripped-down RefDiffRWKV: Ref Feature Extractor Only
# ======================================================================


class RefDiffRWKV(nn.Module):
    """
    RefDiffRWKV 精简版 — 仅保留 ref 特征提取管线。

    完整数据流 (Full Data Flow):
        ┌─────────────────────────────────────────────────────────┐
        │                                                         │
        │  x_t (noisy image, pixel)    LR (low-res)    Ref        │
        │       │                         │              │        │
        │       │                         ▼              │        │
        │       │                  lr_upsampler          │        │
        │       │                         │              │        │
        │       └───── concat ────────────┘              │        │
        │                     │                          │        │
        │                     ▼                          ▼        │
        │           patch_embed_main            patch_embed_ref   │
        │                     │                          │        │
        │                     ▼                          ▼        │
        │              main_tokens                ref_tokens      │
        │                     │ + pos_embed        │ + pos_embed  │
        │                     └──────┬─────────────┘              │
        │                            ▼                            │
        │                      CrossFusion                        │
        │                            │                            │
        │                   ┌────────┴────────┐                   │
        │                   ▼                 ▼                   │
        │             main_tokens'       ref_tokens'              │
        │                                       │                 │
        │                                       ▼                 │
        │                           RefMultiScaleProcessor        │
        │                                      │                  │
        │                          rf1, rf2, rf3                  │
        │                            │    │    │                  │
        │                            ▼    ▼    ▼                  │
        │                    SD2 down_blocks[0][1][2]             │
        └─────────────────────────────────────────────────────────┘

    Args:
        patch_size:       patch 边长（默认 4）
        embed_dim:        嵌入维度（默认 384）
        channels:         图像通道数（默认 3，RGB）
        upsample_mode:    LR 上采样模式: "bilinear" / "cnn" / "pixelshuffle"
        global_semantic:  可选的 GlobalSemanticModule（DINOv2 语义提取器）
    """

    def __init__(
        self,
        patch_size: int = 4,
        embed_dim: int = 384,
        channels: int = 3,
        upsample_mode: str = "bilinear",
        global_semantic: nn.Module = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.channels = channels
        self.global_semantic = global_semantic

        # ── LR 上采样器 ──
        # 根据 upsample_mode 选择不同的上采样策略
        if upsample_mode == "bilinear":
            self.lr_upsampler = lr_upsample_bilinear
        elif upsample_mode == "cnn":
            self.lr_upsampler = LRUpsamplerCNN(
                in_ch=channels, out_ch=channels, scale_factor=10, hidden_ch=64
            )
        elif upsample_mode == "pixelshuffle":
            self.lr_upsampler = LRUpsamplerPixelShuffle(
                in_ch=channels, out_ch=channels, hidden_ch=64
            )
        else:
            raise ValueError(f"Unsupported upsample_mode: {upsample_mode}")

        # ── Patch Embedding（双支路）──
        # main 支路: 输入 = concat(noisy_image, LR_up) → 6 通道
        # ref  支路: 输入 = Ref → 3 通道
        self.patch_embed_main = PatchEmbed(
            patch_size, in_chans=channels * 2, embed_dim=embed_dim
        )
        self.patch_embed_ref = PatchEmbed(
            patch_size, in_chans=channels, embed_dim=embed_dim
        )

        # ── 跨图融合 ──
        # main token 与 ref token 通过门控机制交互
        self.cross_fusion = CrossFusion(embed_dim)

        # ── Ref 多尺度处理器 ──
        # 将融合后的 ref token 转为 3 个尺度的 2D 特征图
        # dims = (embed_dim, embed_dim*2, embed_dim*4) = (384, 768, 1536)
        self.ref_ms_processor = RefMultiScaleProcessor(
            embed_dim=embed_dim,
            dims=(embed_dim, embed_dim * 2, embed_dim * 4),
        )

    # ==================================================================
    #  核心方法: extract_ref_features
    #  一站式提取 ref 多尺度特征，供外部 SD2 UNet 注入使用
    # ==================================================================

    def extract_ref_features(
        self,
        x_t: torch.Tensor,  # (B, 3, H, W) noisy image in pixel space（VAE 解码后）
        LR: torch.Tensor,  # (B, 3, h, w) 低分辨率输入图像
        Ref: torch.Tensor,  # (B, 3, H, W) 参考图像（与 x_t 同分辨率）
    ) -> tuple:
        """
        从 noisy image、LR、Ref 中提取多尺度 ref 特征。

        这是 RefDiffRWKV 精简版的核心对外接口。
        调用者（sd2_control_ldm.py）在训练/推理循环中调用此方法获取 rf1/rf2/rf3，
        再通过 RWKV_Ref_Adapter 注入 SD2 UNet 的 down_blocks。

        完整流程 (Complete Workflow):
            1. LR 上采样到与 x_t 相同的分辨率
            2. main 支路: concat(x_t, LR_up) → patch_embed_main → main_tokens
            3. ref  支路: Ref → patch_embed_ref → ref_tokens
            4. 动态位置编码（根据 patch 网格尺寸动态生成，无需插值）
            5. CrossFusion: main_tokens 与 ref_tokens 门控交互
            6. RefMultiScaleProcessor: ref_tokens → rf1, rf2, rf3（多尺度 2D 特征图）

        Args:
            x_t:  (B, 3, H, W) 当前 noisy image（像素空间，值域 [-1, 1]）
                  注意: 这应该是 SD2 VAE 对 noisy_latent 解码后的像素图！
                  Note: This must be the VAE-decoded pixel image from SD2's noisy_latent!
            LR:   (B, 3, h, w) 低分辨率输入（如 48×48，将被上采样到 H×W）
            Ref:  (B, 3, H, W) 参考图像（像素空间，应与 x_t 同分辨率）

        Returns:
            rf1: (B, embed_dim,     H/p,   W/p)    ← 高分辨率 ref 特征 (~384ch)
            rf2: (B, embed_dim*2,   H/2p,  W/2p)   ← 中分辨率 ref 特征 (~768ch)
            rf3: (B, embed_dim*4,   H/4p,  W/4p)   ← 低分辨率 ref 特征 (~1536ch)
            其中 p = patch_size

            如果 global_semantic 不为 None，额外返回:
            sem_pyramid: dict with keys "e1","e2","e3","latent"

        Raises:
            AssertionError: 如果 x_t 的 H/W 不能被 patch_size 整除

        示例 (Example):
            >>> model = RefDiffRWKV(patch_size=4, embed_dim=384)
            >>> # x_t 来自: vae.decode(noisy_latent).sample
            >>> rf1, rf2, rf3 = model.extract_ref_features(
            ...     x_t=noisy_image,   # (B, 3, 480, 480)
            ...     LR=lr_image,       # (B, 3, 48, 48)
            ...     Ref=ref_image      # (B, 3, 480, 480)
            ... )
            >>> rf1.shape  # (B, 384, 120, 120)   for 480/4=120
            >>> rf2.shape  # (B, 768, 60,  60)
            >>> rf3.shape  # (B, 1536, 30, 30)
        """
        B, _, H, W = x_t.shape

        # ── 检查分辨率合法性 ──
        assert (
            H % self.patch_size == 0 and W % self.patch_size == 0
        ), f"Input size {H}x{W} must be divisible by patch_size {self.patch_size}"

        patch_h = H // self.patch_size  # patch 网格高度
        patch_w = W // self.patch_size  # patch 网格宽度

        # ── Step 1: LR 上采样到当前分辨率 ──
        # 将低分辨率输入（如 48×48）升采样到与 x_t 相同（如 480×480）
        LR_up = self.lr_upsampler(LR)

        # ── Step 2: Patch Embedding（双支路）──
        # main 支路: concat(noisy_image, LR_up) → 6 通道 → patch tokens
        # 将 noisy image 和上采样后的 LR 在通道维拼接，提供"当前状态 + 退化信息"
        main_input = torch.cat([x_t, LR_up], dim=1)  # (B, 6, H, W)
        main_tokens = self.patch_embed_main(main_input)  # (B, N, embed_dim)

        # ref 支路: Ref → 3 通道 → patch tokens
        # 参考图像独立嵌入，保留其纹理和结构信息
        ref_tokens = self.patch_embed_ref(Ref)  # (B, N, embed_dim)

        # ── Step 3: 动态位置编码 ──
        # 根据实际 patch 网格尺寸动态生成，支持任意分辨率，无需插值
        # Dynamic position encoding: generated on-the-fly for any resolution
        pos_embed_np = get_2d_sincos_pos_embed(self.embed_dim, patch_h, patch_w)
        pos_embed = torch.from_numpy(pos_embed_np).float().to(x_t.device).unsqueeze(0)

        main_tokens = main_tokens + pos_embed
        ref_tokens = ref_tokens + pos_embed

        # ── Step 3.5: 全局语义提取（可选）──
        # 如果传入了 GlobalSemanticModule，在此提取 DINOv2 语义特征
        # 注意: 精简版中 sem tokens 不在内部使用（无 BiBlock），
        #       由调用者决定是否注入 SD2 UNet
        sem_pyramid = None
        if self.global_semantic is not None:
            # 仅在 Ref 非全零时提取（避免对空 Ref 做无意义计算）
            if Ref.abs().sum() > 1e-6:
                sem_pyramid = self.global_semantic(Ref)

        # ── Step 4: 跨图融合 ──
        # main token 与 ref token 通过双向 WKV + 门控机制交互
        # 输出: main_tokens'（融合后，当前未使用）、ref_tokens'（ref 信息增强后）
        main_tokens, ref_tokens = self.cross_fusion(
            main_tokens, ref_tokens, (patch_h, patch_w)
        )
        # 注: main_tokens 在当前架构中被丢弃，
        #     因为 SD2 UNet 不需要 main token（它有自己的 encoder）

        # ── Step 5: Ref 多尺度特征提取 ──
        # 将交互后的 ref token 转为 3 个尺度的 2D 特征图
        rf1, rf2, rf3 = self.ref_ms_processor(ref_tokens, patch_h, patch_w)

        # ── 返回 ──
        if sem_pyramid is not None:
            return rf1, rf2, rf3, sem_pyramid
        return rf1, rf2, rf3

    # ==================================================================
    #  兼容性方法: forward
    #  保留 forward 作为 extract_ref_features 的别名，
    #  方便在已有代码中直接替换原 RefDiffRWKV 调用
    # ==================================================================

    def forward(self, x_t, timesteps, LR, Ref):
        """
        兼容性 forward（忽略 timesteps 参数）。

        原 RefDiffRWKV.forward() 的签名: forward(x_t, timesteps, LR, Ref)
        精简版不需要 timesteps（仅 ref 提取，不做扩散去噪），
        但保留此参数以兼容已有调用代码。

        Args:
            x_t:        noisy image in pixel space
            timesteps:  (ignored) 扩散时间步，精简版不需要
            LR:         low-resolution input
            Ref:        reference image

        Returns:
            rf1, rf2, rf3
        """
        return self.extract_ref_features(x_t=x_t, LR=LR, Ref=Ref)

    @classmethod
    def from_args(cls, args):
        """
        从参数对象构建模型（兼容原版 API）。
        Build model from an argument object (backward-compatible).
        """
        return cls(
            patch_size=getattr(args, "patch_size", 4),
            embed_dim=getattr(args, "embed_dim", 384),
            channels=getattr(args, "channels", 3),
            upsample_mode=getattr(args, "upsample_mode", "bilinear"),
            global_semantic=getattr(args, "global_semantic", None),
        )

    def get_parameter_count(self) -> dict:
        """
        统计各子模块的可训练参数量。
        Count trainable parameters per sub-module.
        """
        counts = {}
        total = 0

        # 统计各子模块
        submodules = {
            "lr_upsampler": (
                self.lr_upsampler if isinstance(self.lr_upsampler, nn.Module) else None
            ),
            "patch_embed_main": self.patch_embed_main,
            "patch_embed_ref": self.patch_embed_ref,
            "cross_fusion": self.cross_fusion,
            "ref_ms_processor": self.ref_ms_processor,
        }

        for name, mod in submodules.items():
            if mod is None:
                counts[name] = 0
            else:
                n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
                counts[name] = n
                total += n

        counts["total"] = total
        return counts


# ======================================================================
#  9. 测试代码
#     Test / Sanity Check
# ======================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 构建精简版 RefDiffRWKV ──
    model = RefDiffRWKV(
        patch_size=4,
        embed_dim=384,
        channels=3,
        upsample_mode="bilinear",
    ).to(device)
    model.eval()

    # ── 打印参数量 ──
    param_counts = model.get_parameter_count()
    print("=" * 65)
    print("RefDiffRWKV (精简版) — Parameter Count")
    print("-" * 65)
    for k, v in param_counts.items():
        print(f"  {k:25s}: {v:>10,}")
    print("-" * 65)
    print(
        f"  {'Total':25s}: {param_counts['total']:>10,} "
        f"(~{param_counts['total']/1e6:.1f}M)"
    )
    print("=" * 65)

    # ── 模拟输入 ──
    # 假设: 原始图像 480×480, patch_size=4 → patch 网格 120×120
    #       VAE 8× 下采样后 latent 为 60×60
    #       此处 x_t 是 VAE 解码回像素空间的 noisy image
    B = 2
    H, W = 480, 480  # 像素空间分辨率
    LR_H, LR_W = 48, 48  # 低分辨率输入 (10× 上采样到 480)

    x_t = torch.randn(B, 3, H, W).to(device)  # noisy image (pixel space)
    LR = torch.randn(B, 3, LR_H, LR_W).to(device)  # low-res input
    Ref = torch.randn(B, 3, H, W).to(device)  # reference image

    print(f"Input shapes:")
    print(f"  x_t:  {x_t.shape}   (noisy image, pixel space)")
    print(f"  LR:   {LR.shape}    (low-res input)")
    print(f"  Ref:  {Ref.shape}   (reference image)")

    # ── 提取 ref 特征 ──
    with torch.no_grad():
        rf1, rf2, rf3 = model.extract_ref_features(x_t=x_t, LR=LR, Ref=Ref)

    print(f"Output shapes:")
    print(f"  rf1:  {rf1.shape}   ← SD2 down_blocks[0] (320ch via Adapter)")
    print(f"  rf2:  {rf2.shape}   ← SD2 down_blocks[1] (640ch via Adapter)")
    print(f"  rf3:  {rf3.shape}   ← SD2 down_blocks[2] (1280ch via Adapter)")

    # ── 验证分辨率正确性 ──
    patch_h, patch_w = H // 4, W // 4  # patch_size=4 → 120×120
    assert (
        rf1.shape[2] == patch_h and rf1.shape[3] == patch_w
    ), f"rf1 resolution mismatch: expected ({patch_h},{patch_w}), got ({rf1.shape[2]},{rf1.shape[3]})"
    assert (
        rf2.shape[2] == patch_h // 2 and rf2.shape[3] == patch_w // 2
    ), f"rf2 resolution mismatch"
    assert (
        rf3.shape[2] == patch_h // 4 and rf3.shape[3] == patch_w // 4
    ), f"rf3 resolution mismatch"

    print(f"✓ All resolution checks passed.")

    # ── 验证通道数 ──
    assert rf1.shape[1] == 384, f"rf1 channels: expected 384, got {rf1.shape[1]}"
    assert rf2.shape[1] == 768, f"rf2 channels: expected 768, got {rf2.shape[1]}"
    assert rf3.shape[1] == 1536, f"rf3 channels: expected 1536, got {rf3.shape[1]}"

    print(f"✓ All channel checks passed.")

    # ── 验证兼容性 forward ──
    rf1_fwd, rf2_fwd, rf3_fwd = model(x_t=x_t, timesteps=None, LR=LR, Ref=Ref)
    assert torch.equal(
        rf1, rf1_fwd
    ), "forward() and extract_ref_features() mismatch for rf1"
    print(f"✓ Backward-compatible forward() verified.")

    print(f"{'='*65}")
    print(f"All tests passed! RefDiffRWKV (精简版) is ready.")
    print(f"{'='*65}")
