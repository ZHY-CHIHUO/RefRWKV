# Copyright (c) Shanghai AI Lab. All rights reserved.
import math, os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import sys
from pathlib import Path
from torch.utils.checkpoint import checkpoint

# 添加项目根目录到 sys.path
root_dir = str(Path(__file__).parent.parent)  # 从 models/ 向上到 RefRWKV/
sys.path.insert(0, root_dir)
from models.RefSRWKV import RUN_CUDA, OmniShift


##########################################################################
## 正弦余弦 2D 位置编码
def get_2d_sincos_pos_embed(embed_dim, h, w, cls_token=False, extra_tokens=0):
    """支持任意高度和宽度的2D位置编码"""
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, h, w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


##########################################################################
## 时间步嵌入器
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


##########################################################################
## 图像patch分块与重组
def patchify(imgs, patch_size):
    x = rearrange(
        imgs, "B C (h p1) (w p2) -> B (h w) (p1 p2 C)", p1=patch_size, p2=patch_size
    )
    return x


def unpatchify(x, channels=3, h=None, w=None):
    """支持动态 h,w"""
    if h is None or w is None:
        patch_size = int((x.shape[2] // channels) ** 0.5)
        h = w = int(x.shape[1] ** 0.5)
    else:
        patch_size = int((x.shape[2] // channels) ** 0.5)

    x = rearrange(
        x,
        "B (h w) (p1 p2 C) -> B C (h p1) (w p2)",
        h=h,
        w=w,
        p1=patch_size,
        p2=patch_size,
    )
    return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


##########################################################################
## 随机深度（Stochastic Depth / DropPath）正则化
def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class VRWKV_SpatialMix(nn.Module):
    """
    双向 WKV 空间混合模块（用于图像 token 序列）
    - 用 OmniShift 提供多尺度局部邻域信息
    - 使用两次 WKV 扫描（正向 + 转置方向）实现近似全局双向交互
    - 完全移除原有的 q_shift、spatial_mix_k/v/r 及与层数相关的初始化
    """

    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = n_embd
        self.recurrence = 2  # 双向扫描次数

        # 局部增强（替代 q_shift）
        self.omni_shift = OmniShift(dim=n_embd)

        # 标准 RWKV 投影
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

        # 双向的衰减和补偿系数，形状 (2, n_embd)
        # 简单随机初始化，也可根据任务进一步调节
        with torch.no_grad():
            self.spatial_decay = nn.Parameter(
                torch.randn(self.recurrence, n_embd) * 0.1
            )
            self.spatial_first = nn.Parameter(
                torch.randn(self.recurrence, n_embd) * 0.1
            )

    def jit_func(self, x: torch.Tensor, resolution: tuple):
        """
        生成 k, v, r 和门控信号 sr
        x: (B, N, C)  token 序列
        resolution: (H, W)  2D 网格尺寸
        """
        h, w = resolution

        # 转为 2D 特征图，应用 OmniShift 局部增强，再展平回序列
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)  # 最终门控

        return sr, k, v

    def forward(self, x: torch.Tensor, resolution: tuple):
        B, T, C = x.shape
        sr, k, v = self.jit_func(x, resolution)

        s = C**0.5  # 用 sqrt(C) 缩放，替代 /T

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
    """
    RWKV 通道混合模块（FFN），融入 OmniShift 局部增强。

    与 Diffusion-RWKV 原版的区别：
    - 移除了 q_shift + spatial_mix_k/r 的插值混合。
    - 改用 OmniShift（可重参数化多尺度深度卷积）提供局部空间信息。
    - 不再依赖 n_layer / layer_id 进行初始化，参数更干净。
    - 保留了 RWKV 的平方激活门控：k = relu(k)^2。
    """

    def __init__(self, n_embd: int, hidden_rate: int = 4):
        super().__init__()
        self.n_embd = n_embd
        hidden_sz = int(hidden_rate * n_embd)

        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        # 局部空间增强（替代 q_shift）
        self.omni_shift = OmniShift(dim=n_embd)

    def forward(self, x: torch.Tensor, resolution: tuple):
        """
        x: (B, N, C)  token 序列
        resolution: (H, W)  二维网格的高和宽
        """
        h, w = resolution

        # 转为 2D 特征图 → 应用 OmniShift → 展平回序列
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        # 标准 RWKV 通道混合计算
        k = self.key(x)
        k = torch.square(torch.relu(k))  # 平方激活
        kv = self.value(k)  # 值投影
        r = torch.sigmoid(self.receptance(x))  # 门控信号

        return r * kv


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BiBlock(nn.Module):
    """
    用于 Bi-DiffRWKV 的基础构建块（支持扩散条件注入 + 语义 Cross-Attention）。
    """

    def __init__(
        self,
        n_embd: int,
        hidden_rate: int = 4,
        drop_path: float = 0.0,
        use_adaLN: bool = True,
        use_cross_attn: bool = True,  # 是否启用语义 Cross-Attention
        num_heads: int = 8,  # 交叉注意力头数
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(n_embd)
        self.ffn = VRWKV_ChannelMix(n_embd, hidden_rate=hidden_rate)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # 条件调制层（adaLN）
        self.use_adaLN = use_adaLN
        if use_adaLN:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(n_embd, 6 * n_embd, bias=True),
            )
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            self.gamma1 = nn.Parameter(torch.ones(n_embd))
            self.gamma2 = nn.Parameter(torch.ones(n_embd))

        # ========== 新增：语义 Cross-Attention ==========
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=n_embd,
                num_heads=num_heads,
                batch_first=True,
            )
            self.cross_norm = nn.LayerNorm(n_embd)
            # 可学习的缩放因子（初始化为 1）
            self.cross_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor = None,
        sem_tokens: torch.Tensor = None,  # 新增：语义 Token
    ):
        B, C, H, W = x.shape
        resolution = (H, W)

        # ---------- 生成调制系数 ----------
        if self.use_adaLN:
            assert c is not None, "BiBlock with adaLN requires condition c"
            shift_att, scale_att, gate_att, shift_ffn, scale_ffn, gate_ffn = (
                self.adaLN_modulation(c).chunk(6, dim=1)
            )
        else:
            gate_att = self.gamma1
            gate_ffn = self.gamma2
            shift_att = scale_att = shift_ffn = scale_ffn = 0.0

        # ---------- 注意力分支 ----------
        # 1. 转为序列并归一化 + 调制
        x_flat = rearrange(x, "b c h w -> b (h w) c")
        x_flat = modulate(self.ln1(x_flat), shift_att, scale_att)

        # ========== 关键修改：在 WKV 之前加入语义 Cross-Attention ==========
        if self.use_cross_attn and sem_tokens is not None:
            # 确保 sem_tokens 是 (B, N, C) 格式
            if sem_tokens.dim() == 4:
                sem_tokens = sem_tokens.flatten(2).transpose(1, 2)
            x_flat_norm = self.cross_norm(x_flat)
            attn_out, _ = self.cross_attn(
                x_flat_norm,
                sem_tokens,
                sem_tokens,
            )
            x_flat = x_flat + self.cross_scale * attn_out

        # ================================================================

        # 2. 双向 WKV 空间混合（语义信息已注入）
        x_flat = self.att(x_flat, resolution)

        # 3. 门控 + 残差
        x_flat = gate_att.unsqueeze(1) * x_flat
        x_flat = rearrange(x_flat, "b (h w) c -> b c h w", h=H, w=W)
        x = x + self.drop_path(x_flat)

        # ---------- FFN 分支（保持不变） ----------
        x_flat = rearrange(x, "b c h w -> b (h w) c")
        x_flat = modulate(self.ln2(x_flat), shift_ffn, scale_ffn)
        x_flat = self.ffn(x_flat, resolution)
        x_flat = gate_ffn.unsqueeze(1) * x_flat
        x_flat = rearrange(x_flat, "b (h w) c -> b c h w", h=H, w=W)
        x = x + self.drop_path(x_flat)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


##########################################################################
## 输出头
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels, condition=True):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )

        if condition == True:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )

    def forward(self, x, c=None):
        if c is not None:
            c = self.adaLN_modulation(c)
            shift, scale = c.chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
            x = self.linear(x)
        else:
            x = self.norm_final(x)
            x = self.linear(x)
        return x


##########################################################################
## lr上采样
def lr_upsample_bilinear(lr: torch.Tensor, scale: int = 10):
    _, _, h, w = lr.shape
    return F.interpolate(
        lr, size=(h * scale, w * scale), mode="bilinear", align_corners=False
    )


class LRUpsamplerCNN(nn.Module):
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
    def __init__(self, in_ch=3, out_ch=3, hidden_ch=64):
        super().__init__()
        # 阶段1: 2x 上采样
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),  # 输出 hidden_ch, 96×96
            nn.ReLU(inplace=True),
        )
        # 阶段2: 5x 上采样 (96 → 480)
        self.stage2 = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch * 25, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(5),  # 输出 hidden_ch, 480×480
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.stage1(x)  # → (B, 64, 96, 96)
        x = self.stage2(x)  # → (B, 3, 480, 480)
        return x


class RefMultiScaleProcessor(nn.Module):
    """
    将交互后的 Ref tokens 转换为多尺度 2D 特征图，
    分别对应 U-Net 编码器输出的 e1, e2, e3 尺度。

    Downsample 输出通道 = 输入通道 × 2，
    因此需要显式 adapt 层对齐到目标通道数。
    """

    def __init__(self, embed_dim, dims):
        super().__init__()
        d1, d2, d3 = dims

        # Level 1: embed_dim → d1
        self.proj1 = nn.Conv2d(embed_dim, d1, 1)

        # Level 2: d1 → down(d1) → d1*2 → adapt → d2
        self.down1 = Downsample(d1)  # d1 → d1*2
        self.adapt1 = nn.Conv2d(d1 * 2, d2, 1)  # d1*2 → d2

        # Level 3: d2 → down(d2) → d2*2 → adapt → d3
        self.down2 = Downsample(d2)  # d2 → d2*2
        self.adapt2 = nn.Conv2d(d2 * 2, d3, 1)  # d2*2 → d3

    def forward(self, ref_tokens, H, W):
        B, _, C = ref_tokens.shape
        x = ref_tokens.transpose(1, 2).reshape(B, C, H, W)

        f1 = self.proj1(x)  # (B, d1, H,   W)
        f2 = self.adapt1(self.down1(f1))  # (B, d2, H/2, W/2)
        f3 = self.adapt2(self.down2(f2))  # (B, d3, H/4, W/4)

        return f1, f2, f3


class CrossFusion(nn.Module):
    """
    跨图融合模块（替代原 cross_fusion 方法）。

    main 和 ref 各自独立做双向 WKV 扫描，保留空间结构。
    然后通过门控机制融合：out = main + gate * fused
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.n_embd = embed_dim

        # main 和 ref 各自独立的双向 WKV
        self.wkv_main = VRWKV_SpatialMix(embed_dim)
        self.wkv_ref = VRWKV_SpatialMix(embed_dim)

        # 融合层：拼接 → 投影 → 门控
        self.fuse_proj = nn.Linear(embed_dim * 2, embed_dim, bias=False)
        self.fuse_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.fuse_norm = nn.LayerNorm(embed_dim)

        # 零初始化 → 训练初期 ref 贡献为零
        nn.init.zeros_(self.fuse_proj.weight)
        nn.init.constant_(self.fuse_gate[0].bias, -2.0)  # sigmoid(-2) ≈ 0.12

    def forward(self, main_tokens, ref_tokens, resolution):
        """
        Args:
            main_tokens: (B, N, C)
            ref_tokens:  (B, N, C)
            resolution:  (H, W)
        Returns:
            main_out: (B, N, C)  融合后的 main
            ref_out:  (B, N, C)  处理后的 ref（供后续 RefMultiScaleProcessor）
        """
        # 各自独立做双向 WKV 扫描
        main_out = self.wkv_main(main_tokens, resolution)
        ref_out = self.wkv_ref(ref_tokens, resolution)

        # 门控融合：拼接 + 投影 + 门控 + 残差
        concat = torch.cat([main_out, ref_out], dim=-1)  # (B, N, 2C)
        fused = self.fuse_proj(concat)  # (B, N, C)
        gate = self.fuse_gate(concat)  # (B, N, C)
        main_out = main_tokens + gate * self.fuse_norm(fused)  # 残差连接

        return main_out, ref_out


##########################################################################
## RefDiffRWKV
class RefDiffRWKV(nn.Module):
    """
    RefDiffRWKV: 支持不同分辨率图片的 Reference-guided Diffusion 模型
    - 同一个模型可处理多种分辨率（256, 384, 480, 512 等）
    - 动态位置编码 + 动态 LR 上采样
    """

    def __init__(
        self,
        patch_size: int = 4,
        embed_dim: int = 384,
        channels: int = 3,
        enc_blocks=(6, 6, 6),
        dec_blocks=(6, 6, 6),
        latent_blocks=6,
        drop_path_rate: float = 0.1,
        hidden_rate: int = 4,
        learn_sigma: bool = False,
        upsample_mode: str = "bilinear",  # 可选 "bilinear", "cnn", "pixelshuffle"
        global_semantic: nn.Module = None,
        use_checkpoint: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.channels = channels
        self.global_semantic = global_semantic
        self.use_checkpoint = use_checkpoint

        # ---------- LR 上采样器 ----------
        if upsample_mode == "bilinear":
            self.lr_upsampler = lr_upsample_bilinear
        elif upsample_mode == "cnn":
            self.lr_upsampler = LRUpsamplerCNN(
                in_ch=channels,
                out_ch=channels,
                scale_factor=10,
                hidden_ch=64,
            )
        elif upsample_mode == "pixelshuffle":
            self.lr_upsampler = LRUpsamplerPixelShuffle(
                in_ch=channels, out_ch=channels, hidden_ch=64
            )
        else:
            raise ValueError(f"Unsupported upsample_mode: {upsample_mode}")

        # Patch Embed（双支路）
        self.patch_embed_main = PatchEmbed(
            patch_size, in_chans=channels * 2, embed_dim=embed_dim
        )
        self.patch_embed_ref = PatchEmbed(
            patch_size, in_chans=channels, embed_dim=embed_dim
        )

        # 跨图融合（传入当前 patch 分辨率）
        self.cross_fusion = CrossFusion(embed_dim)

        # 时间嵌入
        self.time_embed = TimestepEmbedder(embed_dim)

        # U-Net 结构
        dim = embed_dim

        self.proj_c_enc1 = nn.Linear(embed_dim, dim)
        self.proj_c_enc2 = nn.Linear(embed_dim, dim * 2)
        self.proj_c_enc3 = nn.Linear(embed_dim, dim * 4)
        self.proj_c_latent = nn.Linear(embed_dim, dim * 8)
        self.proj_c_dec3 = nn.Linear(embed_dim, dim * 4)
        self.proj_c_dec2 = nn.Linear(embed_dim, dim * 2)
        self.proj_c_dec1 = nn.Linear(embed_dim, dim)

        self.down1 = Downsample(dim)
        self.down2 = Downsample(dim * 2)
        self.down3 = Downsample(dim * 4)
        self.up3 = Upsample(dim * 8)
        self.up2 = Upsample(dim * 4)
        self.up1 = Upsample(dim * 2)

        # 随机深度
        total_blocks = sum(enc_blocks) + latent_blocks + sum(dec_blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks + 1)]
        dp_idx = [0]

        def next_dp():
            val = dpr[dp_idx[0]]
            dp_idx[0] += 1
            return val

        self.enc1 = nn.ModuleList(
            [
                BiBlock(dim, hidden_rate, drop_path=next_dp())
                for _ in range(enc_blocks[0])
            ]
        )
        self.enc2 = nn.ModuleList(
            [
                BiBlock(dim * 2, hidden_rate, drop_path=next_dp())
                for _ in range(enc_blocks[1])
            ]
        )
        self.enc3 = nn.ModuleList(
            [
                BiBlock(dim * 4, hidden_rate, drop_path=next_dp())
                for _ in range(enc_blocks[2])
            ]
        )
        self.latent = nn.ModuleList(
            [
                BiBlock(dim * 8, hidden_rate, drop_path=next_dp())
                for _ in range(latent_blocks)
            ]
        )

        self.dec3 = nn.ModuleList(
            [
                BiBlock(dim * 4, hidden_rate, drop_path=next_dp())
                for _ in range(dec_blocks[0])
            ]
        )
        self.dec2 = nn.ModuleList(
            [
                BiBlock(dim * 2, hidden_rate, drop_path=next_dp())
                for _ in range(dec_blocks[1])
            ]
        )
        self.dec1 = nn.ModuleList(
            [
                BiBlock(dim, hidden_rate, drop_path=next_dp())
                for _ in range(dec_blocks[2])
            ]
        )

        # 输入：main 特征 + 对应尺度 Ref 特征（通道数均为 dim * mult）
        # 输出：融合后通道数减半（还原为该尺度标准通道数）
        self.fuse_e1 = nn.Conv2d(dim * 2, dim, 1)  # cat(e1, rf1) -> dim
        self.fuse_e2 = nn.Conv2d(dim * 4, dim * 2, 1)  # cat(e2, rf2) -> dim*2
        self.fuse_e3 = nn.Conv2d(dim * 8, dim * 4, 1)  # cat(e3, rf3) -> dim*4

        # 跳跃连接通道调整
        self.reduce_chan3 = nn.Conv2d(dim * 8, dim * 4, 1)
        self.reduce_chan2 = nn.Conv2d(dim * 4, dim * 2, 1)
        self.reduce_chan1 = nn.Conv2d(dim * 2, dim, 1)

        # 输出头
        self.out_channels = channels * 2 if learn_sigma else channels
        self.final_layer = FinalLayer(
            embed_dim, patch_size, self.out_channels, condition=True
        )

        # ---------- Ref 多尺度处理器（处理交互后的 Ref tokens）----------
        self.ref_ms_processor = RefMultiScaleProcessor(
            embed_dim=embed_dim,
            dims=(dim, dim * 2, dim * 4),  # 对应 e1, e2, e3 的通道数
        )

        self.initialize_weights()

    def initialize_weights(self):
        # 基础初始化
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # PatchEmbed 初始化
        for pe in [self.patch_embed_main, self.patch_embed_ref]:
            w = pe.proj.weight.data
            nn.init.xavier_uniform_(w.view(w.shape[0], -1))
            if pe.proj.bias is not None:
                nn.init.constant_(pe.proj.bias, 0)

        # 时间嵌入
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[3].weight, std=0.02)

        # adaLN + FinalLayer 零初始化（独立步骤，不依赖全局遍历）
        self._zero_init_adaLN()

    def _zero_init_adaLN(self):
        """所有 adaLN 调制层和输出层零初始化。"""
        for module in self.modules():
            if isinstance(module, BiBlock) and module.use_adaLN:
                nn.init.constant_(module.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(module.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x_t, timesteps, LR, Ref):
        B, _, H, W = x_t.shape

        # 检查分辨率合法性
        assert (
            H % self.patch_size == 0 and W % self.patch_size == 0
        ), f"Input size {H}x{W} must be divisible by patch_size {self.patch_size}"

        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        # 1. LR 上采样到当前分辨率
        LR_up = self.lr_upsampler(LR)

        # 2. Patch Embedding
        main_input = torch.cat([x_t, LR_up], dim=1)
        main_tokens = self.patch_embed_main(main_input)
        ref_tokens = self.patch_embed_ref(Ref)

        # 3. 动态位置编码（核心修改）
        pos_embed_np = get_2d_sincos_pos_embed(self.embed_dim, patch_h, patch_w)
        pos_embed = torch.from_numpy(pos_embed_np).float().to(x_t.device).unsqueeze(0)

        main_tokens = main_tokens + pos_embed
        ref_tokens = ref_tokens + pos_embed

        # 4. 时间条件
        c = self.time_embed(timesteps)

        # 初始化语义变量
        sem_e1 = sem_e2 = sem_e3 = sem_lat = None
        if self.global_semantic is not None:
            if Ref.abs().sum() > 1e-6:
                sem_pyramid = self.global_semantic(Ref)
                sem_e1 = sem_pyramid["e1"]
                sem_e2 = sem_pyramid["e2"]
                sem_e3 = sem_pyramid["e3"]
                sem_lat = sem_pyramid["latent"]

        # 5. 跨图融合
        main_tokens, ref_tokens = self.cross_fusion(
            main_tokens, ref_tokens, (patch_h, patch_w)
        )

        # 6. 多尺度特征提取
        rf1, rf2, rf3 = self.ref_ms_processor(ref_tokens, patch_h, patch_w)

        # 7. 转为特征图（动态分辨率）
        x = main_tokens.transpose(1, 2).reshape(B, self.embed_dim, patch_h, patch_w)

        # ====================== U-Net 主干 ======================
        def _run_blocks(blocks, x, c, sem):
            """遍历 BlockList，训练时用 checkpoint，推理时直接执行。"""
            for blk in blocks:
                if self.use_checkpoint and self.training:
                    x = checkpoint(blk, x, c, sem, use_reentrant=False)
                else:
                    x = blk(x, c, sem)
            return x

        c1 = self.proj_c_enc1(c)
        x = _run_blocks(self.enc1, x, c1, sem_e1)
        e1_raw = x
        e1 = self.fuse_e1(torch.cat([e1_raw, rf1], dim=1))
        x = self.down1(e1)

        c2 = self.proj_c_enc2(c)
        x = _run_blocks(self.enc2, x, c2, sem_e2)
        e2_raw = x
        e2 = self.fuse_e2(torch.cat([e2_raw, rf2], dim=1))
        x = self.down2(e2)

        c3 = self.proj_c_enc3(c)
        x = _run_blocks(self.enc3, x, c3, sem_e3)
        e3_raw = x
        e3 = self.fuse_e3(torch.cat([e3_raw, rf3], dim=1))
        x = self.down3(e3)

        c_latent = self.proj_c_latent(c)
        x = _run_blocks(self.latent, x, c_latent, sem_lat)

        # 解码器
        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.reduce_chan3(x)
        c_d3 = self.proj_c_dec3(c)
        x = _run_blocks(self.dec3, x, c_d3, sem_e3)

        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.reduce_chan2(x)
        c_d2 = self.proj_c_dec2(c)
        x = _run_blocks(self.dec2, x, c_d2, sem_e2)

        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.reduce_chan1(x)
        c_d1 = self.proj_c_dec1(c)
        x = _run_blocks(self.dec1, x, c_d1, sem_e1)

        # 7. 输出
        x_flat = rearrange(x, "b c h w -> b (h w) c")
        out = self.final_layer(x_flat, c)
        out = unpatchify(out, self.out_channels, h=patch_h, w=patch_w)

        return out

    @classmethod
    def from_args(cls, args):
        return cls(
            patch_size=getattr(args, "patch_size", 4),
            embed_dim=getattr(args, "embed_dim", 384),
            channels=getattr(args, "channels", 3),
            enc_blocks=getattr(args, "enc_blocks", (6, 6, 6)),
            dec_blocks=getattr(args, "dec_blocks", (6, 6, 6)),
            latent_blocks=getattr(args, "latent_blocks", 6),
            drop_path_rate=getattr(args, "drop_path_rate", 0.1),
            hidden_rate=getattr(args, "hidden_rate", 4),
            learn_sigma=getattr(args, "learn_sigma", False),
            upsample_mode=getattr(args, "upsample_mode", "bilinear"),
            global_semantic=getattr(args, "global_semantic", None),
            use_checkpoint=getattr(args, "use_checkpoint", True),
        )


