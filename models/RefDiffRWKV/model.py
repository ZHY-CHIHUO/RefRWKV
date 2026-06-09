# Copyright (c) Shanghai AI Lab. All rights reserved.
import math, os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from torch.utils.cpp_extension import load

wkv_cuda = load(
    name="bi_wkv",
    sources=["./cuda/bi_wkv.cpp", "./cuda/bi_wkv_kernel.cu"],
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


class OmniShift(nn.Module):
    def __init__(self, dim):
        super(OmniShift, self).__init__()
        # Define the layers for training
        self.conv1x1 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False
        )
        self.conv3x3 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=False,
        )
        self.conv5x5 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=False,
        )
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True)

        # Define the layers for testing
        self.conv5x5_reparam = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=False,
        )
        self.repram_flag = True

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)
        # import pdb
        # pdb.set_trace()

        out = (
            self.alpha[0] * x
            + self.alpha[1] * out1x1
            + self.alpha[2] * out3x3
            + self.alpha[3] * out5x5
        )
        return out

    def reparam_5x5(self):
        # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution

        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))

        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2))

        combined_weight = (
            self.alpha[0] * identity_weight
            + self.alpha[1] * padded_weight_1x1
            + self.alpha[2] * padded_weight_3x3
            + self.alpha[3] * self.conv5x5.weight
        )

        device = self.conv5x5_reparam.weight.device

        combined_weight = combined_weight.to(device)

        self.conv5x5_reparam.weight = nn.Parameter(combined_weight)

    def forward(self, x):

        if self.training:
            self.repram_flag = True
            out = self.forward_train(x)
        elif self.training == False and self.repram_flag == True:
            self.reparam_5x5()
            self.repram_flag = False
            out = self.conv5x5_reparam(x)
        elif self.training == False and self.repram_flag == False:
            out = self.conv5x5_reparam(x)

        return out


class BiWKV_Linear(nn.Module):
    """纯线性双向 WKV，用于跨图融合，不进行 2D reshape。"""

    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = n_embd
        self.recurrence = 2
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

        with torch.no_grad():
            self.spatial_decay = nn.Parameter(
                torch.randn(self.recurrence, n_embd) * 0.1
            )
            self.spatial_first = nn.Parameter(
                torch.randn(self.recurrence, n_embd) * 0.1
            )

    def forward(self, x: torch.Tensor, resolution=None):
        B, T, C = x.shape
        k = self.key(x)
        v = self.value(x)
        r = torch.sigmoid(self.receptance(x))

        for j in range(self.recurrence):
            if j % 2 == 0:
                v = RUN_CUDA(
                    self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v
                )  # 正向扫描
            else:
                # 反向扫描：翻转序列
                k_rev = torch.flip(k, dims=[1])
                v_rev = torch.flip(v, dims=[1])
                v_rev = RUN_CUDA(
                    self.spatial_decay[j] / T, self.spatial_first[j] / T, k_rev, v_rev
                )
                v = torch.flip(v_rev, dims=[1])
                k = torch.flip(k_rev, dims=[1])  # 保持 k 一致

        x = r * v
        x = self.output(x)
        return x


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
        """
        x: (B, N, C)  输入 token 序列
        resolution: (H, W)  对应二维网格的高和宽
        """
        B, T, C = x.shape
        sr, k, v = self.jit_func(x, resolution)

        for j in range(self.recurrence):
            if j % 2 == 0:
                # 正向扫描：行优先顺序 (h, w)
                v = RUN_CUDA(self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
            else:
                # 反向扫描：转置为列优先 (w, h) 再扫描，获得另一个方向的信息
                h, w = resolution
                k = rearrange(k, "b (h w) c -> b (w h) c", h=h, w=w)
                v = rearrange(v, "b (h w) c -> b (w h) c", h=h, w=w)
                v = RUN_CUDA(self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
                # 恢复原顺序
                k = rearrange(k, "b (w h) c -> b (h w) c", h=h, w=w)
                v = rearrange(v, "b (w h) c -> b (h w) c", h=h, w=w)

        # 门控 + 输出投影
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
    用于 Bi-DiffRWKV 的基础构建块（支持扩散条件注入）。

    特性：
    - 输入/输出为二维特征图 (B, C, H, W)，直接适配 U-Net 的上下采样结构。
    - 内部使用双向 WKV + OmniShift 进行空间混合和通道混合。
    - 通过 adaLN 机制接收时间步条件 c，动态调制注意力与 FFN 分支。
    - 支持随机深度 (DropPath) 和可选的跳跃连接 (skip connection)。
    """

    def __init__(
        self,
        n_embd: int,
        hidden_rate: int = 4,
        drop_path: float = 0.0,
        use_adaLN: bool = True,
    ):
        super().__init__()

        # 层归一化
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        # 空间混合（双向 WKV + OmniShift）
        self.att = VRWKV_SpatialMix(n_embd)

        # 通道混合（FFN + OmniShift）
        self.ffn = VRWKV_ChannelMix(n_embd, hidden_rate=hidden_rate)

        # 随机深度
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # 条件调制层（adaLN）
        self.use_adaLN = use_adaLN
        if use_adaLN:
            # 输出 6 个调制参数：att 的 shift/scale/gate，ffn 的 shift/scale/gate
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(n_embd, 6 * n_embd, bias=True),
            )
            # 零初始化（扩散模型训练稳定性的关键）
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            # 若无外部条件，退化为简单的可学习缩放系数（与原 Restore‑RWKV 兼容）
            self.gamma1 = nn.Parameter(torch.ones(n_embd))
            self.gamma2 = nn.Parameter(torch.ones(n_embd))

    def forward(self, x: torch.Tensor, c: torch.Tensor = None):
        """
        x: 输入特征图 (B, C, H, W)
        c: 条件向量 (B, C) ，若 use_adaLN=True 则必须提供，否则可为 None
        返回: 输出特征图 (B, C, H, W)
        """
        B, C, H, W = x.shape
        resolution = (H, W)

        # ---------- 生成调制系数 ----------
        if self.use_adaLN:
            assert c is not None, "BiBlock with adaLN requires condition c"
            shift_att, scale_att, gate_att, shift_ffn, scale_ffn, gate_ffn = (
                self.adaLN_modulation(c).chunk(6, dim=1)
            )
        else:
            # 简单缩放模式
            gate_att = self.gamma1
            gate_ffn = self.gamma2
            shift_att = scale_att = shift_ffn = scale_ffn = 0.0

        # ---------- 注意力分支 ----------
        # 1. 转为序列并归一化 + 调制
        x_flat = rearrange(x, "b c h w -> b (h w) c")
        x_flat = modulate(self.ln1(x_flat), shift_att, scale_att)
        # 2. 双向 WKV 空间混合
        x_flat = self.att(x_flat, resolution)
        # 3. 门控 + 残差
        x_flat = gate_att.unsqueeze(1) * x_flat
        x_flat = rearrange(x_flat, "b (h w) c -> b c h w", h=H, w=W)
        x = x + self.drop_path(x_flat)

        # ---------- FFN 分支 ----------
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
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
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
            c = self.adaLN_modulation(c).squeeze(1)
            shift, scale = c.chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
            x = self.linear(x)
        else:
            x = self.norm_final(x)
            x = self.linear(x)
        return x


##########################################################################
## lr上采样
def lr_upsample_bilinear(lr: torch.Tensor, target_size: tuple = (480, 480)):
    """
    lr: (B, 3, 48, 48)
    返回: (B, 3, 480, 480)
    """
    return F.interpolate(lr, size=target_size, mode="bilinear", align_corners=False)


class LRUpsamplerCNN(nn.Module):
    """轻量 CNN 上采样器：将 48×48 放大到 480×480（10倍）"""

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
        """
        x: (B, 3, 48, 48)
        返回: (B, 3, 480, 480)
        """
        return self.body(x)


class LRUpsamplerPixelShuffle(nn.Module):
    """用 PixelShuffle 分步上采样 10 倍（2× + 5×）"""

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
        """
        x: (B, 3, 48, 48)
        返回: (B, 3, 480, 480)
        """
        x = self.stage1(x)  # → (B, 64, 96, 96)
        x = self.stage2(x)  # → (B, 3, 480, 480)
        return x


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
        img_size: int = 480,  # 仅作为默认参考值
        patch_size: int = 4,
        embed_dim: int = 384,
        channels: int = 3,
        enc_blocks=(6, 6, 6),
        dec_blocks=(6, 6, 6),
        latent_blocks=6,
        drop_path_rate: float = 0.1,
        hidden_rate: int = 4,
        learn_sigma: bool = False,
        upsample_mode: str = "bicubic",
        **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.channels = channels
        self.patch_resolution = (
            img_size // patch_size,
            img_size // patch_size,
        )  # 默认值

        # ---------- LR 上采样器 ----------
        if upsample_mode == 'bilinear':
            self.lr_upsampler = lambda x: F.interpolate(
                x, size=(img_size, img_size), mode='bilinear', align_corners=False
            )
        elif upsample_mode == 'cnn':
            self.lr_upsampler = LRUpsamplerCNN(in_ch=channels, out_ch=channels,
                                               scale_factor=img_size // 48, hidden_ch=64)
        elif upsample_mode == 'pixelshuffle':
            self.lr_upsampler = LRUpsamplerPixelShuffle(in_ch=channels, out_ch=channels, hidden_ch=64)
        else:
            raise ValueError(f"Unsupported upsample_mode: {upsample_mode}")

        # Patch Embed（双支路）
        self.patch_embed_main = PatchEmbed(
            patch_size, in_chans=channels * 2, embed_dim=embed_dim
        )
        self.patch_embed_ref = PatchEmbed(
            patch_size, in_chans=channels, embed_dim=embed_dim
        )

        # 时间嵌入
        self.time_embed = TimestepEmbedder(embed_dim)

        # 位置编码不再固定为 Parameter，改为动态生成
        # self.pos_embed = nn.Parameter(...)  # 已移除

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

        # 跳跃连接通道调整
        self.reduce_chan3 = nn.Conv2d(dim * 8, dim * 4, 1)
        self.reduce_chan2 = nn.Conv2d(dim * 4, dim * 2, 1)
        self.reduce_chan1 = nn.Conv2d(dim * 2, dim, 1)

        # 输出头
        self.out_channels = channels * 2 if learn_sigma else channels
        self.final_layer = FinalLayer(
            embed_dim, patch_size, self.out_channels, condition=True
        )

        # 跨图融合
        self.fuse_att = BiWKV_Linear(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
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

        # 时间嵌入初始化
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # adaLN 零初始化
        for module in self.modules():
            if isinstance(module, BiBlock) and module.use_adaLN:
                nn.init.constant_(module.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(module.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def cross_fusion(self, main_tokens, ref_tokens):
        B, N, C = main_tokens.shape
        interleaved = torch.stack([main_tokens, ref_tokens], dim=2).reshape(B, 2 * N, C)
        interleaved = self.fuse_att(interleaved)
        main_tokens = interleaved[:, 0::2, :]
        return main_tokens

    def forward(self, x_t, timesteps, LR, Ref):
        B, _, H, W = x_t.shape

        # 检查分辨率合法性
        assert (
            H % self.patch_size == 0 and W % self.patch_size == 0
        ), f"Input size {H}x{W} must be divisible by patch_size {self.patch_size}"

        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        # 1. LR 上采样到当前分辨率
        LR_up = self.lr_upsampler(LR, (H, W))

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

        # 5. 跨图融合
        main_tokens = self.cross_fusion(main_tokens, ref_tokens)

        # 6. 转为特征图（动态分辨率）
        x = main_tokens.transpose(1, 2).reshape(B, self.embed_dim, patch_h, patch_w)

        # ====================== U-Net 主干 ======================
        c1 = self.proj_c_enc1(c)
        for blk in self.enc1:
            x = blk(x, c1)
        e1 = x
        x = self.down1(x)

        c2 = self.proj_c_enc2(c)
        for blk in self.enc2:
            x = blk(x, c2)
        e2 = x
        x = self.down2(x)

        c3 = self.proj_c_enc3(c)
        for blk in self.enc3:
            x = blk(x, c3)
        e3 = x
        x = self.down3(x)

        c_latent = self.proj_c_latent(c)
        for blk in self.latent:
            x = blk(x, c_latent)

        # 解码器
        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.reduce_chan3(x)
        c_d3 = self.proj_c_dec3(c)
        for blk in self.dec3:
            x = blk(x, c_d3)

        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.reduce_chan2(x)
        c_d2 = self.proj_c_dec2(c)
        for blk in self.dec2:
            x = blk(x, c_d2)

        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.reduce_chan1(x)
        c_d1 = self.proj_c_dec1(c)
        for blk in self.dec1:
            x = blk(x, c_d1)

        # 7. 输出
        x_flat = rearrange(x, "b c h w -> b (h w) c")
        out = self.final_layer(x_flat, c)
        out = unpatchify(out, self.out_channels, h=patch_h, w=patch_w)

        return out

    @classmethod
    def from_args(cls, args):
        return cls(
            img_size=getattr(args, "img_size", 480),
            patch_size=getattr(args, "patch_size", 4),
            embed_dim=getattr(args, "embed_dim", 384),
            channels=getattr(args, "channels", 3),
            enc_blocks=getattr(args, "enc_blocks", (6, 6, 6)),
            dec_blocks=getattr(args, "dec_blocks", (6, 6, 6)),
            latent_blocks=getattr(args, "latent_blocks", 6),
            drop_path_rate=getattr(args, "drop_path_rate", 0.1),
            hidden_rate=getattr(args, "hidden_rate", 4),
            learn_sigma=getattr(args, "learn_sigma", False),
            upsample_mode=getattr(args, "upsample_mode", "bicubic"),
        )


# ---------- 便捷的模型构建函数 ----------
def refdiffrwkv_s(**kwargs):
    return RefDiffRWKV(
        embed_dim=64,
        enc_blocks=[3, 3, 3],
        dec_blocks=[3, 3, 3],
        latent_blocks=4,
        **kwargs,
    )


def refdiffrwkv_b(**kwargs):
    return RefDiffRWKV(
        embed_dim=64,
        enc_blocks=[4, 6, 6],
        dec_blocks=[6, 6, 4],
        latent_blocks=8,
        **kwargs,
    )


def refdiffrwkv_l(**kwargs):
    return RefDiffRWKV(
        embed_dim=128,
        enc_blocks=[8, 10, 10],
        dec_blocks=[10, 10, 8],
        latent_blocks=12,
        **kwargs,
    )


import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR


class RefDiffRWKV_PL(pl.LightningModule):
    """
    PyTorch Lightning 包装器，用于 RefDiffRWKV 的训练、验证和推理
    """

    def __init__(
        self,
        model: "RefDiffRWKV",  # 传入已实例化的 RefDiffRWKV
        lr: float = 4e-4,
        weight_decay: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        warmup_steps: int = 10000,
        total_steps: int = 500000,
        scheduler: str = "cosine",  # "cosine" 或 "linear"
        num_timesteps: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.scheduler_type = scheduler
        self.num_timesteps = num_timesteps

    def forward(self, x_t, timesteps, LR, Ref):
        return self.model(x_t, timesteps, LR, Ref)

    def _add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        """余弦噪声调度"""
        s = 0.008
        T = self.num_timesteps
        alpha_bar = torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

    def _compute_loss(self, batch, stage: str = "train"):
        hr, lr, ref = batch  # ← 这里接收 Dataset 返回的 (HR, LR, Ref)
        B = hr.shape[0]

        t = torch.randint(0, self.num_timesteps, (B,), device=self.device)
        noise = torch.randn_like(hr)

        x_t = self._add_noise(hr, noise, t)  # 对 HR 加噪
        pred_noise = self.model(x_t, t, lr, ref)  # 模型预测噪声

        loss = F.mse_loss(pred_noise, noise)

        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    # ====================== Training / Validation ======================
    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, "train")

        # 记录学习率
        if self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", current_lr, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, "test")
        return loss

    # ====================== Epoch End Logging ======================
    def on_train_epoch_end(self):
        if self.trainer.global_rank == 0:
            train_loss = self.trainer.callback_metrics.get("train/loss_epoch", 0.0)
            print(f"Epoch {self.current_epoch:04d} | Train Loss: {train_loss:.6f}")

    def on_validation_epoch_end(self):
        if self.trainer.global_rank == 0:
            val_loss = self.trainer.callback_metrics.get("val/loss", 0.0)
            current_lr = (
                self.trainer.optimizers[0].param_groups[0]["lr"]
                if self.trainer.optimizers
                else 0.0
            )
            print(
                f"Epoch {self.current_epoch:04d} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}"
            )

    # ====================== Optimizer ======================
    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.weight_decay,
            eps=1e-8,
        )

        if self.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.total_steps - self.warmup_steps,
                eta_min=self.lr * 0.01,
            )
        else:  # linear warmup + cosine decay

            def lr_lambda(current_step: int):
                if current_step < self.warmup_steps:
                    return float(current_step) / float(max(1, self.warmup_steps))
                progress = float(current_step - self.warmup_steps) / float(
                    max(1, self.total_steps - self.warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_train_start(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ RefDiffRWKV_PL Training Started!")
        print(f"   Total Parameters: {total_params / 1e6:.2f}M")
        print(f"   Learning Rate: {self.lr}")
        print(f"   Warmup Steps: {self.warmup_steps}")
