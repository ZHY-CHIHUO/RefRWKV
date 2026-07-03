"""
RefDiffRWKV_no_xt.py — 去除 x_t 依赖的 Ref 特征提取管线
（借鉴 CRefDiff modules.py 的设计：只输入 LR + Ref）

改动说明 (Changelog):
    原版 RefDiffRWKV:
        输入: x_t(noisy image) + LR + Ref  ← x_t 每步采样都变，需重复提取
        main支路: concat(x_t, LR_up) → 6ch PatchEmbed
    
    改造后 (本文件):
        输入: LR + Ref  ← 条件信号只提取一次，全程复用
        main支路: LR_up → 3ch PatchEmbed  ← 与 CRefDiff 的 SR_Ref_Encoder_LCA 一致
    
    不变的模块 (完全保留):
        - VRWKV_SpatialMix / VRWKV_ChannelMix (RWKV 核心)
        - CrossFusion (门控融合)
        - RefMultiScaleProcessor (多尺度提取)
        - 显式余弦相似度加权 (安全兜底)
        - 所有位置编码 / 下采样 / 上采样模块

用法 (Usage):
    model = RefDiffRWKV_NoXT(patch_size=4, embed_dim=384)
    rf1, rf2, rf3 = model.extract_ref_features(
        LR=lr_image,             # 低分辨率输入 (B, 3, 48, 48)
        Ref=ref_image            # 参考图像   (B, 3, 480, 480)
    )
    # rf1: (B, 384, H/8, W/8)   → SD2 down_blocks[0]
    # rf2: (B, 768, H/16, W/16) → SD2 down_blocks[1]
    # rf3: (B, 1536, H/32, W/32)→ SD2 down_blocks[2]

与原版的调用差异:
    # 原版 (每步采样都要调):
    for t in reversed(timesteps):
        x_t = vae.decode(latent)
        rf1,rf2,rf3 = model(x_t=x_t, LR=lr, Ref=ref)  # x_t每步不同!
        latent = unet(latent, t, rf1,rf2,rf3)

    # 改版 (只提取一次):
    rf1,rf2,rf3 = model(LR=lr, Ref=ref)  # 在采样循环之前调用一次
    for t in reversed(timesteps):
        latent = unet(latent, t, rf1,rf2,rf3)  # 全程复用
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import sys
from pathlib import Path

# 添加项目根目录到 sys.path，确保能导入 RefSRWKV 中的 CUDA 算子
root_dir = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, root_dir)
from models.RefSRWKV import RUN_CUDA, OmniShift


# ======================================================================
#  1. 正弦余弦 2D 位置编码（不变）
# ======================================================================


def get_2d_sincos_pos_embed(embed_dim, h, w, cls_token=False, extra_tokens=0):
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h, indexing="xy")
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
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


# ======================================================================
#  2. Patch Embedding — 图像分块嵌入（不变）
# ======================================================================


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H % self.patch_size == 0 and W % self.patch_size == 0
        ), f"Input size {H}x{W} must be divisible by patch_size {self.patch_size}"
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


# ======================================================================
#  3. VRWKV 空间混合 & 通道混合（完全不变）
# ======================================================================


class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = n_embd
        self.recurrence = 2
        self.omni_shift = OmniShift(dim=n_embd)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

        with torch.no_grad():
            decay_base = torch.linspace(-1.0, -6.0, n_embd)
            self.spatial_decay = nn.Parameter(
                decay_base.unsqueeze(0).expand(self.recurrence, -1).clone()
            )
            self.spatial_first = nn.Parameter(torch.zeros(self.recurrence, n_embd))

    def jit_func(self, x: torch.Tensor, resolution: tuple):
        h, w = resolution
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)
        return sr, k, v

    def forward(self, x: torch.Tensor, resolution: tuple):
        B, T, C = x.shape
        sr, k, v = self.jit_func(x, resolution)
        s = C**0.5

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
                k_t = rearrange(k.clone(), "b (h w) c -> b (w h) c", h=h, w=w)
                v_t = rearrange(v, "b (h w) c -> b (w h) c", h=h, w=w)
                v_t = RUN_CUDA(
                    self.spatial_decay[j] / s,
                    self.spatial_first[j] / s,
                    k_t,
                    v_t,
                )
                v = rearrange(v_t, "b (w h) c -> b (h w) c", h=h, w=w)

        x = sr * v
        x = self.output(x)
        return x


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd: int, hidden_rate: int = 4):
        super().__init__()
        self.n_embd = n_embd
        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)
        self.omni_shift = OmniShift(dim=n_embd)

    def forward(self, x: torch.Tensor, resolution: tuple):
        h, w = resolution
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        k = self.key(x)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        r = torch.sigmoid(self.receptance(x))
        return r * kv


# ======================================================================
#  4. 上下采样模块（不变）
# ======================================================================


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


# ======================================================================
#  5. LR 上采样器（不变）
# ======================================================================


def lr_upsample_bilinear(lr: torch.Tensor, scale: int = 10):
    _, _, h, w = lr.shape
    return F.interpolate(lr, size=(h * scale, w * scale), mode="bilinear", align_corners=False)


class LRUpsamplerCNN(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, scale_factor=10, hidden_ch=64):
        super().__init__()
        self.scale_factor = scale_factor
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.body(x)


class LRUpsamplerPixelShuffle(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, hidden_ch=64):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch * 25, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(5),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        return x


# ======================================================================
#  6. Ref 多尺度处理器（不变）
# ======================================================================


class RefMultiScaleProcessor(nn.Module):
    def __init__(self, embed_dim, dims):
        super().__init__()
        d1, d2, d3 = dims

        self.proj1 = nn.Conv2d(embed_dim, d1, 1)
        self.channel_mix1 = VRWKV_ChannelMix(d1)

        self.down1 = Downsample(d1)
        self.adapt1 = nn.Conv2d(d1 * 2, d2, 1)
        self.channel_mix2 = VRWKV_ChannelMix(d2)

        self.down2 = Downsample(d2)
        self.adapt2 = nn.Conv2d(d2 * 2, d3, 1)
        self.channel_mix3 = VRWKV_ChannelMix(d3)

    def forward(self, ref_tokens, H, W):
        B, _, C = ref_tokens.shape
        x = ref_tokens.transpose(1, 2).reshape(B, C, H, W)

        f1 = self.proj1(x)
        f1 = self._apply_channel_mix(f1, self.channel_mix1)

        f2 = self.adapt1(self.down1(f1))
        f2 = self._apply_channel_mix(f2, self.channel_mix2)

        f3 = self.adapt2(self.down2(f2))
        f3 = self._apply_channel_mix(f3, self.channel_mix3)

        f1 = F.interpolate(f1, scale_factor=0.5, mode="bilinear", align_corners=False)
        f2 = F.interpolate(f2, scale_factor=0.5, mode="bilinear", align_corners=False)
        f3 = F.interpolate(f3, scale_factor=0.5, mode="bilinear", align_corners=False)

        return f1, f2, f3

    @staticmethod
    def _apply_channel_mix(x: torch.Tensor, mix: VRWKV_ChannelMix) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = mix(x, (H, W))
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


# ======================================================================
#  7. 跨图融合模块（接口微调：main_tokens 含义变更）
# ======================================================================


class CrossFusion(nn.Module):
    """
    跨图融合模块: main(LR特征) 与 ref(参考特征) 各自经过级联双向 WKV 扫描后，
    通过门控机制融合。

    ★ 改版变化 (vs 原版):
        原版: main_tokens = concat(x_t, LR_up) 的 patch embed  — 携带 noisy image 状态
        改版: main_tokens = LR_up 的 patch embed              — 仅携带 LR 退化信息

    零初始化策略 (Zero-Init): 不变
        fuse_proj.weight = 0, gate bias = -2 (sigmoid(-2) ≈ 0.12)
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.n_embd = embed_dim

        self.wkv_main = VRWKV_SpatialMix(embed_dim)
        self.wkv_ref = VRWKV_SpatialMix(embed_dim)

        self.fuse_proj = nn.Linear(embed_dim * 2, embed_dim, bias=False)
        self.fuse_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.fuse_norm = nn.LayerNorm(embed_dim)

        # 零初始化: 训练初期 ref 贡献 ≈ 0
        nn.init.zeros_(self.fuse_proj.weight)
        nn.init.constant_(self.fuse_gate[0].bias, -2.0)

    def forward(self, main_tokens, ref_tokens, resolution):
        """
        Args:
            main_tokens: (B, N, C) 主分支 token（★改版: 仅 LR_up 的 patch embed）
            ref_tokens:  (B, N, C) 参考分支 token（Ref 的 patch embed）
            resolution:  (H, W) 2D 网格尺寸
        """
        main_out = self.wkv_main(main_tokens, resolution)
        ref_out = self.wkv_ref(ref_tokens, resolution)

        concat = torch.cat([main_out, ref_out], dim=-1)
        fused = self.fuse_proj(concat)
        gate = self.fuse_gate(concat)
        main_out = main_tokens + gate * self.fuse_norm(fused)

        return main_out, ref_out


# ======================================================================
#  8. RefDiffRWKV_NoXT — 去除 x_t 依赖的纯 Ref 特征提取器
#     （核心改造在此类）
# ======================================================================


class RefDiffRWKV_NoXT(nn.Module):
    """
    RefDiffRWKV 去除 x_t 版本 — 仅输入 LR + Ref 的 Ref 特征提取管线。

    ═══════════════════════════════════════════════════════════════
    与原版的核心差异 (Core Difference vs Original):
    ═══════════════════════════════════════════════════════════════

                        原版 RefDiffRWKV          改版 RefDiffRWKV_NoXT
                        ─────────────────          ────────────────────
    输入:               x_t + LR + Ref             LR + Ref
    main支路:           cat(x_t, LR_up) → 6ch      LR_up → 3ch
    提取时机:           每步采样都重新提取          采样前提取一次，全程复用
    工作模式:           动态感知去噪状态             静态条件注入（同CRefDiff）
    对应CRefDiff模块:   无直接对应                   SR_Ref_Encoder_LCA

    ═══════════════════════════════════════════════════════════════
    管线流程 (Pipeline):
    ═══════════════════════════════════════════════════════════════
        LR (48×48) ──► Upsample ──► PatchEmbed ──► main_tokens
        Ref (480×480) ──────────────► PatchEmbed ──► ref_tokens
            │
            ▼
        CrossFusion（门控融合 main 与 ref token）
            │
            ▼
        RefMultiScaleProcessor → rf1, rf2, rf3（多尺度 2D 特征图）
            │
            ▼
        显式余弦相似度加权（安全兜底，抑制不匹配区域）
            │
            ▼
        返回 rf1, rf2, rf3 → 注入 SD2 UNet down_blocks

    Args:
        patch_size:    patch 边长（默认 4）
        embed_dim:     嵌入维度（默认 384）
        channels:      图像通道数（默认 3，RGB）
        upsample_mode: LR 上采样模式: "bilinear" / "cnn" / "pixelshuffle"
    """

    def __init__(
        self,
        patch_size: int = 4,
        embed_dim: int = 384,
        channels: int = 3,
        upsample_mode: str = "bilinear",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.channels = channels

        # ── LR 上采样器 ──
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

        # ════════════════════════════════════════════════
        # ★ 改造点1: PatchEmbedding（双支路）
        # ════════════════════════════════════════════════
        # 原版: main支路 in_chans=channels*2 (6ch, 因为 concat[x_t, LR_up])
        # 改版: main支路 in_chans=channels   (3ch, 只有 LR_up, 同 CRefDiff 的 SR 分支)
        #
        # ref 支路不变: in_chans=channels (3ch, 只有 Ref)
        # ════════════════════════════════════════════════
        self.patch_embed_main = PatchEmbed(
            patch_size, in_chans=channels, embed_dim=embed_dim  # ★ 3ch (原版6ch)
        )
        self.patch_embed_ref = PatchEmbed(
            patch_size, in_chans=channels, embed_dim=embed_dim
        )

        # ── 跨图融合 ──
        self.cross_fusion = CrossFusion(embed_dim)

        # ── Ref 多尺度处理器 ──
        self.ref_ms_processor = RefMultiScaleProcessor(
            embed_dim=embed_dim,
            dims=(embed_dim, embed_dim * 2, embed_dim * 4),
        )

    # ==================================================================
    #  核心方法: extract_ref_features
    #  ★★ 改造重点: 去除 x_t 参数，仅用 LR + Ref ★★
    # ==================================================================

    def extract_ref_features(
        self,
        LR: torch.Tensor,   # (B, 3, h, w) 低分辨率输入图像
        Ref: torch.Tensor,  # (B, 3, H, W) 参考图像
    ) -> tuple:
        """
        从 LR 和 Ref 中提取多尺度 ref 纹理特征（无需 x_t）。

        完整流程 (Complete Workflow):
            1. LR 上采样到与 Ref 相同的分辨率
            2. main 支路: LR_up → patch_embed_main → main_tokens  (★ 3ch, 非6ch)
            3. ref  支路: Ref → patch_embed_ref → ref_tokens
            4. 动态位置编码（根据 patch 网格尺寸动态生成）
            5. CrossFusion: main_tokens 与 ref_tokens 门控交互
            6. RefMultiScaleProcessor: ref_tokens → rf1, rf2, rf3
            7. 显式相似度图加权: LR_up 与 Ref 的余弦相似度 → 逐位置抑制

        Args:
            LR:  (B, 3, h, w) 低分辨率输入（如 48×48，将被上采样到 H×W）
            Ref: (B, 3, H, W) 参考图像（像素空间，目标分辨率）

        Returns:
            rf1: (B, embed_dim,     H/2p,   W/2p)    ← 高分辨率 ref 特征 (~384ch)
            rf2: (B, embed_dim*2,   H/4p,  W/4p)   ← 中分辨率 ref 特征 (~768ch)
            rf3: (B, embed_dim*4,   H/8p,  W/8p)   ← 低分辨率 ref 特征 (~1536ch)
            其中 p = patch_size

        示例 (Example):
            >>> model = RefDiffRWKV_NoXT(patch_size=4, embed_dim=384)
            >>> rf1, rf2, rf3 = model.extract_ref_features(
            ...     LR=lr_image,       # (B, 3, 48, 48)
            ...     Ref=ref_image      # (B, 3, 480, 480)
            ... )
            >>> rf1.shape  # (B, 384, 60, 60)   for 480/4=120→/2=60
            >>> rf2.shape  # (B, 768, 30, 30)
            >>> rf3.shape  # (B, 1536, 15, 15)
        """

        # ════════════════════════════════════════════════
        # ★ 改造点2: 从 Ref 获取分辨率和设备信息（原版从 x_t 获取）
        # ════════════════════════════════════════════════
        B, _, H, W = Ref.shape

        assert (
            H % self.patch_size == 0 and W % self.patch_size == 0
        ), f"Input size {H}x{W} must be divisible by patch_size {self.patch_size}"

        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        # ── Step 1: LR 上采样到 Ref 分辨率 ──
        LR_up = self.lr_upsampler(LR)

        # ════════════════════════════════════════════════
        # ★ 改造点3: main 支路不再拼接 x_t
        # ════════════════════════════════════════════════
        # 原版: main_input = cat([x_t, LR_up], dim=1)  # (B, 6, H, W)
        # 改版: main_input = LR_up                     # (B, 3, H, W)
        #        → 对应 CRefDiff 中 SR_Ref_Encoder_LCA 的 SR 分支
        # ════════════════════════════════════════════════
        main_input = LR_up  # (B, 3, H, W)  ← ★ 直接用上采样后的 LR
        main_tokens = self.patch_embed_main(main_input)  # (B, N, embed_dim)

        # ref 支路不变: Ref → 3 通道 → patch tokens
        ref_tokens = self.patch_embed_ref(Ref)  # (B, N, embed_dim)

        # ── Step 3: 动态位置编码 ──
        # ★ 改造点4: 设备信息从 Ref 获取（原版从 x_t.device）
        pos_embed_np = get_2d_sincos_pos_embed(self.embed_dim, patch_h, patch_w)
        pos_embed = torch.from_numpy(pos_embed_np).float().to(Ref.device).unsqueeze(0)

        main_tokens = main_tokens + pos_embed
        ref_tokens = ref_tokens + pos_embed

        # ── Step 4: 跨图融合 ──
        # 接口不变，但 main_tokens 的语义从 "noisy+LR" 变为 "仅LR"
        main_tokens, ref_tokens = self.cross_fusion(
            main_tokens, ref_tokens, (patch_h, patch_w)
        )

        # ── Step 5: Ref 多尺度特征提取 ──
        rf1, rf2, rf3 = self.ref_ms_processor(ref_tokens, patch_h, patch_w)

        # ── Step 6: 显式相似度图加权（不变，完全保留）──
        with torch.no_grad():
            lr_for_sim = F.interpolate(
                LR_up, size=rf1.shape[2:], mode="bilinear", align_corners=False
            )
            ref_for_sim = F.interpolate(
                Ref, size=rf1.shape[2:], mode="bilinear", align_corners=False
            )
            sim_map = (
                F.cosine_similarity(
                    lr_for_sim.flatten(2), ref_for_sim.flatten(2), dim=1
                )
                .reshape(rf1.shape[0], 1, rf1.shape[2], rf1.shape[3])
                .clamp(min=0.0)
            )

        rf1 = rf1 * F.interpolate(
            sim_map, size=rf1.shape[2:], mode="bilinear", align_corners=False
        )
        rf2 = rf2 * F.interpolate(
            sim_map, size=rf2.shape[2:], mode="bilinear", align_corners=False
        )
        rf3 = rf3 * F.interpolate(
            sim_map, size=rf3.shape[2:], mode="bilinear", align_corners=False
        )

        return rf1, rf2, rf3

    # ==================================================================
    #  兼容性方法: forward
    #  ★ 签名变更: 去掉 x_t 和 timesteps 参数
    # ==================================================================

    def forward(self, LR, Ref):
        """
        兼容性 forward（签名简化）。
        原版: forward(x_t, timesteps, LR, Ref)
        改版: forward(LR, Ref)
        """
        return self.extract_ref_features(LR=LR, Ref=Ref)

    @classmethod
    def from_args(cls, args):
        """从参数对象构建模型（兼容原版 API）。"""
        return cls(
            patch_size=getattr(args, "patch_size", 4),
            embed_dim=getattr(args, "embed_dim", 384),
            channels=getattr(args, "channels", 3),
            upsample_mode=getattr(args, "upsample_mode", "bilinear"),
        )

    def get_parameter_count(self) -> dict:
        """统计各子模块的参数量。"""
        counts = {}
        total = 0

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
                n = sum(p.numel() for p in mod.parameters())
                counts[name] = n
                total += n

        counts["total"] = total
        return counts
