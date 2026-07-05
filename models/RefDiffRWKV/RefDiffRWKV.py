import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import sys
from pathlib import Path

root_dir = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, root_dir)

# 保留 RWKV 原生 CUDA 算子
from models.RefSRWKV import RUN_CUDA, OmniShift

# 参考 SR_Ref_Encoder_LCA，导入其融合模块
from .modules import LocalCrossAttention, MaskAttention


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


class Downsample(nn.Module):
    """空间减半、通道翻倍。H/4 x W/4 -> H/8 x W/8, C -> 2C。"""

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


class RWKVBlock(nn.Module):
    """
    一个 RWKV 编码块：SpatialMix + ChannelMix，残差连接。
    对应 SR_Ref_Encoder_LCA 中的 Resblock/ConvBlock 层级。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.spatial = VRWKV_SpatialMix(dim)
        self.channel = VRWKV_ChannelMix(dim)

    def forward(self, x: torch.Tensor, resolution: tuple):
        x = x + self.spatial(x, resolution)
        x = x + self.channel(x, resolution)
        return x


class RefDiffRWKV(nn.Module):
    """
    单尺度 RWKV Ref 特征编码器。

    设计思路（参考 SR_Ref_Encoder_LCA）：
        1. LR、Ref 分别做 patch embed + 位置编码
        2. 分 3 个尺度做 RWKV 编码 + LCA/MaskAttention 融合
        3. 最终输出单尺度特征图 (B, out_channel, H/8, W/8)
           与 CRefDiff 编码器输出维度一致，可直接接入 SD2 后处理链

    空间分辨率（以 Ref 480x480, patch_size=4 为例）：
        scale1: 120x120
        scale2:  60x60  (H/8)
        scale3:  60x60  (H/8)
    输出: 60x60

    注意：数据集尺寸固定（Ref = ref_size × ref_size），位置编码在 __init__
    中预计算并 register_buffer，前向直接取用，避免每次 numpy 计算与 CPU→GPU 拷贝。
    """

    def __init__(
        self,
        out_channel: int = 192,
        in_channels: int = 3,
        patch_size: int = 4,
        embed_dim: int = 384,
        upsample_mode: str = "bilinear",
        ref_size: int = 480,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_channel = out_channel
        self.ref_size = ref_size

        if upsample_mode == "bilinear":
            self.lr_upsampler = lr_upsample_bilinear
        elif upsample_mode == "cnn":
            self.lr_upsampler = LRUpsamplerCNN(
                in_ch=in_channels, out_ch=in_channels, scale_factor=10, hidden_ch=64
            )
        elif upsample_mode == "pixelshuffle":
            self.lr_upsampler = LRUpsamplerPixelShuffle(
                in_ch=in_channels, out_ch=in_channels, hidden_ch=64
            )
        else:
            raise ValueError(f"Unsupported upsample_mode: {upsample_mode}")

        # LR / Ref 各自的 patch embed
        self.lr_patch_embed = PatchEmbed(
            patch_size, in_chans=in_channels, embed_dim=embed_dim
        )
        self.ref_patch_embed = PatchEmbed(
            patch_size, in_chans=in_channels, embed_dim=embed_dim
        )

        # ===================== Scale 1: H/4 =====================
        self.lr_block1 = RWKVBlock(embed_dim)
        self.ref_block1 = RWKVBlock(embed_dim)
        self.lca1 = LocalCrossAttention(embed_dim, window_size=8, num_heads=4)
        self.mask1 = MaskAttention(embed_dim)
        # H/4 -> H/8, 通道 embed_dim -> 2*embed_dim
        # SR(融合后) 与 Ref(纯特征) 语义不同，各用一套下采样权重
        self.down1 = Downsample(embed_dim)  # 作用于融合后的 SR 分支
        self.down1_ref = Downsample(embed_dim)  # 作用于 Ref 分支

        # ===================== Scale 2: H/8 =====================
        self.lr_block2 = RWKVBlock(embed_dim * 2)
        self.ref_block2 = RWKVBlock(embed_dim * 2)
        self.lca2 = LocalCrossAttention(embed_dim * 2, window_size=4, num_heads=4)
        self.mask2 = MaskAttention(embed_dim * 2)

        # ===================== Scale 3: H/8 =====================
        self.lr_block3 = RWKVBlock(embed_dim * 2)
        self.ref_block3 = RWKVBlock(embed_dim * 2)
        self.lca3 = LocalCrossAttention(embed_dim * 2, window_size=4, num_heads=4)
        self.mask3 = MaskAttention(embed_dim * 2)

        # 输出投影：concat(LCA, Mask) -> out_channel
        self.last_linear = nn.Conv2d(embed_dim * 4, out_channel, 1, bias=False)

        # ===== 预计算固定尺寸的 sincos 位置编码 =====
        # 数据集尺寸固定：Ref = ref_size × ref_size，patch_h/patch_w 固定
        patch_h = ref_size // patch_size
        patch_w = ref_size // patch_size
        pos_embed_np = get_2d_sincos_pos_embed(embed_dim, patch_h, patch_w)
        pos_embed = torch.from_numpy(pos_embed_np).float().unsqueeze(0)  # [1, N, C]
        # persistent=False：可随时重算，不写入 state_dict，避免污染 checkpoint
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    @staticmethod
    def _to_spatial(tokens, H, W):
        B, N, C = tokens.shape
        return tokens.transpose(1, 2).reshape(B, C, H, W)

    @staticmethod
    def _to_tokens(feat):
        B, C, H, W = feat.shape
        return feat.flatten(2).transpose(1, 2)

    def _call_lca(self, lca, sr_feat, ref_feat, sim_lamuda, return_map):
        if return_map:
            return lca(
                sr_feat,
                ref_feat,
                sim_lamuda=sim_lamuda,
                return_cos_sim_map=True,
            )
        else:
            return lca(sr_feat, ref_feat, sim_lamuda=sim_lamuda), None

    def _call_mask(self, mask, sr_feat, ref_feat, sim_lamuda, return_map):
        if return_map:
            return mask(
                sr_feat,
                ref_feat,
                sim_lamuda=sim_lamuda,
                return_learned_sim_map=True,
            )
        else:
            return mask(sr_feat, ref_feat, sim_lamuda=sim_lamuda), None

    def forward(
        self,
        LR: torch.Tensor,
        Ref: torch.Tensor,
        sim_lamuda=1,
        return_cos_sim_map=False,
        return_learned_sim_map=False,
        **kwargs,
    ):
        B, _, H, W = Ref.shape
        assert (
            H % self.patch_size == 0 and W % self.patch_size == 0
        ), f"Input size {H}x{W} must be divisible by patch_size {self.patch_size}"
        # 位置编码为固定尺寸预计算，校验输入与预设一致
        assert H == self.ref_size and W == self.ref_size, (
            f"输入 Ref 尺寸 {H}x{W} 与预设 ref_size={self.ref_size} 不一致，"
            f"预计算的 pos_embed 无法对齐"
        )

        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        # LR 上采样到 Ref 尺寸
        LR_up = self.lr_upsampler(LR)

        # Patch embed 到 tokens
        sr_tokens = self.lr_patch_embed(LR_up)
        ref_tokens = self.ref_patch_embed(Ref)

        # 位置编码：__init__ 预计算好的 buffer，随模型自动在正确 device/dtype 上
        sr_tokens = sr_tokens + self.pos_embed
        ref_tokens = ref_tokens + self.pos_embed

        # 用于收集返回的 map
        cos_maps = []
        learned_maps = []

        # ===================== Scale 1: H/4 =====================
        sr_tokens = self.lr_block1(sr_tokens, (patch_h, patch_w))
        ref_tokens = self.ref_block1(ref_tokens, (patch_h, patch_w))

        sr_feat = self._to_spatial(sr_tokens, patch_h, patch_w)
        ref_feat = self._to_spatial(ref_tokens, patch_h, patch_w)

        sr_cond1, cos_map1 = self._call_lca(
            self.lca1, sr_feat, ref_feat, sim_lamuda, return_cos_sim_map
        )
        sr_cond2, learned_map1 = self._call_mask(
            self.mask1, sr_feat, ref_feat, sim_lamuda, return_learned_sim_map
        )
        if cos_map1 is not None:
            cos_maps.append(cos_map1)
        if learned_map1 is not None:
            learned_maps.append(learned_map1)

        # 先融合再下采样，对应 LCA 的 sr_cond = lca + mask
        sr_fused = sr_cond1 + sr_cond2

        # LR(融合后) 与 Ref 分支各用独立的下采样权重
        sr_fused = self.down1(sr_fused)
        ref_feat = self.down1_ref(ref_feat)
        patch_h2, patch_w2 = patch_h // 2, patch_w // 2

        sr_tokens = self._to_tokens(sr_fused)
        ref_tokens = self._to_tokens(ref_feat)

        # ===================== Scale 2: H/8 =====================
        sr_tokens = self.lr_block2(sr_tokens, (patch_h2, patch_w2))
        ref_tokens = self.ref_block2(ref_tokens, (patch_h2, patch_w2))

        sr_feat = self._to_spatial(sr_tokens, patch_h2, patch_w2)
        ref_feat = self._to_spatial(ref_tokens, patch_h2, patch_w2)

        sr_cond1, cos_map2 = self._call_lca(
            self.lca2, sr_feat, ref_feat, sim_lamuda, return_cos_sim_map
        )
        sr_cond2, learned_map2 = self._call_mask(
            self.mask2, sr_feat, ref_feat, sim_lamuda, return_learned_sim_map
        )
        if cos_map2 is not None:
            cos_maps.append(cos_map2)
        if learned_map2 is not None:
            learned_maps.append(learned_map2)

        # 这一层不下采样
        sr_fused = sr_cond1 + sr_cond2
        sr_tokens = self._to_tokens(sr_fused)
        ref_tokens = self._to_tokens(ref_feat)

        # ===================== Scale 3: H/8 =====================
        sr_tokens = self.lr_block3(sr_tokens, (patch_h2, patch_w2))
        ref_tokens = self.ref_block3(ref_tokens, (patch_h2, patch_w2))

        sr_feat = self._to_spatial(sr_tokens, patch_h2, patch_w2)
        ref_feat = self._to_spatial(ref_tokens, patch_h2, patch_w2)

        sr_cond1, cos_map3 = self._call_lca(
            self.lca3, sr_feat, ref_feat, sim_lamuda, return_cos_sim_map
        )
        sr_cond2, learned_map3 = self._call_mask(
            self.mask3, sr_feat, ref_feat, sim_lamuda, return_learned_sim_map
        )
        if cos_map3 is not None:
            cos_maps.append(cos_map3)
        if learned_map3 is not None:
            learned_maps.append(learned_map3)

        # 最后一层像 LCA 一样 concat 后投影
        sr_cond = torch.cat([sr_cond1, sr_cond2], dim=1)
        out = self.last_linear(sr_cond)

        if not return_cos_sim_map and not return_learned_sim_map:
            return out
        elif return_cos_sim_map:
            return out, cos_maps
        elif return_learned_sim_map:
            return out, learned_maps
        else:
            # 两个 flag 都为 True 的情况，按 LCA 风格返回
            return out, cos_maps, learned_maps

    @classmethod
    def from_args(cls, args):
        return cls(
            out_channel=getattr(args, "out_channel", 192),
            in_channels=getattr(args, "in_channels", 3),
            patch_size=getattr(args, "patch_size", 4),
            embed_dim=getattr(args, "embed_dim", 384),
            upsample_mode=getattr(args, "upsample_mode", "bilinear"),
            ref_size=getattr(args, "ref_size", 480),
        )

    def get_parameter_count(self) -> dict:
        counts = {}
        total = 0

        submodules = {
            "lr_upsampler": (
                self.lr_upsampler if isinstance(self.lr_upsampler, nn.Module) else None
            ),
            "lr_patch_embed": self.lr_patch_embed,
            "ref_patch_embed": self.ref_patch_embed,
            "blocks": nn.ModuleList(
                [
                    self.lr_block1,
                    self.ref_block1,
                    self.lr_block2,
                    self.ref_block2,
                    self.lr_block3,
                    self.ref_block3,
                ]
            ),
            "lca_mask": nn.ModuleList(
                [self.lca1, self.mask1, self.lca2, self.mask2, self.lca3, self.mask3]
            ),
            "down1": nn.ModuleList([self.down1, self.down1_ref]),
            "last_linear": self.last_linear,
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RefDiffRWKV(out_channel=192, embed_dim=384).to(device)

    B = 2
    LR = torch.randn(B, 3, 48, 48).to(device)
    Ref = torch.randn(B, 3, 480, 480).to(device)

    with torch.no_grad():
        out = model(LR, Ref)
    print(f"Input:  LR={LR.shape}, Ref={Ref.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
