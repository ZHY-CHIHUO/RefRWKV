import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import sys
from pathlib import Path

root_dir = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, root_dir)
from models.RefSRWKV import RUN_CUDA, OmniShift


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


class CrossFusion(nn.Module):

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

        nn.init.zeros_(self.fuse_proj.weight)
        nn.init.constant_(self.fuse_gate[0].bias, -2.0)

    def forward(self, main_tokens, ref_tokens, resolution):
        main_out = self.wkv_main(main_tokens, resolution)
        ref_out = self.wkv_ref(ref_tokens, resolution)

        concat = torch.cat([main_out, ref_out], dim=-1)
        fused = self.fuse_proj(concat)
        gate = self.fuse_gate(concat)
        main_out = main_tokens + gate * self.fuse_norm(fused)

        return main_out, ref_out


class RefDiffRWKV(nn.Module):

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

        self.patch_embed_main = PatchEmbed(
            patch_size, in_chans=channels, embed_dim=embed_dim
        )
        self.patch_embed_ref = PatchEmbed(
            patch_size, in_chans=channels, embed_dim=embed_dim
        )

        self.cross_fusion = CrossFusion(embed_dim)

        self.ref_ms_processor = RefMultiScaleProcessor(
            embed_dim=embed_dim,
            dims=(embed_dim, embed_dim * 2, embed_dim * 4),
        )

    def extract_ref_features(
        self,
        LR: torch.Tensor,
        Ref: torch.Tensor,
    ) -> tuple:

        B, _, H, W = Ref.shape

        assert (
            H % self.patch_size == 0 and W % self.patch_size == 0
        ), f"Input size {H}x{W} must be divisible by patch_size {self.patch_size}"

        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        LR_up = self.lr_upsampler(LR)

        main_input = LR_up
        main_tokens = self.patch_embed_main(main_input)

        ref_tokens = self.patch_embed_ref(Ref)

        pos_embed_np = get_2d_sincos_pos_embed(self.embed_dim, patch_h, patch_w)
        pos_embed = torch.from_numpy(pos_embed_np).float().to(Ref.device).unsqueeze(0)

        main_tokens = main_tokens + pos_embed
        ref_tokens = ref_tokens + pos_embed

        main_tokens, ref_tokens = self.cross_fusion(
            main_tokens, ref_tokens, (patch_h, patch_w)
        )

        rf1, rf2, rf3 = self.ref_ms_processor(ref_tokens, patch_h, patch_w)

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

    def forward(self, LR, Ref):

        return self.extract_ref_features(LR=LR, Ref=Ref)

    @classmethod
    def from_args(cls, args):

        return cls(
            patch_size=getattr(args, "patch_size", 4),
            embed_dim=getattr(args, "embed_dim", 384),
            channels=getattr(args, "channels", 3),
            upsample_mode=getattr(args, "upsample_mode", "bilinear"),
        )

    def get_parameter_count(self) -> dict:

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
