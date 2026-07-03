import cmath
import math
from torch import nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import random

"""
    CNN-SR
"""


def cosine_attention_map(tensor1, tensor2, eps=1e-8):
    """
    输入:
        tensor1, tensor2: 形状为 (B, C, H, W) 的两个张量
    输出:
        cosine_distance_map: 形状为 (B, 1, H, W) 的注意力图
    """
    B, C, H, W = tensor1.shape

    # 展平通道维，准备计算每个像素位置的余弦相似度
    x1 = tensor1.view(B, C, -1)  # (B, C, H*W)
    x2 = tensor2.view(B, C, -1)

    # L2 归一化
    x1_norm = F.normalize(x1, p=2, dim=1)  # (B, C, H*W)
    x2_norm = F.normalize(x2, p=2, dim=1)

    # 对应位置点积（余弦相似度）
    cos_sim = torch.sum(x1_norm * x2_norm, dim=1, keepdim=True)  # (B, 1, H*W)

    # reshape 回 (B, 1, H, W)
    return cos_sim.view(B, 1, H, W)


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc=3):
        super().__init__()
        # self.param_free_norm = SyncBatchNorm.convert_sync_batchnorm(nn.BatchNorm2d(norm_nc, affine=True))
        # self.param_free_norm = nn.GroupNorm(32, norm_nc)
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=True)
        # self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=True)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        ks = 3
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        # segmap = segmap[str(x.size(-1))]
        normalized = self.param_free_norm(x)
        # normalized = x

        # Part 2. produce scaling and bias conditioned on semantic map
        # segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SpadeResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.spade1 = SPADE(norm_nc=dim)
        self.silu1 = nn.SiLU()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

        self.spade2 = SPADE(norm_nc=dim)
        self.silu2 = nn.SiLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, input):
        x, segmap = input[0], input[1]
        out = self.spade1(x, segmap)
        out = self.silu1(out)
        out = self.conv1(out)

        out = self.spade2(out, segmap)
        out = self.silu2(out)
        out = self.conv2(out)
        return out + x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.acti = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.acti(out)
        return out


class Resblock(nn.Module):
    def __init__(self, n_feat, kernel_size=3, stride=1, padding=1):
        super(Resblock, self).__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(
                in_channels=n_feat,
                out_channels=n_feat,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(
                in_channels=n_feat,
                out_channels=n_feat,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

    def forward(self, x):
        identity = x
        out = self.res_block(x)
        return out + identity


class Downsample(nn.Module):
    def __init__(self, in_channels, scale):
        super(Downsample, self).__init__()
        self.downsample = nn.Sequential(
            nn.PixelUnshuffle(scale),
            nn.Conv2d(
                in_channels=in_channels * scale * scale,
                out_channels=in_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        return self.downsample(x)


class SR_Encoder(pl.LightningModule):
    def __init__(self, out_channel=8, in_channel=3):
        super(SR_Encoder, self).__init__()

        self.first_layer_sr = nn.Conv2d(
            in_channels=in_channel, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # 256 256 32

        # (1) cnn encoder
        self.layer1_sr = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),  # 256 256 64
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1
            ),  ##128 128 64
            nn.LeakyReLU(),
        )

        self.layer2_sr = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),  # 128 128 128
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1
            ),  # 64 64 128
            nn.LeakyReLU(),
        )

        self.layer3_sr = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # 64 64 256
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1
            ),  # 32 32 256
            nn.LeakyReLU(),
        )

        # (3)out
        self.last_linear = nn.Conv2d(256, out_channel, 3, bias=False, padding=1)

    def forward(self, sr):
        # (1)cnn encoder
        # b 3 256 256 -> b 256 32 32
        sr_cond = self.first_layer_sr(sr)
        sr_cond = self.layer1_sr(sr_cond)
        sr_cond = self.layer2_sr(sr_cond)
        sr_cond = self.layer3_sr(sr_cond)
        out = self.last_linear(sr_cond)
        # 4 32 32
        return out


class LocalCrossAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.ln_ref = nn.LayerNorm(dim)
        self.ln_sr = nn.LayerNorm(dim)

    def forward(self, sr, ref, return_cos_sim_map=False, sim_lamuda=1):
        B, C, H, W = sr.shape
        assert (
            H % self.window_size == 0 and W % self.window_size == 0
        ), "H and W must be divisible by window_size"

        attn = (cosine_attention_map(sr, ref) + 1) / 2  # B 1 H W
        # Step 1: unfold to non-overlapping windows
        sr_windows = F.unfold(
            sr, kernel_size=self.window_size, stride=self.window_size
        )  # B, C*win*win, N_win
        ref_windows = F.unfold(
            ref, kernel_size=self.window_size, stride=self.window_size
        )

        # Now shape: B, C*win*win, Num_windows → reshape to B*N, win*win, C
        B, _, N = sr_windows.shape
        win_area = self.window_size * self.window_size

        sr_windows = (
            sr_windows.transpose(1, 2).reshape(B * N, C, win_area).permute(0, 2, 1)
        )  # [B*N, win_area, C]
        ref_windows = (
            ref_windows.transpose(1, 2).reshape(B * N, C, win_area).permute(0, 2, 1)
        )

        # Step 2: cross-attention: Q=sr, K/V=ref
        sr_windows = self.ln_sr(sr_windows)
        ref_windows = self.ln_ref(ref_windows)
        fused_windows, _ = self.attn(
            query=sr_windows, key=ref_windows, value=ref_windows
        )  # [B*N, win_area, C]

        # Step 3: reshape back
        fused_windows = (
            fused_windows.permute(0, 2, 1).reshape(B, N, C * win_area).transpose(1, 2)
        )  # B, C*win_area, N
        out = F.fold(
            fused_windows,
            output_size=(H, W),
            kernel_size=self.window_size,
            stride=self.window_size,
        )  # B, C, H, W

        if self.training and random.random() < 0.2:
            sim_lamuda = 0

        if isinstance(sim_lamuda, float):
            attn = (attn * sim_lamuda).clip(0, 1)
        elif isinstance(sim_lamuda, torch.Tensor):
            sim_lamuda = torch.nn.functional.interpolate(
                sim_lamuda.unsqueeze(0).unsqueeze(0),
                size=(attn.shape[2], attn.shape[3]),
            )
            attn = (attn * sim_lamuda).clip(0, 1)

        out = attn * out + (1 - attn) * sr

        if not return_cos_sim_map:
            return out
        else:
            return out, attn


class MaskAttention(nn.Module):
    def __init__(self, channels):
        super(MaskAttention, self).__init__()

        # 分别处理 sr 和 ref 的卷积模块（共享结构）
        self.sr_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.ref_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 拼接后生成注意力图
        self.attention = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, sr, ref, sim_lamuda=1, return_learned_sim_map=False):
        sr_feat = self.sr_conv(sr)  # B C H W
        ref_feat = self.ref_conv(ref)  # B C H W

        fused = torch.cat([sr_feat, ref_feat], dim=1)  # B 2C H W
        attn = self.attention(fused)  # B C H W, in [0, 1]

        # 加权融合
        if self.training and random.random() < 0.2:
            sim_lamuda = 0

        if isinstance(sim_lamuda, float):
            attn = (attn * sim_lamuda).clip(0, 1)
        elif isinstance(sim_lamuda, torch.Tensor):
            sim_lamuda = torch.nn.functional.interpolate(
                sim_lamuda.unsqueeze(0).unsqueeze(0),
                size=(attn.shape[2], attn.shape[3]),
            )
            attn = (attn * sim_lamuda).clip(0, 1)

        # attn = (attn ** sim_lamuda).clip(0, 1)
        output = attn * ref_feat + (1 - attn) * sr_feat  # B C H W
        if not return_learned_sim_map:
            return output
        else:
            return output, torch.mean(attn, dim=1, keepdim=True)


class SR_Ref_Encoder_LCA(pl.LightningModule):
    def __init__(self, out_channel=8, in_sr_channel=3, in_ref_channel=3):
        super(SR_Ref_Encoder_LCA, self).__init__()

        self.first_layer_sr = ConvBlock(
            in_channels=in_sr_channel,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )  # 256 256 64

        self.first_layer_ref = ConvBlock(
            in_channels=in_ref_channel,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )  # 256 256 64

        # (1) cnn encoder
        self.layer1_sr = nn.Sequential(
            Resblock(n_feat=32, kernel_size=3, stride=1, padding=1),  # 256 256 128
            ConvBlock(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),  # 256 256 128
            Downsample(in_channels=64, scale=2),  # 128 128 128
        )

        self.layer2_sr = nn.Sequential(
            Resblock(n_feat=64, kernel_size=3, stride=1, padding=1),  # 128 128 128
            ConvBlock(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),  # 128 128 128
            Downsample(in_channels=128, scale=2),  # 64 64 128
        )

        self.layer3_sr = nn.Sequential(
            Resblock(n_feat=128, kernel_size=3, stride=1, padding=1),  # 64 64 256
            ConvBlock(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # 64 64 256
            Downsample(in_channels=256, scale=2),  # 32 32 256
        )

        self.layer1_ref = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 256 256 64
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Downsample(in_channels=64, scale=2),  # 128 128 64
        )

        self.layer2_ref = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 128 128 128
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Downsample(in_channels=128, scale=2),  # 64 64 128
        )

        self.layer3_ref = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 64 64 256
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Downsample(in_channels=256, scale=2),  # 32 32 256
        )

        self.lca1 = LocalCrossAttention(64, 8, 4)
        self.lca2 = LocalCrossAttention(128, 4, 4)
        self.lca3 = LocalCrossAttention(256, 2, 4)

        self.mask_attn1 = MaskAttention(64)
        self.mask_attn2 = MaskAttention(128)
        self.mask_attn3 = MaskAttention(256)

        self.last_linear = nn.Conv2d(512, out_channel, 1, bias=False)

    def forward(
        self,
        sr,
        ref,
        return_cos_sim_map=False,
        return_learned_sim_map=False,
        sim_lamuda=1,
    ):
        sr_cond = self.first_layer_sr(sr)
        ref_cond = self.first_layer_ref(ref)

        ref_cond = self.layer1_ref(ref_cond)
        sr_cond = self.layer1_sr(sr_cond)

        if return_cos_sim_map:
            sr_cond1, cos_map1 = self.lca1(
                sr_cond,
                ref_cond,
                return_cos_sim_map=return_cos_sim_map,
                sim_lamuda=sim_lamuda,
            )
        else:
            sr_cond1 = self.lca1(
                sr_cond,
                ref_cond,
                return_cos_sim_map=return_cos_sim_map,
                sim_lamuda=sim_lamuda,
            )

        if return_learned_sim_map:
            sr_cond2, learned_map1 = self.mask_attn1(
                sr_cond,
                ref_cond,
                sim_lamuda=sim_lamuda,
                return_learned_sim_map=return_learned_sim_map,
            )
        else:
            sr_cond2 = self.mask_attn1(
                sr_cond,
                ref_cond,
                sim_lamuda=sim_lamuda,
                return_learned_sim_map=return_learned_sim_map,
            )
        sr_cond = sr_cond1 + sr_cond2

        ref_cond = self.layer2_ref(ref_cond)
        sr_cond = self.layer2_sr(sr_cond)

        if return_cos_sim_map:
            sr_cond1, cos_map2 = self.lca2(
                sr_cond,
                ref_cond,
                return_cos_sim_map=return_cos_sim_map,
                sim_lamuda=sim_lamuda,
            )
        else:
            sr_cond1 = self.lca2(
                sr_cond,
                ref_cond,
                return_cos_sim_map=return_cos_sim_map,
                sim_lamuda=sim_lamuda,
            )

        if return_learned_sim_map:
            sr_cond2, learned_map2 = self.mask_attn2(
                sr_cond,
                ref_cond,
                sim_lamuda=sim_lamuda,
                return_learned_sim_map=return_learned_sim_map,
            )
        else:
            sr_cond2 = self.mask_attn2(
                sr_cond,
                ref_cond,
                sim_lamuda=sim_lamuda,
                return_learned_sim_map=return_learned_sim_map,
            )
        sr_cond = sr_cond1 + sr_cond2

        ref_cond = self.layer3_ref(ref_cond)
        sr_cond = self.layer3_sr(sr_cond)

        if return_cos_sim_map:
            sr_cond1, cos_map3 = self.lca3(
                sr_cond,
                ref_cond,
                return_cos_sim_map=return_cos_sim_map,
                sim_lamuda=sim_lamuda,
            )
        else:
            sr_cond1 = self.lca3(
                sr_cond,
                ref_cond,
                return_cos_sim_map=return_cos_sim_map,
                sim_lamuda=sim_lamuda,
            )
        if return_learned_sim_map:
            sr_cond2, learned_map3 = self.mask_attn3(
                sr_cond,
                ref_cond,
                sim_lamuda=sim_lamuda,
                return_learned_sim_map=return_learned_sim_map,
            )
        else:
            sr_cond2 = self.mask_attn3(
                sr_cond,
                ref_cond,
                sim_lamuda=sim_lamuda,
                return_learned_sim_map=return_learned_sim_map,
            )
        sr_cond = torch.cat([sr_cond1, sr_cond2], dim=1)

        out = self.last_linear(sr_cond)

        if not return_cos_sim_map and not return_learned_sim_map:
            return out
        elif return_cos_sim_map:
            return out, [cos_map1, cos_map2, cos_map3]
        elif return_learned_sim_map:
            return out, [learned_map1, learned_map2, learned_map3]


class ImplicitPromptModule(nn.Module):

    def __init__(
        self,
        image_feat_dim=1280,
        proj_dim=1024,
        num_queries=256,
        embed_dim=1024,
        num_heads=8,
    ):
        super().__init__()

        # Projector (MLP)
        self.projector = nn.Sequential(
            nn.Linear(image_feat_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, embed_dim),
        )

        # Learnable Queries
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim))

        # Transformer-style layers
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, image_feat, sim_lamuda=None):
        # 2. 通过 MLP projector 映射到目标维度
        vis_feat = self.projector(image_feat)  # [B, N, proj_dim]

        # 3. Learnable Queries
        B = image_feat.size(0)
        queries = self.queries.unsqueeze(0).expand(
            B, -1, -1
        )  # [B, num_queries, embed_dim]

        # 5. Cross-Attention: queries attend to visual features
        q_norm = self.layernorm(queries)
        vis_norm = self.layernorm(vis_feat)

        if isinstance(sim_lamuda, float):
            factor = sim_lamuda
            q_ca, _ = self.cross_attn(q_norm, vis_norm, vis_norm)
            queries = queries + factor * q_ca  # residual
        elif isinstance(sim_lamuda, torch.Tensor):
            mask = torch.nn.functional.interpolate(
                sim_lamuda.unsqueeze(0).unsqueeze(0),
                size=(
                    int(math.sqrt(vis_feat.shape[1])),
                    int(math.sqrt(vis_feat.shape[1])),
                ),
            )
            mask = mask.reshape(1, -1).repeat(queries.shape[1], 1)  # [num_queries, N]
            attn_mask = mask.masked_fill(mask == 0, float("-inf"))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
            q_ca, _ = self.cross_attn(q_norm, vis_norm, vis_norm, attn_mask=attn_mask)
            queries = queries + q_ca  # residual
        else:
            q_ca, _ = self.cross_attn(q_norm, vis_norm, vis_norm)
            queries = queries + q_ca  # residual

        # 6. Feed-Forward Network
        q_norm = self.layernorm(queries)
        out = queries + self.ffn(q_norm)  # residual

        # 输出的隐式文本特征：F_txt_imp
        return out  # shape: [B, num_queries, embed_dim]


class SR_Ref_Encoder_Spade(nn.Module):
    """
    SPADE 融合编码器 — 用于 Spade_Adapter。

    结构: LR 和 Ref 分别经独立卷积编码后，
          在每个尺度上用 SPADE 归一化调制将 Ref 信息注入 LR 特征。

    对应 adapters.py 中 Spade_Adapter 的 merge_encoder。
    """

    def __init__(self, out_channel=192, in_ref_channel=3):
        super().__init__()

        # LR 分支: 与 SR_Encoder 类似的 CNN 编码链
        self.first_layer_sr = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer1_sr = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
        )  # H/2
        self.layer2_sr = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
        )  # H/4
        self.layer3_sr = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
        )  # H/8

        # Ref 分支: 独立轻量编码
        self.first_layer_ref = nn.Conv2d(in_ref_channel, 32, kernel_size=3, padding=1)
        self.layer1_ref = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
        )
        self.layer2_ref = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
        )
        self.layer3_ref = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
        )

        # 三个尺度的 SPADE 融合层
        self.spade_fuse1 = SPADE(norm_nc=64, label_nc=64)  # 尺度1: 64ch
        self.spade_conv1 = nn.Conv2d(64, 64, 3, padding=1)

        self.spade_fuse2 = SPADE(norm_nc=128, label_nc=128)  # 尺度2: 128ch
        self.spade_conv2 = nn.Conv2d(128, 128, 3, padding=1)

        self.spade_fuse3 = SPADE(norm_nc=256, label_nc=256)  # 尺度3: 256ch
        self.spade_conv3 = nn.Conv2d(256, 256, 3, padding=1)

        # 输出投影
        self.last_linear = nn.Conv2d(256 * 2, out_channel, 3, padding=1)

    def forward(self, sr, ref, **kwargs):
        # LR 分支编码
        sr_feat = self.first_layer_sr(sr)
        sr_feat = self.layer1_sr(sr_feat)
        sr_64 = sr_feat
        sr_feat = self.layer2_sr(sr_feat)
        sr_128 = sr_feat
        sr_feat = self.layer3_sr(sr_feat)
        sr_256 = sr_feat

        # Ref 分支编码
        ref_feat = self.first_layer_ref(ref)
        ref_feat = self.layer1_ref(ref_feat)
        ref_64 = ref_feat
        ref_feat = self.layer2_ref(ref_feat)
        ref_128 = ref_feat
        ref_feat = self.layer3_ref(ref_feat)
        ref_256 = ref_feat

        # 三尺度 SPADE 融合: 用 Ref 特征调制 LR 特征
        fuse1 = self.spade_fuse1(sr_64, ref_64)
        fuse1 = torch.relu(fuse1)
        fuse1 = self.spade_conv1(fuse1)

        fuse2 = self.spade_fuse2(sr_128, ref_128)
        fuse2 = torch.relu(fuse2)
        fuse2 = self.spade_conv2(fuse2)

        fuse3 = self.spade_fuse3(sr_256, ref_256)
        fuse3 = torch.relu(fuse3)
        fuse3 = self.spade_conv3(fuse3)

        # 拼接最后两个尺度 + 投影输出
        out = torch.cat([fuse2, fuse3], dim=1)
        out = self.last_linear(out)
        return out


class SR_Ref_Encoder_Cos_Sim(nn.Module):
    """
    余弦相似度加权融合编码器 — 用于 Cos_Sim_Adapter。

    结构: LR 和 Ref 共享一个拼接输入的编码器，
          在每个尺度上用余弦相似度加权融合两个分支的特征。

    与 Dual_Adapter 的区别: 这里是在编码器内部做余弦加权，
    而 Dual 是在两个完全独立的编码器之后才做加权。
    """

    def __init__(self, out_channel=192, in_channel=6):
        super().__init__()

        # 共享编码器 (输入为 concat(LR, Ref) = 6ch)
        self.first_layer = nn.Conv2d(in_channel, 32, kernel_size=3, padding=1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
        )
        self.last_linear = nn.Conv2d(256, out_channel, 3, padding=1)

    def forward(self, sr, ref, sim_lamuda=None, **kwargs):
        # 拼接 LR 和 Ref 为 6 通道输入
        x = torch.cat([sr, ref], dim=1)

        x = self.first_layer(x)
        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x

        out = self.last_linear(f3)
        return out
