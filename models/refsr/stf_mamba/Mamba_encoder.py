import torch
import torch.nn as nn
from .vss_blocks import dual_vss_block

class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=6, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, H//2, W//2, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x.permute(0, 3, 1, 2)

class Encoder(nn.Module):
    def __init__(self, channel_first=False, norm_layer="LN", ssm_act_layer="silu", mlp_act_layer="gelu", **kwargs):
        super(Encoder, self).__init__()

        in_chans = int(kwargs.get("in_chans", 4))
        embed_dim = kwargs.get("dims", 96)
        if isinstance(embed_dim, (list, tuple)):
            embed_dim = embed_dim[0] if embed_dim else 96
        vss = dict(
            channel_first=channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            kwargs=kwargs,
        )
        self.patch_emb = PatchEmbed2D(in_chans=in_chans, embed_dim=int(embed_dim))
        self.downsample_1 = PatchMerging2D(96)
        self.st_block_1 = nn.ModuleList([dual_vss_block(96, 32, **vss) for _ in range(2)])
        self.downsample_2 = PatchMerging2D(192)
        self.st_block_2 = nn.ModuleList([dual_vss_block(192, 16, **vss) for _ in range(2)])
        self.downsample_3 = PatchMerging2D(192 * 2)
        self.st_block_3 = nn.ModuleList([dual_vss_block(192 * 2, 8, **vss) for _ in range(2)])
        self.st_block_4 = nn.ModuleList([dual_vss_block(192 * 4, 4, **vss) for _ in range(1)])

    def forward(self, x):
        output = []
        x = self.patch_emb(x)
        for index, module in enumerate(self.st_block_1):
            x = module(x)
        output.append(x)
        x = self.downsample_1(x)
        for index, module in enumerate(self.st_block_2):
            x = module(x)
        output.append(x)
        x = self.downsample_2(x)
        for index, module in enumerate(self.st_block_3):
            x = module(x)
        output.append(x)
        x = self.downsample_3(x)
        for index, module in enumerate(self.st_block_4):
            x = module(x)
        output.append(x)
        return output
