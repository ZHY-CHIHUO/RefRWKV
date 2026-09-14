"""Cross-attention fusion of the two STFMamba decoder hypotheses.

Upstream hard-codes 6-band embeddings and a 128x128 canvas.  Wuhan is 4-band
and is evaluated by 128 tiles, so both the channel count and the spatial size
are taken from the tensors instead of literals.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def split_image_tensor(input_tensor: torch.Tensor, block_size: int) -> torch.Tensor:
    batch_size, channels, height, width = input_tensor.shape
    if height % block_size or width % block_size:
        raise ValueError(
            f"STFMamba cross-attention needs H/W divisible by {block_size}, "
            f"got {height}x{width}; use data.eval_tile_size=128 on Wuhan"
        )
    unfolded = input_tensor.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
    unfolded = unfolded.permute(0, 2, 3, 1, 4, 5).contiguous()
    return unfolded.view(-1, channels, block_size, block_size)


def combine_image_tensor(
    input_tensor: torch.Tensor,
    block_size: int,
    original_height: int,
    original_width: int,
) -> torch.Tensor:
    num_blocks_height = original_height // block_size
    num_blocks_width = original_width // block_size
    batch_size = input_tensor.size(0) // (num_blocks_height * num_blocks_width)
    channels = input_tensor.size(1)
    input_tensor = input_tensor.view(
        batch_size, num_blocks_height, num_blocks_width, channels, block_size, block_size
    )
    input_tensor = input_tensor.permute(0, 3, 1, 4, 2, 5).contiguous()
    return input_tensor.view(batch_size, channels, original_height, original_width)


class Cross_MultiAttention(nn.Module):
    def __init__(self, in_channels: int, emb_dim: int = 16, num_heads: int = 4, block_size: int = 16):
        super().__init__()
        self.in_channels = int(in_channels)
        self.emb_dim = int(emb_dim)
        self.num_heads = int(num_heads)
        self.block_size = int(block_size)
        self.scale = emb_dim ** -0.5
        if self.emb_dim % self.num_heads:
            raise ValueError("emb_dim must be divisible by num_heads")
        self.depth = self.emb_dim // self.num_heads
        self.embedding = nn.Linear(self.in_channels, self.emb_dim)
        self.embedding_2 = nn.Linear(self.in_channels * 2, self.emb_dim)
        self.Wq = nn.Linear(self.emb_dim, self.emb_dim)
        self.Wk = nn.Linear(self.emb_dim, self.emb_dim)
        self.Wv = nn.Linear(self.emb_dim, self.emb_dim)
        self.proj_out = nn.Conv2d(self.emb_dim * 2, self.in_channels, kernel_size=1)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor, pad_mask=None) -> torch.Tensor:
        _, _, height, width = img1.shape
        img1 = split_image_tensor(img1, self.block_size)
        img2 = split_image_tensor(img2, self.block_size)
        batch, channels, block_h, block_w = img1.shape
        img1_flat = img1.view(batch, channels, block_h * block_w).permute(0, 2, 1)
        img2_flat = img2.view(batch, channels, block_h * block_w).permute(0, 2, 1)
        img_cat = torch.cat((img1_flat, img2_flat), dim=2)
        img1_emb = self.embedding(img1_flat)
        img2_emb = self.embedding(img2_flat)
        img_emb = self.embedding_2(img_cat)

        query1 = self.Wq(img1_emb)
        query2 = self.Wq(img2_emb)
        key = self.Wk(img_emb)
        value = self.Wv(img_emb)

        query1 = query1.view(batch, -1, self.num_heads, self.depth).transpose(1, 2)
        query2 = query2.view(batch, -1, self.num_heads, self.depth).transpose(1, 2)
        key = key.view(batch, -1, self.num_heads, self.depth).transpose(1, 2)
        value = value.view(batch, -1, self.num_heads, self.depth).transpose(1, 2)

        weights = torch.einsum("bnid,bnjd -> bnij", query1, key) * self.scale
        out = torch.einsum("bnij, bnjd -> bnid", F.softmax(weights, dim=-1), value)
        weights2 = torch.einsum("bnid,bnjd -> bnij", query2, key) * self.scale
        out2 = torch.einsum("bnij, bnjd -> bnid", F.softmax(weights2, dim=-1), value)

        out = out.transpose(1, 2).contiguous().view(batch, -1, self.emb_dim)
        out = out.permute(0, 2, 1).view(batch, self.emb_dim, block_h, block_w)
        out2 = out2.transpose(1, 2).contiguous().view(batch, -1, self.emb_dim)
        out2 = out2.permute(0, 2, 1).view(batch, self.emb_dim, block_h, block_w)
        fused = self.proj_out(torch.cat((out, out2), dim=1))
        return combine_image_tensor(fused, self.block_size, height, width)
