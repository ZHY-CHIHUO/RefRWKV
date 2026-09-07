"""Dependency-light SR baseline networks.

The project keeps the data/optimizer/evaluation contract independent from a
particular paper implementation.  The classes in this module are compact,
faithful *architecture families* for the classical baselines (EDSR, RCAN,
HAT, and MambaIRv2).  They deliberately use only PyTorch so that a baseline
can be trained in the repository's ``rwkv7`` environment.  The adapters expose
the same ``[-1, 1]`` tensor contract as SwinIR.

For exact paper numbers, use the official bridge environments documented in
``docs/models/baselines.md`` and load the corresponding upstream checkpoint.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scale_factors(scale: int) -> tuple[int, ...]:
    if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
        raise ValueError(f"scale must be a positive integer, got {scale!r}")
    factors: list[int] = []
    remainder = int(scale)
    for prime in (2, 3):
        while remainder % prime == 0:
            factors.append(prime)
            remainder //= prime
    if remainder > 1:
        factors.append(remainder)
    return tuple(factors)


class Upsample(nn.Sequential):
    """Pixel-shuffle upsampler supporting every integer scale."""

    def __init__(self, scale: int, channels: int) -> None:
        layers: list[nn.Module] = []
        for factor in _scale_factors(scale):
            layers.extend(
                [
                    nn.Conv2d(channels, channels * factor * factor, 3, 1, 1),
                    nn.PixelShuffle(factor),
                    nn.ReLU(inplace=True),
                ]
            )
        super().__init__(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, res_scale: float = 1.0) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.res_scale = float(res_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // int(reduction))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.body(self.pool(x))


class RCAB(nn.Module):
    """Residual channel-attention block used by RCAN."""

    def __init__(self, channels: int, reduction: int = 16, res_scale: float = 1.0) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            ChannelAttention(channels, reduction),
        )
        self.res_scale = float(res_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class ResidualGroup(nn.Module):
    def __init__(self, channels: int, blocks: int, reduction: int = 16) -> None:
        super().__init__()
        self.body = nn.Sequential(
            *(RCAB(channels, reduction) for _ in range(int(blocks))),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class HybridAttentionBlock(nn.Module):
    """A local hybrid attention block for the HAT compatibility baseline.

    HAT combines window self-attention and channel attention.  A depthwise
    local branch plus channel gating gives the same inductive bias without a
    dependency on BasicSR/timm and remains valid for arbitrary image sizes
    (including tiny smoke-test tensors).
    """

    def __init__(self, channels: int, reduction: int = 16, window_size: int = 16) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.norm = nn.GroupNorm(1, channels)
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.channel = ChannelAttention(channels, reduction)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )
        self.gamma = nn.Parameter(torch.ones(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.local(y)
        y = self.channel(y)
        x = x + self.gamma[0] * y
        return x + self.gamma[1] * self.ffn(self.norm(x))


class MambaLikeBlock(nn.Module):
    """Portable approximation of the MambaIRv2 local state-space block.

    The official model uses ``mamba_ssm`` selective scan.  This fallback keeps
    the same gated, directional/local processing shape and can be replaced by
    the upstream implementation in the optional bridge environment.
    """

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = max(channels, int(channels * expansion))
        self.norm = nn.GroupNorm(1, channels)
        self.in_proj = nn.Conv2d(channels, hidden * 2, 1)
        self.state = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.out_proj = nn.Conv2d(hidden, channels, 1)
        self.gate = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        value, gate = self.in_proj(z).chunk(2, dim=1)
        value = F.silu(self.state(value))
        value = self.out_proj(value)
        return x + value * torch.sigmoid(self.gate(z) + gate.mean(dim=1, keepdim=True))


class _SRBase(nn.Module):
    def __init__(self, scale: int, channels: int, out_channels: int = 3) -> None:
        super().__init__()
        self.scale = int(scale)
        self.channels = int(channels)
        self.out_channels = int(out_channels)
        self.head = nn.Conv2d(out_channels, channels, 3, 1, 1)
        self.tail = nn.Conv2d(channels, out_channels, 3, 1, 1)
        self.upsample = Upsample(self.scale, channels)

    def _finish(self, lr: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        residual = self.tail(self.upsample(features))
        base = F.interpolate(
            lr, scale_factor=self.scale, mode="bicubic", align_corners=False, recompute_scale_factor=False
        )
        if residual.shape[-2:] != base.shape[-2:]:
            residual = F.interpolate(residual, size=base.shape[-2:], mode="bilinear", align_corners=False)
        return (base + residual).clamp(-1.0, 1.0)

    @staticmethod
    def _check_input(lr: torch.Tensor, channels: int = 3) -> None:
        if not torch.is_tensor(lr) or lr.ndim != 4 or lr.shape[1] != channels:
            raise ValueError(f"SR expects NCHW RGB input, got {getattr(lr, 'shape', None)}")


class EDSRNet(_SRBase):
    def __init__(
        self,
        *,
        scale: int = 4,
        dim: int = 64,
        num_blocks: int = 16,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__(scale, dim)
        self.body = nn.Sequential(
            *(ResidualBlock(dim, res_scale=res_scale) for _ in range(int(num_blocks))),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        self._check_input(lr)
        features = self.head(lr)
        features = features + self.body(features)
        return self._finish(lr, features)


class RCANNet(_SRBase):
    def __init__(
        self,
        *,
        scale: int = 4,
        dim: int = 64,
        num_groups: int = 10,
        blocks_per_group: int = 20,
        reduction: int = 16,
    ) -> None:
        super().__init__(scale, dim)
        self.body = nn.Sequential(
            *(ResidualGroup(dim, blocks_per_group, reduction) for _ in range(int(num_groups))),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        self._check_input(lr)
        features = self.head(lr)
        features = features + self.body(features)
        return self._finish(lr, features)


class HATNet(_SRBase):
    def __init__(
        self,
        *,
        scale: int = 4,
        dim: int = 96,
        num_blocks: int = 12,
        reduction: int = 16,
        window_size: int = 16,
    ) -> None:
        super().__init__(scale, dim)
        self.body = nn.Sequential(
            *(HybridAttentionBlock(dim, reduction, window_size) for _ in range(int(num_blocks))),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        self._check_input(lr)
        features = self.head(lr)
        features = features + self.body(features)
        return self._finish(lr, features)


class MambaIRv2Net(_SRBase):
    def __init__(
        self,
        *,
        scale: int = 4,
        dim: int = 64,
        num_blocks: int = 12,
        expansion: int = 2,
    ) -> None:
        super().__init__(scale, dim)
        self.body = nn.Sequential(
            *(MambaLikeBlock(dim, expansion) for _ in range(int(num_blocks))),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        self._check_input(lr)
        features = self.head(lr)
        features = features + self.body(features)
        return self._finish(lr, features)


class BicubicSR(nn.Module):
    """Reference-only, parameter-free bicubic baseline."""

    trainable = False

    def __init__(self, *, scale: int = 4) -> None:
        super().__init__()
        self.scale = int(scale)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(lr) or lr.ndim != 4 or lr.shape[1] != 3:
            raise ValueError(f"Bicubic expects NCHW RGB input, got {getattr(lr, 'shape', None)}")
        return F.interpolate(
            lr, scale_factor=self.scale, mode="bicubic", align_corners=False, recompute_scale_factor=False
        ).clamp(-1.0, 1.0)


__all__ = [
    "BicubicSR",
    "EDSRNet",
    "HATNet",
    "MambaIRv2Net",
    "RCANNet",
]
