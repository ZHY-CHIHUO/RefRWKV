"""Portable reference-SR comparison baselines.

The native RefSR contract in this repository is intentionally narrow:
``forward(lr, ref) -> sr``, where all tensors use RGB ``[-1, 1]`` values and
the reference/output resolution is ``lr * scale``.  TTSR, MASA-SR and DATSR
were released with substantially different and, in DATSR's case, legacy CUDA
dependency stacks.  The networks below keep the distinguishing *algorithmic
families* runnable in the main environment:

* TTSR: low-resolution correspondence followed by gated high-resolution
  texture transfer;
* MASA-SR: coarse matching plus learned spatial adaptation of reference
  details;
* DATSR: local correspondence, offset-based alignment and restoration.

They are registered as ``native_compatibility`` implementations.  They are
not checkpoint-compatible copies of their official repositories; the exact
upstream environments are documented in ``environments/`` and
``docs/models/baselines.md``.
"""

from __future__ import annotations

import math

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


def _check_rgb_pair(lr: torch.Tensor, ref: torch.Tensor, scale: int, name: str) -> tuple[int, int]:
    if not torch.is_tensor(lr) or not torch.is_tensor(ref):
        raise TypeError(f"{name} expects tensor lr/ref inputs")
    if lr.ndim != 4 or ref.ndim != 4 or lr.shape[1] != 3 or ref.shape[1] != 3:
        raise ValueError(f"{name} expects RGB NCHW lr/ref tensors, got {tuple(lr.shape)} and {tuple(ref.shape)}")
    if lr.shape[0] != ref.shape[0]:
        raise ValueError(f"{name} LR/Ref batch sizes differ: {lr.shape[0]} vs {ref.shape[0]}")
    expected = (int(lr.shape[-2]) * int(scale), int(lr.shape[-1]) * int(scale))
    if tuple(ref.shape[-2:]) != expected:
        raise ValueError(f"{name} Ref must be LR * x{scale}: {tuple(ref.shape[-2:])} vs {expected}")
    return expected


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class _Upsample(nn.Sequential):
    """Pixel-shuffle upsampler for every positive integer scale."""

    def __init__(self, scale: int, channels: int) -> None:
        layers: list[nn.Module] = []
        for factor in _scale_factors(scale):
            layers.extend(
                (
                    nn.Conv2d(channels, channels * factor * factor, 3, padding=1),
                    nn.PixelShuffle(factor),
                    nn.GELU(),
                )
            )
        super().__init__(*layers)


class _LocalCorrespondence(nn.Module):
    """Memory-bounded differentiable local feature matching.

    Matching happens only at LR resolution.  Keeping ``match_dim`` and
    ``transfer_dim`` compact makes the comparison profiles practical on the
    same hardware as SwinIR while preserving differentiable local patch
    selection.  ``return_confidence`` exposes the maximum correspondence
    probability for safe high-resolution texture fusion.
    """

    def __init__(
        self,
        channels: int,
        window_size: int = 7,
        *,
        match_dim: int | None = None,
        transfer_dim: int | None = None,
    ) -> None:
        super().__init__()
        if window_size < 3 or window_size % 2 != 1:
            raise ValueError("match_window must be an odd integer >= 3")
        if channels < 8:
            raise ValueError("channels must be >= 8")
        self.window_size = int(window_size)
        self.match_dim = int(match_dim or min(16, channels))
        self.transfer_dim = int(transfer_dim or min(16, channels))
        if self.match_dim < 1 or self.transfer_dim < 1:
            raise ValueError("match_dim and transfer_dim must be positive")
        self.query = nn.Conv2d(channels, self.match_dim, 1)
        self.key = nn.Conv2d(channels, self.match_dim, 1)
        self.value = nn.Conv2d(channels, self.transfer_dim, 1)
        self.out = nn.Conv2d(self.transfer_dim, channels, 1)
        # A learnable positive multiplier is more stable than an unrestricted
        # temperature when the projected query/key vectors are L2 normalized.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(8.0)))

    def forward(
        self,
        lr_feature: torch.Tensor,
        ref_feature: torch.Tensor,
        *,
        return_confidence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if lr_feature.shape[0] != ref_feature.shape[0]:
            raise ValueError("local correspondence requires equal batch sizes")
        if lr_feature.shape[-2:] != ref_feature.shape[-2:]:
            ref_feature = F.interpolate(ref_feature, size=lr_feature.shape[-2:], mode="bilinear", align_corners=False)
        query = F.normalize(self.query(lr_feature), dim=1, eps=1.0e-6)
        key = F.normalize(self.key(ref_feature), dim=1, eps=1.0e-6)
        value = self.value(ref_feature)
        batch, _channels, height, width = query.shape
        locations = height * width
        neighborhood = self.window_size * self.window_size
        padding = self.window_size // 2
        key_patches = F.unfold(key, self.window_size, padding=padding).reshape(
            batch, self.match_dim, neighborhood, locations
        )
        logits = (query.reshape(batch, self.match_dim, 1, locations) * key_patches).sum(dim=1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        weights = torch.softmax(logits * scale, dim=1)
        value_patches = F.unfold(value, self.window_size, padding=padding).reshape(
            batch, self.transfer_dim, neighborhood, locations
        )
        matched = (value_patches * weights.unsqueeze(1)).sum(dim=2).reshape(
            batch, self.transfer_dim, height, width
        )
        matched = self.out(matched)
        if not return_confidence:
            return matched
        confidence = weights.amax(dim=1).reshape(batch, 1, height, width)
        return matched, confidence


class _FlowWarp(nn.Module):
    """Grid-sample fallback for DCNv2-style feature aggregation."""

    def __init__(self, channels: int, max_offset: float = 2.0) -> None:
        super().__init__()
        if max_offset <= 0:
            raise ValueError("max_offset must be positive")
        self.max_offset = float(max_offset)
        self.offset = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 2, 3, padding=1),
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.shape[0] != target.shape[0] or source.shape[1] != target.shape[1]:
            raise ValueError("flow warp source/target must have equal batch and channel dimensions")
        if source.shape[-2:] != target.shape[-2:]:
            source = F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)
        batch, _channels, height, width = target.shape
        offsets = torch.tanh(self.offset(torch.cat((target, source), dim=1))) * self.max_offset
        y = torch.linspace(-1.0, 1.0, height, device=target.device, dtype=target.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=target.device, dtype=target.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        base = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        # align_corners=True maps a one-pixel offset to 2/(size - 1).
        offset_x = offsets[:, 0] * (2.0 / max(1, width - 1))
        offset_y = offsets[:, 1] * (2.0 / max(1, height - 1))
        grid = base + torch.stack((offset_x, offset_y), dim=-1)
        return F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=True)


class _ReferenceGate(nn.Module):
    """Fuse an aligned reference feature only where it is useful."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        target: torch.Tensor,
        reference: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if reference.shape[-2:] != target.shape[-2:]:
            reference = F.interpolate(reference, size=target.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.gate(torch.cat((target, reference), dim=1))
        if confidence is not None:
            if confidence.shape[-2:] != target.shape[-2:]:
                confidence = F.interpolate(confidence, size=target.shape[-2:], mode="bilinear", align_corners=False)
            gate = gate * confidence.clamp(0.0, 1.0)
        return target + gate * reference


class _RefSRCompatibilityBase(nn.Module):
    def __init__(self, scale: int, dim: int) -> None:
        super().__init__()
        if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
            raise ValueError("scale must be a positive integer")
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 8:
            raise ValueError("dim must be an integer >= 8")
        self.scale = int(scale)
        self.dim = int(dim)

    def _finish(self, lr: torch.Tensor, feature: torch.Tensor, tail: nn.Module) -> torch.Tensor:
        expected = (lr.shape[-2] * self.scale, lr.shape[-1] * self.scale)
        residual = tail(feature)
        if tuple(residual.shape[-2:]) != expected:
            residual = F.interpolate(residual, size=expected, mode="bilinear", align_corners=False)
        base = F.interpolate(lr, size=expected, mode="bicubic", align_corners=False)
        return (base + residual).clamp(-1.0, 1.0)


class TTSRCompatibilityNet(_RefSRCompatibilityBase):
    """TTSR-family texture-transfer baseline runnable without VGG downloads."""

    def __init__(
        self,
        *,
        scale: int = 4,
        dim: int = 64,
        num_blocks: int = 8,
        texture_blocks: int = 3,
        match_window: int = 7,
        match_dim: int = 16,
    ) -> None:
        super().__init__(scale, dim)
        if num_blocks < 1 or texture_blocks < 1:
            raise ValueError("num_blocks and texture_blocks must be positive")
        self.lr_head = nn.Sequential(nn.Conv2d(3, dim, 3, padding=1), nn.GELU())
        self.ref_low_head = nn.Sequential(nn.Conv2d(3, dim, 3, padding=1), nn.GELU())
        self.correspondence = _LocalCorrespondence(dim, match_window, match_dim=match_dim)
        self.low_fuse = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 3, padding=1))
        self.low_body = nn.Sequential(*(_ResidualBlock(dim) for _ in range(int(num_blocks))))
        self.upsample = _Upsample(self.scale, dim)
        self.texture_encoder = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1),
            nn.GELU(),
            *(_ResidualBlock(dim) for _ in range(int(texture_blocks))),
        )
        self.texture_project = nn.Conv2d(dim, dim, 1)
        self.texture_gate = _ReferenceGate(dim)
        self.hr_body = nn.Sequential(_ResidualBlock(dim), _ResidualBlock(dim))
        self.tail = nn.Conv2d(dim, 3, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, lr: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        _check_rgb_pair(lr, ref, self.scale, "TTSR")
        lr_feature = self.lr_head(lr)
        ref_low = F.interpolate(ref, size=lr.shape[-2:], mode="bicubic", align_corners=False)
        ref_feature = self.ref_low_head(ref_low)
        matched, confidence = self.correspondence(lr_feature, ref_feature, return_confidence=True)
        feature = self.low_fuse(torch.cat((lr_feature, matched), dim=1))
        feature = feature + self.low_body(feature)
        feature = self.upsample(feature)
        texture = self.texture_project(self.texture_encoder(ref))
        feature = self.texture_gate(feature, texture, confidence)
        feature = feature + self.hr_body(feature)
        return self._finish(lr, feature, self.tail)


class MASACompatibilityNet(_RefSRCompatibilityBase):
    """MASA-SR-family matching acceleration and spatial-adaptation baseline."""

    def __init__(
        self,
        *,
        scale: int = 4,
        dim: int = 64,
        num_blocks: int = 8,
        refine_blocks: int = 4,
        match_window: int = 7,
        match_dim: int = 16,
        max_offset: float = 2.0,
    ) -> None:
        super().__init__(scale, dim)
        if num_blocks < 1 or refine_blocks < 1:
            raise ValueError("num_blocks and refine_blocks must be positive")
        self.lr_head = nn.Sequential(nn.Conv2d(3, dim, 3, padding=1), nn.GELU())
        self.ref_low_head = nn.Sequential(nn.Conv2d(3, dim, 3, padding=1), nn.GELU())
        self.coarse_match = _LocalCorrespondence(dim, match_window, match_dim=match_dim)
        self.coarse_fuse = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 3, padding=1))
        self.low_body = nn.Sequential(*(_ResidualBlock(dim) for _ in range(int(num_blocks))))
        self.upsample = _Upsample(self.scale, dim)
        self.ref_detail = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1),
            nn.GELU(),
            _ResidualBlock(dim),
        )
        self.spatial_adaptor = _FlowWarp(dim, max_offset=max_offset)
        self.detail_gate = _ReferenceGate(dim)
        self.refine = nn.Sequential(*(_ResidualBlock(dim) for _ in range(int(refine_blocks))))
        self.tail = nn.Conv2d(dim, 3, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, lr: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        _check_rgb_pair(lr, ref, self.scale, "MASA-SR")
        lr_feature = self.lr_head(lr)
        ref_low = F.interpolate(ref, size=lr.shape[-2:], mode="bicubic", align_corners=False)
        ref_feature = self.ref_low_head(ref_low)
        matched, confidence = self.coarse_match(lr_feature, ref_feature, return_confidence=True)
        feature = self.coarse_fuse(torch.cat((lr_feature, matched), dim=1))
        feature = feature + self.low_body(feature)
        feature = self.upsample(feature)
        detail = self.ref_detail(ref)
        aligned = self.spatial_adaptor(detail, feature)
        feature = self.detail_gate(feature, aligned, confidence)
        feature = feature + self.refine(feature)
        return self._finish(lr, feature, self.tail)


class DATSRCompatibilityNet(_RefSRCompatibilityBase):
    """DATSR-family deformable-attention baseline without MMCV/DCNv2.

    The official generator uses legacy MMCV and a compiled DCNv2 extension.
    This native model replaces that dependency with local correspondence and a
    differentiable offset field built from ``grid_sample``.
    """

    def __init__(
        self,
        *,
        scale: int = 4,
        dim: int = 64,
        num_blocks: int = 10,
        match_window: int = 7,
        match_dim: int = 16,
        max_offset: float = 2.0,
    ) -> None:
        super().__init__(scale, dim)
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        self.head = nn.Sequential(nn.Conv2d(3, dim, 3, padding=1), nn.GELU())
        self.ref_head = nn.Sequential(nn.Conv2d(3, dim, 3, padding=1), nn.GELU())
        self.correspondence = _LocalCorrespondence(dim, match_window, match_dim=match_dim)
        self.low_warp = _FlowWarp(dim, max_offset=max_offset)
        self.low_fuse = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        self.body = nn.Sequential(*(_ResidualBlock(dim) for _ in range(int(num_blocks))))
        self.upsample = _Upsample(self.scale, dim)
        self.ref_detail = nn.Sequential(nn.Conv2d(3, dim, 3, padding=1), nn.GELU(), _ResidualBlock(dim))
        self.detail_warp = _FlowWarp(dim, max_offset=max_offset)
        self.detail_gate = _ReferenceGate(dim)
        self.tail = nn.Conv2d(dim, 3, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, lr: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        _check_rgb_pair(lr, ref, self.scale, "DATSR")
        lr_feature = self.head(lr)
        ref_low = F.interpolate(ref, size=lr.shape[-2:], mode="bicubic", align_corners=False)
        ref_feature = self.ref_head(ref_low)
        matched, confidence = self.correspondence(lr_feature, ref_feature, return_confidence=True)
        aligned = self.low_warp(matched, lr_feature)
        feature = self.low_fuse(torch.cat((lr_feature, matched, aligned), dim=1))
        feature = feature + self.body(feature)
        feature = self.upsample(feature)
        detail = self.detail_warp(self.ref_detail(ref), feature)
        feature = self.detail_gate(feature, detail, confidence)
        return self._finish(lr, feature, self.tail)


__all__ = ["DATSRCompatibilityNet", "MASACompatibilityNet", "TTSRCompatibilityNet"]
