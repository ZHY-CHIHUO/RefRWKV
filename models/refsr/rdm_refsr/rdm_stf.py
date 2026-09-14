"""Dedicated same-band spatio-temporal fusion network.

Classic STF is ``(C0, F0, C1) -> F1``.  Python call is ``forward(lr, ref, c0)``
with ``lr=C1``, ``ref=F0``, ``c0=C0``.  ``c0`` is optional:

- Wuhan / full STF: pass ``C0`` (``batch[lr_t1]``).  Mamba sees C1+C0, the
  change gate uses true ``ΔC = C1 - C0``.
- Two-input (HRMS, or ``use_c0=false``): ``forward(C1, F0)``.  Mamba sees
  only C1; F0 stays on the RWKV stream.  The gate uses ``C1 - lowpass(F0)``
  as a mismatch prior, not a fake temporal C0 concatenated into Mamba.

F0 is scanned with RWKV, C1 (and C0 when present) with Mamba.  Training-only
regularizers: RWKV block shuffle, and random spectral dropout on the coarse
stream.  ChangeGate both scales the C1 residual and, as a 1-channel
reliability map, gates F0→Mamba inside ``StfFusion``.  C1→F0 is never gated.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rdm_pan import Downsample, DualStreamFusion, Upsample
from .rdm_refsr import normalize_reference_kind


_IGNORED_KEYS = {
    "depths",
    "decoder_depths",
    "channel_multipliers",
    "mamba_stages",
    "reference_condition_stages",
    "detail_injection_stages",
    "high_order",
    "match_window",
    "match_dim",
    "match_grid",
    "temporal_match_confidence_floor",
    "alignment",
    "max_offset",
    "response_matrix",
    "sensor_response",
    "target_channels",
    "variant",
    "family",
    "implementation",
    "name",
}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _align(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(value.shape[-2:]) == size:
        return value
    return F.interpolate(value, size=size, mode="bilinear", align_corners=False)


class StfFusion(DualStreamFusion):
    """STF dual-stream unit: F0/RWKV and C1/Mamba, F0→C1 reliability-gated."""

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor, reliability: torch.Tensor):
        if reliability is None:
            raise ValueError("StfFusion requires a reliability map")
        return super().forward(fine, coarse, reliability)


class ChangeGate(nn.Module):
    """Per-pixel / per-band residual gate from ``C1-C0`` or ``C1-lowpass(F0)``.

    The same tensor, averaged over bands, is the fusion reliability map.
    """

    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 3, 1, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.net(delta)


class RDMStf(nn.Module):
    """Same-band STF: ``out = skip(C1) + trunk(F0, C1[, C0]) * ChangeGate(Δ)``."""

    def __init__(
        self,
        inp_channels: int = 4,
        out_channels: int | None = None,
        ref_channels: int | None = None,
        dim: int = 32,
        scale: int = 1,
        mamba_d_state: int = 8,
        mamba_d_conv: int = 4,
        mamba_expand: int = 1,
        allow_cpu_mamba: bool = True,
        clamp_output: bool = True,
        use_reference: bool = True,
        reference_kind: str | None = None,
        shuffle_prob: float = 0.15,
        shuffle_block: int = 4,
        spectral_drop_bands: int = 1,
        **unused: Any,
    ) -> None:
        super().__init__()
        leftover = {key: value for key, value in unused.items() if key not in _IGNORED_KEYS}
        if leftover:
            raise TypeError(f"RDMStf got unexpected arguments: {sorted(leftover)}")
        if not bool(use_reference):
            raise ValueError("RDMStf requires the fine reference F0")
        if reference_kind is not None and normalize_reference_kind(reference_kind) != "stf":
            raise ValueError("RDMStf is the spatio-temporal model; use rdm_pan or rdm_mhf")
        inp_channels = _positive_int(inp_channels, "inp_channels")
        if ref_channels is None:
            ref_channels = inp_channels
        ref_channels = _positive_int(ref_channels, "ref_channels")
        if out_channels is None:
            out_channels = inp_channels
        out_channels = _positive_int(out_channels, "out_channels")
        if ref_channels != inp_channels:
            raise ValueError(
                "RDMStf expects same-band LR/Ref "
                f"(inp_channels={inp_channels}, ref_channels={ref_channels})"
            )
        if out_channels != inp_channels:
            raise ValueError("RDMStf out_channels must match inp_channels")
        dim = _positive_int(dim, "dim")
        if dim < 16 or dim % 16:
            raise ValueError("dim must be a multiple of 16 and >= 16 for the WKV kernel")
        scale = _positive_int(scale, "scale")
        if not 0.0 <= float(shuffle_prob) <= 1.0:
            raise ValueError("shuffle_prob must be in [0, 1]")
        if isinstance(spectral_drop_bands, bool) or not isinstance(spectral_drop_bands, int) or spectral_drop_bands < 0:
            raise ValueError("spectral_drop_bands must be a non-negative integer")
        dim0 = dim
        dim1 = dim * 2
        dim2 = dim * 4
        for width in (dim0, dim1, dim2):
            if width < 16 or width % 16:
                raise ValueError(f"stage width must be a multiple of 16, got {width}")

        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.ref_channels = ref_channels
        self.dim = dim
        self.scale = scale
        self.reference_kind = "stf"
        self.use_reference = True
        self.clamp_output = bool(clamp_output)
        self.channel_multipliers = (1, 2, 4)
        self.high_order = False
        self.shuffle_prob = float(shuffle_prob)
        self.shuffle_block = _positive_int(shuffle_block, "shuffle_block")
        self.spectral_drop_bands = int(spectral_drop_bands)

        self.raise_fine = nn.Sequential(
            nn.Conv2d(ref_channels, dim0, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.raise_c1 = nn.Sequential(
            nn.Conv2d(inp_channels, dim0, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.raise_c0 = nn.Sequential(
            nn.Conv2d(inp_channels, dim0, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        nn.init.zeros_(self.raise_c0[0].weight)
        nn.init.zeros_(self.raise_c0[0].bias)
        fusion_kwargs = dict(
            allow_cpu_mamba=bool(allow_cpu_mamba),
            mamba_d_state=_positive_int(mamba_d_state, "mamba_d_state"),
            mamba_d_conv=_positive_int(mamba_d_conv, "mamba_d_conv"),
            mamba_expand=_positive_int(mamba_expand, "mamba_expand"),
            shuffle_prob=self.shuffle_prob,
            shuffle_block=self.shuffle_block,
        )
        self.stage0 = StfFusion(dim0, **fusion_kwargs)
        self.down_fine0 = Downsample(dim0, dim1)
        self.down_coarse0 = Downsample(dim0, dim1)
        self.stage1 = StfFusion(dim1, **fusion_kwargs)
        self.down_fine1 = Downsample(dim1, dim2)
        self.down_coarse1 = Downsample(dim1, dim2)
        self.stage2 = StfFusion(dim2, **fusion_kwargs)
        self.up_fine2 = Upsample(dim2, dim1)
        self.up_coarse2 = Upsample(dim2, dim1)
        self.stage3 = StfFusion(dim1, **fusion_kwargs)
        self.up_fine3 = Upsample(dim1, dim0)
        self.up_coarse3 = Upsample(dim1, dim0)
        self.stage4 = StfFusion(dim0, final=True, **fusion_kwargs)
        self.change_gate = ChangeGate(inp_channels, dim0)
        self.to_hr = nn.Sequential(
            nn.Conv2d(dim0, dim0, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim0, out_channels, 3, 1, 1),
        )
        nn.init.uniform_(self.to_hr[-1].weight, -1.0e-4, 1.0e-4)
        nn.init.zeros_(self.to_hr[-1].bias)

    def _spectral_keep_mask(self, value: torch.Tensor) -> torch.Tensor | None:
        channels = int(value.shape[1])
        if not self.training or self.spectral_drop_bands <= 0 or channels < 2:
            return None
        count = min(int(self.spectral_drop_bands), max(1, channels // 4), channels - 1)
        if count < 1:
            return None
        scores = torch.rand(value.shape[0], channels, device=value.device)
        _, index = scores.topk(count, dim=1, largest=False)
        mask = torch.ones(value.shape[0], channels, device=value.device, dtype=value.dtype)
        mask.scatter_(1, index, 0.0)
        return mask.view(value.shape[0], channels, 1, 1)

    def _drop_spectrum(self, value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = self._spectral_keep_mask(value)
        if mask is None:
            return value
        return value * mask

    def _coarse_inputs(
        self,
        lr: torch.Tensor,
        ref: torch.Tensor,
        c0: torch.Tensor | None,
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        skip = _align(lr, output_size)
        if c0 is None:
            low = F.interpolate(ref, size=tuple(lr.shape[-2:]), mode="area")
            prior = _align(low, output_size)
            return skip, skip - prior, None
        if c0.ndim != 4 or c0.shape[0] != lr.shape[0] or c0.shape[1] != self.inp_channels:
            raise ValueError(
                f"c0 must be NCHW with {self.inp_channels} channels, got {tuple(c0.shape)}"
            )
        coarse_0 = _align(c0, output_size)
        return skip, skip - coarse_0, coarse_0

    def forward(
        self,
        lr: torch.Tensor,
        ref: torch.Tensor | None = None,
        c0: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if lr.ndim != 4 or lr.shape[1] != self.inp_channels:
            raise ValueError(
                f"lr/C1 must be NCHW with {self.inp_channels} channels, got {tuple(lr.shape)}"
            )
        if ref is None or ref.ndim != 4:
            raise ValueError("RDMStf requires a 4-D F0 tensor")
        output_size = (int(lr.shape[-2]) * self.scale, int(lr.shape[-1]) * self.scale)
        if ref.shape[0] != lr.shape[0] or ref.shape[1] != self.ref_channels:
            raise ValueError(
                f"lr/ref batch or channels mismatch: lr={tuple(lr.shape)}, ref={tuple(ref.shape)}"
            )
        if tuple(ref.shape[-2:]) != output_size:
            raise ValueError(
                f"F0 spatial size must be LR*x{self.scale}: {tuple(ref.shape[-2:])} vs {output_size}"
            )

        skip, delta, coarse_0 = self._coarse_inputs(lr, ref, c0, output_size)
        gate = self.change_gate(delta)
        reliability = gate.mean(dim=1, keepdim=True)
        spectral_mask = self._spectral_keep_mask(skip)
        query = self._drop_spectrum(skip, spectral_mask)
        fine = self.raise_fine(ref)
        coarse = self.raise_c1(query)
        if coarse_0 is not None:
            coarse = coarse + self.raise_c0(self._drop_spectrum(coarse_0, spectral_mask))

        fine, coarse = self.stage0(fine, coarse, reliability)
        fine_skip0, coarse_skip0 = fine, coarse
        fine, coarse = self.down_fine0(fine), self.down_coarse0(coarse)

        fine, coarse = self.stage1(fine, coarse, reliability)
        fine_skip1, coarse_skip1 = fine, coarse
        fine, coarse = self.down_fine1(fine), self.down_coarse1(coarse)

        fine, coarse = self.stage2(fine, coarse, reliability)
        fine, coarse = self.up_fine2(fine, fine_skip1), self.up_coarse2(coarse, coarse_skip1)

        fine, coarse = self.stage3(fine, coarse, reliability)
        fine, coarse = self.up_fine3(fine, fine_skip0), self.up_coarse3(coarse, coarse_skip0)

        fused = self.stage4(fine, coarse, reliability)
        output = skip + self.to_hr(fused) * gate
        if self.clamp_output:
            output = output.clamp(0.0, 1.0)
        if return_aux:
            return output, {
                "delta": delta,
                "gate": gate,
                "reliability": reliability,
                "used_c0": coarse_0 is not None,
            }
        return output
