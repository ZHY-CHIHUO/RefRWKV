"""Dedicated same-band spatio-temporal fusion network.

This is not the old gated dual-grid RefSR.  The contract is classic STF:

    ``(C0, F0, C1) -> F1``

Python call is ``forward(lr, ref, c0)`` with ``lr=C1``, ``ref=F0``, ``c0=C0``.
Wuhan supplies all three on one 4-band grid (``scale=1``, ``[0, 1]``) and
the trainer must pass ``C0``.  HRMS may omit ``c0``; the network then
synthesizes it from F0.  F0 is scanned with RWKV, (C0, C1) with Mamba, and
``ΔC = C1 - C0`` gates how much F0 detail is added onto the C1 skip.
Matcher / Haar / reliability leftovers stay out.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rdm_pan import Downsample, PanMsFusion, Upsample
from .rdm_refsr import normalize_reference_kind


_IGNORED_KEYS = {
    "depths",
    "decoder_depths",
    "channel_multipliers",
    "mamba_stages",
    "reference_condition_stages",
    "detail_injection_stages",
    "high_order",
    "shuffle_prob",
    "shuffle_block",
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


class ChangeGate(nn.Module):
    """Per-pixel gate from the coarse temporal difference ``C1 - C0``."""

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
    """Same-band STF: ``out = skip(C1) + trunk(F0, C0, C1) * ChangeGate(ΔC)``."""

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

        self.raise_fine = nn.Sequential(
            nn.Conv2d(ref_channels, dim0, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.raise_coarse = nn.Sequential(
            nn.Conv2d(inp_channels * 2, dim0, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        mamba_kwargs = dict(
            allow_cpu_mamba=bool(allow_cpu_mamba),
            mamba_d_state=_positive_int(mamba_d_state, "mamba_d_state"),
            mamba_d_conv=_positive_int(mamba_d_conv, "mamba_d_conv"),
            mamba_expand=_positive_int(mamba_expand, "mamba_expand"),
        )
        self.stage0 = PanMsFusion(dim0, **mamba_kwargs)
        self.down_fine0 = Downsample(dim0, dim1)
        self.down_coarse0 = Downsample(dim0, dim1)
        self.stage1 = PanMsFusion(dim1, **mamba_kwargs)
        self.down_fine1 = Downsample(dim1, dim2)
        self.down_coarse1 = Downsample(dim1, dim2)
        self.stage2 = PanMsFusion(dim2, **mamba_kwargs)
        self.up_fine2 = Upsample(dim2, dim1)
        self.up_coarse2 = Upsample(dim2, dim1)
        self.stage3 = PanMsFusion(dim1, **mamba_kwargs)
        self.up_fine3 = Upsample(dim1, dim0)
        self.up_coarse3 = Upsample(dim1, dim0)
        self.stage4 = PanMsFusion(dim0, final=True, **mamba_kwargs)
        self.change_gate = ChangeGate(inp_channels, dim0)
        self.to_hr = nn.Sequential(
            nn.Conv2d(dim0, dim0, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim0, out_channels, 3, 1, 1),
        )
        nn.init.uniform_(self.to_hr[-1].weight, -1.0e-4, 1.0e-4)
        nn.init.zeros_(self.to_hr[-1].bias)

    def _coarse_pair(
        self, lr: torch.Tensor, ref: torch.Tensor, c0: torch.Tensor | None, output_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        skip = _align(lr, output_size)
        if c0 is None:
            low = F.interpolate(ref, size=tuple(lr.shape[-2:]), mode="area")
            coarse_0 = _align(low, output_size)
        else:
            if c0.ndim != 4 or c0.shape[0] != lr.shape[0] or c0.shape[1] != self.inp_channels:
                raise ValueError(
                    f"c0 must be NCHW with {self.inp_channels} channels, got {tuple(c0.shape)}"
                )
            coarse_0 = _align(c0, output_size)
        return skip, coarse_0

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

        skip, coarse_0 = self._coarse_pair(lr, ref, c0, output_size)
        delta = skip - coarse_0
        fine = self.raise_fine(ref)
        coarse = self.raise_coarse(torch.cat((skip, coarse_0), dim=1))

        fine, coarse = self.stage0(fine, coarse)
        fine_skip0, coarse_skip0 = fine, coarse
        fine, coarse = self.down_fine0(fine), self.down_coarse0(coarse)

        fine, coarse = self.stage1(fine, coarse)
        fine_skip1, coarse_skip1 = fine, coarse
        fine, coarse = self.down_fine1(fine), self.down_coarse1(coarse)

        fine, coarse = self.stage2(fine, coarse)
        fine, coarse = self.up_fine2(fine, fine_skip1), self.up_coarse2(coarse, coarse_skip1)

        fine, coarse = self.stage3(fine, coarse)
        fine, coarse = self.up_fine3(fine, fine_skip0), self.up_coarse3(coarse, coarse_skip0)

        fused = self.stage4(fine, coarse)
        output = skip + self.to_hr(fused) * self.change_gate(delta)
        if self.clamp_output:
            output = output.clamp(0.0, 1.0)
        if return_aux:
            return output, {"delta": delta, "gate": self.change_gate(delta)}
        return output
