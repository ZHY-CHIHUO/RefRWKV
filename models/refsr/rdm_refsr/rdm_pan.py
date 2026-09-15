"""Dedicated MS+PAN pansharpening network.

This is not a gated RefSR leftover.  The contract follows FusionMamba:
upsample MS onto the PAN grid, keep two equal streams, fuse them with RWKV
(spatial / PAN) and Mamba (spectral / MS), then add a residual onto the
upsampled MS.  There is no matcher, reliability field, or change prior.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rdm_refsr import (
    FourDirectionMamba,
    RMSNorm2d,
    SharedDirectionalRWKV,
    normalize_reference_kind,
)


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


class SpectralGate(nn.Module):
    """Channel-wise gate from LR MS, analogue of FusionMamba SpeAttention.

    An optional fused-feature branch starts at zero, so init is still a
    global 0.5 scale; later training can make the gate spatial.
    """

    def __init__(self, ms_channels: int, hidden: int, spatial_dim: int | None = None) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.logit = nn.Sequential(
            nn.Conv2d(ms_channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, ms_channels, 1),
        )
        nn.init.zeros_(self.logit[-1].bias)
        self.spatial = None
        if spatial_dim is not None:
            self.spatial = nn.Conv2d(int(spatial_dim), ms_channels, 1, bias=True)
            nn.init.zeros_(self.spatial.weight)
            nn.init.zeros_(self.spatial.bias)

    def forward(self, lrms: torch.Tensor, fused: torch.Tensor | None = None) -> torch.Tensor:
        logit = self.logit(self.pool(lrms))
        if self.spatial is not None and fused is not None:
            logit = logit + self.spatial(fused)
        return torch.sigmoid(logit)


class DualStreamFusion(nn.Module):
    """Shared RWKV+Mamba dual-stream unit.

    Stream A (``pan`` / fine / spatial) is RWKV; stream B (``ms`` / coarse /
    spectral) is Mamba.  Cross projections start at 0.  If ``reliability`` is
    given, only A→B is gated; B→A is never gated.  PAN never passes it.
    """

    def __init__(
        self,
        dim: int,
        *,
        final: bool = False,
        allow_cpu_mamba: bool = True,
        mamba_d_state: int = 8,
        mamba_d_conv: int = 4,
        mamba_expand: int = 1,
        shuffle_prob: float = 0.0,
        shuffle_block: int = 4,
    ) -> None:
        super().__init__()
        self.final = bool(final)
        self.norm_pan = RMSNorm2d(dim)
        self.norm_ms = RMSNorm2d(dim)
        self.norm_pan_cross = RMSNorm2d(dim)
        self.norm_ms_cross = RMSNorm2d(dim)
        self.pan_scan = SharedDirectionalRWKV(
            dim, shuffle_prob=float(shuffle_prob), shuffle_block=int(shuffle_block), high_order=False
        )
        self.ms_scan = FourDirectionMamba(
            dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            allow_cpu_reference=allow_cpu_mamba,
        )
        self.pan_from_ms = nn.Conv2d(dim, dim, 1, bias=False)
        self.ms_from_pan = nn.Conv2d(dim, dim, 1, bias=False)
        self.out_proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.pan_from_ms.weight)
        nn.init.zeros_(self.ms_from_pan.weight)

    def forward(
        self,
        pan: torch.Tensor,
        ms: torch.Tensor,
        reliability: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        pan = pan + self.pan_scan(self.norm_pan(pan))
        ms = ms + self.ms_scan(self.norm_ms(ms))
        spa = pan + self.pan_scan(self.norm_pan_cross(pan + self.pan_from_ms(ms)))
        injected = pan
        if reliability is not None:
            if reliability.ndim != 4 or reliability.shape[0] != pan.shape[0]:
                raise ValueError(
                    f"reliability must be NCHW with batch {pan.shape[0]}, got {tuple(reliability.shape)}"
                )
            if tuple(reliability.shape[-2:]) != tuple(pan.shape[-2:]):
                reliability = F.interpolate(reliability, size=tuple(pan.shape[-2:]), mode="area")
            injected = pan * reliability.to(dtype=pan.dtype)
        spe = ms + self.ms_scan(self.norm_ms_cross(ms + self.ms_from_pan(injected)))
        fused = self._fuse(spa, spe)
        if self.final:
            return fused
        return (spa + fused) * 0.5, (spe + fused) * 0.5

    def _fuse(self, spa: torch.Tensor, spe: torch.Tensor) -> torch.Tensor:
        return self.out_proj((spa + spe) * 0.5)


class PanMsFusion(DualStreamFusion):
    """Pansharpening unit: same weights, no reliability argument.

    ``adaptive_final`` replaces the last 50/50 average with a spatial softmax
    mix of the PAN and MS streams.  Zero-init logits keep the 50/50 start.
    """

    def __init__(self, dim: int, *, adaptive_final: bool = False, **kwargs: Any) -> None:
        super().__init__(dim, **kwargs)
        self.adaptive_final = bool(adaptive_final) and bool(self.final)
        if self.adaptive_final:
            self.mix_logit = nn.Conv2d(dim * 2, 2, 1, bias=True)
            nn.init.zeros_(self.mix_logit.weight)
            nn.init.zeros_(self.mix_logit.bias)

    def forward(self, pan: torch.Tensor, ms: torch.Tensor):  # type: ignore[override]
        return super().forward(pan, ms, reliability=None)

    def _fuse(self, spa: torch.Tensor, spe: torch.Tensor) -> torch.Tensor:
        if not self.adaptive_final:
            return super()._fuse(spa, spe)
        weight = torch.softmax(self.mix_logit(torch.cat((spa, spe), dim=1)), dim=1)
        mixed = spa * weight[:, :1] + spe * weight[:, 1:2]
        return self.out_proj(mixed)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 2, 2, 0),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2, 0),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = _align(self.up(x), skip.shape[-2:])
        return self.mix(x + skip)


class RDMPan(nn.Module):
    """MS LR + co-registered PAN HR -> MS HR.

    FusionMamba-style U2 body on the HR grid:

    ``out = to_hrms(mix(PAN, MS_up)) * SpeGate(MS, fused) + bicubic(MS)``

    The last mix is a spatial softmax of the two streams; it starts at 50/50.

    PAN is a first-class stream, never a reliability-gated reference residual.
    """

    def __init__(
        self,
        inp_channels: int = 8,
        out_channels: int | None = None,
        ref_channels: int = 1,
        dim: int = 32,
        scale: int = 4,
        mamba_d_state: int = 8,
        mamba_d_conv: int = 4,
        mamba_expand: int = 1,
        allow_cpu_mamba: bool = True,
        clamp_output: bool = True,
        use_reference: bool = True,
        reference_kind: str | None = None,
        shuffle_prob: float = 0.0,
        shuffle_block: int = 4,
        spectral_drop_bands: int = 0,
        highpass_pan: bool = False,
        **unused: Any,
    ) -> None:
        super().__init__()
        leftover = {
            key: value for key, value in unused.items() if key not in _IGNORED_KEYS
        }
        if leftover:
            raise TypeError(f"RDMPan got unexpected arguments: {sorted(leftover)}")
        if not bool(use_reference):
            raise ValueError(
                "RDMPan is a pansharpening model and requires the PAN reference"
            )
        if reference_kind is not None and normalize_reference_kind(reference_kind) != "pan":
            raise ValueError("RDMPan is the pansharpening model; use rdm_stf or rdm_mhf")
        inp_channels = _positive_int(inp_channels, "inp_channels")
        ref_channels = _positive_int(ref_channels, "ref_channels")
        if out_channels is None:
            out_channels = inp_channels
        out_channels = _positive_int(out_channels, "out_channels")
        if ref_channels != 1:
            raise ValueError(
                f"RDMPan expects a 1-channel PAN, got ref_channels={ref_channels}"
            )
        if out_channels != inp_channels:
            raise ValueError("RDMPan out_channels must match inp_channels (MS bands)")
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
        self.reference_kind = "pan"
        self.use_reference = True
        self.clamp_output = bool(clamp_output)
        self.spectral_drop_bands = int(spectral_drop_bands)
        self.highpass_pan = bool(highpass_pan)
        self.channel_multipliers = (1, 2, 4)
        self.high_order = False

        self.raise_pan = nn.Sequential(
            nn.Conv2d(ref_channels, dim0, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.raise_ms = nn.Sequential(
            nn.Conv2d(inp_channels, dim0, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        mamba_kwargs = dict(
            allow_cpu_mamba=bool(allow_cpu_mamba),
            mamba_d_state=_positive_int(mamba_d_state, "mamba_d_state"),
            mamba_d_conv=_positive_int(mamba_d_conv, "mamba_d_conv"),
            mamba_expand=_positive_int(mamba_expand, "mamba_expand"),
            shuffle_prob=float(shuffle_prob),
            shuffle_block=_positive_int(shuffle_block, "shuffle_block"),
        )
        self.stage0 = PanMsFusion(dim0, **mamba_kwargs)
        self.down_pan0 = Downsample(dim0, dim1)
        self.down_ms0 = Downsample(dim0, dim1)
        self.stage1 = PanMsFusion(dim1, **mamba_kwargs)
        self.down_pan1 = Downsample(dim1, dim2)
        self.down_ms1 = Downsample(dim1, dim2)
        self.stage2 = PanMsFusion(dim2, **mamba_kwargs)
        self.up_pan2 = Upsample(dim2, dim1)
        self.up_ms2 = Upsample(dim2, dim1)
        self.stage3 = PanMsFusion(dim1, **mamba_kwargs)
        self.up_pan3 = Upsample(dim1, dim0)
        self.up_ms3 = Upsample(dim1, dim0)
        self.stage4 = PanMsFusion(dim0, final=True, adaptive_final=True, **mamba_kwargs)
        self.spe_gate = SpectralGate(inp_channels, dim0, spatial_dim=dim0)
        self.to_hrms = nn.Sequential(
            nn.Conv2d(dim0, dim0, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim0, out_channels, 3, 1, 1),
        )
        nn.init.uniform_(self.to_hrms[-1].weight, -1.0e-4, 1.0e-4)
        nn.init.zeros_(self.to_hrms[-1].bias)

    def _drop_spectrum(self, value: torch.Tensor) -> torch.Tensor:
        channels = int(value.shape[1])
        if not self.training or self.spectral_drop_bands <= 0 or channels < 2:
            return value
        count = min(int(self.spectral_drop_bands), max(1, channels // 4), channels - 1)
        if count < 1:
            return value
        scores = torch.rand(value.shape[0], channels, device=value.device)
        _, index = scores.topk(count, dim=1, largest=False)
        mask = torch.ones(value.shape[0], channels, device=value.device, dtype=value.dtype)
        mask.scatter_(1, index, 0.0)
        return value * mask.view(value.shape[0], channels, 1, 1)

    def forward(
        self,
        lr: torch.Tensor,
        ref: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if lr.ndim != 4 or lr.shape[1] != self.inp_channels:
            raise ValueError(
                f"lr must be NCHW with {self.inp_channels} channels, got {tuple(lr.shape)}"
            )
        if ref is None or ref.ndim != 4:
            raise ValueError("RDMPan requires a 4-D PAN tensor")
        output_size = (int(lr.shape[-2]) * self.scale, int(lr.shape[-1]) * self.scale)
        if ref.shape[0] != lr.shape[0] or ref.shape[1] != self.ref_channels:
            raise ValueError(
                f"lr/ref batch or channels mismatch: lr={tuple(lr.shape)}, ref={tuple(ref.shape)}"
            )
        if tuple(ref.shape[-2:]) != output_size:
            raise ValueError(
                f"PAN spatial size must be LR*x{self.scale}: {tuple(ref.shape[-2:])} vs {output_size}"
            )

        skip = F.interpolate(lr, size=output_size, mode="bicubic", align_corners=False)
        pan_in = ref
        if self.highpass_pan:
            pan_low = F.interpolate(ref, size=lr.shape[-2:], mode="area")
            pan_low = F.interpolate(pan_low, size=output_size, mode="bicubic", align_corners=False)
            pan_in = ref - pan_low
        pan = self.raise_pan(pan_in)
        ms_in = self._drop_spectrum(skip)
        ms = self.raise_ms(ms_in)

        pan, ms = self.stage0(pan, ms)
        pan_skip0, ms_skip0 = pan, ms
        pan, ms = self.down_pan0(pan), self.down_ms0(ms)

        pan, ms = self.stage1(pan, ms)
        pan_skip1, ms_skip1 = pan, ms
        pan, ms = self.down_pan1(pan), self.down_ms1(ms)

        pan, ms = self.stage2(pan, ms)
        pan, ms = self.up_pan2(pan, pan_skip1), self.up_ms2(ms, ms_skip1)

        pan, ms = self.stage3(pan, ms)
        pan, ms = self.up_pan3(pan, pan_skip0), self.up_ms3(ms, ms_skip0)

        fused = self.stage4(pan, ms)
        output = skip + self.to_hrms(fused) * self.spe_gate(lr, fused)
        if self.clamp_output:
            output = output.clamp(0.0, 1.0)
        if return_aux:
            return output, {}
        return output
