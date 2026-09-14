"""Reliability-guided dual-grid RWKV/Mamba reference super-resolution.

This module deliberately has a different fusion contract from the historical
``RefSRWKV`` implementation.  The query (LR) stream owns the reconstructed
low-frequency signal.  A high-resolution reference is decomposed with a fixed
Haar transform and can enter the query stream only as a reliability-weighted
high-frequency residual.

The CUDA path uses the official ``mamba_ssm`` selective scan and the shared
project Bi-WKV kernel.  A small differentiable selective-scan/WKV reference
implementation is kept for CPU shape and gradient tests; it is never used on
CUDA and is not a convolutional approximation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels.wkv import OmniShift, RUN_CUDA

try:  # The dependency is installed in the project's rwkv7 environment.
    from mamba_ssm import Mamba as _OfficialMamba
except Exception as _MAMBA_IMPORT_ERROR:  # pragma: no cover - environment dependent
    _OfficialMamba = None
else:
    _MAMBA_IMPORT_ERROR = None


_REF_MODE_ALIASES = {
    "pan": "pan",
    "pansharpening": "pan",
    "pan_guided": "pan",
    "stf": "stf",
    "temporal": "stf",
    "cross_temporal": "stf",
    "hrms": "stf",
    "mhf": "mhf",
    "hsi_msi": "mhf",
    # Keep one canonical branch for both spellings used in HSI/MSI papers.
    "msi_hsi": "mhf",
    "hyperspectral_multispectral": "mhf",
    "generic": "generic",
    "aligned": "generic",
}
_CANONICAL_REFERENCE_KINDS = ("pan", "stf", "mhf", "generic")

# GPU uses the original Vision-RWKV 32-way bidirectional kernel for any T.
# CPU tests keep the exclusive recurrence in ``_reference_biwkv``.


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _odd_int(value: Any, name: str, minimum: int = 3) -> int:
    value = _positive_int(value, name)
    if value < minimum or value % 2 == 0:
        raise ValueError(f"{name} must be an odd integer >= {minimum}, got {value!r}")
    return value


def normalize_reference_kind(value: Any) -> str:
    normalized = str(value).strip().lower()
    try:
        return _REF_MODE_ALIASES[normalized]
    except KeyError as exc:
        options = ", ".join(_CANONICAL_REFERENCE_KINDS)
        raise ValueError(
            f"reference kind must be one of {options}, got {value!r}"
        ) from exc


class RMSNorm2d(nn.Module):
    """Per-pixel RMS normalization that preserves channel/DC information."""

    def __init__(self, channels: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        channels = _positive_int(channels, "channels")
        if not math.isfinite(float(eps)) or eps <= 0:
            raise ValueError("eps must be positive and finite")
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.weight.numel():
            raise ValueError(
                f"RMSNorm2d expects NCHW with {self.weight.numel()} channels, got {tuple(x.shape)}"
            )
        value = x.float()
        value = value * value.square().mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        value = value * self.weight.float().view(1, -1, 1, 1)
        return value.to(dtype=x.dtype)


def haar_dwt2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Fixed orthonormal 2-D Haar transform.

    The last row/column is replicate-padded for odd dimensions.  The original
    spatial size is returned so that inverse synthesis can crop exactly.
    """

    if x.ndim != 4:
        raise ValueError(f"haar_dwt2d expects NCHW, got {tuple(x.shape)}")
    h, w = int(x.shape[-2]), int(x.shape[-1])
    if h < 1 or w < 1:
        raise ValueError("haar_dwt2d input must have positive spatial dimensions")
    pad_h, pad_w = h % 2, w % 2
    padded = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate") if (pad_h or pad_w) else x
    x00 = padded[..., 0::2, 0::2]
    x01 = padded[..., 0::2, 1::2]
    x10 = padded[..., 1::2, 0::2]
    x11 = padded[..., 1::2, 1::2]
    scale = 0.5
    ll = (x00 + x01 + x10 + x11) * scale
    lh = (x00 - x01 + x10 - x11) * scale
    hl = (x00 + x01 - x10 - x11) * scale
    hh = (x00 - x01 - x10 + x11) * scale
    return ll, torch.cat((lh, hl, hh), dim=1), (h, w)


def haar_idwt2d(
    low: torch.Tensor,
    detail: torch.Tensor,
    output_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Inverse of :func:`haar_dwt2d` for concatenated LH/HL/HH details."""

    if low.ndim != 4 or detail.ndim != 4 or detail.shape[1] != low.shape[1] * 3:
        raise ValueError(
            "haar_idwt2d expects low=B,C,h,w and detail=B,3C,h,w, "
            f"got {tuple(low.shape)} and {tuple(detail.shape)}"
        )
    lh, hl, hh = detail.chunk(3, dim=1)
    a = (low + lh + hl + hh) * 0.5
    b = (low - lh + hl - hh) * 0.5
    c = (low + lh - hl - hh) * 0.5
    d = (low - lh - hl + hh) * 0.5
    batch, channels, height, width = low.shape
    rows = torch.stack((a, b), dim=-1).reshape(batch, channels, height, width * 2)
    rows2 = torch.stack((c, d), dim=-1).reshape(batch, channels, height, width * 2)
    output = torch.stack((rows, rows2), dim=-2).reshape(
        batch, channels, height * 2, width * 2
    )
    if output_size is not None:
        output = output[..., : int(output_size[0]), : int(output_size[1])]
    return output


def _gray_gradient(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Sobel x/y and magnitude for a tensor averaged over channels."""

    gray = x.float().mean(dim=1, keepdim=True)
    kx = gray.new_tensor(((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))).view(
        1, 1, 3, 3
    )
    ky = gray.new_tensor(((-1.0, -2.0, -1.0), (0.0, 0.0, 0.0), (1.0, 2.0, 1.0))).view(
        1, 1, 3, 3
    )
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    # The reliability CNN may be running in bf16/fp16 under Lightning's mixed
    # precision autocast.  Convert the float32 Sobel result back to the input
    # dtype before concatenating it with learned feature tensors.
    magnitude = torch.sqrt(gx.square() + gy.square() + 1.0e-6)
    return gx.to(dtype=x.dtype), gy.to(dtype=x.dtype), magnitude.to(dtype=x.dtype)


def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _translate_no_wrap(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Translate a feature map with replicate-free zero boundary handling."""

    h, w = x.shape[-2:]
    pad_l, pad_r = max(dx, 0), max(-dx, 0)
    pad_t, pad_b = max(dy, 0), max(-dy, 0)
    padded = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
    y0 = max(-dy, 0)
    x0 = max(-dx, 0)
    return padded[..., y0 : y0 + h, x0 : x0 + w]


class QShift(nn.Module):
    """Non-circular four-way shift used at the coefficient-resolution head."""

    def __init__(self, channels: int, shift: int = 1) -> None:
        super().__init__()
        self.channels = _positive_int(channels, "channels")
        self.shift = _positive_int(shift, "shift")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"QShift expects {self.channels} NCHW channels, got {tuple(x.shape)}"
            )
        chunks = torch.tensor_split(x, 4, dim=1)
        shifted = (
            _translate_no_wrap(chunks[0], 0, 0),
            _translate_no_wrap(chunks[1], self.shift, 0),
            _translate_no_wrap(chunks[2], 0, self.shift),
            _translate_no_wrap(chunks[3], self.shift, self.shift),
        )
        return torch.cat(shifted, dim=1)


def _block_shuffle(
    seq: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Shuffle contiguous blocks while returning an inverse index."""

    length = int(seq.shape[1])
    if length <= block_size:
        return None
    blocks = [
        torch.arange(start, min(start + block_size, length), device=seq.device)
        for start in range(0, length, block_size)
    ]
    order = torch.randperm(len(blocks), device=seq.device)
    indices = torch.cat([blocks[int(index)] for index in order.tolist()])
    inverse = torch.argsort(indices)
    return seq.index_select(1, indices), inverse


def _reference_biwkv(
    decay: torch.Tensor,
    first: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Differentiable O(T) reference implementation of the Bi-WKV recurrence.

    ``decay`` is the positive distance penalty ``w`` used by the CUDA kernel
    (the kernel evaluates ``exp(-w * distance)``).  Keeping that convention
    explicit is important: passing the learnable softplus parameter itself
    here would make the CPU and CUDA paths disagree by one exponential.
    """

    batch, length, channels = key.shape
    if length < 1:
        raise ValueError("Bi-WKV requires at least one token")
    # Keep fp32 accumulation for half/bfloat16, while preserving float64 for
    # numerical checks and high-precision research runs.
    work_dtype = torch.float64 if key.dtype == torch.float64 else torch.float32
    decay_factor = torch.exp(-decay.to(work_dtype)).view(1, channels)
    key_exp = torch.exp(key.to(work_dtype).clamp(-20.0, 20.0))
    value_float = value.to(work_dtype)
    past_num: list[torch.Tensor] = []
    past_den: list[torch.Tensor] = []
    num = value_float.new_zeros(batch, channels)
    den = value_float.new_zeros(batch, channels)
    for index in range(length):
        past_num.append(num)
        past_den.append(den)
        weight = key_exp[:, index]
        # Store the state *before* token ``index``.  For the next token every
        # existing contribution moves one more position away, including the
        # token just consumed; this ordering matches ``k_i-w*(t-i)`` in the
        # CUDA kernel (the previous implementation was off by one distance).
        num = (num + weight * value_float[:, index]) * decay_factor
        den = (den + weight) * decay_factor
    future_num: list[torch.Tensor] = [
        value_float.new_zeros(batch, channels) for _ in range(length)
    ]
    future_den: list[torch.Tensor] = [
        value_float.new_zeros(batch, channels) for _ in range(length)
    ]
    num = value_float.new_zeros(batch, channels)
    den = value_float.new_zeros(batch, channels)
    for index in range(length - 1, -1, -1):
        future_num[index] = num
        future_den[index] = den
        weight = key_exp[:, index]
        num = (num + weight * value_float[:, index]) * decay_factor
        den = (den + weight) * decay_factor
    outputs = []
    first_float = first.to(work_dtype).view(1, channels)
    for index in range(length):
        current = torch.exp(
            (first_float + key[:, index].to(work_dtype)).clamp(-20.0, 20.0)
        )
        numerator = (
            past_num[index] + future_num[index] + current * value_float[:, index]
        )
        denominator = past_den[index] + future_den[index] + current + 1.0e-6
        outputs.append(numerator / denominator)
    return torch.stack(outputs, dim=1).to(dtype=value.dtype)


class SharedDirectionalRWKV(nn.Module):
    """Four-direction RWKV with shared WKV parameters and optional block shuffle."""

    def __init__(
        self,
        channels: int,
        *,
        shuffle_prob: float = 0.0,
        shuffle_block: int = 4,
        high_order: bool = True,
    ) -> None:
        super().__init__()
        channels = _positive_int(channels, "channels")
        if channels < 16 or channels % 16:
            raise ValueError("RWKV channels must be a multiple of 16 and >= 16")
        if not 0.0 <= float(shuffle_prob) <= 1.0:
            raise ValueError("shuffle_prob must be in [0, 1]")
        self.channels = channels
        self.shuffle_prob = float(shuffle_prob)
        self.shuffle_block = _positive_int(shuffle_block, "shuffle_block")
        self.shift = OmniShift(channels)
        self.key = nn.Conv2d(channels, channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.receptance = nn.Conv2d(channels, channels, 1, bias=False)
        self.output = nn.Conv2d(channels, channels, 1, bias=False)
        # A fixed average of horizontal and vertical scans retains a residual
        # raster-direction bias on elongated structures.  This gate learns a
        # per-pixel, per-channel mixture while starting exactly at 0.5, so it
        # preserves the symmetric four-direction initialization and can then
        # specialize to local grain/orientation (as in multi-grain RWKV work).
        self.direction_gate = nn.Conv2d(channels + 1, channels, 1, bias=True)
        nn.init.zeros_(self.direction_gate.weight)
        nn.init.zeros_(self.direction_gate.bias)
        # The same decay/first parameters are used by horizontal and vertical
        # scans, and by both scan directions.
        self.decay = nn.Parameter(torch.full((channels,), math.log(math.expm1(0.5))))
        self.first = nn.Parameter(torch.zeros(channels))
        self.high_order = bool(high_order)
        if self.high_order:
            self.high_a = nn.Conv2d(channels, channels // 4, 1, bias=False)
            self.high_b = nn.Conv2d(channels, channels // 4, 1, bias=False)
            self.high_out = nn.Conv2d(channels // 4, channels, 1, bias=False)
            self.high_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    @staticmethod
    def _horizontal(x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        return x.permute(0, 2, 3, 1).reshape(batch * height, width, channels)

    @staticmethod
    def _vertical(x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        return x.permute(0, 3, 2, 1).reshape(batch * width, height, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"SharedDirectionalRWKV expects {self.channels} channels, got {tuple(x.shape)}"
            )
        shifted = self.shift(x)
        key = self.key(shifted)
        value = self.value(shifted)
        receptance = torch.sigmoid(self.receptance(shifted))
        horizontal = self._horizontal(key)
        vertical = self._vertical(key)
        h_value = self._horizontal(value)
        v_value = self._vertical(value)

        # The WKV core receives separate key/value tensors; scan layouts are
        # reconstructed below.  Reusing the same core parameters is the
        # directional weight sharing proposed by RS-RWKV.
        def run(k: torch.Tensor, v: torch.Tensor, reverse: bool) -> torch.Tensor:
            if reverse:
                k = k.flip(1)
                v = v.flip(1)
            inverse = None
            if (
                self.training
                and self.shuffle_prob > 0.0
                and bool(torch.rand((), device=k.device) < self.shuffle_prob)
            ):
                shuffled = _block_shuffle(k, self.shuffle_block)
                if shuffled is not None:
                    # ``_block_shuffle`` returns the inverse map that restores
                    # the original order.  Its inverse is the permutation
                    # needed to apply exactly the same shuffle to values.
                    k, inverse = shuffled
                    permutation = torch.argsort(inverse)
                    v = v.index_select(1, permutation)
            decay = F.softplus(self.decay.float()) / math.sqrt(float(self.channels))
            decay = decay.to(dtype=k.dtype)
            first = (self.first.float() / math.sqrt(float(self.channels))).to(
                dtype=k.dtype
            )
            input_dtype = k.dtype
            if k.is_cuda:
                y = RUN_CUDA(decay, first, k.contiguous(), v.contiguous())
            else:
                y = _reference_biwkv(decay, first, k, v)
            # RUN_CUDA deliberately promotes its extension inputs to fp32;
            # restore the feature dtype before the result reaches bf16/fp16
            # convolutions in a mixed-precision model.
            y = y.to(dtype=input_dtype)
            if inverse is not None:
                y = y.index_select(1, inverse)
            if reverse:
                y = y.flip(1)
            return y

        h_out = (run(horizontal, h_value, False) + run(horizontal, h_value, True)) * 0.5
        v_out = (run(vertical, v_value, False) + run(vertical, v_value, True)) * 0.5
        batch, channels, height, width = x.shape
        h_out = h_out.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        v_out = v_out.reshape(batch, width, height, channels).permute(0, 3, 2, 1)
        # Feed a cheap local-frequency statistic to the direction selector.
        # It makes the scan grain-aware: smooth regions can prefer the long
        # axis while edges/high-frequency texture can switch orientation.
        local_mean = F.avg_pool2d(shifted, kernel_size=3, stride=1, padding=1)
        local_frequency = (shifted - local_mean).abs().mean(dim=1, keepdim=True)
        direction = torch.sigmoid(
            self.direction_gate(torch.cat((shifted, local_frequency), dim=1))
        )
        output = receptance * (direction * h_out + (1.0 - direction) * v_out)
        if self.high_order:
            second = torch.tanh(self.high_a(shifted)) * torch.tanh(self.high_b(shifted))
            output = output + self.high_scale * self.high_out(second)
        return self.output(output)


class _ReferenceSelectiveScan(nn.Module):
    """Small exact selective SSM used only for CPU tests."""

    def __init__(self, channels: int, d_state: int) -> None:
        super().__init__()
        self.channels = channels
        self.d_state = d_state
        self.in_proj = nn.Linear(channels, channels * 2)
        self.param_proj = nn.Linear(channels, 1 + 2 * d_state)
        self.dt_proj = nn.Linear(1, d_state)
        self.a_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        )
        self.skip = nn.Parameter(torch.ones(channels))
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(sequence).chunk(2, dim=-1)
        params = self.param_proj(value)
        dt_raw, b_param, c_param = params.split((1, self.d_state, self.d_state), dim=-1)
        delta = F.softplus(self.dt_proj(dt_raw))
        decay = torch.exp(-torch.exp(self.a_log).view(1, 1, self.d_state) * delta)
        state = sequence.new_zeros(sequence.shape[0], self.channels, self.d_state)
        outputs = []
        for index in range(sequence.shape[1]):
            state = state * decay[:, index].unsqueeze(1)
            state = state + b_param[:, index].unsqueeze(1) * value[:, index].unsqueeze(
                -1
            )
            current = (state * c_param[:, index].unsqueeze(1)).sum(dim=-1)
            outputs.append(current + self.skip * value[:, index])
        output = torch.stack(outputs, dim=1)
        return self.out_proj(output * torch.sigmoid(gate))


class TrueMambaScan(nn.Module):
    """Official Mamba on CUDA, with a selective-scan reference path on CPU."""

    def __init__(
        self,
        channels: int,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        allow_cpu_reference: bool = True,
    ) -> None:
        super().__init__()
        channels = _positive_int(channels, "channels")
        d_state = _positive_int(d_state, "d_state")
        d_conv = _positive_int(d_conv, "d_conv")
        expand = _positive_int(expand, "expand")
        if _OfficialMamba is None and not allow_cpu_reference:
            detail = (
                f": {_MAMBA_IMPORT_ERROR}" if _MAMBA_IMPORT_ERROR is not None else ""
            )
            raise ImportError(
                "RDMRefSR requires the official mamba_ssm package for CUDA execution"
                + detail
            )
        self.official = (
            _OfficialMamba(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            if _OfficialMamba is not None
            else None
        )
        self.reference = _ReferenceSelectiveScan(channels, d_state)
        self.allow_cpu_reference = bool(allow_cpu_reference)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.is_cuda:
            if self.official is None:
                raise RuntimeError(
                    "TrueMambaScan received a CUDA tensor but mamba_ssm is unavailable"
                )
            return self.official(sequence)
        if not self.allow_cpu_reference:
            raise RuntimeError(
                "TrueMambaScan requires CUDA; enable allow_cpu_reference for CPU tests"
            )
        return self.reference(sequence)


class FourDirectionMamba(nn.Module):
    """Shared-parameter horizontal/vertical bidirectional selective scan."""

    def __init__(self, channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.channels = channels
        self.scan = TrueMambaScan(channels, **kwargs)
        self.output = nn.Conv2d(channels, channels, 1, bias=False)

    @staticmethod
    def _horizontal(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return x.permute(0, 2, 3, 1).reshape(b * h, w, c)

    @staticmethod
    def _vertical(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return x.permute(0, 3, 2, 1).reshape(b * w, h, c)

    def _run(self, sequence: torch.Tensor, reverse: bool) -> torch.Tensor:
        if reverse:
            sequence = sequence.flip(1)
        output = self.scan(sequence)
        return output.flip(1) if reverse else output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        horizontal = self._horizontal(x)
        vertical = self._vertical(x)
        h_out = (self._run(horizontal, False) + self._run(horizontal, True)) * 0.5
        v_out = (self._run(vertical, False) + self._run(vertical, True)) * 0.5
        h_out = h_out.reshape(b, h, w, c).permute(0, 3, 1, 2)
        v_out = v_out.reshape(b, w, h, c).permute(0, 3, 2, 1)
        return self.output((h_out + v_out) * 0.5)


class TriTokenConditioner(nn.Module):
    """Spectral, structure and reliability tokens plus a register token."""

    def __init__(self, channels: int, reference_dim: int) -> None:
        super().__init__()
        channels = _positive_int(channels, "channels")
        reference_dim = _positive_int(reference_dim, "reference_dim")
        token_dim = max(16, min(64, channels))
        self.q_proj = nn.Linear(channels, token_dim)
        self.low_proj = nn.Linear(reference_dim, token_dim)
        self.detail_proj = nn.Linear(reference_dim, token_dim)
        self.reliability_proj = nn.Linear(1, token_dim)
        self.register_token = nn.Parameter(torch.zeros(1, token_dim))
        self.mlp = nn.Sequential(
            nn.Linear(token_dim * 5, max(channels, token_dim * 2)),
            nn.GELU(),
            nn.Linear(max(channels, token_dim * 2), channels * 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        ref_low: torch.Tensor,
        ref_detail: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_token = self.q_proj(x.mean(dim=(-2, -1)))
        low_token = self.low_proj(ref_low.mean(dim=(-2, -1)))
        detail_token = self.detail_proj(ref_detail.mean(dim=(-2, -1)))
        reliability_token = self.reliability_proj(reliability.mean(dim=(-2, -1)))
        register = self.register_token.expand(x.shape[0], -1)
        gamma, beta = self.mlp(
            torch.cat(
                (q_token, low_token, detail_token, reliability_token, register), dim=1
            )
        ).chunk(2, dim=1)
        return gamma[..., None, None], beta[..., None, None]


class HybridStateBlock(nn.Module):
    """RWKV content branch plus optional official Mamba detail branch."""

    def __init__(
        self,
        channels: int,
        reference_dim: int,
        *,
        use_mamba: bool,
        shuffle_prob: float,
        shuffle_block: int,
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_expand: int,
        allow_cpu_mamba: bool,
        high_order: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.norm1 = RMSNorm2d(channels)
        self.condition = TriTokenConditioner(channels, reference_dim)
        self.rwkv = SharedDirectionalRWKV(
            channels,
            shuffle_prob=shuffle_prob,
            shuffle_block=shuffle_block,
            high_order=high_order,
        )
        self.use_mamba = bool(use_mamba)
        self.mamba = (
            FourDirectionMamba(
                channels,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                allow_cpu_reference=allow_cpu_mamba,
            )
            if self.use_mamba
            else None
        )
        self.detail_proj = nn.Conv2d(reference_dim, channels, 1, bias=False)
        self.mix_gate = nn.Sequential(
            nn.Conv2d(channels + 1, max(8, channels // 4), 1),
            nn.GELU(),
            nn.Conv2d(max(8, channels // 4), channels, 1),
            nn.Sigmoid(),
        )
        self.norm2 = RMSNorm2d(channels)
        hidden = max(channels * 2, 32)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.res_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.ffn_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(
        self,
        x: torch.Tensor,
        ref_low: torch.Tensor,
        ref_detail: torch.Tensor,
        reliability: torch.Tensor,
        *,
        detail: torch.Tensor | None = None,
        use_reference: bool = True,
    ) -> torch.Tensor:
        # A stage can keep the query-only RWKV/Mamba path while receiving the
        # same tensor shapes.  Zeroing the conditioning inputs here avoids a
        # second copy of the block and makes the reference schedule explicit.
        if not use_reference:
            ref_low = torch.zeros_like(ref_low)
            ref_detail = torch.zeros_like(ref_detail)
            reliability = torch.zeros_like(reliability)
            detail = None
        ref_low = _resize(ref_low, x.shape[-2:])
        ref_detail = _resize(ref_detail, x.shape[-2:])
        reliability = _resize(reliability, x.shape[-2:])
        normalized = self.norm1(x)
        gamma, beta = self.condition(normalized, ref_low, ref_detail, reliability)
        # Reference statistics are allowed to modulate content only where the
        # reliability field says the reference is trustworthy.  This keeps a
        # changed/misaligned temporal reference from becoming an unconditional
        # shortcut through every U-Net stage.
        condition_gate = reliability.mean(dim=(-2, -1), keepdim=True).clamp(0.0, 1.0)
        normalized = normalized * (1.0 + 0.1 * torch.tanh(gamma) * condition_gate)
        normalized = normalized + 0.1 * torch.tanh(beta) * condition_gate
        rwkv_value = self.rwkv(normalized)
        if detail is not None:
            detail = _resize(detail, x.shape[-2:])
            mamba_input = normalized + self.detail_proj(detail) * reliability
        else:
            mamba_input = normalized
        if self.mamba is not None:
            mamba_value = self.mamba(mamba_input)
            gate = self.mix_gate(torch.cat((normalized, reliability), dim=1))
            state_value = gate * rwkv_value + (1.0 - gate) * mamba_value
        else:
            state_value = rwkv_value
        x = x + self.res_scale * state_value
        x = x + self.ffn_scale * self.ffn(self.norm2(x))
        return x


class Stage(nn.Module):
    def __init__(self, blocks: Sequence[HybridStateBlock]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(list(blocks))

    def forward(
        self,
        x: torch.Tensor,
        ref_low: torch.Tensor,
        ref_detail: torch.Tensor,
        reliability: torch.Tensor,
        *,
        detail: torch.Tensor | None = None,
        use_reference: bool = True,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(
                x,
                ref_low,
                ref_detail,
                reliability,
                detail=detail,
                use_reference=use_reference,
            )
        return x


class SelectiveDetailInjector(nn.Module):
    """Inject only a reliability-weighted reference residual."""

    def __init__(self, channels: int, reference_dim: int) -> None:
        super().__init__()
        self.detail = nn.Conv2d(reference_dim, channels, 1, bias=False)
        hidden = max(8, channels // 4)
        self.gate = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        # A small non-zero path keeps the reference encoder trainable from the
        # first step while the tiny projection initialization below keeps the
        # initial prediction effectively bicubic.
        self.alpha = nn.Parameter(torch.full((1, channels, 1, 1), 0.05))

    def forward(
        self,
        x: torch.Tensor,
        detail: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        detail = _resize(detail, x.shape[-2:])
        reliability = _resize(reliability, x.shape[-2:])
        candidate = self.detail(detail)
        gate = self.gate(torch.cat((x, reliability), dim=1))
        return x + self.alpha * gate * reliability * candidate


class WaveletSynthesisHead(nn.Module):
    """Predict query residual coefficients and gated reference detail."""

    def __init__(
        self, feature_dim: int, input_channels: int, output_channels: int
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_projection = (
            nn.Conv2d(input_channels, output_channels, 1, bias=False)
            if input_channels != output_channels
            else nn.Identity()
        )
        self.low_delta = nn.Conv2d(
            feature_dim, output_channels, 3, padding=1, bias=True
        )
        self.detail_delta = nn.Conv2d(
            feature_dim, output_channels * 3, 3, padding=1, bias=True
        )
        self.reference_detail = nn.Conv2d(
            feature_dim, output_channels * 3, 1, bias=False
        )
        self.band_gate = nn.Conv2d(feature_dim + 1, output_channels, 1, bias=True)
        # A second, physically constrained detail path is supplied by the
        # response-matrix projection in ``RDMRefSR``.  It gives PAN/MSI a
        # direct route to the Haar coefficients while the learned matcher
        # handles semantic retrieval.  The negative bias and small scale keep
        # this path conservative at initialization.
        self.response_gate = nn.Conv2d(feature_dim + 1, output_channels, 1, bias=True)
        self.response_detail_scale = nn.Parameter(
            torch.full((1, output_channels, 1, 1), 0.03)
        )
        self.reference_scale = nn.Parameter(
            torch.full((1, output_channels, 1, 1), 0.10)
        )
        nn.init.normal_(self.low_delta.weight, std=1.0e-4)
        nn.init.zeros_(self.low_delta.bias)
        nn.init.normal_(self.detail_delta.weight, std=1.0e-4)
        nn.init.zeros_(self.detail_delta.bias)
        nn.init.normal_(self.reference_detail.weight, std=5.0e-3)
        nn.init.constant_(self.band_gate.bias, -0.5)
        nn.init.constant_(self.response_gate.bias, -1.0)
        if isinstance(self.base_projection, nn.Conv2d):
            nn.init.zeros_(self.base_projection.weight)
            with torch.no_grad():
                for index in range(min(input_channels, output_channels)):
                    self.base_projection.weight[index, index, 0, 0] = 1.0

    def forward(
        self,
        lr: torch.Tensor,
        feature: torch.Tensor,
        reference_detail: torch.Tensor,
        reliability: torch.Tensor,
        output_size: tuple[int, int],
        response_detail: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base = self.base_projection(lr)
        # Match the repository's SISR baseline exactly at initialization. The
        # generic feature resize is bilinear, but the reconstruction base is
        # deliberately bicubic because that is the physical LR->HR prior used
        # by all RefSR comparisons.
        base = F.interpolate(
            base, size=output_size, mode="bicubic", align_corners=False
        )
        base_low, base_high, original_size = haar_dwt2d(base)
        feature = _resize(feature, base_low.shape[-2:])
        reliability = _resize(reliability, base_low.shape[-2:])
        reference_detail = _resize(reference_detail, base_low.shape[-2:])
        low = base_low + self.low_delta(feature)
        high = base_high + self.detail_delta(feature)
        candidate = self.reference_detail(reference_detail)
        gate = torch.sigmoid(self.band_gate(torch.cat((feature, reliability), dim=1)))
        reference_scale = self.reference_scale.repeat_interleave(3, dim=1)
        high = (
            high
            + reference_scale
            * gate.repeat_interleave(3, dim=1)
            * reliability
            * candidate
        )
        if response_detail is not None:
            response_detail = _resize(response_detail, base_low.shape[-2:])
            response_gate = torch.sigmoid(
                self.response_gate(torch.cat((feature, reliability), dim=1))
            )
            response_scale = self.response_detail_scale.repeat_interleave(3, dim=1)
            high = (
                high
                + response_scale
                * response_gate.repeat_interleave(3, dim=1)
                * reliability
                * response_detail
            )
        return haar_idwt2d(low, high, original_size)


class RDMRefSR(nn.Module):
    """Full dual-grid reference SR model with true RWKV and Mamba branches.

    Parameters intentionally expose ``target_channels`` independently from
    ``ref_channels``.  This supports LR-HSI + HR-MSI as well as PAN and
    cross-temporal RGB without assuming a channel ordering relationship.
    ``channel_multipliers`` keeps the four-stage U-Net but can cap the
    bottleneck width.  Pansharpening is implemented by ``RDMPan``, not this
    gated dual-grid model.
    """

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int | None = None,
        ref_channels: int = 3,
        target_channels: int | None = None,
        dim: int = 48,
        depths: Sequence[int] = (2, 2, 3, 4),
        decoder_depths: Sequence[int] = (3, 2, 2),
        channel_multipliers: Sequence[int] = (1, 2, 4, 8),
        scale: int = 4,
        reference_kind: str = "stf",
        mamba_stages: Sequence[str] = ("enc2", "latent", "dec2"),
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        high_order: bool = True,
        allow_cpu_mamba: bool = True,
        shuffle_prob: float = 0.15,
        shuffle_block: int = 4,
        match_window: int = 5,
        match_dim: int = 16,
        match_grid: str = "coefficient",
        temporal_match_confidence_floor: float = 0.15,
        alignment: bool | None = None,
        max_offset: float = 3.0,
        response_matrix: Sequence[Sequence[float]] | None = None,
        sensor_response: Sequence[float] | None = None,
        use_reference: bool = True,
        clamp_output: bool = True,
        reference_condition_stages: Sequence[str] = ("enc2", "latent", "dec1", "coeff"),
        detail_injection_stages: Sequence[str] = ("dec1", "coeff"),
    ) -> None:
        super().__init__()
        inp_channels = _positive_int(inp_channels, "inp_channels")
        ref_channels = _positive_int(ref_channels, "ref_channels")
        if out_channels is None:
            out_channels = (
                target_channels if target_channels is not None else inp_channels
            )
        out_channels = _positive_int(out_channels, "out_channels")
        if target_channels is not None and int(target_channels) != out_channels:
            raise ValueError(
                "target_channels and out_channels must agree when both are provided"
            )
        dim = _positive_int(dim, "dim")
        if dim < 16 or dim % 16:
            raise ValueError(
                "dim must be a multiple of 16 and >= 16 for the WKV kernel"
            )
        scale = _positive_int(scale, "scale")
        if len(depths) != 4 or any(
            _positive_int(value, "depths item") < 1 for value in depths
        ):
            raise ValueError("depths must contain four positive integers")
        if len(decoder_depths) != 3 or any(
            _positive_int(value, "decoder_depths item") < 1 for value in decoder_depths
        ):
            raise ValueError("decoder_depths must contain three positive integers")
        multipliers = tuple(
            _positive_int(value, "channel_multipliers item")
            for value in channel_multipliers
        )
        if len(multipliers) != 4:
            raise ValueError("channel_multipliers must contain four positive integers")
        channels = tuple(dim * value for value in multipliers)
        for index, width in enumerate(channels):
            if width < 16 or width % 16:
                raise ValueError(
                    "dim * channel_multipliers must be a multiple of 16 and >= 16 "
                    f"at stage {index}, got {width}"
                )
        if isinstance(mamba_stages, (str, bytes)):
            raise ValueError("mamba_stages must be a sequence of stage names")
        valid_stages = {
            "enc0",
            "enc1",
            "enc2",
            "latent",
            "dec2",
            "dec1",
            "dec0",
            "coeff",
        }
        normalized_mamba_stages = frozenset(
            str(name).strip().lower() for name in mamba_stages
        )
        unknown_stages = normalized_mamba_stages.difference(valid_stages)
        if unknown_stages:
            raise ValueError(
                "mamba_stages contains unknown stages: "
                + ", ".join(sorted(unknown_stages))
            )
        for schedule_name, schedule in (
            ("reference_condition_stages", reference_condition_stages),
            ("detail_injection_stages", detail_injection_stages),
        ):
            if isinstance(schedule, (str, bytes)):
                raise ValueError(f"{schedule_name} must be a sequence of stage names")
            unknown = frozenset(
                str(name).strip().lower() for name in schedule
            ).difference(valid_stages)
            if unknown:
                raise ValueError(
                    f"{schedule_name} contains unknown stages: "
                    + ", ".join(sorted(unknown))
                )
        normalized_reference_condition_stages = frozenset(
            str(name).strip().lower() for name in reference_condition_stages
        )
        normalized_detail_injection_stages = frozenset(
            str(name).strip().lower() for name in detail_injection_stages
        )
        kind = normalize_reference_kind(reference_kind)
        if kind == "pan":
            raise ValueError(
                "RDMRefSR no longer implements pansharpening; use RDMPan "
                "(model.name=rdm_pan). Matcher/reliability gating is only for "
                "STF/MHF, where the reference can be wrong."
            )
        if alignment is None:
            alignment = kind in {"stf", "mhf"}
        if not isinstance(alignment, bool):
            raise ValueError("alignment must be bool or None")
        if max_offset <= 0 or not math.isfinite(float(max_offset)):
            raise ValueError("max_offset must be positive and finite")
        if not isinstance(use_reference, bool):
            raise ValueError("use_reference must be bool")
        if not isinstance(clamp_output, bool):
            raise ValueError("clamp_output must be bool")
        if not isinstance(high_order, bool):
            raise ValueError("high_order must be bool")
        for name, value in (
            ("mamba_d_state", mamba_d_state),
            ("mamba_d_conv", mamba_d_conv),
            ("mamba_expand", mamba_expand),
        ):
            _positive_int(value, name)
        match_grid = str(match_grid).strip().lower()
        if match_grid not in {"lr", "coefficient"}:
            raise ValueError("match_grid must be lr or coefficient")
        if (
            isinstance(temporal_match_confidence_floor, bool)
            or not isinstance(temporal_match_confidence_floor, (int, float))
            or not math.isfinite(float(temporal_match_confidence_floor))
            or not 0.0 <= float(temporal_match_confidence_floor) < 1.0
        ):
            raise ValueError("temporal_match_confidence_floor must be in [0, 1)")
        self.inp_channels = inp_channels
        self.ref_channels = ref_channels
        self.out_channels = out_channels
        self.target_channels = out_channels
        self.dim = dim
        self.scale = scale
        self.reference_kind = kind
        self.use_reference = use_reference
        self.alignment = bool(alignment)
        self.max_offset = float(max_offset)
        self.clamp_output = bool(clamp_output)
        self.mamba_stages = normalized_mamba_stages
        self.reference_condition_stages = normalized_reference_condition_stages
        self.detail_injection_stages = normalized_detail_injection_stages
        self.match_grid = match_grid
        self.temporal_match_confidence_floor = float(temporal_match_confidence_floor)
        self.channel_multipliers = multipliers
        self.high_order = bool(high_order)

        self.ms_stem = nn.Sequential(
            nn.Conv2d(inp_channels, dim, 3, padding=1, bias=False),
            RMSNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
        )
        self.ref_low_encoder = nn.Sequential(
            nn.Conv2d(ref_channels, dim, 3, padding=1, bias=False),
            RMSNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
        )
        self.ref_detail_encoder = nn.Sequential(
            nn.Conv2d(ref_channels * 3, dim, 3, padding=1, bias=False),
            RMSNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
        )
        self.detail_matcher = _LocalDetailMatcher(
            dim,
            window_size=match_window,
            match_dim=min(match_dim, dim),
            transfer_dim=dim,
        )

        if kind == "pan":
            if sensor_response is None:
                initial_response = torch.ones(inp_channels, dtype=torch.float32)
            else:
                initial_response = torch.as_tensor(sensor_response, dtype=torch.float32)
                if initial_response.numel() != inp_channels:
                    raise ValueError("sensor_response length must equal inp_channels")
                if (
                    not torch.isfinite(initial_response).all()
                    or (initial_response <= 0).any()
                ):
                    raise ValueError(
                        "sensor_response must contain positive finite values"
                    )
            self.sensor_logits = nn.Parameter(
                torch.log(initial_response.clamp_min(1.0e-4))
            )
        else:
            self.register_parameter("sensor_logits", None)
        if response_matrix is None:
            response = torch.full(
                (out_channels, ref_channels), 1.0 / float(ref_channels)
            )
            if out_channels == ref_channels:
                response = torch.eye(out_channels)
        else:
            response = torch.as_tensor(response_matrix, dtype=torch.float32)
            if tuple(response.shape) != (out_channels, ref_channels):
                raise ValueError(
                    "response_matrix must have shape (out_channels, ref_channels), "
                    f"got {tuple(response.shape)}"
                )
            if not torch.isfinite(response).all() or (response < 0).any():
                raise ValueError(
                    "response_matrix must contain finite non-negative values"
                )
        self.reference_lift_logits = nn.Parameter(response.clamp_min(1.0e-4).log())
        if kind == "stf":
            # Cross-date radiometric drift is an STF problem; PAN/MHF skip it.
            self.radiometric = nn.Sequential(
                nn.Linear(4, 32),
                nn.GELU(),
                nn.Linear(32, 2),
            )
            nn.init.zeros_(self.radiometric[-1].weight)
            nn.init.zeros_(self.radiometric[-1].bias)
        else:
            self.radiometric = None
        if alignment:
            self.offset_net = nn.Sequential(
                nn.Conv2d(2, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 2, 3, padding=1),
            )
            nn.init.zeros_(self.offset_net[-1].weight)
            nn.init.zeros_(self.offset_net[-1].bias)
        else:
            self.offset_net = None
        self.reliability = ReliabilityField(
            kind,
            temporal_match_confidence_floor=self.temporal_match_confidence_floor,
        )

        self.enc0 = self._make_stage(
            "enc0",
            channels[0],
            depths[0],
            allow_cpu_mamba,
            shuffle_prob,
            shuffle_block,
            mamba_d_state,
            mamba_d_conv,
            mamba_expand,
        )
        self.down01 = _Downsample(channels[0], channels[1])
        self.enc1 = self._make_stage(
            "enc1",
            channels[1],
            depths[1],
            allow_cpu_mamba,
            shuffle_prob,
            shuffle_block,
            mamba_d_state,
            mamba_d_conv,
            mamba_expand,
        )
        self.down12 = _Downsample(channels[1], channels[2])
        self.enc2 = self._make_stage(
            "enc2",
            channels[2],
            depths[2],
            allow_cpu_mamba,
            shuffle_prob,
            shuffle_block,
            mamba_d_state,
            mamba_d_conv,
            mamba_expand,
        )
        self.down23 = _Downsample(channels[2], channels[3])
        self.latent = self._make_stage(
            "latent",
            channels[3],
            depths[3],
            allow_cpu_mamba,
            shuffle_prob,
            shuffle_block,
            mamba_d_state,
            mamba_d_conv,
            mamba_expand,
        )
        self.up32 = _Upsample(channels[3], channels[2])
        self.dec2 = self._make_stage(
            "dec2",
            channels[2],
            decoder_depths[0],
            allow_cpu_mamba,
            shuffle_prob,
            shuffle_block,
            mamba_d_state,
            mamba_d_conv,
            mamba_expand,
        )
        self.up21 = _Upsample(channels[2], channels[1])
        self.dec1 = self._make_stage(
            "dec1",
            channels[1],
            decoder_depths[1],
            allow_cpu_mamba,
            shuffle_prob,
            shuffle_block,
            mamba_d_state,
            mamba_d_conv,
            mamba_expand,
        )
        self.up10 = _Upsample(channels[1], channels[0])
        self.dec0 = self._make_stage(
            "dec0",
            channels[0],
            decoder_depths[2],
            allow_cpu_mamba,
            shuffle_prob,
            shuffle_block,
            mamba_d_state,
            mamba_d_conv,
            mamba_expand,
        )
        self.coeff_lift = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 3, padding=1, bias=False),
            RMSNorm2d(channels[0]),
            nn.GELU(),
        )
        self.qshift = QShift(channels[0], shift=1)
        self.coeff_stage = self._make_stage(
            "coeff",
            channels[0],
            1,
            allow_cpu_mamba,
            shuffle_prob,
            shuffle_block,
            mamba_d_state,
            mamba_d_conv,
            mamba_expand,
        )
        self.inject_dec1 = SelectiveDetailInjector(channels[1], dim)
        self.inject_coeff = SelectiveDetailInjector(channels[0], dim)
        self.synthesis = WaveletSynthesisHead(channels[0], inp_channels, out_channels)

        self.apply(self._init_weights)
        # ``apply`` above initializes every convolution; restore the exact
        # symmetric direction prior for all RWKV blocks afterwards.
        for module in self.modules():
            if isinstance(module, SharedDirectionalRWKV):
                nn.init.zeros_(module.direction_gate.weight)
                nn.init.zeros_(module.direction_gate.bias)
        self._init_reference_residuals()

    def _make_stage(
        self,
        stage_name: str,
        channels: int,
        depth: int,
        allow_cpu_mamba: bool,
        shuffle_prob: float,
        shuffle_block: int,
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_expand: int,
    ) -> Stage:
        blocks = [
            HybridStateBlock(
                channels,
                self.dim,
                use_mamba=stage_name.lower() in self.mamba_stages,
                shuffle_prob=shuffle_prob,
                shuffle_block=shuffle_block,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                allow_cpu_mamba=allow_cpu_mamba,
                high_order=self.high_order,
            )
            for _ in range(depth)
        ]
        return Stage(blocks)

    def _init_reference_residuals(self) -> None:
        """Restore conservative STF/MHF residuals after generic conv init.

        ``apply(_init_weights)`` zeroes every conv bias, including the
        reliability head.  Temporal/HSI-MSI stay close to bicubic until the
        matcher and reliability field have evidence.  Pansharpening lives in
        ``RDMPan`` and does not use this gated residual path.
        """
        nn.init.constant_(self.reliability.net[-1].bias, -1.0)
        nn.init.constant_(self.reliability.change_net[-1].bias, -1.0)
        for module in (self.synthesis.low_delta, self.synthesis.detail_delta):
            nn.init.normal_(module.weight, std=1.0e-4)
            nn.init.zeros_(module.bias)
        nn.init.normal_(self.synthesis.reference_detail.weight, std=5.0e-3)
        nn.init.constant_(self.synthesis.band_gate.bias, -0.5)
        nn.init.constant_(self.synthesis.response_gate.bias, -1.0)
        nn.init.constant_(self.synthesis.reference_scale, 0.10)
        nn.init.constant_(self.synthesis.response_detail_scale, 0.03)
        for injector in (self.inject_dec1, self.inject_coeff):
            nn.init.normal_(injector.detail.weight, std=1.0e-3)
            nn.init.constant_(injector.alpha, 0.05)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _query_intensity(
        self, lr: torch.Tensor, output_size: tuple[int, int]
    ) -> torch.Tensor:
        upsampled = F.interpolate(
            lr, size=output_size, mode="bicubic", align_corners=False
        )
        if self.sensor_logits is not None:
            weights = torch.softmax(self.sensor_logits.float(), dim=0).to(
                dtype=upsampled.dtype
            )
            return (upsampled * weights.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        return upsampled.mean(dim=1, keepdim=True)

    def _reference_intensity(self, ref: torch.Tensor) -> torch.Tensor:
        lift = torch.softmax(self.reference_lift_logits.float(), dim=1).to(
            dtype=ref.dtype
        )
        projected = torch.einsum("or,brhw->bohw", lift, ref)
        return projected.mean(dim=1, keepdim=True)

    def _sensor_band_gains(self) -> torch.Tensor:
        """Return unit-mean PAN response gains for target output bands."""

        if self.sensor_logits is None:
            return self.reference_lift_logits.new_ones(self.out_channels)
        logits = self.sensor_logits.float().view(1, 1, -1)
        if logits.shape[-1] != self.out_channels:
            logits = F.interpolate(
                logits, size=self.out_channels, mode="linear", align_corners=False
            )
        weights = torch.softmax(logits.flatten(1), dim=1).squeeze(0)
        return weights * float(self.out_channels)

    def _project_reference_detail(self, detail: torch.Tensor) -> torch.Tensor:
        """Project each Haar detail band with the learned sensor response.

        The same response matrix used for reference intensity is applied to
        LH/HL/HH independently.  This is deliberately a residual-only path:
        the query stream still owns the low-frequency reconstruction and the
        caller supplies reliability before the coefficients are synthesized.
        """

        if detail.ndim != 4 or detail.shape[1] != self.ref_channels * 3:
            raise ValueError(
                "reference detail must contain three Haar bands per reference channel, "
                f"got {tuple(detail.shape)}"
            )
        if self.reference_kind == "pan":
            # PAN has one spatial measurement but a broad spectral response.
            # Use the learnable sensor prior to assign relative detail gains to
            # output bands; unlike a softmax over a one-channel response this
            # remains trainable for the usual 8-band MS + 1-band PAN contract.
            gain = self._sensor_band_gains().to(dtype=detail.dtype)
            lift = gain[:, None].expand(self.out_channels, self.ref_channels)
            lift = lift / float(max(self.ref_channels, 1))
        else:
            lift = torch.softmax(self.reference_lift_logits.float(), dim=1).to(
                dtype=detail.dtype
            )
        projected = [
            torch.einsum("or,brhw->bohw", lift, band) for band in detail.chunk(3, dim=1)
        ]
        return torch.cat(projected, dim=1)

    def _warp(self, image: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=image.device, dtype=image.dtype),
            torch.linspace(-1.0, 1.0, w, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        grid = (
            torch.stack((xx, yy), dim=-1)
            .unsqueeze(0)
            .expand(image.shape[0], -1, -1, -1)
            .clone()
        )
        normalized = _resize(offsets, (h, w))
        normalized_x = normalized[:, 0] / max(float(w - 1), 1.0) * 2.0
        normalized_y = normalized[:, 1] / max(float(h - 1), 1.0) * 2.0
        grid = grid + torch.stack((normalized_x, normalized_y), dim=-1)
        return F.grid_sample(
            image, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

    def _prepare_reference(
        self,
        lr: torch.Tensor,
        ref: torch.Tensor,
        output_size: tuple[int, int],
        query_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ref_low_raw, ref_detail_raw, _ = haar_dwt2d(ref)
        query_intensity = self._query_intensity(lr, output_size)
        ref_intensity = _resize(self._reference_intensity(ref), output_size)
        if self.reference_kind == "stf":
            stats = torch.cat(
                (
                    query_intensity.mean(dim=(-2, -1)),
                    query_intensity.std(dim=(-2, -1), unbiased=False),
                    ref_intensity.mean(dim=(-2, -1)),
                    ref_intensity.std(dim=(-2, -1), unbiased=False),
                ),
                dim=1,
            )
            affine = self.radiometric(stats)
            gain = 1.0 + 0.2 * torch.tanh(affine[:, :1, None, None])
            bias = 0.1 * torch.tanh(affine[:, 1:, None, None])
            ref_low_raw = ref_low_raw * gain + bias
            ref_intensity = _resize(ref_low_raw.mean(dim=1, keepdim=True), output_size)
        if self.alignment:
            coarse_size = (max(1, output_size[0] // 4), max(1, output_size[1] // 4))
            q_coarse = _resize(query_intensity, coarse_size)
            r_coarse = _resize(ref_intensity, coarse_size)
            offset = (
                torch.tanh(self.offset_net(torch.cat((q_coarse, r_coarse), dim=1)))
                * self.max_offset
            )
            # ``offset`` is predicted in HR pixels.  Haar LL/detail live on
            # the HR/2 coefficient grid, so pass offsets in that grid's pixel
            # units; using HR values directly doubles the physical motion.
            coeff_h, coeff_w = ref_low_raw.shape[-2:]
            scale_y = float(coeff_h) / float(max(output_size[0], 1))
            scale_x = float(coeff_w) / float(max(output_size[1], 1))
            coeff_offset = offset.clone()
            coeff_offset[:, 0] = coeff_offset[:, 0] * scale_x
            coeff_offset[:, 1] = coeff_offset[:, 1] * scale_y
            ref_low_raw = self._warp(ref_low_raw, coeff_offset)
            ref_detail_raw = self._warp(ref_detail_raw, coeff_offset)
            ref_intensity = _resize(ref_low_raw.mean(dim=1, keepdim=True), output_size)
        ref_low_feature = self.ref_low_encoder(ref_low_raw)
        ref_detail_feature = self.ref_detail_encoder(ref_detail_raw)
        response_detail = self._project_reference_detail(ref_detail_raw)
        coefficient_size = (
            max(1, (output_size[0] + 1) // 2),
            max(1, (output_size[1] + 1) // 2),
        )
        # Haar detail lives on the coefficient grid (HR/2).  Matching it on
        # the LR grid and then interpolating back discards the very phase
        # information that the detail path is supposed to recover.  The
        # coefficient option retains that grid; ``lr`` remains available for
        # unusually large scale factors where the memory tradeoff dominates.
        match_size = (
            coefficient_size
            if self.match_grid == "coefficient"
            else query_feature.shape[-2:]
        )
        query_match = _resize(query_feature, match_size)
        low_match = _resize(ref_low_feature, match_size)
        detail_match = _resize(ref_detail_feature, match_size)
        matched_detail, match_conf = self.detail_matcher(
            query_match, low_match, detail_match, return_confidence=True
        )
        matched_detail = _resize(matched_detail, coefficient_size)
        match_conf = _resize(match_conf, coefficient_size)
        reliability, change = self.reliability(
            query_intensity,
            ref_intensity,
            ref_detail_feature,
            match_conf,
        )
        return {
            "low": ref_low_feature,
            "detail": ref_detail_feature,
            "response_detail": response_detail,
            "matched_detail": matched_detail,
            "reliability": reliability,
            "match_confidence": match_conf,
            "change": change,
            "query_intensity": query_intensity,
            "ref_intensity": ref_intensity,
        }

    def _stage(
        self,
        stage_name: str,
        module: Stage,
        x: torch.Tensor,
        reference: dict[str, torch.Tensor],
        *,
        detail: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized_name = str(stage_name).strip().lower()
        return module(
            x,
            reference["low"],
            reference["detail"],
            reference["reliability"],
            detail=detail if normalized_name in self.detail_injection_stages else None,
            use_reference=(
                self.use_reference
                and normalized_name in self.reference_condition_stages
            ),
        )

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
        output_size = (int(lr.shape[-2]) * self.scale, int(lr.shape[-1]) * self.scale)
        if self.use_reference:
            if ref is None or ref.ndim != 4:
                raise ValueError(
                    "RDMRefSR requires a 4-D ref tensor when use_reference=true"
                )
            if ref.shape[0] != lr.shape[0] or ref.shape[1] != self.ref_channels:
                raise ValueError(
                    f"lr/ref batch or channels mismatch: lr={tuple(lr.shape)}, ref={tuple(ref.shape)}"
                )
            if tuple(ref.shape[-2:]) != output_size:
                raise ValueError(
                    f"ref spatial size must be LR*x{self.scale}: {tuple(ref.shape[-2:])} vs {output_size}"
                )
        elif ref is not None and ref.ndim == 4 and tuple(ref.shape[-2:]) != output_size:
            raise ValueError("optional ref spatial size must match LR*scale")

        query = self.ms_stem(lr)
        if self.use_reference:
            reference = self._prepare_reference(lr, ref, output_size, query)
        else:
            zero = query.new_zeros(
                query.shape[0],
                self.dim,
                max(1, output_size[0] // 2),
                max(1, output_size[1] // 2),
            )
            reference = {
                "low": zero,
                "detail": zero,
                "response_detail": query.new_zeros(
                    query.shape[0],
                    self.out_channels * 3,
                    max(1, (output_size[0] + 1) // 2),
                    max(1, (output_size[1] + 1) // 2),
                ),
                "matched_detail": zero,
                "reliability": query.new_zeros(query.shape[0], 1, *output_size),
                "match_confidence": query.new_zeros(
                    query.shape[0],
                    1,
                    max(1, output_size[0] // 2),
                    max(1, output_size[1] // 2),
                ),
                "change": query.new_zeros(query.shape[0], 1, *output_size),
            }
        e0 = self._stage("enc0", self.enc0, query, reference)
        e1 = self._stage("enc1", self.enc1, self.down01(e0), reference)
        e2 = self._stage("enc2", self.enc2, self.down12(e1), reference)
        latent = self._stage("latent", self.latent, self.down23(e2), reference)
        d2 = self.up32(latent, e2.shape[-2:]) + e2
        d2 = self._stage("dec2", self.dec2, d2, reference)
        d1 = self.up21(d2, e1.shape[-2:]) + e1
        if self.use_reference and "dec1" in self.detail_injection_stages:
            d1 = self.inject_dec1(
                d1, reference["matched_detail"], reference["reliability"]
            )
        d1 = self._stage(
            "dec1", self.dec1, d1, reference, detail=reference["matched_detail"]
        )
        d0 = self.up10(d1, e0.shape[-2:]) + e0
        d0 = self._stage("dec0", self.dec0, d0, reference)
        coefficient_size = (
            max(1, (output_size[0] + 1) // 2),
            max(1, (output_size[1] + 1) // 2),
        )
        coefficient_feature = self.coeff_lift(_resize(d0, coefficient_size))
        coefficient_feature = self.qshift(coefficient_feature)
        if self.use_reference and "coeff" in self.detail_injection_stages:
            coefficient_feature = self.inject_coeff(
                coefficient_feature,
                reference["matched_detail"],
                reference["reliability"],
            )
        coefficient_feature = self._stage(
            "coeff",
            self.coeff_stage,
            coefficient_feature,
            reference,
            detail=reference["matched_detail"],
        )
        output = self.synthesis(
            lr,
            coefficient_feature,
            reference["matched_detail"],
            reference["reliability"],
            output_size,
            response_detail=reference["response_detail"],
        )
        if self.clamp_output:
            output = output.clamp(0.0, 1.0)
        if not return_aux:
            return output
        return output, {
            "reliability": reference["reliability"],
            "match_confidence": reference["match_confidence"],
            "change": reference["change"],
            "query_intensity": reference.get(
                "query_intensity", query.new_zeros(query.shape[0], 1, *output_size)
            ),
            "ref_intensity": reference.get(
                "ref_intensity", query.new_zeros(query.shape[0], 1, *output_size)
            ),
        }


class _Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            RMSNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            RMSNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return self.proj(_resize(x, size))


class _LocalDetailMatcher(nn.Module):
    """Low-resolution local matching used before high-resolution injection."""

    def __init__(
        self, channels: int, *, window_size: int, match_dim: int, transfer_dim: int
    ) -> None:
        super().__init__()
        self.channels = channels
        self.window_size = _odd_int(window_size, "match_window")
        self.radius = self.window_size // 2
        self.kernel = self.window_size * self.window_size
        self.match_dim = _positive_int(match_dim, "match_dim")
        self.transfer_dim = _positive_int(transfer_dim, "transfer_dim")
        self.query = nn.Conv2d(channels, self.match_dim, 1, bias=False)
        self.key = nn.Conv2d(channels, self.match_dim, 1, bias=False)
        self.value = nn.Conv2d(channels, self.transfer_dim, 1, bias=False)
        self.output = nn.Conv2d(self.transfer_dim, channels, 1, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(8.0)))

    def forward(
        self,
        query_feature: torch.Tensor,
        reference_low: torch.Tensor,
        reference_detail: torch.Tensor,
        *,
        return_confidence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (
            query_feature.shape != reference_low.shape
            or reference_detail.shape != query_feature.shape
        ):
            raise ValueError("local detail matcher inputs must have identical shapes")
        b, _, h, w = query_feature.shape
        query = F.normalize(self.query(query_feature), dim=1, eps=1.0e-6)
        key = F.normalize(self.key(reference_low), dim=1, eps=1.0e-6)
        value = self.value(reference_detail)
        key = F.unfold(key, self.window_size, padding=self.radius).reshape(
            b, self.match_dim, self.kernel, h * w
        )
        value = F.unfold(value, self.window_size, padding=self.radius).reshape(
            b, self.transfer_dim, self.kernel, h * w
        )
        query = query.reshape(b, self.match_dim, 1, h * w)
        logits = (query * key).sum(dim=1) * self.logit_scale.float().exp().clamp(
            1.0, 100.0
        )
        valid = F.unfold(
            query_feature.new_ones((b, 1, h, w)), self.window_size, padding=self.radius
        )
        valid = valid.reshape(b, self.kernel, h * w)
        logits = logits.masked_fill(valid < 0.5, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=1)
        matched = (
            (value * attention.unsqueeze(1))
            .sum(dim=2)
            .reshape(b, self.transfer_dim, h, w)
        )
        matched = self.output(matched)
        if not return_confidence:
            return matched
        entropy = -(
            attention.clamp_min(1.0e-6) * attention.clamp_min(1.0e-6).log()
        ).sum(dim=1)
        confidence = (
            (1.0 - entropy / math.log(float(self.kernel)))
            .clamp(0.0, 1.0)
            .reshape(b, 1, h, w)
        )
        return matched, confidence


class ReliabilityField(nn.Module):
    """Physics-conditioned spatial reference reliability."""

    def __init__(
        self,
        reference_kind: str,
        *,
        temporal_match_confidence_floor: float = 0.15,
    ) -> None:
        super().__init__()
        if (
            isinstance(temporal_match_confidence_floor, bool)
            or not isinstance(temporal_match_confidence_floor, (int, float))
            or not math.isfinite(float(temporal_match_confidence_floor))
            or not 0.0 <= float(temporal_match_confidence_floor) < 1.0
        ):
            raise ValueError("temporal_match_confidence_floor must be in [0, 1)")
        self.reference_kind = reference_kind
        self.temporal_match_confidence_floor = float(temporal_match_confidence_floor)
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )
        nn.init.constant_(self.net[-1].bias, -1.0)
        self.change_net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )
        nn.init.constant_(self.change_net[-1].bias, -1.0)

    def forward(
        self,
        query_intensity: torch.Tensor,
        reference_intensity: torch.Tensor,
        reference_detail: torch.Tensor,
        match_confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_size = query_intensity.shape[-2:]
        reference_intensity = _resize(reference_intensity, output_size)
        reference_detail = _resize(reference_detail, output_size)
        match_confidence = _resize(match_confidence, output_size)
        qx, qy, qmag = _gray_gradient(query_intensity)
        rx, ry, rmag = _gray_gradient(reference_intensity)
        correlation = (qx * rx + qy * ry) / (qmag * rmag + 1.0e-4)
        correlation = correlation.clamp(-1.0, 1.0)
        if self.reference_kind == "pan":
            # PAN and MS are different radiometric sensors.  Their absolute
            # low-frequency values are not a valid confidence signal, so
            # compare locally standardized intensities instead.
            q_mean = query_intensity.mean(dim=(-2, -1), keepdim=True)
            r_mean = reference_intensity.mean(dim=(-2, -1), keepdim=True)
            q_std = query_intensity.std(
                dim=(-2, -1), keepdim=True, unbiased=False
            ).clamp_min(1.0e-3)
            r_std = reference_intensity.std(
                dim=(-2, -1), keepdim=True, unbiased=False
            ).clamp_min(1.0e-3)
            low_difference = (
                (query_intensity - q_mean) / q_std
                - (reference_intensity - r_mean) / r_std
            ).abs()
        else:
            low_difference = (query_intensity - reference_intensity).abs()
        detail_energy = reference_detail.abs().mean(dim=1, keepdim=True)
        features = torch.cat(
            (low_difference, correlation, detail_energy, qmag, rmag, match_confidence),
            dim=1,
        )
        learned = torch.sigmoid(self.net(features))
        if self.reference_kind == "pan":
            prior = torch.sigmoid(
                3.0 * correlation - 1.0 * low_difference - 0.15 * detail_energy
            )
            change = torch.zeros_like(prior)
        else:
            prior = torch.sigmoid(2.0 * correlation - 4.0 * low_difference)
            change = torch.sigmoid(
                self.change_net(torch.cat((low_difference, (qmag - rmag).abs()), dim=1))
            )
            if self.reference_kind == "stf":
                prior = prior * (1.0 - change)
        # A co-registered sensor reference (PAN or MSI) should not be muted
        # solely because the learned matcher is still untrained.  Temporal
        # references keep the strict confidence product because misalignment
        # and scene change are genuine failure modes.
        if self.reference_kind == "pan":
            match_factor = 0.5 + 0.5 * match_confidence
        elif self.reference_kind == "mhf":
            match_factor = 0.25 + 0.75 * match_confidence
        elif self.reference_kind == "stf":
            # Query/key features are unaligned at initialization, so entropy
            # confidence alone collapses every reference path to nearly zero.
            # Apply the floor only after the temporal change prior: changed
            # regions still retain a near-zero reliability.
            match_factor = (
                self.temporal_match_confidence_floor
                + (1.0 - self.temporal_match_confidence_floor) * match_confidence
            )
        else:
            match_factor = match_confidence
        reliability = (prior * (0.5 + 0.5 * learned) * match_factor).clamp(0.0, 1.0)
        return reliability, change


class HaarDWT2D(nn.Module):
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        return haar_dwt2d(x)


class HaarIDWT2D(nn.Module):
    def forward(
        self,
        low: torch.Tensor,
        detail: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        return haar_idwt2d(low, detail, output_size)


# RDMPan lives in rdm_pan.py.  RDMStf lives in rdm_stf.py.  Both are
# dedicated dual-stream networks, not gated dual-grid RefSR specialists.


class RDMMhf(RDMRefSR):
    """Multispectral/hyperspectral fusion specialist (LR-HSI + HR-MSI)."""

    def __init__(self, **kwargs: Any) -> None:
        kind = kwargs.get("reference_kind")
        if kind is not None and normalize_reference_kind(kind) != "mhf":
            raise ValueError(
                "RDMMhf is the MS/HS fusion model; use rdm_pan or rdm_stf"
            )
        kwargs.setdefault("alignment", True)
        kwargs["reference_kind"] = "mhf"
        super().__init__(**kwargs)


__all__ = [
    "FourDirectionMamba",
    "HaarDWT2D",
    "HaarIDWT2D",
    "HybridStateBlock",
    "QShift",
    "RDMMhf",
    "RDMRefSR",
    "RMSNorm2d",
    "ReliabilityField",
    "SharedDirectionalRWKV",
    "TrueMambaScan",
    "haar_dwt2d",
    "haar_idwt2d",
    "normalize_reference_kind",
]
