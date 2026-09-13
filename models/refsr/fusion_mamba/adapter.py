"""Project adapter for the official FusionMamba pansharpening network."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..registry import RefSRModelAdapter, register_adapter
from .u2net import U2Net


class FusionMambaRefSR(U2Net):
    """Official U2Net on the repository's ``[0, 1]`` tensor contract.

    Parameter names stay identical to the upstream model so official
    checkpoints such as ``420.pth`` load without key remapping.
    """

    def forward(self, lr: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if lr.ndim != 4 or ref.ndim != 4:
            raise ValueError("FusionMamba expects NCHW LR/MS and PAN tensors")
        out01 = super().forward(lr.float().clamp(0.0, 1.0), ref.float().clamp(0.0, 1.0))
        return out01.clamp(0.0, 1.0)


_MODEL_FIELDS = {"dim", "pan_dim", "ms_dim", "input_h", "input_w", "scale"}


class FusionMambaAdapter(RefSRModelAdapter):
    name = "fusion_mamba"

    def build(self, model_config: Mapping[str, Any], *, scale: int) -> FusionMambaRefSR:
        dim = int(model_config.get("dim", 32))
        pan_dim = int(model_config.get("pan_dim", model_config.get("ref_channels", 1)))
        ms_dim = int(model_config.get("ms_dim", model_config.get("inp_channels", 8)))
        input_h = int(model_config.get("input_h", 64))
        input_w = int(model_config.get("input_w", 64))
        configured_scale = int(model_config.get("scale", scale))
        if configured_scale != int(scale):
            configured_scale = int(scale)
        if input_h < 1 or input_w < 1:
            raise ValueError(f"FusionMamba input_h/input_w must be positive, got {input_h}x{input_w}")
        return FusionMambaRefSR(
            dim=dim,
            pan_dim=pan_dim,
            ms_dim=ms_dim,
            H=input_h,
            W=input_w,
            scale=configured_scale,
        )


register_adapter(FusionMambaAdapter())

__all__ = ["FusionMambaAdapter", "FusionMambaRefSR"]
