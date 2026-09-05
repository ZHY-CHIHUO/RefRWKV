"""RefSRWKV adapter for the direct RefSR model registry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from ..registry import RefSRModelAdapter, register_adapter
from .model import RefSRWKV


_MODEL_FIELDS = {
    "inp_channels",
    "out_channels",
    "dim",
    "num_blocks",
    "num_refinement_blocks",
    "upsampler",
    "color_match",
    "drop_path_rate",
    "hidden_rate",
    "ref_channels",
    "windows",
    "fusion_match",
    "decoder_refusion",
    "global_latent_blocks",
    "ref_encoder",
}


class RefSRWKVAdapter(RefSRModelAdapter):
    """Build RefSRWKV without leaking its constructor into other families."""

    name = "refsrwkv"

    def build(self, model_config: Mapping[str, Any], *, scale: int) -> nn.Module:
        kwargs = {key: value for key, value in model_config.items() if key in _MODEL_FIELDS}
        kwargs["scale"] = int(scale)
        return RefSRWKV(**kwargs)


register_adapter(RefSRWKVAdapter())

__all__ = ["RefSRWKVAdapter"]
