"""Registry adapter for the dual-grid RDMRefSR model.

The adapter intentionally filters the YAML mapping before construction.  This
keeps runtime metadata (``family``, ``variant`` and dataset-only fields) out of
the neural-network constructor while allowing the three physical reference
modes to share one implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from .rdm_refsr import RDMRefSR, normalize_reference_kind
from ..registry import RefSRModelAdapter, register_adapter


_MODEL_FIELDS = {
    "inp_channels",
    "out_channels",
    "ref_channels",
    "target_channels",
    "dim",
    "depths",
    "decoder_depths",
    "scale",
    "reference_kind",
    "mamba_stages",
    "reference_condition_stages",
    "detail_injection_stages",
    "mamba_d_state",
    "mamba_d_conv",
    "mamba_expand",
    "allow_cpu_mamba",
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
    "use_reference",
    "clamp_output",
}


class RDMRefSRAdapter(RefSRModelAdapter):
    """Build the research RDMRefSR architecture from a materialized config."""

    name = "rdm_refsr"

    def build(self, model_config: Mapping[str, Any], *, scale: int) -> nn.Module:
        kwargs = {key: value for key, value in model_config.items() if key in _MODEL_FIELDS}
        # The run's data.scale is authoritative.  A stale model.scale in an
        # inherited YAML must not silently produce a different output geometry.
        kwargs["scale"] = int(scale)
        if "reference_kind" in kwargs:
            kwargs["reference_kind"] = normalize_reference_kind(kwargs["reference_kind"])
        return RDMRefSR(**kwargs)

    def describe(self, model_config: Mapping[str, Any], *, scale: int) -> dict[str, Any]:
        result = super().describe(model_config, scale=scale)
        result.update(
            {
                "implementation": "dual_grid_rwkv_mamba",
                "reference_kind": normalize_reference_kind(
                    model_config.get("reference_kind", "temporal")
                ),
                "official_mamba_backend": "mamba_ssm.Mamba",
                "wkv_backend": "kernels.wkv.RUN_CUDA",
            }
        )
        return result


register_adapter(RDMRefSRAdapter())

__all__ = ["RDMRefSRAdapter"]
