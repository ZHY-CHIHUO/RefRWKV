"""Registry adapters for the three RDM specialists.

``rdm_pan``, ``rdm_stf`` and ``rdm_mhf`` are separate models with locked
physical contracts.  ``rdm_refsr`` remains as a compatibility entry that
still reads ``reference_kind``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from .rdm_pan import RDMPan
from .rdm_stf import RDMStf
from .rdm_refsr import RDMMhf, RDMRefSR, normalize_reference_kind
from ..registry import RefSRModelAdapter, register_adapter


_MODEL_FIELDS = {
    "inp_channels",
    "out_channels",
    "ref_channels",
    "target_channels",
    "dim",
    "depths",
    "decoder_depths",
    "channel_multipliers",
    "scale",
    "reference_kind",
    "mamba_stages",
    "reference_condition_stages",
    "detail_injection_stages",
    "mamba_d_state",
    "mamba_d_conv",
    "mamba_expand",
    "high_order",
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

_RDM_TRAIN_NAMES = ("rdm_refsr", "rdm_pan", "rdm_stf", "rdm_mhf")


def _filtered_kwargs(model_config: Mapping[str, Any], *, scale: int) -> dict[str, Any]:
    kwargs = {key: value for key, value in model_config.items() if key in _MODEL_FIELDS}
    kwargs["scale"] = int(scale)
    return kwargs


class RDMRefSRAdapter(RefSRModelAdapter):
    """Compatibility adapter that still dispatches on ``reference_kind``."""

    name = "rdm_refsr"

    def build(self, model_config: Mapping[str, Any], *, scale: int) -> nn.Module:
        kwargs = _filtered_kwargs(model_config, scale=scale)
        if "reference_kind" in kwargs:
            kwargs["reference_kind"] = normalize_reference_kind(kwargs["reference_kind"])
        if kwargs.get("reference_kind") == "pan":
            raise ValueError(
                "model.name=rdm_refsr no longer supports pansharpening; "
                "use model.name=rdm_pan (configs/models/refsr/rdm_pan.yaml)"
            )
        return RDMRefSR(**kwargs)

    def describe(self, model_config: Mapping[str, Any], *, scale: int) -> dict[str, Any]:
        result = super().describe(model_config, scale=scale)
        result.update(
            {
                "implementation": "dual_grid_rwkv_mamba",
                "reference_kind": normalize_reference_kind(
                    model_config.get("reference_kind", "stf")
                ),
                "family": "rdm",
                "official_mamba_backend": "mamba_ssm.Mamba",
                "wkv_backend": "kernels.wkv.RUN_CUDA",
            }
        )
        return result


class RDMPanAdapter(RefSRModelAdapter):
    """Pansharpening specialist: MS LR + PAN HR."""

    name = "rdm_pan"

    def build(self, model_config: Mapping[str, Any], *, scale: int) -> nn.Module:
        kwargs = _filtered_kwargs(model_config, scale=scale)
        kwargs.pop("reference_kind", None)
        return RDMPan(**kwargs)

    def describe(self, model_config: Mapping[str, Any], *, scale: int) -> dict[str, Any]:
        result = super().describe(model_config, scale=scale)
        result.update(
            {
                "implementation": "hr_dual_stream_rwkv_mamba",
                "reference_kind": "pan",
                "family": "rdm",
                "task": "pansharpening",
                "official_mamba_backend": "mamba_ssm.Mamba",
                "wkv_backend": "kernels.wkv.RUN_CUDA",
            }
        )
        return result


class RDMStfAdapter(RefSRModelAdapter):
    """Spatio-temporal fusion specialist: same-band LR/Ref, different date/GSD."""

    name = "rdm_stf"

    def build(self, model_config: Mapping[str, Any], *, scale: int) -> nn.Module:
        kwargs = _filtered_kwargs(model_config, scale=scale)
        kwargs.pop("reference_kind", None)
        return RDMStf(**kwargs)

    def describe(self, model_config: Mapping[str, Any], *, scale: int) -> dict[str, Any]:
        result = super().describe(model_config, scale=scale)
        result.update(
            {
                "implementation": "hr_dual_stream_rwkv_mamba_stf",
                "reference_kind": "stf",
                "family": "rdm",
                "task": "spatio_temporal_fusion",
                "inputs": "(c0, f0, c1)",
                "call": "forward(lr=C1, ref=F0, c0=C0)",
                "official_mamba_backend": "mamba_ssm.Mamba",
                "wkv_backend": "kernels.wkv.RUN_CUDA",
            }
        )
        return result


class RDMMhfAdapter(RefSRModelAdapter):
    """Multispectral/hyperspectral fusion specialist."""

    name = "rdm_mhf"

    def build(self, model_config: Mapping[str, Any], *, scale: int) -> nn.Module:
        kwargs = _filtered_kwargs(model_config, scale=scale)
        kwargs.pop("reference_kind", None)
        return RDMMhf(**kwargs)

    def describe(self, model_config: Mapping[str, Any], *, scale: int) -> dict[str, Any]:
        result = super().describe(model_config, scale=scale)
        result.update(
            {
                "implementation": "dual_grid_rwkv_mamba",
                "reference_kind": "mhf",
                "family": "rdm",
                "task": "ms_hs_fusion",
                "official_mamba_backend": "mamba_ssm.Mamba",
                "wkv_backend": "kernels.wkv.RUN_CUDA",
            }
        )
        return result


register_adapter(RDMRefSRAdapter())
register_adapter(RDMPanAdapter())
register_adapter(RDMStfAdapter())
register_adapter(RDMMhfAdapter())

__all__ = [
    "RDMMhfAdapter",
    "RDMPanAdapter",
    "RDMRefSRAdapter",
    "RDMStfAdapter",
    "_RDM_TRAIN_NAMES",
]
