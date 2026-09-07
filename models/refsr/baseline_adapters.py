"""Registry adapters for portable TTSR, MASA-SR and DATSR baselines."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from .baselines import DATSRCompatibilityNet, MASACompatibilityNet, TTSRCompatibilityNet
from .registry import RefSRModelAdapter, register_adapter


def _int(config: Mapping[str, Any], key: str, default: int, minimum: int = 1) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"model.{key} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _float(config: Mapping[str, Any], key: str, default: float, minimum: float | None = None) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"model.{key} must be numeric, got {value!r}")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"model.{key} must be >= {minimum}, got {value!r}")
    return result


class _Adapter(RefSRModelAdapter):
    """Shared metadata and validation helpers for direct RefSR baselines."""

    implementation = "native_compatibility"
    official_repo = ""

    def describe(self, model_config: Mapping[str, Any], *, scale: int) -> dict[str, Any]:
        return {
            "name": self.name,
            "variant": str(model_config.get("variant", self.name)),
            "scale": int(scale),
            "implementation": self.implementation,
            "official_repo": self.official_repo,
        }


class TTSRAdapter(_Adapter):
    name = "ttsr"
    official_repo = "https://github.com/researchmm/TTSR"

    def build(self, model_config: Mapping[str, Any], *, scale: int) -> nn.Module:
        return TTSRCompatibilityNet(
            scale=int(scale),
            dim=_int(model_config, "dim", 64, 8),
            num_blocks=_int(model_config, "num_blocks", 8),
            texture_blocks=_int(model_config, "texture_blocks", 3),
            match_window=_int(model_config, "match_window", 7, 3),
            match_dim=_int(model_config, "match_dim", 16),
        )

    def describe(self, model_config: Mapping[str, Any], *, scale: int) -> dict[str, Any]:
        result = super().describe(model_config, scale=scale)
        result.update(
            dim=_int(model_config, "dim", 64, 8),
            num_blocks=_int(model_config, "num_blocks", 8),
            texture_blocks=_int(model_config, "texture_blocks", 3),
            match_window=_int(model_config, "match_window", 7, 3),
        )
        return result


class MASASRAdapter(_Adapter):
    name = "masa_sr"
    official_repo = "https://github.com/JIA-Lab-research/MASA-SR"

    def build(self, model_config: Mapping[str, Any], *, scale: int) -> nn.Module:
        return MASACompatibilityNet(
            scale=int(scale),
            dim=_int(model_config, "dim", 64, 8),
            num_blocks=_int(model_config, "num_blocks", 8),
            refine_blocks=_int(model_config, "refine_blocks", 4),
            match_window=_int(model_config, "match_window", 7, 3),
            match_dim=_int(model_config, "match_dim", 16),
            max_offset=_float(model_config, "max_offset", 2.0, 1.0e-6),
        )

    def describe(self, model_config: Mapping[str, Any], *, scale: int) -> dict[str, Any]:
        result = super().describe(model_config, scale=scale)
        result.update(
            dim=_int(model_config, "dim", 64, 8),
            num_blocks=_int(model_config, "num_blocks", 8),
            refine_blocks=_int(model_config, "refine_blocks", 4),
            match_window=_int(model_config, "match_window", 7, 3),
            max_offset=_float(model_config, "max_offset", 2.0, 1.0e-6),
        )
        return result


class DATSRAdapter(_Adapter):
    name = "datsr"
    official_repo = "https://github.com/caojiezhang/DATSR"

    def build(self, model_config: Mapping[str, Any], *, scale: int) -> nn.Module:
        return DATSRCompatibilityNet(
            scale=int(scale),
            dim=_int(model_config, "dim", 64, 8),
            num_blocks=_int(model_config, "num_blocks", 10),
            match_window=_int(model_config, "match_window", 7, 3),
            match_dim=_int(model_config, "match_dim", 16),
            max_offset=_float(model_config, "max_offset", 2.0, 1.0e-6),
        )

    def describe(self, model_config: Mapping[str, Any], *, scale: int) -> dict[str, Any]:
        result = super().describe(model_config, scale=scale)
        result.update(
            dim=_int(model_config, "dim", 64, 8),
            num_blocks=_int(model_config, "num_blocks", 10),
            match_window=_int(model_config, "match_window", 7, 3),
            max_offset=_float(model_config, "max_offset", 2.0, 1.0e-6),
        )
        return result


for _adapter in (TTSRAdapter(), MASASRAdapter(), DATSRAdapter()):
    register_adapter(_adapter)


__all__ = ["DATSRAdapter", "MASASRAdapter", "TTSRAdapter"]
