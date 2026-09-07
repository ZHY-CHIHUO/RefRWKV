"""Registry adapters for the portable SR comparison baselines."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from .baselines import BicubicSR, EDSRNet, HATNet, MambaIRv2Net, RCANNet
from .registry import SRModelAdapter, register_adapter


def _int(config: Mapping[str, Any], key: str, default: int, minimum: int = 1) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"model.{key} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _float(config: Mapping[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"model.{key} must be numeric, got {value!r}")
    return float(value)


class _Adapter(SRModelAdapter):
    implementation = "native_compatibility"
    official_repo = ""

    def _description(self, config: Mapping[str, Any], scale: int, **extra: Any) -> dict[str, Any]:
        result = {
            "name": self.name,
            "scale": int(scale),
            "implementation": self.implementation,
            "official_repo": self.official_repo,
        }
        result.update(extra)
        return result


class EDSRAdapter(_Adapter):
    name = "edsr"
    official_repo = "https://github.com/sanghyun-son/EDSR-PyTorch"

    def build(self, model_config: Mapping[str, Any], *, scale: int) -> nn.Module:
        return EDSRNet(
            scale=int(scale),
            dim=_int(model_config, "dim", 64, 8),
            num_blocks=_int(model_config, "num_blocks", 16),
            res_scale=_float(model_config, "res_scale", 0.1),
        )

    def describe(self, model_config: Mapping[str, Any], *, scale: int) -> dict[str, Any]:
        return self._description(
            model_config,
            scale,
            variant=str(model_config.get("variant", "EDSR")),
            dim=_int(model_config, "dim", 64, 8),
            num_blocks=_int(model_config, "num_blocks", 16),
            res_scale=_float(model_config, "res_scale", 0.1),
        )


class RCANAdapter(_Adapter):
    name = "rcan"
    official_repo = "https://github.com/yulunzhang/RCAN"

    def build(self, model_config: Mapping[str, Any], *, scale: int) -> nn.Module:
        return RCANNet(
            scale=int(scale),
            dim=_int(model_config, "dim", 64, 8),
            num_groups=_int(model_config, "num_groups", 10),
            blocks_per_group=_int(model_config, "blocks_per_group", 20),
            reduction=_int(model_config, "reduction", 16),
        )

    def describe(self, model_config: Mapping[str, Any], *, scale: int) -> dict[str, Any]:
        return self._description(
            model_config,
            scale,
            variant=str(model_config.get("variant", "RCAN")),
            dim=_int(model_config, "dim", 64, 8),
            num_groups=_int(model_config, "num_groups", 10),
            blocks_per_group=_int(model_config, "blocks_per_group", 20),
            reduction=_int(model_config, "reduction", 16),
        )


class HATAdapter(_Adapter):
    name = "hat"
    official_repo = "https://github.com/XPixelGroup/HAT"

    def build(self, model_config: Mapping[str, Any], *, scale: int) -> nn.Module:
        return HATNet(
            scale=int(scale),
            dim=_int(model_config, "dim", 96, 8),
            num_blocks=_int(model_config, "num_blocks", 12),
            reduction=_int(model_config, "reduction", 16),
            window_size=_int(model_config, "window_size", 16),
        )

    def describe(self, model_config: Mapping[str, Any], *, scale: int) -> dict[str, Any]:
        return self._description(
            model_config,
            scale,
            variant=str(model_config.get("variant", "HAT-S")),
            dim=_int(model_config, "dim", 96, 8),
            num_blocks=_int(model_config, "num_blocks", 12),
            reduction=_int(model_config, "reduction", 16),
            window_size=_int(model_config, "window_size", 16),
        )


class MambaIRv2Adapter(_Adapter):
    name = "mambairv2"
    official_repo = "https://github.com/csguoh/MambaIR"

    def build(self, model_config: Mapping[str, Any], *, scale: int) -> nn.Module:
        return MambaIRv2Net(
            scale=int(scale),
            dim=_int(model_config, "dim", 64, 8),
            num_blocks=_int(model_config, "num_blocks", 12),
            expansion=_int(model_config, "expansion", 2),
        )

    def describe(self, model_config: Mapping[str, Any], *, scale: int) -> dict[str, Any]:
        return self._description(
            model_config,
            scale,
            variant=str(model_config.get("variant", "MambaIRv2")),
            dim=_int(model_config, "dim", 64, 8),
            num_blocks=_int(model_config, "num_blocks", 12),
            expansion=_int(model_config, "expansion", 2),
            official_backend=str(model_config.get("official_backend", "mamba_ssm")),
        )


class BicubicAdapter(_Adapter):
    name = "bicubic"
    implementation = "reference_only"
    official_repo = ""

    def build(self, model_config: Mapping[str, Any], *, scale: int) -> nn.Module:
        return BicubicSR(scale=int(scale))

    def describe(self, model_config: Mapping[str, Any], *, scale: int) -> dict[str, Any]:
        return self._description(model_config, scale, trainable=False, params=0)


for _adapter in (BicubicAdapter(), EDSRAdapter(), RCANAdapter(), HATAdapter(), MambaIRv2Adapter()):
    register_adapter(_adapter)


__all__ = [
    "BicubicAdapter",
    "EDSRAdapter",
    "HATAdapter",
    "MambaIRv2Adapter",
    "RCANAdapter",
]
