"""Small registry for HR/LR single-image SR model families.

Every SR model adapter builds a model from the same run configuration.  The
runner can therefore keep data loading, losses, EMA, checkpoints, and metrics
identical while model implementations stay isolated in this package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch.nn as nn


class SRModelAdapter(ABC):
    """Interface implemented by one SISR architecture family."""

    name: str

    @abstractmethod
    def build(self, model_config: Mapping[str, Any], *, scale: int) -> nn.Module:
        """Construct an untrained model for ``scale``."""

    def describe(self, model_config: Mapping[str, Any], *, scale: int) -> dict[str, Any]:
        """Return architecture metadata that can be written to results."""
        return {
            "name": self.name,
            "scale": int(scale),
            "config": dict(model_config),
        }


_ADAPTERS: dict[str, SRModelAdapter] = {}
_BUILTINS_LOADED = False


def register_adapter(adapter: SRModelAdapter) -> SRModelAdapter:
    name = str(adapter.name).strip().lower()
    if not name:
        raise ValueError("SR model adapter name must be non-empty")
    if name in _ADAPTERS:
        raise ValueError(f"SR model adapter already registered: {name}")
    adapter.name = name
    _ADAPTERS[name] = adapter
    return adapter


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    # Importing registers built-ins and keeps this core module independent of
    # any particular architecture's optional dependencies.
    from . import swinir  # noqa: F401

    _BUILTINS_LOADED = True


def get_adapter(name: str) -> SRModelAdapter:
    _load_builtins()
    normalized = str(name).strip().lower()
    try:
        return _ADAPTERS[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(_ADAPTERS)) or "none"
        raise KeyError(f"unknown SR model {name!r}; available: {available}") from exc


def build_model(model_config: Mapping[str, Any], *, scale: int) -> nn.Module:
    if not isinstance(model_config, Mapping):
        raise TypeError("model config must be a mapping")
    name = model_config.get("name") or model_config.get("id")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model.name must be a non-empty string")
    return get_adapter(name).build(model_config, scale=int(scale))


def list_models() -> tuple[str, ...]:
    _load_builtins()
    return tuple(sorted(_ADAPTERS))
