"""Registry for direct reference-based super-resolution model families.

Registered models follow the project RefSR tensor contract:
``forward(lr, ref) -> sr`` with inputs and output in ``[-1, 1]`` and an
output spatial size of ``lr * scale``.  Diffusion systems remain a separate
RefSR model family because their inference interface includes a sampler.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch.nn as nn


class RefSRModelAdapter(ABC):
    """Construction interface for one direct RefSR architecture family."""

    name: str

    @abstractmethod
    def build(self, model_config: Mapping[str, Any], *, scale: int) -> nn.Module:
        """Construct an untrained ``forward(lr, ref)`` model for ``scale``."""


_ADAPTERS: dict[str, RefSRModelAdapter] = {}
_BUILTINS_LOADED = False


def register_adapter(adapter: RefSRModelAdapter) -> RefSRModelAdapter:
    """Register one direct RefSR adapter under its normalized model name."""
    name = str(adapter.name).strip().lower()
    if not name:
        raise ValueError("RefSR model adapter name must be non-empty")
    if name in _ADAPTERS:
        raise ValueError(f"RefSR model adapter already registered: {name}")
    adapter.name = name
    _ADAPTERS[name] = adapter
    return adapter


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    # Importing package adapters performs registration while keeping this
    # registry independent of concrete model implementations.
    from . import refsrwkv  # noqa: F401

    _BUILTINS_LOADED = True


def get_adapter(name: str) -> RefSRModelAdapter:
    """Return the adapter registered for a direct RefSR model name."""
    _load_builtins()
    normalized = str(name).strip().lower()
    try:
        return _ADAPTERS[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(_ADAPTERS)) or "none"
        raise KeyError(
            f"unknown direct RefSR model {name!r}; available: {available}"
        ) from exc


def build_model(model_config: Mapping[str, Any], *, scale: int) -> nn.Module:
    """Build a direct RefSR model from a materialized model configuration."""
    if not isinstance(model_config, Mapping):
        raise TypeError("RefSR model config must be a mapping")
    name = model_config.get("name") or model_config.get("id")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model.name must be a non-empty string")
    return get_adapter(name).build(model_config, scale=int(scale))


def list_models() -> tuple[str, ...]:
    """Return all registered direct RefSR model names."""
    _load_builtins()
    return tuple(sorted(_ADAPTERS))


__all__ = [
    "RefSRModelAdapter",
    "build_model",
    "get_adapter",
    "list_models",
    "register_adapter",
]
