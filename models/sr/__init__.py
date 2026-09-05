"""Single-image super-resolution model families and registry."""

from .registry import SRModelAdapter, build_model, get_adapter, list_models, register_adapter

__all__ = [
    "SRModelAdapter",
    "build_model",
    "get_adapter",
    "list_models",
    "register_adapter",
]
