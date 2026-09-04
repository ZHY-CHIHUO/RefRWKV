"""Baseline models and the shared SISR experiment protocol."""

from .registry import build_model, get_adapter, list_models

__all__ = ["build_model", "get_adapter", "list_models"]
