"""Reference-based super-resolution model families.

Only the two supported model families are exposed here: ``RefSRWKV`` and
``RefDiffRWKV``.  Heavy diffusion dependencies remain lazy.
"""

from .registry import RefSRModelAdapter, build_model, get_adapter, list_models, register_adapter
from .refsrwkv.model import RefSRWKV

__all__ = [
    "RefDiffRWKV",
    "RefSRModelAdapter",
    "RefSRWKV",
    "build_model",
    "get_adapter",
    "list_models",
    "register_adapter",
]


def __getattr__(name):
    if name == "RefDiffRWKV":
        from .RefDiffRWKV import RefDiffRWKV

        return RefDiffRWKV
    raise AttributeError(name)
