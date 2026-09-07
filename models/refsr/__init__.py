"""Reference-based super-resolution model families.

Direct models use the shared ``forward(lr, ref) -> sr`` contract and are
registered lazily.  RefDiffRWKV remains lazy because importing its diffusion
stack is intentionally optional.
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
