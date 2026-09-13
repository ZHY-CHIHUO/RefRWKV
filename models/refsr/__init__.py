"""Reference-based super-resolution model families.

Direct models use the shared ``forward(lr, ref) -> sr`` contract and are
registered lazily.  RefDiffRWKV remains lazy because importing its diffusion
stack is intentionally optional.
"""

from .registry import RefSRModelAdapter, build_model, get_adapter, list_models, register_adapter
from .rdm_refsr import (
    FourDirectionMamba,
    HaarDWT2D,
    HaarIDWT2D,
    RDMMhf,
    RDMPan,
    RDMRefSR,
    RDMStf,
    SharedDirectionalRWKV,
    TrueMambaScan,
    normalize_reference_kind,
)
from .refsrwkv.model import RefSRWKV

__all__ = [
    "RefDiffRWKV",
    "RefSRModelAdapter",
    "RefSRWKV",
    "RDMMhf",
    "RDMPan",
    "RDMRefSR",
    "RDMStf",
    "FourDirectionMamba",
    "HaarDWT2D",
    "HaarIDWT2D",
    "SharedDirectionalRWKV",
    "TrueMambaScan",
    "normalize_reference_kind",
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
