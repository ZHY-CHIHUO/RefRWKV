"""Reference-based SR training engines."""

from .refsrwkv_trainer import RefSRWKVTrainer
from .rdm_refsr_trainer import RDMRefSRTrainer
from .trainer import RefSRTrainer

__all__ = ["RDMRefSRTrainer", "RefSRTrainer", "RefSRWKVTrainer"]


def __getattr__(name):
    if name == "RefDiffRWKVTrainer":
        from .refdiff_trainer import RefDiffRWKVTrainer

        return RefDiffRWKVTrainer
    raise AttributeError(name)
