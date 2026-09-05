"""Reference-based SR training engines."""

from .refsrwkv_trainer import RefSRWKVTrainer

__all__ = ["RefSRWKVTrainer"]


def __getattr__(name):
    if name in {"RefDiffRWKVTrainer", "RefSRTrainer"}:
        from .refdiff_trainer import RefDiffRWKVTrainer

        return RefDiffRWKVTrainer if name == "RefDiffRWKVTrainer" else RefSRWKVTrainer
    raise AttributeError(name)
