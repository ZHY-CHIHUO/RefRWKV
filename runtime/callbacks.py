"""Reusable PyTorch Lightning callbacks for training control."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint


logger = logging.getLogger(__name__)


class StopOnLearningRate(Callback):
    """Stop once every optimizer parameter group reaches the configured floor."""

    def __init__(self, min_lr: float, *, eps: float = 1.0e-12) -> None:
        super().__init__()
        if min_lr <= 0:
            raise ValueError("min_lr must be positive")
        self.min_lr = float(min_lr)
        self.eps = float(eps)
        self._stopped = False

    def _check(self, trainer) -> None:
        if self._stopped or not getattr(trainer, "optimizers", None):
            return
        learning_rates = [
            float(group["lr"])
            for optimizer in trainer.optimizers
            for group in optimizer.param_groups
        ]
        if learning_rates and max(learning_rates) <= self.min_lr + self.eps:
            trainer.should_stop = True
            self._stopped = True
            logger.info(
                "Stopping training because all learning rates reached %.3e",
                self.min_lr,
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._check(trainer)

    def on_validation_end(self, trainer, pl_module) -> None:
        self._check(trainer)


class IntervalEpochCheckpoint(Callback):
    """Save ``{epoch}.ckpt`` every N completed epochs, 1-based like FusionMamba.

    Official FusionMamba writes ``20.pth``, ``40.pth``, … ``420.pth`` and does
    not delete older files from a ``val/loss`` top-k rule.  Lightning epoch
    indices are 0-based, so this callback stores ``current_epoch + 1``.
    """

    def __init__(self, dirpath: str | Path, every_n_epochs: int = 20) -> None:
        super().__init__()
        if isinstance(every_n_epochs, bool) or not isinstance(every_n_epochs, int) or every_n_epochs < 1:
            raise ValueError("every_n_epochs must be a positive integer")
        self.dirpath = Path(dirpath)
        self.every_n_epochs = int(every_n_epochs)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        epoch = int(trainer.current_epoch) + 1
        if epoch % self.every_n_epochs:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(self.dirpath / f"{epoch}.ckpt"))


def build_checkpoint_callbacks(train: dict[str, Any], checkpoint_dir: str | Path) -> list[Callback]:
    """Build checkpoint callbacks.

    ``train.ckpt_every_n_epochs`` follows FusionMamba: keep every Nth epoch
    forever, plus ``last.ckpt``.  Otherwise fall back to ``val/loss`` top-k.
    Extra ``train.checkpoint_monitors`` copies never delete the periodic files.
    """
    if not isinstance(train, dict):
        raise TypeError("train must be a mapping")
    checkpoint_dir = Path(checkpoint_dir)
    callbacks: list[Callback] = []
    ckpt_every = train.get("ckpt_every_n_epochs")
    if ckpt_every is not None:
        if isinstance(ckpt_every, bool) or not isinstance(ckpt_every, int) or ckpt_every < 1:
            raise ValueError("train.ckpt_every_n_epochs must be a positive integer")
        callbacks.append(IntervalEpochCheckpoint(checkpoint_dir, ckpt_every))
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="epoch={epoch:04d}-step={step:06d}",
                auto_insert_metric_name=False,
                save_top_k=0,
                save_last=True,
            )
        )
        monitors = train.get("checkpoint_monitors") or ()
        if isinstance(monitors, dict):
            monitors = [monitors]
        if not isinstance(monitors, (list, tuple)):
            raise ValueError("train.checkpoint_monitors must be a list of mappings")
        for spec in monitors:
            if not isinstance(spec, dict) or "monitor" not in spec:
                raise ValueError("each checkpoint monitor needs a monitor name")
            mode = str(spec.get("mode", "min")).strip().lower()
            if mode not in {"min", "max"}:
                raise ValueError("checkpoint monitor mode must be min or max")
            filename = str(spec.get("filename") or spec["monitor"]).replace("/", "_")
            callbacks.append(
                ModelCheckpoint(
                    dirpath=str(checkpoint_dir),
                    filename=filename + "-{epoch:04d}",
                    auto_insert_metric_name=False,
                    monitor=str(spec["monitor"]),
                    mode=mode,
                    save_top_k=int(spec.get("save_top_k", 1)),
                    save_last=False,
                )
            )
        return callbacks

    callbacks.append(
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="epoch={epoch:04d}-step={step:06d}",
            auto_insert_metric_name=False,
            monitor="val/loss",
            mode="min",
            save_top_k=int(train.get("save_top_k", 3)),
            save_last=True,
        )
    )
    return callbacks


__all__ = [
    "IntervalEpochCheckpoint",
    "StopOnLearningRate",
    "build_checkpoint_callbacks",
]
