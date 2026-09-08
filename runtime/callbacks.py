"""Reusable PyTorch Lightning callbacks for training control."""

from __future__ import annotations

import logging

from pytorch_lightning import Callback


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


__all__ = ["StopOnLearningRate"]
