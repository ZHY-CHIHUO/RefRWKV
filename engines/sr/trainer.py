"""Trainer for registry-based single-image SR models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from engines.base_trainer import BaseTrainer
from models.sr import build_model


class SRTrainer(BaseTrainer):
    """Common SR optimization around any model registered in ``models.sr``."""

    def __init__(self, model: nn.Module, config: Mapping[str, Any]) -> None:
        data = config.get("data", {})
        super().__init__(model, config, lr_key=str(data.get("lr_key", "lr")), hr_key=str(data.get("hr_key", "hr")))
        loss = config.get("loss", {})
        loss_name = str(loss.get("name", "l1")).lower()
        self.loss_name = "mse" if loss_name == "l2" else loss_name
        if self.loss_name not in {"l1", "mse", "charbonnier"}:
            raise ValueError("loss.name must be l1, mse, or charbonnier")
        self.charbonnier_eps = float(loss.get("eps", 1.0e-3))
        if self.charbonnier_eps <= 0:
            raise ValueError("loss.eps must be positive")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SRTrainer":
        data = config.get("data", {})
        model = build_model(config["model"], scale=int(data["scale"]))
        return cls(model, config)

    def _unpack(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError("SR batches must be mappings")
        if self.lr_key not in batch or self.hr_key not in batch:
            raise KeyError(f"SR batch must contain {self.lr_key!r} and {self.hr_key!r}")
        return batch[self.lr_key], batch[self.hr_key]

    def _loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(f"SR output/HR geometry mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
        residual = prediction - target
        if self.loss_name == "l1":
            return residual.abs().mean()
        if self.loss_name == "mse":
            return residual.square().mean()
        return (residual.square() + self.charbonnier_eps**2).sqrt().mean()

    def _train_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        lr, hr = self._unpack(batch)
        return self._loss(self(lr), hr)

    def _eval_step(self, batch: Any, batch_idx: int, *, stage: str) -> dict[str, torch.Tensor]:
        lr, hr = self._unpack(batch)
        prediction = self(lr)
        return {"loss": self._loss(prediction, hr), **self.image_metrics(prediction, hr)}

