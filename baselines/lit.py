"""Lightning module shared by all HR/LR single-image SR baselines."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn

from .common import EMA, gaussian_ssim, per_image_psnr


class LitSISRBaseline(pl.LightningModule):
    """Train one registry model with a common loss, EMA, and LR schedule."""

    def __init__(self, model: nn.Module, config: Mapping[str, Any]) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self.model = model
        self.config = copy.deepcopy(dict(config))
        train = self.config.get("train", {})
        loss = self.config.get("loss", {})
        data = self.config.get("data", {})
        if not isinstance(train, dict) or not isinstance(loss, dict) or not isinstance(data, dict):
            raise ValueError("train, loss, and data must be mappings")

        self.learning_rate = float(train.get("learning_rate", 1.0e-4))
        self.weight_decay = float(train.get("weight_decay", 0.0))
        self.betas = tuple(float(value) for value in train.get("adam_betas", [0.9, 0.999]))
        if self.learning_rate <= 0 or self.weight_decay < 0 or len(self.betas) != 2:
            raise ValueError("invalid Adam optimizer configuration")
        if any(value < 0 or value >= 1 for value in self.betas):
            raise ValueError("Adam betas must be in [0, 1)")

        self.loss_name = str(loss.get("name", "l1")).lower()
        if self.loss_name == "l2":
            self.loss_name = "mse"
        if self.loss_name not in {"l1", "mse", "charbonnier"}:
            raise ValueError("loss.name must be l1, mse, or charbonnier")
        self.charbonnier_eps = float(loss.get("eps", 1.0e-3))
        if self.charbonnier_eps <= 0:
            raise ValueError("loss.eps must be positive")
        self.lr_key = str(data.get("lr_key", "lr"))
        self.hr_key = str(data.get("hr_key", "hr"))
        self.use_ema = bool(train.get("use_ema", True))
        self.ema = EMA(float(train.get("ema_decay", 0.999))) if self.use_ema else None
        self._ema_last_step = -1
        self._plateau_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
        self._signature: dict[str, Any] | None = None
        self.save_hyperparameters(
            {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "adam_betas": self.betas,
                "loss_name": self.loss_name,
                "use_ema": self.use_ema,
            }
        )

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        return self.model(lr)

    def _unpack_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError("baseline batch must be a mapping with lr and hr")
        missing = [key for key in (self.lr_key, self.hr_key) if key not in batch]
        if missing:
            raise KeyError(f"baseline batch is missing keys: {missing}")
        return batch[self.lr_key], batch[self.hr_key]

    def _loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"model output and HR target must match: {tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        if self.loss_name == "l1":
            return (prediction - target).abs().mean()
        if self.loss_name == "mse":
            return (prediction - target).square().mean()
        return ((prediction - target).square() + self.charbonnier_eps**2).sqrt().mean()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        lr, hr = self._unpack_batch(batch)
        prediction = self(lr)
        loss = self._loss(prediction, hr)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=hr.shape[0])
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        lr, hr = self._unpack_batch(batch)
        prediction = self(lr)
        loss = self._loss(prediction, hr)
        metric_prediction = prediction.clamp(-1.0, 1.0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=hr.shape[0])
        self.log(
            "val_psnr",
            per_image_psnr(metric_prediction, hr).mean(),
            on_step=False,
            on_epoch=True,
            batch_size=hr.shape[0],
        )
        self.log(
            "val_ssim",
            gaussian_ssim(metric_prediction, hr).mean(),
            on_step=False,
            on_epoch=True,
            batch_size=hr.shape[0],
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        lr, hr = self._unpack_batch(batch)
        prediction = self(lr)
        loss = self._loss(prediction, hr)
        metric_prediction = prediction.clamp(-1.0, 1.0)
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=hr.shape[0])
        self.log(
            "test_psnr",
            per_image_psnr(metric_prediction, hr).mean(),
            on_step=False,
            on_epoch=True,
            batch_size=hr.shape[0],
        )
        self.log(
            "test_ssim",
            gaussian_ssim(metric_prediction, hr).mean(),
            on_step=False,
            on_epoch=True,
            batch_size=hr.shape[0],
        )
        return loss

    def on_train_start(self) -> None:
        if self.ema is not None:
            self.ema._init(self.model)

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self.ema is None:
            return
        current_step = int(self.global_step)
        if current_step > self._ema_last_step:
            self.ema.update(self.model)
            self._ema_last_step = current_step

    def _apply_ema(self) -> None:
        if self.ema is not None:
            self.ema.apply(self.model)

    def _restore_ema(self) -> None:
        if self.ema is not None:
            self.ema.restore(self.model)

    def on_validation_model_eval(self) -> None:
        self._apply_ema()

    def on_validation_model_train(self) -> None:
        self._restore_ema()

    def on_validation_start(self) -> None:
        self._apply_ema()

    def on_validation_end(self) -> None:
        self._restore_ema()

    def on_test_model_eval(self) -> None:
        self._apply_ema()

    def on_test_model_train(self) -> None:
        self._restore_ema()

    def on_test_start(self) -> None:
        self._apply_ema()

    def on_test_end(self) -> None:
        self._restore_ema()

    def on_validation_epoch_end(self) -> None:
        self._step_plateau_scheduler()

    def _step_plateau_scheduler(self) -> None:
        train = self.config["train"]
        if str(train.get("lr_scheduler", "plateau")).lower() != "plateau":
            return
        try:
            trainer = self.trainer
        except RuntimeError:
            return
        if trainer is None or getattr(trainer, "sanity_checking", False):
            return
        run_fn = getattr(getattr(trainer, "state", None), "fn", None)
        run_fn = getattr(run_fn, "value", run_fn)
        if run_fn is not None and str(run_fn).lower() not in {"fit", "fitting"}:
            return
        scheduler = self._plateau_scheduler
        if scheduler is None:
            for item in getattr(trainer, "lr_scheduler_configs", ()):
                candidate = getattr(item, "scheduler", None)
                if isinstance(candidate, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler = candidate
                    self._plateau_scheduler = candidate
                    break
        if scheduler is None:
            return
        metric = (getattr(trainer, "callback_metrics", {}) or {}).get("val_loss")
        if metric is None:
            raise RuntimeError("ReduceLROnPlateau requires val_loss after every validation run")
        if torch.is_tensor(metric):
            metric = metric.detach().float().item()
        metric = float(metric)
        if not math.isfinite(metric):
            raise RuntimeError(f"val_loss must be finite, got {metric}")
        scheduler.step(metric)

    def lr_scheduler_step(self, scheduler: Any, metric: Any) -> None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # ``on_validation_epoch_end`` owns the only plateau update.  This
            # avoids Lightning demanding val_loss on sparse-validation epochs.
            return
        scheduler.step(metric) if metric is not None else scheduler.step()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        train = self.config["train"]
        scheduler_name = str(train.get("lr_scheduler", "plateau")).lower()
        if scheduler_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(train.get("lr_factor", 0.5)),
                patience=int(train.get("lr_patience", 3)),
                threshold=float(train.get("lr_threshold", 1.0e-5)),
                threshold_mode="abs",
                min_lr=float(train.get("lr_min", 1.0e-7)),
            )
            self._plateau_scheduler = scheduler
            return [optimizer], [
                {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "reduce_on_plateau": False,
                }
            ]
        if scheduler_name != "cosine":
            raise ValueError("train.lr_scheduler must be plateau or cosine")
        max_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0)
        if max_steps < 1:
            max_steps = max(1, int(train.get("max_steps", 100000)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_steps,
            eta_min=float(train.get("lr_min", 1.0e-7)),
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        configured = self.config["train"].get("grad_clip_norm", 1.0)
        clip_value = configured if gradient_clip_val is None else gradient_clip_val
        if clip_value is not None and float(clip_value) > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=float(clip_value),
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["baseline_config"] = copy.deepcopy(self.config)
        if self._signature is not None:
            checkpoint["baseline_experiment_signature"] = copy.deepcopy(self._signature)
        if self.ema is not None:
            checkpoint["baseline_ema_state"] = self.ema.state_dict()
        if str(self.config["train"].get("lr_scheduler", "plateau")).lower() == "plateau":
            checkpoint["plateau_step_unit"] = "validation"

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.ema is not None and isinstance(checkpoint.get("baseline_ema_state"), Mapping):
            self.ema.load_state_dict(checkpoint["baseline_ema_state"])
