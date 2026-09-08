"""Common Lightning lifecycle for all trainable model families.

The engine owns the parts that must stay identical across SR and RefSR:
optimizer construction, optional EMA, validation metrics, checkpoint metadata,
gradient clipping and the ``train/val/test`` dispatch.  Model-specific engines
only implement ``_train_step`` and ``_eval_step``.
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn

from runtime.common import EMA, gaussian_ssim, per_image_psnr
from runtime.tiling import tiled_forward


class BaseTrainer(pl.LightningModule, ABC):
    """Reusable training lifecycle shared by SR and RefSR engines."""

    def __init__(
        self,
        model: nn.Module,
        config: Mapping[str, Any] | None = None,
        *,
        lr_key: str = "lr",
        hr_key: str = "hr",
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self.model = model
        self.config = copy.deepcopy(dict(config or {}))
        train = self.config.setdefault("train", {})
        if not isinstance(train, dict):
            raise ValueError("train must be a mapping")
        self.lr_key = str(lr_key)
        self.hr_key = str(hr_key)
        self.learning_rate = float(train.get("learning_rate", 1.0e-4))
        self.weight_decay = float(train.get("weight_decay", 0.0))
        self.betas = tuple(float(v) for v in train.get("adam_betas", (0.9, 0.999)))
        if self.learning_rate <= 0 or self.weight_decay < 0 or len(self.betas) != 2:
            raise ValueError("invalid optimizer configuration")
        if any(not 0 <= v < 1 for v in self.betas):
            raise ValueError("adam_betas must be in [0, 1)")
        self.use_ema = bool(train.get("use_ema", True))
        self.ema = EMA(float(train.get("ema_decay", 0.999))) if self.use_ema else None
        self._ema_last_step = -1
        self._plateau_scheduler = None
        self._checkpoint_train_config: dict[str, Any] | None = None
        data = self.config.get("data", {})
        dataset = self.config.get("dataset", {})
        if not isinstance(data, Mapping):
            data = {}
        if not isinstance(dataset, Mapping):
            dataset = {}
        dataset_markers = {
            str(value).strip().lower()
            for value in (
                dataset.get("id"),
                dataset.get("kind"),
                data.get("dataset_format"),
                data.get("dataset_kind"),
                data.get("format"),
                str(data.get("root", "")).split("/")[-1],
            )
            if value is not None
        }
        self.is_wuhan = bool(
            any("wuhan" in marker for marker in dataset_markers)
            or "wuhan_stf_tiff" in dataset_markers
        )
        self.metric_resolution_ratio = float(data.get("physical_resolution_ratio", 30.0 / 8.0))
        raw_tile_size = data.get("eval_tile_size")
        raw_tile_overlap = data.get("eval_tile_overlap", 0)
        if raw_tile_size is None:
            self.eval_tile_size: int | None = None
        elif isinstance(raw_tile_size, bool) or not isinstance(raw_tile_size, int) or raw_tile_size < 1:
            raise ValueError("data.eval_tile_size must be a positive integer or null")
        else:
            self.eval_tile_size = int(raw_tile_size)
        if isinstance(raw_tile_overlap, bool) or not isinstance(raw_tile_overlap, int) or raw_tile_overlap < 0:
            raise ValueError("data.eval_tile_overlap must be a non-negative integer")
        self.eval_tile_overlap = int(raw_tile_overlap)
        if self.eval_tile_size is not None and self.eval_tile_overlap >= self.eval_tile_size:
            raise ValueError("data.eval_tile_overlap must be smaller than data.eval_tile_size")
        self.save_hyperparameters(
            {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "adam_betas": self.betas,
                "use_ema": self.use_ema,
            }
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def predict_for_eval(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Run bounded-memory full-grid inference when an eval tile is configured."""
        if self.eval_tile_size is None:
            return self(*inputs)
        scale = int(self.config.get("data", {}).get("scale", 1))
        return tiled_forward(
            self,
            *inputs,
            scale=scale,
            tile_size=self.eval_tile_size,
            overlap=self.eval_tile_overlap,
        )

    @abstractmethod
    def _train_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Return the scalar loss for one optimization batch."""

    @abstractmethod
    def _eval_step(self, batch: Any, batch_idx: int, *, stage: str) -> dict[str, torch.Tensor]:
        """Return scalar metrics for validation or test."""

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self._train_step(batch, batch_idx)
        if not torch.is_tensor(loss) or loss.ndim != 0:
            raise ValueError("_train_step must return a scalar tensor")
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _shared_eval(self, batch: Any, batch_idx: int, stage: str) -> torch.Tensor:
        metrics = self._eval_step(batch, batch_idx, stage=stage)
        if not isinstance(metrics, Mapping) or "loss" not in metrics:
            raise ValueError("_eval_step must return a mapping containing loss")
        for name, value in metrics.items():
            if torch.is_tensor(value):
                self.log(
                    f"{stage}/{name}", value,
                    on_step=False, on_epoch=True,
                    prog_bar=name == "loss",
                )
        return metrics["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_eval(batch, batch_idx, "test")

    @staticmethod
    def image_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction = prediction.clamp(-1.0, 1.0)
        return {
            "psnr": per_image_psnr(prediction, target).mean(),
            "ssim": gaussian_ssim(prediction, target).mean(),
        }

    def benchmark_image_metrics(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return standard metrics plus Wuhan's six-band-quality metrics."""
        metrics = self.image_metrics(prediction, target)
        if self.is_wuhan:
            from metrics.wuhan import wuhan_metric_tensors

            values = wuhan_metric_tensors(
                prediction,
                target,
                resolution_ratio=self.metric_resolution_ratio,
                value_range="minus_one_one",
            )
            metrics.update(
                {
                    key: values[key].mean()
                    for key in ("rmse", "uiqi", "psnr", "sam_rad", "sam_deg", "ergas")
                }
            )
        return metrics

    def on_train_start(self) -> None:
        if self.ema is not None:
            self.ema._init(self.model)
        self._sync_resumed_optimization_config()

    def _sync_resumed_optimization_config(self) -> None:
        """Apply edited optimization settings after checkpoint restoration.

        Lightning restores optimizer and scheduler state after ``on_fit_start``.
        This hook runs later, so edits made to an experiment snapshot are not
        overwritten by the serialized state.
        """
        scheduler = self._plateau_scheduler
        train = self.config["train"]
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.factor = float(train.get("lr_factor", scheduler.factor))
            scheduler.patience = int(train.get("lr_patience", scheduler.patience))
            scheduler.threshold = float(train.get("lr_threshold", scheduler.threshold))
            scheduler.threshold_mode = "abs"
            min_lr = float(train.get("lr_min", min(scheduler.min_lrs)))
            scheduler.min_lrs = [min_lr for _ in scheduler.optimizer.param_groups]
            scheduler.default_min_lr = min_lr

        # Preserve a scheduler-decayed LR when only the run budget/plateau
        # policy changed.  Explicitly changing learning_rate starts the
        # resumed optimizer from the newly requested value.
        saved_train = self._checkpoint_train_config
        if not saved_train or "learning_rate" not in train:
            return
        old_lr = saved_train.get("learning_rate")
        new_lr = float(train["learning_rate"])
        if old_lr is not None and float(old_lr) == new_lr:
            return
        optimizers = getattr(self.trainer, "optimizers", None) or []
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = new_lr

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self.ema is None:
            return
        step = int(self.global_step)
        if step > self._ema_last_step:
            self.ema.update(self.model)
            self._ema_last_step = step

    def _apply_ema(self) -> None:
        if self.ema is not None:
            self.ema.apply(self.model)

    def _restore_ema(self) -> None:
        if self.ema is not None:
            self.ema.restore(self.model)

    on_validation_model_eval = lambda self: self._apply_ema()
    on_validation_model_train = lambda self: self._restore_ema()
    on_validation_start = lambda self: self._apply_ema()
    on_validation_end = lambda self: self._restore_ema()
    on_test_model_eval = lambda self: self._apply_ema()
    on_test_model_train = lambda self: self._restore_ema()
    on_test_start = lambda self: self._apply_ema()
    on_test_end = lambda self: self._restore_ema()

    def on_validation_epoch_end(self) -> None:
        if str(self.config["train"].get("lr_scheduler", "plateau")).lower() != "plateau":
            return
        scheduler = self._plateau_scheduler
        if scheduler is None:
            return
        metric = (getattr(self.trainer, "callback_metrics", {}) or {}).get("val/loss")
        if metric is None:
            metric = (getattr(self.trainer, "callback_metrics", {}) or {}).get("val_loss")
        if metric is None:
            return
        value = float(metric.detach().float().item() if torch.is_tensor(metric) else metric)
        if math.isfinite(value):
            scheduler.step(value)

    def lr_scheduler_step(self, scheduler: Any, metric: Any) -> None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return
        scheduler.step(metric) if metric is not None else scheduler.step()

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError(
                "model has no trainable parameters; use the evaluation entry point for a reference-only baseline such as Bicubic"
            )
        optimizer = torch.optim.AdamW(
            parameters,
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
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "reduce_on_plateau": False}]
        if scheduler_name != "cosine":
            raise ValueError("train.lr_scheduler must be plateau or cosine")
        max_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0) or train.get("max_steps", 100000))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, max_steps), eta_min=float(train.get("lr_min", 1.0e-7))
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        value = self.config["train"].get("grad_clip_norm", 1.0)
        if gradient_clip_val is not None:
            value = gradient_clip_val
        if value is not None and float(value) > 0:
            self.clip_gradients(optimizer, gradient_clip_val=float(value), gradient_clip_algorithm=gradient_clip_algorithm or "norm")

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["trainer_config"] = copy.deepcopy(self.config)
        checkpoint["trainer_class"] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        if self.ema is not None:
            checkpoint["ema_state"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        trainer_config = checkpoint.get("trainer_config")
        if isinstance(trainer_config, Mapping):
            train_config = trainer_config.get("train")
            if isinstance(train_config, Mapping):
                self._checkpoint_train_config = dict(train_config)
        state = checkpoint.get("ema_state")
        if self.ema is not None and isinstance(state, Mapping):
            self.ema.load_state_dict(dict(state))
