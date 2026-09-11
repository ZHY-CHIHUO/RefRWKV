"""Trainer for the standalone ``RefSRWKV`` reference SR model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from engines.base_trainer import BaseTrainer
from models.refsr import build_model as build_refsr_model
from models.refsr.refsrwkv.model import RefSRWKV
from runtime.common import adapt_reference_channels, gaussian_ssim
from runtime.config import normalize_reference_mode, validate_refsr_reference_contract


class RefSRWKVTrainer(BaseTrainer):
    """Train RefSRWKV with paired references or an LR-derived reference."""

    def __init__(self, model: RefSRWKV, config: Mapping[str, Any]) -> None:
        validate_refsr_reference_contract(config)
        data = config.get("data", {})
        super().__init__(model, config, lr_key=str(data.get("lr_key", "lr")), hr_key=str(data.get("hr_key", "hr")))
        self.ref_key = str(data.get("ref_key", "ref"))
        self.reference_mode = normalize_reference_mode(data.get("reference_mode", "paired"))
        loss = config.get("loss", {})
        self.loss_name = str(loss.get("name", "l1")).lower()
        if self.loss_name == "l2":
            self.loss_name = "mse"
        model_cfg = config.get("model", {})
        self.use_reference = bool(model_cfg.get("use_reference", True))
        self.ssim_weight = float(loss.get("ssim_weight", model_cfg.get("ssim_weight", 0.0)))
        self.fft_weight = float(loss.get("fft_weight", model_cfg.get("fft_weight", 0.0)))
        self.ref_drop_prob = float(loss.get("ref_drop_prob", model_cfg.get("ref_drop_prob", 0.0)))
        if self.ssim_weight < 0.0 or self.fft_weight < 0.0:
            raise ValueError("loss.ssim_weight and loss.fft_weight must be non-negative")
        if not 0.0 <= self.ref_drop_prob <= 1.0:
            raise ValueError("loss.ref_drop_prob must be in [0, 1]")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RefSRWKVTrainer":
        data = config.get("data", {})
        model = build_refsr_model(config["model"], scale=int(data["scale"]))
        if not isinstance(model, RefSRWKV):
            raise TypeError("RefSRWKVTrainer requires model.name=RefSRWKV")
        return cls(model, config)

    def _unpack(self, batch: Any):
        if not isinstance(batch, Mapping):
            raise TypeError("RefSRWKV batches must be mappings")
        if self.lr_key not in batch or self.hr_key not in batch:
            raise KeyError(f"batch must contain {self.lr_key!r} and {self.hr_key!r}")
        return batch[self.lr_key], batch[self.hr_key], batch.get(self.ref_key)

    def _reference(self, lr: torch.Tensor, hr: torch.Tensor, ref: torch.Tensor | None) -> torch.Tensor | None:
        expected = (lr.shape[-2] * self.model.scale, lr.shape[-1] * self.model.scale)
        if tuple(hr.shape[-2:]) != expected:
            raise ValueError(f"LR/HR geometry mismatch: {tuple(lr.shape[-2:])} -> {tuple(hr.shape[-2:])}, x{self.model.scale}")
        if not self.use_reference:
            return None
        if self.reference_mode == "lr_up":
            reference = F.interpolate(lr, size=expected, mode="bicubic", align_corners=False)
            return adapt_reference_channels(reference, self.model.ref_channels)
        if self.reference_mode == "none":
            raise ValueError("model.use_reference=true requires paired or lr_up reference_mode")
        if ref is None:
            raise ValueError("paired reference mode requires a ref tensor")
        return ref

    def _loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        residual = prediction - target
        if self.loss_name in {"l1", "mae"}:
            base = residual.abs().mean()
        elif self.loss_name in {"mse", "l2"}:
            base = residual.square().mean()
        else:
            base = (residual.square() + 1.0e-6).sqrt().mean()
        if self.ssim_weight:
            base = base + self.ssim_weight * (1.0 - gaussian_ssim(prediction, target).mean())
        if self.fft_weight:
            pred_fft = torch.fft.rfft2(prediction.float(), norm="ortho")
            target_fft = torch.fft.rfft2(target.float(), norm="ortho")
            base = base + self.fft_weight * (pred_fft - target_fft).abs().mean()
        return base

    def _apply_reference_dropout(self, ref: torch.Tensor | None, lr: torch.Tensor) -> torch.Tensor | None:
        if ref is None:
            return None
        if self.ref_drop_prob <= 0.0 or not self.training:
            return ref
        fallback = F.interpolate(lr, size=ref.shape[-2:], mode="bicubic", align_corners=False)
        fallback = adapt_reference_channels(fallback, ref.shape[1])
        drop = torch.rand(ref.shape[0], 1, 1, 1, device=ref.device) < self.ref_drop_prob
        return torch.where(drop, fallback, ref)

    def _train_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        lr, hr, ref = self._unpack(batch)
        reference = self._reference(lr, hr, ref)
        reference = self._apply_reference_dropout(reference, lr)
        prediction = self(lr) if reference is None else self(lr, reference)
        return self._loss(prediction, hr)

    def _eval_step(self, batch: Any, batch_idx: int, *, stage: str) -> dict[str, torch.Tensor]:
        lr, hr, ref = self._unpack(batch)
        reference = self._reference(lr, hr, ref)
        prediction = (
            self.predict_for_eval(lr)
            if reference is None
            else self.predict_for_eval(lr, reference)
        )
        return {"loss": self._loss(prediction, hr), **self.benchmark_image_metrics(prediction, hr)}
