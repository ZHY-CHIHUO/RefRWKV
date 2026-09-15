"""Common trainer for direct ``forward(lr, ref)`` RefSR models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from engines.base_trainer import BaseTrainer
from models.refsr import build_model as build_refsr_model
from runtime.common import gaussian_ssim
from runtime.config import normalize_reference_mode, validate_refsr_reference_contract


STF_THREE_INPUT_MODELS = frozenset({"stf_mamba", "rdm_stf"})


class RefSRTrainer(BaseTrainer):
    """Train a direct RefSR model with the repository-wide paired contract.

    Unlike ``RefSRWKVTrainer``, this class has no architecture-specific loss
    terms or type check.  Two-input models stay ``forward(lr, ref)``.  STF
    models additionally consume ``c0`` (``batch[data.c0_key]``, default
    ``lr_t1``) when the loader returns a temporal quadruple.
    """

    def __init__(self, model: nn.Module, config: Mapping[str, Any]) -> None:
        validate_refsr_reference_contract(config)
        data = config.get("data", {})
        if not isinstance(data, Mapping):
            raise ValueError("data must be a mapping")
        scale = data.get("scale")
        if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
            raise ValueError("data.scale must be a positive integer")
        super().__init__(
            model,
            config,
            lr_key=str(data.get("lr_key", "lr")),
            hr_key=str(data.get("hr_key", "hr")),
        )
        self.scale = int(scale)
        self.ref_key = str(data.get("ref_key", "ref"))
        self.reference_mode = normalize_reference_mode(data.get("reference_mode", "paired"))
        loss = config.get("loss", {})
        if not isinstance(loss, Mapping):
            raise ValueError("loss must be a mapping")
        loss_name = str(loss.get("name", "l1")).lower()
        self.loss_name = "mse" if loss_name == "l2" else loss_name
        if self.loss_name not in {"l1", "mse", "charbonnier"}:
            raise ValueError("loss.name must be l1, mse, or charbonnier")
        self.charbonnier_eps = float(loss.get("eps", 1.0e-3))
        if self.charbonnier_eps <= 0:
            raise ValueError("loss.eps must be positive")
        self.ssim_weight = float(loss.get("ssim_weight", 0.0))
        self.fft_weight = float(loss.get("fft_weight", 0.0))
        self.consistency_weight = float(loss.get("consistency_weight", 0.0))
        self.pan_struct_weight = float(loss.get("pan_struct_weight", 0.0))
        self.ref_drop_prob = float(loss.get("ref_drop_prob", 0.0))
        if self.ssim_weight < 0.0 or self.fft_weight < 0.0:
            raise ValueError("loss.ssim_weight and loss.fft_weight must be non-negative")
        if self.consistency_weight < 0.0 or self.pan_struct_weight < 0.0:
            raise ValueError("loss.consistency_weight and loss.pan_struct_weight must be non-negative")
        if not 0.0 <= self.ref_drop_prob <= 1.0:
            raise ValueError("loss.ref_drop_prob must be in [0, 1]")
        model_name = str(config.get("model", {}).get("name", "")).strip().lower()
        self.c0_key = str(data.get("c0_key", "lr_t1"))
        self.use_c0 = bool(data.get("use_c0", data.get("return_quadruple", False))) or model_name == "stf_mamba"
        self.require_c0 = model_name in STF_THREE_INPUT_MODELS and self.use_c0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RefSRTrainer":
        data = config.get("data", {})
        if not isinstance(data, Mapping):
            raise ValueError("data must be a mapping")
        model = build_refsr_model(config["model"], scale=int(data["scale"]))
        return cls(model, config)

    def _unpack(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not isinstance(batch, Mapping):
            raise TypeError("RefSR batches must be mappings")
        if self.lr_key not in batch or self.hr_key not in batch:
            raise KeyError(f"RefSR batch must contain {self.lr_key!r} and {self.hr_key!r}")
        return batch[self.lr_key], batch[self.hr_key], batch.get(self.ref_key)

    def _reference(self, lr: torch.Tensor, hr: torch.Tensor, ref: torch.Tensor | None) -> torch.Tensor:
        expected = (lr.shape[-2] * self.scale, lr.shape[-1] * self.scale)
        if tuple(hr.shape[-2:]) != expected:
            raise ValueError(
                f"LR/HR geometry mismatch: {tuple(lr.shape[-2:])} -> {tuple(hr.shape[-2:])}, x{self.scale}"
            )
        if self.reference_mode == "lr_up":
            return F.interpolate(lr, size=expected, mode="bicubic", align_corners=False)
        if ref is None:
            raise ValueError("paired reference mode requires a ref tensor")
        if not torch.is_tensor(ref):
            raise TypeError(f"batch[{self.ref_key!r}] must be a tensor, got {type(ref).__name__}")
        if ref.shape[0] != lr.shape[0] or tuple(ref.shape[-2:]) != expected:
            raise ValueError(
                f"LR/Ref geometry mismatch: LR={tuple(lr.shape)}, Ref={tuple(ref.shape)}, expected Ref spatial={expected}"
            )
        return ref

    @staticmethod
    def _highpass(value: torch.Tensor) -> torch.Tensor:
        kernel = value.new_tensor(((0.0, -1.0, 0.0), (-1.0, 4.0, -1.0), (0.0, -1.0, 0.0)))
        kernel = kernel.view(1, 1, 3, 3).repeat(value.shape[1], 1, 1, 1)
        return F.conv2d(value, kernel, padding=1, groups=value.shape[1])

    def _loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        lr: torch.Tensor | None = None,
        ref: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(f"RefSR output/HR geometry mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
        residual = prediction - target
        if self.loss_name == "l1":
            value = residual.abs().mean()
        elif self.loss_name == "mse":
            value = residual.square().mean()
        else:
            value = (residual.square() + self.charbonnier_eps**2).sqrt().mean()
        if self.ssim_weight:
            value = value + self.ssim_weight * (1.0 - gaussian_ssim(prediction, target).mean())
        if self.fft_weight:
            pred_fft = torch.fft.rfft2(prediction.float(), norm="ortho")
            target_fft = torch.fft.rfft2(target.float(), norm="ortho")
            value = value + self.fft_weight * (pred_fft - target_fft).abs().mean()
        if self.consistency_weight and lr is not None:
            down = F.interpolate(prediction, size=lr.shape[-2:], mode="area")
            value = value + self.consistency_weight * (down - lr).abs().mean()
        if self.pan_struct_weight and ref is not None:
            pred_hp = self._highpass(prediction.mean(1, keepdim=True))
            pan_hp = self._highpass(ref.mean(1, keepdim=True))
            pred_flat = pred_hp.flatten(2)
            pan_flat = pan_hp.flatten(2)
            cosine = (pred_flat * pan_flat).sum(dim=-1) / (
                pred_flat.norm(dim=-1) * pan_flat.norm(dim=-1) + 1.0e-6
            )
            cosine = cosine.clamp(-1.0 + 1.0e-4, 1.0 - 1.0e-4)
            value = value + self.pan_struct_weight * (1.0 - cosine).mean()
        return value

    def _c0(self, batch: Any) -> torch.Tensor | None:
        if not self.use_c0:
            return None
        if not isinstance(batch, Mapping):
            raise TypeError("RefSR batches must be mappings")
        value = batch.get(self.c0_key)
        if value is None and self.require_c0:
            raise KeyError(
                f"STF batch must contain {self.c0_key!r}; set data.return_quadruple=true"
            )
        if value is not None and not torch.is_tensor(value):
            raise TypeError(f"batch[{self.c0_key!r}] must be a tensor")
        return value

    def _forward_model(
        self, lr: torch.Tensor, ref: torch.Tensor, c0: torch.Tensor | None
    ) -> torch.Tensor:
        if c0 is not None or self.require_c0:
            return self(lr, ref, c0)
        return self(lr, ref)

    def _apply_reference_dropout(self, ref: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        if self.ref_drop_prob <= 0.0 or not self.training:
            return ref
        fallback = F.interpolate(lr, size=ref.shape[-2:], mode="bicubic", align_corners=False)
        drop = torch.rand(ref.shape[0], 1, 1, 1, device=ref.device) < self.ref_drop_prob
        return torch.where(drop, fallback, ref)

    def _train_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        lr, hr, ref = self._unpack(batch)
        reference = self._apply_reference_dropout(self._reference(lr, hr, ref), lr)
        prediction = self._forward_model(lr, reference, self._c0(batch))
        return self._loss(prediction, hr, lr=lr, ref=reference)

    def _eval_step(self, batch: Any, batch_idx: int, *, stage: str) -> dict[str, torch.Tensor]:
        lr, hr, ref = self._unpack(batch)
        reference = self._reference(lr, hr, ref)
        c0 = self._c0(batch)
        if c0 is not None or self.require_c0:
            prediction = self.predict_for_eval(lr, reference, c0)
        else:
            prediction = self.predict_for_eval(lr, reference)
        return {"loss": self._loss(prediction, hr, lr=lr, ref=reference), **self.benchmark_image_metrics(prediction, hr)}


__all__ = ["RefSRTrainer"]
