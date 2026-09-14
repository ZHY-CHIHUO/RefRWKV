"""Training engine for the hybrid RDMRefSR model.

The objective follows the physical split in the model: the LR/query stream
must remain spectrally faithful, while the HR reference is supervised mainly
through reliable wavelet detail.  All terms are optional and default to the
conservative values in ``configs/common/rdm_refsr.yaml``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from engines.base_trainer import BaseTrainer
from models.refsr.rdm_refsr.rdm_refsr import RDMRefSR, haar_dwt2d
from runtime.common import adapt_reference_channels, gaussian_ssim
from runtime.config import normalize_reference_mode, validate_refsr_reference_contract


def _nonnegative(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"loss.{name} must be numeric") from exc
    if result < 0.0 or not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"loss.{name} must be finite and non-negative")
    return result


class RDMRefSRTrainer(BaseTrainer):
    """Lightning engine for ``RDMRefSR(lr, ref) -> hr``."""

    def __init__(self, model: RDMRefSR, config: Mapping[str, Any]) -> None:
        if not isinstance(model, RDMRefSR):
            raise TypeError("RDMRefSRTrainer requires an RDM model")
        validate_refsr_reference_contract(config)
        data = config.get("data", {})
        if not isinstance(data, Mapping):
            raise ValueError("data must be a mapping")
        super().__init__(
            model,
            config,
            lr_key=str(data.get("lr_key", "lr")),
            hr_key=str(data.get("hr_key", "hr")),
        )
        self.ref_key = str(data.get("ref_key", "ref"))
        self.reference_mode = normalize_reference_mode(data.get("reference_mode", "paired"))
        self.use_reference = bool(model.use_reference)

        loss = config.get("loss", {})
        if not isinstance(loss, Mapping):
            raise ValueError("loss must be a mapping")
        self.loss_name = str(loss.get("name", "charbonnier")).strip().lower()
        if self.loss_name == "l2":
            self.loss_name = "mse"
        if self.loss_name not in {"l1", "mse", "charbonnier"}:
            raise ValueError("loss.name must be l1, mse, or charbonnier")
        self.eps = float(loss.get("eps", 1.0e-3))
        if self.eps <= 0.0 or not torch.isfinite(torch.tensor(self.eps)):
            raise ValueError("loss.eps must be positive and finite")
        self.ssim_weight = _nonnegative(loss.get("ssim_weight", 0.0), "ssim_weight")
        self.fft_weight = _nonnegative(loss.get("fft_weight", 0.0), "fft_weight")
        self.sam_weight = _nonnegative(loss.get("sam_weight", 0.0), "sam_weight")
        self.wavelet_weight = _nonnegative(loss.get("wavelet_weight", 0.0), "wavelet_weight")
        self.consistency_weight = _nonnegative(
            loss.get("consistency_weight", 0.0), "consistency_weight"
        )
        self.sensor_weight = _nonnegative(loss.get("sensor_weight", 0.0), "sensor_weight")
        self.change_weight = _nonnegative(loss.get("change_weight", 0.0), "change_weight")
        self.ref_drop_prob = _nonnegative(loss.get("ref_drop_prob", 0.0), "ref_drop_prob")
        if self.ref_drop_prob > 1.0:
            raise ValueError("loss.ref_drop_prob must be in [0, 1]")
        self._last_terms: dict[str, torch.Tensor] = {}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RDMRefSRTrainer":
        data = config.get("data", {})
        if not isinstance(data, Mapping) or "scale" not in data:
            raise ValueError("RDMRefSR configuration requires data.scale")
        from models.refsr import build_model as build_refsr_model

        model = build_refsr_model(config["model"], scale=int(data["scale"]))
        if not isinstance(model, RDMRefSR):
            raise TypeError("model.name must resolve to rdm_mhf or rdm_refsr")
        return cls(model, config)

    def _unpack(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not isinstance(batch, Mapping):
            raise TypeError("RDMRefSR batches must be mappings")
        if self.lr_key not in batch or self.hr_key not in batch:
            raise KeyError(f"batch must contain {self.lr_key!r} and {self.hr_key!r}")
        lr, hr, ref = batch[self.lr_key], batch[self.hr_key], batch.get(self.ref_key)
        if not torch.is_tensor(lr) or not torch.is_tensor(hr):
            raise TypeError("lr and hr batch entries must be tensors")
        return lr, hr, ref

    def _reference(
        self, lr: torch.Tensor, hr: torch.Tensor, ref: torch.Tensor | None
    ) -> torch.Tensor | None:
        expected = (int(lr.shape[-2]) * self.model.scale, int(lr.shape[-1]) * self.model.scale)
        if tuple(hr.shape[-2:]) != expected:
            raise ValueError(
                f"LR/HR geometry mismatch: {tuple(lr.shape[-2:])} -> "
                f"{tuple(hr.shape[-2:])}, x{self.model.scale}"
            )
        if not self.use_reference:
            return None
        if self.reference_mode == "lr_up":
            return adapt_reference_channels(
                F.interpolate(lr, size=expected, mode="bicubic", align_corners=False),
                self.model.ref_channels,
            )
        if self.reference_mode == "none":
            raise ValueError("model.use_reference=true requires paired or lr_up reference_mode")
        if ref is None:
            raise ValueError("paired reference mode requires a ref tensor")
        if not torch.is_tensor(ref):
            raise TypeError(f"batch[{self.ref_key!r}] must be a tensor")
        if ref.ndim != 4 or ref.shape[0] != lr.shape[0] or ref.shape[1] != self.model.ref_channels:
            raise ValueError(
                f"reference shape must be [B,{self.model.ref_channels},H,W], got {tuple(ref.shape)}"
            )
        if tuple(ref.shape[-2:]) != expected:
            raise ValueError(f"reference spatial size must be {expected}, got {tuple(ref.shape[-2:])}")
        return ref

    def _apply_reference_dropout(
        self, ref: torch.Tensor | None, lr: torch.Tensor
    ) -> torch.Tensor | None:
        if ref is None or not self.training or self.ref_drop_prob <= 0.0:
            return ref
        fallback = adapt_reference_channels(
            F.interpolate(lr, size=ref.shape[-2:], mode="bicubic", align_corners=False),
            ref.shape[1],
        )
        mask = torch.rand(ref.shape[0], 1, 1, 1, device=ref.device) < self.ref_drop_prob
        return torch.where(mask, fallback, ref)

    @staticmethod
    def _sam(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape[1] < 2:
            return prediction.new_zeros(())
        pred = prediction.float().flatten(2)
        truth = target.float().flatten(2)
        cosine = (pred * truth).sum(dim=1) / (
            pred.square().sum(dim=1).sqrt() * truth.square().sum(dim=1).sqrt() + 1.0e-6
        )
        cosine = torch.nan_to_num(cosine, nan=1.0, posinf=1.0, neginf=-1.0)
        # ``acos`` has an infinite derivative at +/-1.  Half/bfloat16
        # activations can quantize nearly parallel spectra to those exact
        # endpoints, producing Inf/NaN gradients even when the loss weight
        # is small.  Keep a narrow finite margin around both endpoints.
        cosine = cosine.clamp(-1.0 + 1.0e-4, 1.0 - 1.0e-4)
        return torch.acos(cosine).mean()

    @staticmethod
    def _highpass(value: torch.Tensor) -> torch.Tensor:
        # A differentiable Laplacian is less sensitive to image-border
        # padding than a raw pixel comparison and works for every channel.
        kernel = value.new_tensor(((0.0, -1.0, 0.0), (-1.0, 4.0, -1.0), (0.0, -1.0, 0.0)))
        kernel = kernel.view(1, 1, 3, 3).repeat(value.shape[1], 1, 1, 1)
        return F.conv2d(value, kernel, padding=1, groups=value.shape[1])

    def _base_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"RDMRefSR output/HR geometry mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        residual = prediction - target
        if self.loss_name == "l1":
            value = residual.abs().mean()
        elif self.loss_name == "mse":
            value = residual.square().mean()
        else:
            value = (residual.square() + self.eps * self.eps).sqrt().mean()
        if self.ssim_weight:
            value = value + self.ssim_weight * (1.0 - gaussian_ssim(prediction, target).mean())
        if self.fft_weight:
            pred_fft = torch.fft.rfft2(prediction.float(), norm="ortho")
            target_fft = torch.fft.rfft2(target.float(), norm="ortho")
            value = value + self.fft_weight * (pred_fft - target_fft).abs().mean()
        return value

    def _physical_terms(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        lr: torch.Tensor,
        ref: torch.Tensor | None,
        aux: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        zero = prediction.new_zeros(())
        terms: dict[str, torch.Tensor] = {
            "sam": zero,
            "wavelet": zero,
            "consistency": zero,
            "sensor": zero,
            "change": zero,
        }
        if self.sam_weight:
            terms["sam"] = self._sam(prediction, target)
        if self.wavelet_weight:
            _, pred_detail, _ = haar_dwt2d(prediction)
            _, target_detail, _ = haar_dwt2d(target)
            terms["wavelet"] = (pred_detail - target_detail).abs().mean()
        if self.consistency_weight:
            down = F.interpolate(prediction, size=lr.shape[-2:], mode="area")
            if down.shape[1] != lr.shape[1]:
                down = adapt_reference_channels(down, lr.shape[1])
            terms["consistency"] = (down - lr).abs().mean()
        if self.sensor_weight and self.model.reference_kind == "pan" and ref is not None:
            # PAN constrains the spatial high-pass of the sensor-response
            # projection, never each MS band independently.
            weights = torch.softmax(self.model.sensor_logits.float(), dim=0).to(prediction.dtype)
            count = min(int(prediction.shape[1]), int(weights.numel()))
            pred_intensity = (prediction[:, :count] * weights[:count].view(1, -1, 1, 1)).sum(1, keepdim=True)
            ref_intensity = ref.float().mean(1, keepdim=True).to(prediction.dtype)
            terms["sensor"] = (
                self._highpass(pred_intensity) - self._highpass(ref_intensity)
            ).abs().mean()
        if self.change_weight and self.model.reference_kind in {"stf", "mhf"}:
            change = aux.get("change")
            if change is not None:
                change = F.interpolate(change, size=prediction.shape[-2:], mode="bilinear", align_corners=False)
                terms["change"] = (change * (prediction - target).abs()).mean()
        return terms

    def _loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        lr: torch.Tensor,
        ref: torch.Tensor | None,
        aux: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        value = self._base_loss(prediction, target)
        terms = self._physical_terms(prediction, target, lr, ref, aux)
        value = value + self.sam_weight * terms["sam"]
        value = value + self.wavelet_weight * terms["wavelet"]
        value = value + self.consistency_weight * terms["consistency"]
        value = value + self.sensor_weight * terms["sensor"]
        value = value + self.change_weight * terms["change"]
        self._last_terms = terms
        return value

    def _train_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        lr, hr, ref = self._unpack(batch)
        reference = self._apply_reference_dropout(self._reference(lr, hr, ref), lr)
        prediction, aux = self(
            lr,
            reference,
            return_aux=True,
        )
        return self._loss(prediction, hr, lr, reference, aux)

    def _eval_step(self, batch: Any, batch_idx: int, *, stage: str) -> dict[str, torch.Tensor]:
        lr, hr, ref = self._unpack(batch)
        reference = self._reference(lr, hr, ref)
        prediction = self.predict_for_eval(lr, reference) if reference is not None else self.predict_for_eval(lr)
        # Evaluation keeps the objective identical but avoids retaining aux
        # tensors from tiled inference; physical terms are reported only when
        # the model is run directly on a validation patch.
        value = self._base_loss(prediction, hr)
        return {"loss": value, **self.benchmark_image_metrics(prediction, hr)}


__all__ = ["RDMRefSRTrainer"]
