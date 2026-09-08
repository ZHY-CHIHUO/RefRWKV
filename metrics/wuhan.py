"""Metrics used by the Wuhan spatiotemporal-fusion benchmark.

All calculations are performed in reflectance ``[0, 1]``.  The training
pipeline stores tensors in ``[-1, 1]``; pass ``value_range='minus_one_one'``
to convert them.  Functions return one value per image for batched tensors so
callers can aggregate without weighting images by their number of pixels.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def _as_nchw(value: torch.Tensor | np.ndarray) -> torch.Tensor:
    supplied_torch = torch.is_tensor(value)
    tensor = value if supplied_torch else torch.as_tensor(value)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        # Public numpy helpers conventionally receive HWC; torch pipelines
        # conventionally receive CHW.  A four-channel first dimension is an
        # unambiguous CHW signal, while HWC is used otherwise.
        plausible_channels = {1, 2, 3, 4, 5, 6, 8, 10, 12, 16}
        first_is_channel = int(tensor.shape[0]) in plausible_channels
        last_is_channel = int(tensor.shape[-1]) in plausible_channels
        if first_is_channel and not last_is_channel:
            tensor = tensor.unsqueeze(0)
        elif last_is_channel and not first_is_channel:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        elif supplied_torch and tensor.shape[0] <= 16:
            tensor = tensor.unsqueeze(0)
        elif not supplied_torch and tensor.shape[-1] <= 16:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    elif tensor.ndim == 4:
        # Batched tensors are NCHW in the training/evaluation code.  Accept
        # NHWC as a convenience when the last dimension looks like bands.
        if tensor.shape[1] > 16 and tensor.shape[-1] <= 16:
            tensor = tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"expected 2D/3D/4D image tensor, got {tuple(tensor.shape)}")
    if tensor.ndim != 4 or tensor.shape[1] < 1:
        raise ValueError(f"invalid image tensor shape: {tuple(tensor.shape)}")
    return tensor.float()


def to_reflectance(value: torch.Tensor | np.ndarray, value_range: str = "minus_one_one") -> torch.Tensor:
    """Convert an image/batch to NCHW reflectance in ``[0, 1]``."""
    tensor = _as_nchw(value)
    normalized = str(value_range).strip().lower()
    if normalized in {"minus_one_one", "-1_1", "[-1,1]", "neg1_pos1"}:
        tensor = (tensor + 1.0) * 0.5
    elif normalized not in {"zero_one", "0_1", "[0,1]", "reflectance"}:
        raise ValueError("value_range must be minus_one_one or zero_one")
    return tensor.clamp(0.0, 1.0)


def wuhan_metric_tensors(
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    *,
    resolution_ratio: float = 30.0 / 8.0,
    value_range: str = "minus_one_one",
    eps: float = 1.0e-12,
) -> dict[str, torch.Tensor]:
    """Return batched RMSE/UIQI/PSNR/SAM/ERGAS tensors.

    ``sam_rad`` is the primary spectral-angle metric; ``sam_deg`` is provided
    simultaneously for convenient comparison with papers reporting degrees.
    ERGAS uses the physical resolution ratio (30/8 for Wuhan), which need not
    be an integer.
    """
    pred = to_reflectance(prediction, value_range)
    truth = to_reflectance(target, value_range)
    if pred.shape != truth.shape:
        raise ValueError(f"prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(truth.shape)}")
    if not math.isfinite(float(resolution_ratio)) or float(resolution_ratio) <= 0:
        raise ValueError("resolution_ratio must be a positive finite number")

    diff = pred - truth
    mse = diff.square().mean(dim=(1, 2, 3))
    rmse = mse.sqrt()
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(float(eps)))

    pred_mean = pred.mean(dim=(2, 3))
    truth_mean = truth.mean(dim=(2, 3))
    pred_centered = pred - pred_mean[:, :, None, None]
    truth_centered = truth - truth_mean[:, :, None, None]
    pred_var = pred_centered.square().mean(dim=(2, 3))
    truth_var = truth_centered.square().mean(dim=(2, 3))
    covariance = (pred_centered * truth_centered).mean(dim=(2, 3))
    numerator = 4.0 * covariance * pred_mean * truth_mean
    denominator = (pred_var + truth_var) * (pred_mean.square() + truth_mean.square())
    # A pair of identical flat bands has a mathematically 0/0 UIQI; its
    # limiting quality is one, which is also what benchmark implementations
    # expect for an identity prediction.
    uiqi_band = torch.where(
        denominator.abs() <= float(eps),
        torch.ones_like(denominator),
        numerator / denominator.clamp_min(float(eps)),
    )
    uiqi = uiqi_band.clamp(-1.0, 1.0).mean(dim=1)

    dot = (pred * truth).sum(dim=1)
    pred_norm = pred.square().sum(dim=1).sqrt()
    truth_norm = truth.square().sum(dim=1).sqrt()
    denom = pred_norm * truth_norm
    cosine = (dot / denom.clamp_min(float(eps))).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    both_zero = (pred_norm <= float(eps)) & (truth_norm <= float(eps))
    # Round-off can turn the identity cosine into 0.99999999 in float32;
    # explicitly apply the limiting zero angle for stable identity tests.
    angle = torch.where((both_zero) | (cosine >= 1.0 - 1.0e-6), torch.zeros_like(angle), angle)
    sam_rad = angle.mean(dim=(1, 2))
    sam_deg = sam_rad * (180.0 / math.pi)

    band_rmse = diff.square().mean(dim=(2, 3)).sqrt()
    band_mean = truth.mean(dim=(2, 3)).clamp_min(float(eps))
    ergas = (100.0 / float(resolution_ratio)) * (
        (band_rmse / band_mean).square().mean(dim=1).sqrt()
    )
    return {
        "rmse": rmse,
        "uiqi": uiqi,
        "psnr": psnr,
        "sam_rad": sam_rad,
        "sam_deg": sam_deg,
        "ergas": ergas,
        "rmse_per_band": band_rmse,
    }


def compute_wuhan_metrics(
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    *,
    resolution_ratio: float = 30.0 / 8.0,
    value_range: str = "zero_one",
) -> dict[str, Any]:
    """Compute scalar metrics for one image (or mean over a batch).

    The returned mapping includes ``per_band`` RMSE values in addition to the
    six paper metrics, making it suitable for a JSON report.
    """
    values = wuhan_metric_tensors(
        prediction,
        target,
        resolution_ratio=resolution_ratio,
        value_range=value_range,
    )
    result: dict[str, Any] = {}
    for key in ("rmse", "uiqi", "psnr", "sam_rad", "sam_deg", "ergas"):
        result[key] = float(values[key].mean().detach().cpu().item())
    result["rmse_per_band"] = values["rmse_per_band"].mean(dim=0).detach().cpu().tolist()
    result["resolution_ratio"] = float(resolution_ratio)
    return result


def metric_report(
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    *,
    resolution_ratio: float = 30.0 / 8.0,
    value_range: str = "zero_one",
) -> dict[str, Any]:
    """Alias with a descriptive name for external evaluation scripts."""
    return compute_wuhan_metrics(
        prediction,
        target,
        resolution_ratio=resolution_ratio,
        value_range=value_range,
    )


def rmse(prediction, target, *, value_range: str = "zero_one") -> torch.Tensor:
    return wuhan_metric_tensors(prediction, target, value_range=value_range)["rmse"]


def uiqi(prediction, target, *, value_range: str = "zero_one") -> torch.Tensor:
    return wuhan_metric_tensors(prediction, target, value_range=value_range)["uiqi"]


def psnr(prediction, target, *, value_range: str = "zero_one") -> torch.Tensor:
    return wuhan_metric_tensors(prediction, target, value_range=value_range)["psnr"]


def sam(prediction, target, *, value_range: str = "zero_one", degrees: bool = False) -> torch.Tensor:
    key = "sam_deg" if degrees else "sam_rad"
    return wuhan_metric_tensors(prediction, target, value_range=value_range)[key]


def ergas(
    prediction,
    target,
    *,
    resolution_ratio: float = 30.0 / 8.0,
    value_range: str = "zero_one",
) -> torch.Tensor:
    return wuhan_metric_tensors(
        prediction,
        target,
        resolution_ratio=resolution_ratio,
        value_range=value_range,
    )["ergas"]


__all__ = [
    "compute_wuhan_metrics",
    "ergas",
    "metric_report",
    "psnr",
    "rmse",
    "sam",
    "to_reflectance",
    "uiqi",
    "wuhan_metric_tensors",
]
