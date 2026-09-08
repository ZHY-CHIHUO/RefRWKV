"""Standalone Wuhan image-quality metrics.

Historically this module delegated to :mod:`sewar`, which did not expose
ERGAS consistently and only returned SAM in degrees.  The implementation now
uses the repository's dependency-light metric functions and reports both
radian and degree spectral angles.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from metrics.wuhan import compute_wuhan_metrics


def evaluate(
    pred: np.ndarray,
    gt: np.ndarray,
    max_val: float = 1.0,
    *,
    resolution_ratio: float = 30.0 / 8.0,
) -> dict[str, Any]:
    """Evaluate one HWC/CHW image in the requested numeric range.

    ``max_val`` is retained for compatibility. Inputs are converted to
    reflectance ``[0,1]`` before calculating RMSE, UIQI, PSNR, SAM and ERGAS.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"形状不一致: pred {pred.shape}, gt {gt.shape}")
    if max_val <= 0:
        raise ValueError("max_val must be positive")
    pred_value = np.asarray(pred, dtype=np.float32) / float(max_val)
    gt_value = np.asarray(gt, dtype=np.float32) / float(max_val)
    result = compute_wuhan_metrics(
        pred_value,
        gt_value,
        resolution_ratio=resolution_ratio,
        value_range="zero_one",
    )
    # Preserve the old uppercase names and add explicit angular units.
    return {
        "PSNR": result["psnr"],
        "UIQI": result["uiqi"],
        "SAM_rad": result["sam_rad"],
        "SAM_deg": result["sam_deg"],
        "SAM": result["sam_deg"],
        "ERGAS": result["ergas"],
        "RMSE": result["rmse"],
        "RMSE_per_band": result["rmse_per_band"],
        "resolution_ratio": result["resolution_ratio"],
    }


def print_metrics(metrics: dict, title: str = "Evaluation Results") -> None:
    print(f"\n{'=' * 40}\n  {title}\n{'=' * 40}")
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.number)):
            print(f"  {key:16s}: {float(value):.6f}")
        else:
            print(f"  {key:16s}: {value}")
    print("=" * 40)


def evaluate_CHW(
    pred: np.ndarray,
    gt: np.ndarray,
    max_val: float = 1.0,
    print_result: bool = True,
    title: str = "Evaluation (CHW -> HWC)",
    *,
    resolution_ratio: float = 30.0 / 8.0,
) -> dict[str, Any]:
    """Evaluate CHW (or 2D) arrays and optionally print the result."""
    if pred.ndim == 3:
        pred = np.transpose(pred, (1, 2, 0))
        gt = np.transpose(gt, (1, 2, 0))
    result = evaluate(pred, gt, max_val=max_val, resolution_ratio=resolution_ratio)
    if print_result:
        print_metrics(result, title=title)
    return result


def average_metrics(results_list: list[dict]) -> dict:
    """Average numeric metrics while preserving per-band lists."""
    if not results_list:
        return {}
    result: dict[str, Any] = {}
    for key in results_list[0]:
        values = [item[key] for item in results_list if item.get(key) is not None]
        if not values:
            result[key] = None
        elif isinstance(values[0], (list, tuple, np.ndarray)):
            result[key] = np.asarray(values, dtype=np.float64).mean(axis=0).tolist()
        elif isinstance(values[0], (int, float, np.number)):
            result[key] = float(np.mean(values))
        else:
            result[key] = values[0]
    return result


if __name__ == "__main__":  # pragma: no cover
    pred = np.random.rand(4, 8, 8)
    gt = np.random.rand(4, 8, 8)
    print_metrics(evaluate_CHW(pred, gt))
