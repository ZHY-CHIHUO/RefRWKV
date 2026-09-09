#!/usr/bin/env python3
"""Measure how TRefSR responds to controlled reference-image quality.

For each test image this script builds
``Ref(alpha) = alpha * HR + (1 - alpha) * bicubic(LR)`` in memory, runs the
same TRefSR checkpoint, and compares its PSNR with the already evaluated
RefSRWKV-SR arm.  No checkpoint or dataset file is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loaders import build_refsr_test_loader
from evaluation.runner import (
    _build_refsr_model,
    _image_tensor,
    _move_batch,
    _normalize_sample_ids,
    _reference_from_lr,
    select_device,
)
from runtime.checkpoint import load_checkpoint
from runtime.common import per_image_psnr, resolve_path
from runtime.config import load_test_config, normalize_reference_mode, validate_config
from runtime.tiling import tiled_forward


LOGGER = logging.getLogger(__name__)
DEFAULT_ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SPLITS = ("test_easy", "test_hard")
DEFAULT_TREF_CHECKPOINT = (
    PROJECT_ROOT
    / "experiments/train/refsr/refsrwkv/hrms_scd/x4/hrms_scd_trefsr_x4/checkpoints/epoch=0149-step=089400.ckpt"
)
DEFAULT_SR_RUN = PROJECT_ROOT / "experiments/test/refsr/refsrwkv/hrms_scd/x4/hrms_scd_sr_x4"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments/test/comparison/hrms_scd_x4/reference_quality"


def _load_sr_psnr(root: Path, split: str) -> dict[str, float]:
    path = root / split / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"SR arm metrics.json not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("psnr", {}).get("per_image")
    ids = payload.get("sample_ids")
    if not isinstance(values, list) or not isinstance(ids, list) or len(values) != len(ids):
        raise ValueError(f"SR arm metrics must contain aligned psnr.per_image and sample_ids: {path}")
    if any(not isinstance(item, str) for item in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"SR arm sample_ids are invalid or duplicated: {path}")
    return {sample_id: float(value) for sample_id, value in zip(ids, values)}


def _pearson_per_image(reference: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    """Return one Pearson correlation over all channels/pixels per image."""
    ref_flat = reference.float().flatten(1)
    hr_flat = hr.float().flatten(1)
    ref_centered = ref_flat - ref_flat.mean(dim=1, keepdim=True)
    hr_centered = hr_flat - hr_flat.mean(dim=1, keepdim=True)
    numerator = (ref_centered * hr_centered).sum(dim=1)
    denominator = torch.sqrt(
        ref_centered.square().sum(dim=1) * hr_centered.square().sum(dim=1)
    ).clamp_min(1.0e-12)
    return (numerator / denominator).clamp(-1.0, 1.0)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _write_plot(path: Path, rows: list[dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional evaluation dependency
        raise RuntimeError("curve output requires matplotlib; install the evaluation dependencies") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    x = [row["reference_correlation"] for row in rows]
    y = [row["delta_psnr"] for row in rows]
    labels = [row["alpha"] for row in rows]
    plt.figure(figsize=(7.0, 4.8))
    plt.plot(x, y, marker="o", linewidth=2.0)
    for x_value, y_value, alpha in zip(x, y, labels):
        plt.annotate(f"{alpha:.1f}", (x_value, y_value), xytext=(4, 5), textcoords="offset points")
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xlabel("Reference correlation with HR (Pearson)")
    plt.ylabel("TRefSR - SR PSNR (dB)")
    plt.title("Reference quality ablation")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _write_rows(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "alpha",
        "samples",
        "reference_correlation",
        "reference_correlation_std",
        "trefsr_psnr",
        "sr_psnr",
        "delta_psnr",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_alphas(values: list[str] | None) -> list[float]:
    raw = DEFAULT_ALPHAS if values is None else tuple(float(value) for value in values)
    alphas = sorted(set(raw))
    if not alphas or any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in alphas):
        raise ValueError("alpha values must be finite numbers in [0, 1]")
    return alphas


def _scan_split(
    *,
    config: Mapping[str, Any],
    model: torch.nn.Module,
    split: str,
    sr_psnr: dict[str, float],
    device: torch.device,
    alphas: list[float],
    batch_size: int | None,
) -> tuple[list[dict[str, float | int]], list[dict[str, float]]]:
    data = config["data"]
    scale = int(data["scale"])
    loader = build_refsr_test_loader(config, split=split, batch_size=batch_size)
    eval_tile_size = data.get("eval_tile_size") if "wuhan" in str(config.get("dataset", {}).get("id", "")).lower() else None
    eval_tile_overlap = data.get("eval_tile_overlap", 0) if eval_tile_size is not None else 0
    per_alpha: dict[float, dict[str, list[float]]] = {
        alpha: {"correlation": [], "trefsr_psnr": [], "sr_psnr": []} for alpha in alphas
    }
    seen: set[str] = set()
    with torch.inference_mode():
        for batch in loader:
            batch = _move_batch(batch, device)
            lr = batch[data.get("lr_key", "lr")]
            hr = batch[data.get("hr_key", "hr")]
            batch_ids = _normalize_sample_ids(batch.get("sample_id"), int(hr.shape[0]))
            duplicates = seen.intersection(batch_ids)
            if duplicates:
                raise ValueError(f"duplicate sample IDs in {split}: {sorted(duplicates)[:8]}")
            missing = [sample_id for sample_id in batch_ids if sample_id not in sr_psnr]
            if missing:
                raise ValueError(f"SR arm is missing sample IDs in {split}: {missing[:8]}")
            seen.update(batch_ids)
            degraded = _reference_from_lr(lr, hr, scale)
            baseline = [sr_psnr[sample_id] for sample_id in batch_ids]
            for alpha in alphas:
                reference = alpha * hr + (1.0 - alpha) * degraded
                prediction = tiled_forward(
                    model,
                    lr,
                    reference,
                    scale=scale,
                    tile_size=eval_tile_size,
                    overlap=eval_tile_overlap,
                )
                prediction_metric, _ = _image_tensor(prediction, value_range="minus_one_one")
                values = per_image_psnr(prediction_metric, hr).detach().cpu().tolist()
                correlations = _pearson_per_image(reference, hr).detach().cpu().tolist()
                per_alpha[alpha]["trefsr_psnr"].extend(float(value) for value in values)
                per_alpha[alpha]["correlation"].extend(float(value) for value in correlations)
                per_alpha[alpha]["sr_psnr"].extend(float(value) for value in baseline)
    if not seen:
        raise RuntimeError(f"split {split!r} produced no samples")
    if seen != set(sr_psnr):
        missing = sorted(set(sr_psnr) - seen)
        raise ValueError(f"SR arm and TRefSR sample sets differ in {split}: missing={missing[:8]}")
    rows: list[dict[str, float | int]] = []
    per_image: list[dict[str, float]] = []
    for alpha in alphas:
        values = per_alpha[alpha]
        trefsr_mean = _mean(values["trefsr_psnr"])
        sr_mean = _mean(values["sr_psnr"])
        rows.append(
            {
                "alpha": alpha,
                "samples": len(values["trefsr_psnr"]),
                "reference_correlation": _mean(values["correlation"]),
                "reference_correlation_std": _std(values["correlation"]),
                "trefsr_psnr": trefsr_mean,
                "sr_psnr": sr_mean,
                "delta_psnr": trefsr_mean - sr_mean,
            }
        )
    return rows, per_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/test/test.yaml")
    parser.add_argument("--checkpoint", default=str(DEFAULT_TREF_CHECKPOINT))
    parser.add_argument("--sr-run", default=str(DEFAULT_SR_RUN), help="completed RefSRWKV-SR test-run root")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--split", action="append", choices=SPLITS, dest="splits")
    parser.add_argument("--alpha", action="append", dest="alphas", help="reference blend in [0,1]; repeatable")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--raw-weights", action="store_true", help="ignore EMA shadows")
    parser.add_argument("--overrides", nargs="*", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    alphas = _parse_alphas(args.alphas)
    checkpoint_path = resolve_path(args.checkpoint, prefer_cwd=True)
    checkpoint = load_checkpoint(checkpoint_path)
    config = load_test_config(args.config, checkpoint, args.overrides)
    validate_config(config)
    if str(config.get("task", "refsr")).lower() != "refsr":
        raise ValueError("reference_quality_scan.py requires a RefSR config")
    if normalize_reference_mode(config["data"].get("reference_mode", "paired")) != "paired":
        raise ValueError("reference quality scan requires data.reference_mode=paired")
    device = select_device(config, args.device)
    model, value_range, generator = _build_refsr_model(
        config,
        checkpoint,
        device,
        raw_weights=args.raw_weights,
    )
    if value_range != "minus_one_one" or generator is not None:
        raise ValueError("reference quality scan currently supports direct RefSR models only")
    sr_root = resolve_path(args.sr_run, prefer_cwd=True)
    output = resolve_path(args.output, prefer_cwd=True)
    for split in args.splits or list(SPLITS):
        sr_psnr = _load_sr_psnr(sr_root, split)
        rows, _ = _scan_split(
            config=config,
            model=model,
            split=split,
            sr_psnr=sr_psnr,
            device=device,
            alphas=alphas,
            batch_size=args.batch_size,
        )
        _write_rows(output / f"{split}.csv", rows)
        _write_plot(output / f"{split}.png", [{key: float(row[key]) for key in ("alpha", "reference_correlation", "delta_psnr")} for row in rows])
        payload = {
            "split": split,
            "checkpoint": str(checkpoint_path),
            "sr_run": str(sr_root),
            "alphas": alphas,
            "rows": rows,
            "reference_definition": "alpha * HR + (1 - alpha) * bicubic(LR)",
            "correlation": "per-image Pearson correlation over channels and pixels, then averaged",
        }
        (output / f"{split}.json").parent.mkdir(parents=True, exist_ok=True)
        (output / f"{split}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        LOGGER.info("saved reference quality scan for %s to %s", split, output)
        print(output / f"{split}.csv")


if __name__ == "__main__":
    main()
