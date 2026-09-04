#!/usr/bin/env python3
"""Evaluate an HR/LR SISR baseline checkpoint with the shared protocol."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baselines.common import (
    gaussian_ssim,
    json_safe,
    load_config,
    per_image_psnr,
    resolve_path,
    validate_config,
    write_json,
)
from baselines.registry import build_model, get_adapter
from baselines.runtime import checkpoint_config, load_checkpoint, load_model_weights
from SR_data.SRDataset import SRPNGDataset


def _crop_border(tensor: torch.Tensor, border: int) -> torch.Tensor:
    if border == 0:
        return tensor
    if min(tensor.shape[-2:]) <= 2 * border:
        raise ValueError(f"image {tuple(tensor.shape[-2:])} is too small for border={border}")
    return tensor[..., border:-border, border:-border]


def _warmup(model: torch.nn.Module, loader: DataLoader, device: torch.device, runs: int) -> None:
    if runs <= 0:
        return
    batch = next(iter(loader), None)
    if batch is None:
        return
    lr = batch["lr"].to(device, non_blocking=device.type == "cuda")
    with torch.inference_mode():
        for _ in range(runs):
            _ = model(lr)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    scale: int,
    border: int,
) -> dict[str, Any]:
    metrics = {"bicubic": {"psnr": [], "ssim": []}, "model": {"psnr": [], "ssim": []}}
    forward_seconds = 0.0
    seen = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        lr = batch["lr"].to(device, non_blocking=device.type == "cuda")
        hr = batch["hr"].to(device, non_blocking=device.type == "cuda")
        expected = (lr.shape[-2] * scale, lr.shape[-1] * scale)
        if hr.shape[-2:] != expected:
            raise ValueError(
                f"dataset geometry mismatch: LR={tuple(lr.shape[-2:])}, HR={tuple(hr.shape[-2:])}, scale=x{scale}"
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        prediction = model(lr).clamp(-1.0, 1.0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_seconds += time.perf_counter() - start
        bicubic = F.interpolate(lr, size=expected, mode="bicubic", align_corners=False).clamp(-1.0, 1.0)
        metric_hr = _crop_border(hr, border)
        for name, output in (("bicubic", bicubic), ("model", prediction)):
            output = _crop_border(output, border)
            metrics[name]["psnr"].extend(per_image_psnr(output, metric_hr).cpu().tolist())
            metrics[name]["ssim"].extend(gaussian_ssim(output, metric_hr).cpu().tolist())
        seen += lr.shape[0]
    if seen == 0:
        raise ValueError("evaluation split is empty")
    result_metrics = {
        name: {metric: float(sum(values) / len(values)) for metric, values in values_by_metric.items()}
        for name, values_by_metric in metrics.items()
    }
    performance: dict[str, float | int] = {
        "images": seen,
        "model_forward_seconds": forward_seconds,
        "milliseconds_per_image": forward_seconds * 1000.0 / seen,
        "images_per_second": seen / forward_seconds if forward_seconds > 0 else float("inf"),
    }
    if device.type == "cuda":
        performance["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024.0**2)
    return {"metrics": result_metrics, "performance": performance}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="run YAML; otherwise read checkpoint metadata")
    parser.add_argument("--data", default=None, help="override data.root")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=None, help="evaluate the first N sorted samples")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--border", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--raw", action="store_true", help="ignore EMA weights")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", default=None, help="result JSON path")
    args = parser.parse_args()
    if args.n is not None and args.n < 1:
        raise ValueError("--n must be positive")
    if args.batch_size < 1 or args.num_workers < 0 or args.border < 0 or args.warmup < 0:
        raise ValueError("batch size, workers, border, and warmup must be non-negative (batch size positive)")

    checkpoint_path = resolve_path(args.checkpoint, prefer_cwd=True)
    checkpoint = load_checkpoint(checkpoint_path)
    if args.config:
        config = load_config(args.config)
    else:
        config = checkpoint_config(checkpoint)
        if config is None:
            raise ValueError("checkpoint has no saved config; pass --config")
    if args.data:
        config["data"]["root"] = args.data
    validate_config(config)

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config["model"], scale=int(config["data"]["scale"]))
    report = load_model_weights(model, checkpoint, prefer_ema=not args.raw)
    model = model.to(device).eval()
    dataset = SRPNGDataset(
        data_dir=resolve_path(config["data"]["root"]),
        mode=args.split,
        patch_size=None,
        scale=int(config["data"]["scale"]),
        augment=False,
        max_samples=(None, None, None),
        sample_seed=int(config.get("train", {}).get("seed", 42)),
        lr_key=config["data"].get("lr_key", "lr"),
        hr_key=config["data"].get("hr_key", "hr"),
    )
    if args.n is not None:
        dataset.filenames = dataset.filenames[: args.n]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    _warmup(model, loader, device, args.warmup)
    result = evaluate(
        model,
        loader,
        device,
        scale=int(config["data"]["scale"]),
        border=args.border,
    )
    adapter = get_adapter(config["model"]["name"])
    weights = "raw" if args.raw or not report["ema_applied"] else "ema"
    payload: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "weights": weights,
        "load_report": report,
        "dataset": {
            "id": config.get("dataset", {}).get("id", "dataset"),
            "root": str(resolve_path(config["data"]["root"])),
            "split": args.split,
            "scale": int(config["data"]["scale"]),
            "value_range": "[-1,1]",
            "metric": "RGB per-image mean; Gaussian SSIM 11x11; border crop applied before metrics",
            "border": args.border,
        },
        "model": adapter.describe(config["model"], scale=int(config["data"]["scale"])),
        "metrics": result["metrics"],
        "performance": result["performance"],
        "warmup_for_timing": args.warmup,
    }
    output = (
        resolve_path(args.output, prefer_cwd=True)
        if args.output
        else PROJECT_ROOT
        / "results"
        / "baselines"
        / f"{config['run']['name']}_{args.split}_{weights}.json"
    )
    write_json(output, payload)
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
