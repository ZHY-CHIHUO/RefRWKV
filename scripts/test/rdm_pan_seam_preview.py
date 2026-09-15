#!/usr/bin/env python3
"""Save RDM-PAN tiled predictions and RGB/seam previews."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loaders import build_refsr_test_loader
from evaluation.runner import _build_refsr_model
from runtime.checkpoint import load_checkpoint
from runtime.config import load_test_config
from runtime.tiling import tiled_forward

# WorldView-3 8-band: Coastal, Blue, Green, Yellow, Red, Red Edge, NIR1, NIR2
RGB_BANDS = (4, 2, 1)


def _stretch(rgb: np.ndarray) -> np.ndarray:
    out = np.empty_like(rgb, dtype=np.float32)
    for channel in range(3):
        lo, hi = np.percentile(rgb[..., channel], (2.0, 98.0))
        if hi <= lo:
            out[..., channel] = 0.0
        else:
            out[..., channel] = np.clip((rgb[..., channel] - lo) / (hi - lo), 0.0, 1.0)
    return out


def _to_rgb(image: torch.Tensor) -> np.ndarray:
    array = image.detach().float().cpu().clamp(0.0, 1.0).numpy()
    if array.shape[0] >= 5:
        rgb = np.stack([array[idx] for idx in RGB_BANDS], axis=-1)
    elif array.shape[0] == 3:
        rgb = np.transpose(array, (1, 2, 0))
    else:
        gray = array.mean(axis=0)
        rgb = np.stack([gray, gray, gray], axis=-1)
    return _stretch(rgb)


def _save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb * 255.0 + 0.5).clip(0, 255).astype(np.uint8)).save(path)


def _grid_coords(length: int, stride: int) -> list[int]:
    if stride < 8:
        return []
    return [index for index in range(stride, length - 1, stride)]


def _line_score(diff: np.ndarray, lines: list[int], axis: int) -> float:
    if not lines:
        return 0.0
    values = [float(np.mean(np.take(diff, line, axis=axis))) for line in lines]
    return float(np.mean(values))


def seam_report(image: torch.Tensor, stride: int) -> dict[str, float]:
    array = image.detach().float().cpu().numpy()
    dx = np.abs(array[:, :, 1:] - array[:, :, :-1]).mean(axis=0)
    dy = np.abs(array[:, 1:, :] - array[:, :-1, :]).mean(axis=0)
    height, width = dx.shape[0], dy.shape[1]
    x_lines = _grid_coords(width, stride)
    y_lines = _grid_coords(height, stride)
    interior_x = [col for col in range(1, width - 1) if all(abs(col - line) > 1 for line in x_lines)]
    interior_y = [row for row in range(1, height - 1) if all(abs(row - line) > 1 for line in y_lines)]
    grid_x = _line_score(dx, x_lines, axis=1)
    grid_y = _line_score(dy, y_lines, axis=0)
    interior_dx = _line_score(dx, interior_x, axis=1) if interior_x else float(dx.mean())
    interior_dy = _line_score(dy, interior_y, axis=0) if interior_y else float(dy.mean())
    ratio_x = grid_x / max(interior_dx, 1.0e-8)
    ratio_y = grid_y / max(interior_dy, 1.0e-8)
    return {
        "stride": float(stride),
        "grid_dx": grid_x,
        "grid_dy": grid_y,
        "interior_dx": interior_dx,
        "interior_dy": interior_dy,
        "ratio_x": ratio_x,
        "ratio_y": ratio_y,
        "ratio": 0.5 * (ratio_x + ratio_y),
    }


def seam_map(image: torch.Tensor, stride: int) -> np.ndarray:
    array = image.detach().float().cpu().numpy()
    dx = np.abs(array[:, :, 1:] - array[:, :, :-1]).mean(axis=0)
    dy = np.abs(array[:, 1:, :] - array[:, :-1, :]).mean(axis=0)
    grad = np.zeros(array.shape[1:], dtype=np.float32)
    grad[:, 1:] += dx
    grad[1:, :] += dy
    vis = np.clip(grad / max(np.percentile(grad, 99.0), 1.0e-6), 0.0, 1.0)
    vis = np.power(vis, 0.45)
    rgb = np.stack([vis, vis, vis], axis=-1)
    height, width = vis.shape
    for x in _grid_coords(width, stride):
        rgb[:, max(0, x - 1) : x + 2, 0] = 1.0
        rgb[:, max(0, x - 1) : x + 2, 1] *= 0.2
        rgb[:, max(0, x - 1) : x + 2, 2] *= 0.2
    for y in _grid_coords(height, stride):
        rgb[max(0, y - 1) : y + 2, :, 0] = 1.0
        rgb[max(0, y - 1) : y + 2, :, 1] *= 0.2
        rgb[max(0, y - 1) : y + 2, :, 2] *= 0.2
    return rgb


def overlay_grid(rgb: np.ndarray, stride: int) -> np.ndarray:
    out = rgb.copy()
    height, width, _ = out.shape
    for x in _grid_coords(width, stride):
        out[:, x, :] = (0.95, 0.15, 0.15)
    for y in _grid_coords(height, stride):
        out[y, :, :] = (0.95, 0.15, 0.15)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="experiments/train/refsr/rdm_pan/pancollection_wv3/x4/pancollection_wv3_rdm_pan_x4/checkpoints/last.ckpt",
    )
    parser.add_argument("--output", default="experiments/test/refsr/rdm_pan_seams")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    config = load_test_config("configs/test/test.yaml", checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = _build_refsr_model(config, checkpoint, device)
    scale = int(config["data"]["scale"])
    output_root = Path(args.output)
    reports: dict[str, list[dict[str, float | str]]] = {}

    for split in ("test", "test_hard"):
        loader = build_refsr_test_loader(config, split=split, batch_size=1)
        for overlap in (8, 0):
            tag = f"ov{overlap}"
            stride = (args.tile_size - overlap) * scale
            count = 0
            for batch in loader:
                lr = batch["lr"].to(device)
                ref = batch["ref"].to(device)
                sample_id = str(batch["sample_id"][0] if isinstance(batch["sample_id"], (list, tuple)) else batch["sample_id"])
                with torch.inference_mode():
                    prediction = tiled_forward(
                        model,
                        lr,
                        ref,
                        scale=scale,
                        tile_size=args.tile_size,
                        overlap=overlap,
                        input_scales=(1, scale),
                    ).clamp(0.0, 1.0)
                image = prediction[0].cpu()
                rgb = _to_rgb(image)
                stem = output_root / tag / split / sample_id
                _save_png(stem.parent / f"{sample_id}_rgb.png", rgb)
                _save_png(stem.parent / f"{sample_id}_grid.png", overlay_grid(rgb, stride))
                _save_png(stem.parent / f"{sample_id}_seam.png", seam_map(image, stride))
                np.save(stem.parent / f"{sample_id}.npy", image.numpy())
                stats = seam_report(image, stride)
                stats["sample_id"] = sample_id
                stats["split"] = split
                stats["overlap"] = float(overlap)
                reports.setdefault(f"{tag}/{split}", []).append(stats)
                count += 1
                print(
                    f"{tag:4s} {split:9s} {sample_id}  "
                    f"shape={tuple(image.shape)}  seam_ratio={stats['ratio']:.3f}  "
                    f"x={stats['ratio_x']:.3f} y={stats['ratio_y']:.3f}",
                    flush=True,
                )
                if count >= args.max_samples:
                    break

    summary = {}
    for key, rows in reports.items():
        ratios = [float(row["ratio"]) for row in rows]
        summary[key] = {
            "n": len(rows),
            "mean_ratio": float(np.mean(ratios)),
            "max_ratio": float(np.max(ratios)),
            "samples": rows,
        }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "seam_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: {k: v for k, v in value.items() if k != "samples"} for key, value in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
