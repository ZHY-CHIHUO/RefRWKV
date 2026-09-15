#!/usr/bin/env python3
"""Save clean RDM-PAN RGB previews for every test image.

No grid overlay.  Optional grayscale gradient maps make tile seams visible
as bright lines without covering the RGB.
"""

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

RGB_BANDS = (4, 2, 1)


def _sample_id(batch) -> str:
    value = batch.get("sample_id")
    if isinstance(value, (list, tuple)):
        value = value[0]
    text = str(value).strip()
    return text.replace("/", "_").replace("\\", "_")


def _stretch(rgb: np.ndarray) -> np.ndarray:
    out = np.empty_like(rgb, dtype=np.float32)
    for channel in range(3):
        lo, hi = np.percentile(rgb[..., channel], (2.0, 98.0))
        if hi <= lo:
            out[..., channel] = 0.0
        else:
            out[..., channel] = np.clip((rgb[..., channel] - lo) / (hi - lo), 0.0, 1.0)
    return out


def to_rgb(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(image):
        array = image.detach().float().cpu().clamp(0.0, 1.0).numpy()
    else:
        array = np.asarray(image, dtype=np.float32)
        if array.max() > 1.5:
            array = array / 2047.0
        array = np.clip(array, 0.0, 1.0)
    if array.ndim == 3 and array.shape[0] < array.shape[-1]:
        pass
    elif array.ndim == 3 and array.shape[-1] in {1, 3, 4, 8}:
        array = np.transpose(array, (2, 0, 1))
    if array.shape[0] >= 5:
        rgb = np.stack([array[idx] for idx in RGB_BANDS], axis=-1)
    elif array.shape[0] == 3:
        rgb = np.transpose(array, (1, 2, 0))
    else:
        gray = array.mean(axis=0)
        rgb = np.stack((gray, gray, gray), axis=-1)
    return _stretch(rgb)


def gradient_map(image: torch.Tensor) -> np.ndarray:
    array = image.detach().float().cpu().numpy()
    dx = np.abs(array[:, :, 1:] - array[:, :, :-1]).mean(axis=0)
    dy = np.abs(array[:, 1:, :] - array[:, :-1, :]).mean(axis=0)
    grad = np.zeros(array.shape[1:], dtype=np.float32)
    grad[:, 1:] += dx
    grad[1:, :] += dy
    vis = np.clip(grad / max(float(np.percentile(grad, 99.0)), 1.0e-6), 0.0, 1.0)
    vis = np.power(vis, 0.45)
    return np.stack((vis, vis, vis), axis=-1)


def save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb * 255.0 + 0.5).clip(0, 255).astype(np.uint8)).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="experiments/train/refsr/rdm_pan/pancollection_wv3/x4/pancollection_wv3_rdm_pan_x4/checkpoints/last.ckpt",
    )
    parser.add_argument("--output", default="experiments/test/refsr/rdm_pan_preview")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all images")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--overlaps", nargs="+", type=int, default=[8, 0])
    parser.add_argument("--splits", nargs="+", default=["test", "test_hard"])
    parser.add_argument("--save-grad", action="store_true", default=True)
    parser.add_argument("--no-grad", dest="save_grad", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    # Training checkpoints currently bake test_hard -> reduced_examples.
    # Force the OrigScale H5 so FR previews are 512, matching evaluation.
    config = load_test_config(
        "configs/test/test.yaml",
        checkpoint,
        overrides=[
            "dataset.files.test_hard=full_examples/test_wv3_OrigScale_multiExm1.h5",
            "data.files.test_hard=full_examples/test_wv3_OrigScale_multiExm1.h5",
        ],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  ckpt={args.checkpoint}", flush=True)
    model, _, _ = _build_refsr_model(config, checkpoint, device)
    scale = int(config["data"]["scale"])
    # Preview is sequential and WSL /dev/shm is small; worker processes
    # hit "unable to allocate shared memory" on the last few samples.
    config.setdefault("data", {})
    config["data"]["num_workers"] = 0
    config["data"]["val_num_workers"] = 0
    config["data"]["persistent_workers"] = False
    output_root = Path(args.output)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    limit = None if int(args.max_samples) <= 0 else int(args.max_samples)

    for split in args.splits:
        loader = build_refsr_test_loader(config, split=split, batch_size=1)
        for overlap in args.overlaps:
            if overlap < 0 or overlap >= args.tile_size:
                raise ValueError(f"overlap {overlap} must be in [0, {args.tile_size})")
            tag = f"tile{args.tile_size}_ov{overlap}"
            out_dir = output_root / tag / split
            count = 0
            if split == "test_hard":
                h5_path = str(getattr(loader.dataset, "file_path", ""))
                print(f"{tag:16s} {split:9s} h5={h5_path}", flush=True)
                if "OrigScale" not in Path(h5_path).name:
                    raise RuntimeError(
                        f"test_hard must use OrigScale H5, got {h5_path}. "
                        "Do not reuse reduced_examples/test_wv3_multiExm1.h5."
                    )
            for batch in loader:
                lr = batch["lr"].to(device)
                ref = batch["ref"].to(device)
                sample_id = _sample_id(batch)
                if count == 0:
                    print(
                        f"{tag:16s} {split:9s} first batch lr={tuple(lr.shape)} ref={tuple(ref.shape)}",
                        flush=True,
                    )
                    if split == "test_hard" and tuple(ref.shape[-2:]) != (512, 512):
                        raise RuntimeError(
                            f"test_hard PAN must be 512x512, got {tuple(ref.shape)}. "
                            "This is the reduced-res split, not OrigScale."
                        )
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
                save_png(out_dir / f"{sample_id}.png", to_rgb(image))
                if args.save_grad:
                    save_png(out_dir / f"{sample_id}_grad.png", gradient_map(image))
                if "hr" in batch and split != "test_hard":
                    save_png(out_dir / f"{sample_id}_gt.png", to_rgb(batch["hr"][0]))
                count += 1
                print(
                    f"{tag:16s} {split:9s} {sample_id}  {tuple(image.shape)}",
                    flush=True,
                )
                if limit is not None and count >= limit:
                    break
            print(f"saved {count} images -> {out_dir}", flush=True)

    meta = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "tile_size": args.tile_size,
        "overlaps": list(args.overlaps),
        "splits": list(args.splits),
        "note": "PNG is clean RGB, no grid overlay. *_grad.png is a grayscale gradient; seams look like bright straight lines.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preview.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
