#!/usr/bin/env python3
"""Convert official pansharpening/HSI .mat outputs to RGB PNG previews."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SKIP_DIRS = {"panmamba_code"}


def _natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.stem)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def _first_array(path: Path) -> np.ndarray:
    import scipy.io as sio

    try:
        payload = sio.loadmat(path)
        keys = [key for key in payload if not key.startswith("__")]
        if not keys:
            raise ValueError(f"no arrays in {path}")
        return np.array(payload[keys[0]])
    except Exception:
        import h5py

        with h5py.File(path, "r") as handle:
            key = list(handle.keys())[0]
            return np.array(handle[key])


def as_hwc_list(array: np.ndarray, path: Path) -> list[np.ndarray]:
    array = np.asarray(array)
    if array.ndim == 4:
        if array.shape[-1] <= 128:
            return [array[index] for index in range(array.shape[0])]
        if array.shape[1] <= 128:
            return [np.transpose(array[index], (1, 2, 0)) for index in range(array.shape[0])]
    if array.ndim == 3:
        squeezed = np.squeeze(array)
        if squeezed.ndim == 2:
            return [squeezed[..., None]]
        if squeezed.shape[-1] <= 128:
            return [squeezed]
        if squeezed.shape[0] <= 128:
            return [np.transpose(squeezed, (1, 2, 0))]
    raise ValueError(f"unsupported array {array.shape} from {path}")


def rgb_indices(bands: int) -> tuple[int, int, int]:
    if bands >= 8 and bands <= 10:
        return (min(4, bands - 1), min(2, bands - 1), min(1, bands - 1))
    if bands == 4:
        return (2, 1, 0)
    if bands == 3:
        return (0, 1, 2)
    if bands == 1:
        return (0, 0, 0)
    return (
        min(bands - 1, int(round(0.72 * (bands - 1)))),
        int(round(0.45 * (bands - 1))),
        int(round(0.12 * (bands - 1))),
    )


def to_rgb(hwc: np.ndarray) -> np.ndarray:
    if hwc.ndim != 3:
        raise ValueError(f"expected HWC, got {hwc.shape}")
    bands = int(hwc.shape[-1])
    idx = rgb_indices(bands)
    rgb = np.stack((hwc[..., idx[0]], hwc[..., idx[1]], hwc[..., idx[2]]), axis=-1).astype(np.float32)
    out = np.empty_like(rgb, dtype=np.float32)
    for channel in range(3):
        lo, hi = np.percentile(rgb[..., channel], (1.0, 99.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            out[..., channel] = 0.0
        else:
            out[..., channel] = np.clip((rgb[..., channel] - lo) / (hi - lo), 0.0, 1.0)
    return out


def save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb * 255.0 + 0.5).clip(0, 255).astype(np.uint8)).save(path)


def contact_sheet(images: list[np.ndarray], cols: int = 5) -> np.ndarray:
    if not images:
        raise ValueError("no images")
    thumb = []
    max_h = max(image.shape[0] for image in images)
    max_w = max(image.shape[1] for image in images)
    target = 256 if max(max_h, max_w) > 256 else max(max_h, max_w)
    for image in images:
        pil = Image.fromarray((image * 255.0 + 0.5).clip(0, 255).astype(np.uint8))
        pil.thumbnail((target, target))
        canvas = Image.new("RGB", (target, target), (0, 0, 0))
        canvas.paste(pil, ((target - pil.size[0]) // 2, (target - pil.size[1]) // 2))
        thumb.append(np.asarray(canvas).astype(np.float32) / 255.0)
    height = width = target
    rows = (len(thumb) + cols - 1) // cols
    sheet = np.zeros((rows * height, cols * width, 3), dtype=np.float32)
    for index, image in enumerate(thumb):
        row, col = divmod(index, cols)
        sheet[row * height : (row + 1) * height, col * width : (col + 1) * width] = image
    return sheet


def iter_mat_files(src: Path) -> list[Path]:
    files = list(src.glob("*.mat"))
    if (src / "results").is_dir():
        files.extend((src / "results").glob("*.mat"))
    return sorted(set(files), key=_natural_key)


def convert_mat_dir(src: Path, dst: Path, *, skip_existing: bool) -> int:
    files = iter_mat_files(src)
    rgbs: list[np.ndarray] = []
    written = 0
    for path in files:
        cubes = as_hwc_list(_first_array(path), path)
        index_match = re.search(r"(\d+)$", path.stem)
        stem = f"{int(index_match.group(1)):02d}" if index_match else path.stem
        for cube_id, cube in enumerate(cubes):
            name = f"{stem}.png" if len(cubes) == 1 else f"{stem}_{cube_id}.png"
            out_path = dst / name
            if skip_existing and out_path.is_file():
                try:
                    rgbs.append(np.asarray(Image.open(out_path)).astype(np.float32) / 255.0)
                except Exception:
                    pass
                continue
            rgb = to_rgb(cube)
            save_png(out_path, rgb)
            rgbs.append(rgb)
            written += 1
            print(f"  {path.name} -> {out_path.relative_to(dst.parent.parent) if dst.parent.parent.exists() else out_path} {tuple(cube.shape)}", flush=True)
    if rgbs:
        save_png(dst / "_contact.png", contact_sheet(rgbs))
    return written


def convert_wv3_gt(dst: Path, *, skip_existing: bool) -> int:
    import h5py

    h5_path = PROJECT_ROOT / "data/refsr/PanCollection/training_wv3/reduced_examples/test_wv3_multiExm1.h5"
    if not h5_path.is_file():
        print(f"skip WV3 GT, missing {h5_path}")
        return 0
    with h5py.File(h5_path, "r") as handle:
        gt = np.array(handle["gt"])
    rgbs = []
    written = 0
    for index, image in enumerate(gt):
        out_path = dst / f"{index:02d}.png"
        if skip_existing and out_path.is_file():
            rgbs.append(np.asarray(Image.open(out_path)).astype(np.float32) / 255.0)
            continue
        rgb = to_rgb(np.transpose(image, (1, 2, 0)))
        save_png(out_path, rgb)
        rgbs.append(rgb)
        written += 1
        print(f"  GT {index:02d} {tuple(rgb.shape)}", flush=True)
    if rgbs:
        save_png(dst / "_contact.png", contact_sheet(rgbs))
    return written


def discover_jobs(src_root: Path) -> list[tuple[str, str, Path]]:
    jobs = []
    for dataset_dir in sorted(src_root.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.lower() in SKIP_DIRS:
            continue
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if iter_mat_files(model_dir):
                jobs.append((dataset_dir.name, model_dir.name, model_dir))
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="/mnt/d/Download/Results")
    parser.add_argument("--output", default="experiments/vis/official_mat_previews")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = Path(args.src)
    out_root = Path(args.output)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    skip_existing = bool(args.skip_existing) and not bool(args.overwrite)
    jobs = discover_jobs(src_root)
    print(f"found {len(jobs)} model folders under {src_root}", flush=True)
    total = 0
    for dataset, model, src in jobs:
        dst = out_root / dataset / model
        print(f"\n==== {dataset}/{model}", flush=True)
        total += convert_mat_dir(src, dst, skip_existing=skip_existing)
    print("\n==== WV3_rr/GT from PanCollection H5", flush=True)
    total += convert_wv3_gt(out_root / "WV3_rr" / "GT", skip_existing=skip_existing)
    print(f"\nWrote {total} new PNGs under {out_root}", flush=True)


if __name__ == "__main__":
    main()
