#!/usr/bin/env python3
"""Prepare UC Merced or AID images for RefPNGDataset.

The source datasets contain single high-resolution images arranged by class.
This tool creates a deterministic stratified split and synthetic 4x SR pairs:

    output/<split>/HR/*.png
    output/<split>/LR/*.png
    output/<split>/Ref/*.png

The Ref image is the bicubic-upsampled LR image.  This matches the SISR
fallback used during training when ``ref_drop_prob`` replaces a reference.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import random
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageChops

DATASET_DEFAULTS = {
    "ucmerced": {
        "hr_size": 256,
        "expected_classes": 21,
        "expected_images": 2100,
        "source_url": (
            "https://www.kaggle.com/datasets/abdulhasibuddin/"
            "uc-merced-land-use-dataset"
        ),
    },
    "aid": {
        "hr_size": 512,
        "expected_classes": 30,
        "expected_images": 10000,
        "source_url": (
            "https://www.kaggle.com/datasets/jiayuanchengala/"
            "aid-scene-classification-datasets"
        ),
    },
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SourceRecord:
    path: Path
    class_name: str
    stem: str


def _safe_extract(archive: Path, destination: Path) -> Path:
    """Extract a zip archive without allowing paths outside destination."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"unsafe archive member: {member.filename}")
        zf.extractall(destination)
    return destination


def _normalise_class_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return value or "class"


def _unique_name(record: SourceRecord, used: set[str]) -> str:
    base = f"{_normalise_class_name(record.class_name)}__{_normalise_class_name(record.stem)}"
    if base not in used:
        used.add(base)
        return base
    digest = hashlib.sha1(str(record.path).encode("utf-8")).hexdigest()[:8]
    name = f"{base}__{digest}"
    used.add(name)
    return name


def _discover_images(source_dir: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        # Both datasets use one directory per class.  A file directly under
        # the source root is ignored because it has no reliable class label.
        if path.parent == source_dir:
            continue
        records.append(
            SourceRecord(path=path, class_name=path.parent.name, stem=path.stem)
        )
    if not records:
        raise RuntimeError(f"no supported images found below {source_dir}")
    return records


def _split_counts(
    total: int, ratios: tuple[float, float, float]
) -> tuple[int, int, int]:
    if total < 3:
        raise ValueError(f"class has only {total} images; at least 3 are required")
    raw = [total * ratio for ratio in ratios]
    counts = [int(math.floor(value)) for value in raw]
    remainder = total - sum(counts)
    order = sorted(
        range(3), key=lambda i: (raw[i] - counts[i], ratios[i]), reverse=True
    )
    for i in order[:remainder]:
        counts[i] += 1

    active = [i for i, ratio in enumerate(ratios) if ratio > 0]
    if len(active) > total:
        raise ValueError("not enough images for the requested non-empty splits")
    for i in active:
        if counts[i] > 0:
            continue
        donor = max(
            (j for j in active if counts[j] > 1), key=lambda j: counts[j], default=None
        )
        if donor is None:
            raise ValueError("cannot make every requested split non-empty")
        counts[donor] -= 1
        counts[i] = 1
    return tuple(counts)  # type: ignore[return-value]


def _stratified_split(
    records: Iterable[SourceRecord],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[SourceRecord]]:
    grouped: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        grouped[record.class_name].append(record)

    rng = random.Random(seed)
    result = {split: [] for split in SPLITS}
    for class_name in sorted(grouped):
        items = sorted(grouped[class_name], key=lambda item: str(item.path))
        rng.shuffle(items)
        counts = _split_counts(len(items), ratios)
        begin = 0
        for split, count in zip(SPLITS, counts):
            result[split].extend(items[begin : begin + count])
            begin += count
    for split in SPLITS:
        result[split].sort(key=lambda item: (item.class_name, str(item.path)))
    return result


def _center_crop(image: Image.Image, size: int, source: Path) -> Image.Image:
    width, height = image.size
    if width < size or height < size:
        # A few UC Merced mirror files are a pixel or two below the nominal
        # 256x256 size.  Resizing these rare cases keeps the class sample
        # instead of silently dropping it; normal larger images still use a
        # lossless center crop below.
        return image.resize((size, size), Image.Resampling.BICUBIC).convert("RGB")
    left = (width - size) // 2
    top = (height - size) // 2
    return image.crop((left, top, left + size, top + size)).convert("RGB")


def _open_hr(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        return _center_crop(image, size, path)


def _write_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The source archives are large; a fast PNG level keeps preparation
    # practical while preserving lossless RGB pixels for the dataset.
    image.save(path, format="PNG", optimize=False, compress_level=1)


def _validate_output(
    output_dir: Path, expected_sizes: dict[str, tuple[int, int]]
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in SPLITS:
        hr_dir = output_dir / split / "HR"
        lr_dir = output_dir / split / "LR"
        ref_dir = output_dir / split / "Ref"
        names = sorted(path.stem for path in lr_dir.glob("*.png"))
        if not names:
            raise RuntimeError(f"prepared split is empty: {split}")
        directory_counts = {
            directory: len(list((output_dir / split / directory).glob("*.png")))
            for directory in ("LR", "HR", "Ref")
        }
        if len(set(directory_counts.values())) != 1:
            raise RuntimeError(
                f"split directories have different counts: {split}: {directory_counts}"
            )
        for name in names:
            paths = [
                hr_dir / f"{name}.png",
                lr_dir / f"{name}.png",
                ref_dir / f"{name}.png",
            ]
            if not all(path.is_file() for path in paths):
                raise RuntimeError(f"incomplete pair for {split}/{name}")
            with Image.open(paths[0]) as hr, Image.open(paths[1]) as lr, Image.open(
                paths[2]
            ) as ref:
                if hr.mode != "RGB" or lr.mode != "RGB" or ref.mode != "RGB":
                    raise RuntimeError(f"non-RGB output for {split}/{name}")
                if hr.size != expected_sizes["hr"] or ref.size != expected_sizes["hr"]:
                    raise RuntimeError(
                        f"bad HR/Ref size for {split}/{name}: {hr.size}, {ref.size}"
                    )
                if lr.size != expected_sizes["lr"]:
                    raise RuntimeError(f"bad LR size for {split}/{name}: {lr.size}")
                expected_ref = lr.resize(hr.size, Image.Resampling.BICUBIC)
                if ImageChops.difference(ref, expected_ref).getbbox() is not None:
                    raise RuntimeError(
                        f"Ref is not bicubic-upsampled LR for {split}/{name}"
                    )
        counts[split] = len(names)
    return counts


def _prepare_item(
    args: tuple[SourceRecord, str, Path, str, int, int],
) -> dict[str, object]:
    record, name, output_dir, split, hr_size, lr_size = args
    hr = _open_hr(record.path, hr_size)
    lr = hr.resize((lr_size, lr_size), Image.Resampling.BICUBIC)
    # The prepared reference intentionally follows the SISR path: it
    # contains no external semantic information and is exactly the LR input
    # resized back to the HR grid.
    ref = lr.resize((hr_size, hr_size), Image.Resampling.BICUBIC)
    _write_image(output_dir / split / "HR" / f"{name}.png", hr)
    _write_image(output_dir / split / "LR" / f"{name}.png", lr)
    _write_image(output_dir / split / "Ref" / f"{name}.png", ref)
    return {
        "name": name,
        "split": split,
        "class": record.class_name,
        "source": str(record.path),
        "reference_source": None,
        "reference_policy": "bicubic-upsampled LR",
    }


def prepare(
    dataset: str,
    source_dir: Path,
    output_dir: Path,
    hr_size: int,
    scale: int,
    ratios: tuple[float, float, float],
    seed: int,
    overwrite: bool,
    source_archive: Path | None,
    workers: int = 4,
) -> None:
    if hr_size % scale != 0:
        raise ValueError(f"--hr-size {hr_size} must be divisible by --scale {scale}")
    if hr_size % 32 != 0:
        raise ValueError("--hr-size must be divisible by 32 for RefSRWKV")
    if any(ratio <= 0 for ratio in ratios) or not math.isclose(
        sum(ratios), 1.0, abs_tol=1e-6
    ):
        raise ValueError("train/val/test ratios must all be positive and sum to 1")
    if workers < 1:
        raise ValueError("--workers must be at least 1")

    existing_pngs = list(output_dir.glob("*/HR/*.png")) if output_dir.exists() else []
    if existing_pngs and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains prepared data; use --overwrite to rebuild"
        )
    if overwrite:
        for split in SPLITS:
            split_dir = output_dir / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _discover_images(source_dir)
    class_counts = Counter(record.class_name for record in records)
    defaults = DATASET_DEFAULTS[dataset]
    if len(class_counts) != defaults["expected_classes"]:
        print(
            f"warning: found {len(class_counts)} classes; expected {defaults['expected_classes']} for {dataset}",
            file=sys.stderr,
        )
    if len(records) != defaults["expected_images"]:
        print(
            f"warning: found {len(records)} images; expected about {defaults['expected_images']} for {dataset}",
            file=sys.stderr,
        )
    if min(class_counts.values()) < 3:
        raise ValueError("every class must contain at least 3 images")

    split_records = _stratified_split(records, ratios, seed)
    used_names: set[str] = set()
    names_by_record: dict[SourceRecord, str] = {}
    for record in records:
        names_by_record[record] = _unique_name(record, used_names)

    lr_size = hr_size // scale
    all_manifest: list[dict[str, object]] = []
    split_counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for split in SPLITS:
            items = split_records[split]
            split_counts[split] = len(items)
            jobs = (
                (record, names_by_record[record], output_dir, split, hr_size, lr_size)
                for record in items
            )
            for index, manifest_item in enumerate(
                executor.map(_prepare_item, jobs), start=1
            ):
                all_manifest.append(manifest_item)
                if index % 250 == 0 or index == len(items):
                    print(f"[{dataset}/{split}] {index}/{len(items)}", flush=True)

    sizes = {"hr": [hr_size, hr_size], "lr": [lr_size, lr_size]}
    validated = _validate_output(
        output_dir, {"hr": (hr_size, hr_size), "lr": (lr_size, lr_size)}
    )
    metadata = {
        "dataset": dataset,
        "source_url": defaults["source_url"],
        "source_archive": str(source_archive) if source_archive else None,
        "source_dir": str(source_dir),
        "seed": seed,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "scale": scale,
        "sizes": sizes,
        "split_counts": split_counts,
        "validated_counts": validated,
        "class_counts": dict(sorted(class_counts.items())),
        "reference_policy": "bicubic-upsampled LR at HR resolution (SISR mode)",
        "degradation": "PIL bicubic downsampling from HR to LR",
        "loader": "RefSR_data.RefDataset.RefPNGDataset",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as manifest_file:
        for item in all_manifest:
            manifest_file.write(json.dumps(item, ensure_ascii=True) + "\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="extracted class-folder directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, default=None)
    parser.add_argument("--hr-size", type=int, default=None)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers", type=int, default=4, help="并行生成样本的 worker 数"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.hr_size is None:
        args.hr_size = DATASET_DEFAULTS[args.dataset]["hr_size"]
    return args


def main() -> None:
    args = _parse_args()
    prepare(
        dataset=args.dataset,
        source_dir=args.source_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        hr_size=args.hr_size,
        scale=args.scale,
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
        overwrite=args.overwrite,
        source_archive=args.source_archive.resolve() if args.source_archive else None,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
