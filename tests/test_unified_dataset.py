"""Regression tests for the unified HR/LR/Ref Dataset contract."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import SuperResolutionDataset
from data.loaders import build_sr_loaders


def _write_png(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _write_hr(root: Path, split: str = "train", *, include_lr: bool = False, include_ref: bool = False) -> None:
    _write_png(root / split / "HR" / "sample.png", (8, 8), (64, 128, 192))
    if include_lr:
        _write_png(root / split / "LR" / "sample.png", (4, 4), (80, 120, 160))
    if include_ref:
        _write_png(root / split / "Ref" / "sample.png", (8, 8), (192, 128, 64))


class UnifiedDatasetTests(unittest.TestCase):
    def test_hr_only_generates_bicubic_lr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_hr(root)
            dataset = SuperResolutionDataset(
                root, mode="train", scale=2, return_items=("lr", "hr"), lr_provenance="bicubic"
            )
            sample = dataset[0]
            self.assertEqual(sample["lr"].shape, (3, 4, 4))
            self.assertEqual(sample["hr"].shape, (3, 8, 8))

    def test_stored_reference_can_be_selected_without_returning_ref_for_swinir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_hr(root, include_lr=True, include_ref=True)
            ignored = SuperResolutionDataset(
                root, mode="train", scale=2, return_items=("lr", "hr"), lr_provenance="bicubic"
            )
            self.assertEqual(set(ignored[0]), {"lr", "hr"})
            paired = SuperResolutionDataset(
                root,
                mode="train",
                scale=2,
                return_items=("lr", "hr", "ref"),
                reference_source="stored",
                lr_provenance="bicubic",
            )
            self.assertEqual(set(paired[0]), {"lr", "hr", "ref"})

    def test_lr_up_reference_works_without_ref_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_hr(root, include_lr=True)
            dataset = SuperResolutionDataset(
                root,
                mode="train",
                scale=2,
                return_items=("lr", "hr", "ref"),
                reference_source="lr_up",
            )
            sample = dataset[0]
            self.assertEqual(sample["ref"].shape, sample["hr"].shape)

    def test_sensor_lr_rejects_non_native_scale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_hr(root, include_lr=True)
            dataset = SuperResolutionDataset(
                root,
                mode="train",
                scale=2,
                lr_native_scale=4,
                lr_provenance="sensor",
                return_items=("lr", "hr"),
            )
            with self.assertRaisesRegex(ValueError, "native scale"):
                _ = dataset[0]

    def test_loader_accepts_hr_only_bicubic_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_hr(root, "train")
            _write_hr(root, "val")
            config = {
                "task": "sr",
                "model": {"name": "swinir"},
                "data": {
                    "root": str(root),
                    "scale": 2,
                    "lr_source": "auto",
                    "lr_provenance": "bicubic",
                    "batch_size": 1,
                    "val_batch_size": 1,
                    "num_workers": 0,
                    "val_num_workers": 0,
                    "pin_memory": False,
                },
                "train": {"seed": 42},
                "loss": {},
                "output": {},
            }
            train_loader, val_loader = build_sr_loaders(config)
            self.assertEqual(set(next(iter(train_loader))), {"lr", "hr"})
            self.assertEqual(set(next(iter(val_loader))), {"lr", "hr"})


if __name__ == "__main__":
    unittest.main()
