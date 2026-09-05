"""Regression coverage for the two explicit RefSR reference contracts."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loaders import build_refsr_loaders
from runtime.config import load_config, validate_config


PAIRED_ONLY_DATA_KEYS = {
    "augment_ref",
    "ref_aug_strengths",
    "ref_aug_probs",
    "ref_gray_prob",
}


def _write_png(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(64, 128, 192)).save(path)


def _write_split(root: Path, split: str, *, include_ref: bool) -> None:
    _write_png(root / split / "LR" / "sample.png", (2, 2))
    _write_png(root / split / "HR" / "sample.png", (4, 4))
    if include_ref:
        _write_png(root / split / "Ref" / "sample.png", (4, 4))


def _loader_config(root: Path, *, mode: str, model_name: str = "RefSRWKV") -> dict:
    return {
        "task": "refsr",
        "model": {"name": model_name},
        "data": {
            "root": str(root),
            "scale": 2,
            "reference_mode": mode,
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


class ReferenceContractTests(unittest.TestCase):
    def test_lr_up_run_has_no_paired_only_settings(self) -> None:
        config = load_config("configs/runs/refsrwkv/aid_x4.yaml")
        self.assertEqual(config["data"]["reference_mode"], "lr_up")
        self.assertFalse(PAIRED_ONLY_DATA_KEYS & config["data"].keys())
        self.assertNotIn("ref_drop_prob", config["loss"])
        validate_config(config)

    def test_paired_run_inherits_paired_profile(self) -> None:
        config = load_config("configs/runs/refsrwkv/hrms_scd_x4.yaml")
        self.assertEqual(config["data"]["reference_mode"], "paired")
        self.assertTrue(PAIRED_ONLY_DATA_KEYS <= config["data"].keys())
        self.assertIn("ref_drop_prob", config["loss"])
        validate_config(config)

    def test_lr_up_rejects_real_reference_settings(self) -> None:
        config = _loader_config(Path("/tmp/not-used"), mode="lr_up")
        config["data"]["augment_ref"] = True
        with self.assertRaisesRegex(ValueError, "augment_ref"):
            validate_config(config)

    def test_refdiff_rejects_lr_up(self) -> None:
        config = _loader_config(Path("/tmp/not-used"), mode="lr_up", model_name="RefDiffRWKV")
        with self.assertRaisesRegex(ValueError, "RefDiffRWKV"):
            validate_config(config)

    def test_lr_up_loader_uses_hr_lr_pairs_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_split(root, "train", include_ref=False)
            _write_split(root, "val", include_ref=False)
            train_loader, val_loader = build_refsr_loaders(_loader_config(root, mode="lr_up"))
            self.assertEqual(set(next(iter(train_loader))), {"lr", "hr"})
            self.assertEqual(set(next(iter(val_loader))), {"lr", "hr"})

    def test_paired_loader_requires_and_returns_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_split(root, "train", include_ref=True)
            _write_split(root, "val", include_ref=True)
            train_loader, val_loader = build_refsr_loaders(_loader_config(root, mode="paired"))
            self.assertEqual(set(next(iter(train_loader))), {"lr", "hr", "ref"})
            self.assertEqual(set(next(iter(val_loader))), {"lr", "hr", "ref"})


if __name__ == "__main__":
    unittest.main()
