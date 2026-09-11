"""Tests for the PanCollection reduced-resolution H5 contract."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency in minimal envs
    h5py = None

from data.loaders import build_pancollection_loaders
from data.refsr.pancollection import PanCollectionH5Dataset


@unittest.skipIf(h5py is None, "h5py is required for PanCollection tests")
class PanCollectionDatasetTests(unittest.TestCase):
    @staticmethod
    def _write_file(path: Path, count: int = 3) -> None:
        with h5py.File(path, "w") as handle:
            handle.create_dataset("gt", data=np.full((count, 8, 8, 8), 2047, dtype=np.float32))
            handle.create_dataset("ms", data=np.full((count, 8, 2, 2), 1023.5, dtype=np.float32))
            handle.create_dataset("pan", data=np.full((count, 1, 8, 8), 512, dtype=np.float32))
            # Upstream's lms baseline is intentionally ignored by the loader.
            handle.create_dataset("lms", data=np.zeros((count, 8, 8, 8), dtype=np.float32))

    @staticmethod
    def _config(root: Path) -> dict:
        return {
            "task": "refsr",
            "dataset": {
                "id": "pancollection_wv3",
                "kind": "pancollection_h5",
                "format": "pancollection_h5",
                "root": str(root),
                "files": {"train": "train.h5", "val": "valid.h5"},
                "h5_keys": {"lr": "ms", "ref": "pan", "hr": "gt"},
            },
            "model": {
                "name": "RefSRWKV",
                "inp_channels": 8,
                "ref_channels": 1,
                "out_channels": 8,
            },
            "data": {
                "root": str(root),
                "scale": 4,
                "reference_mode": "paired",
                "lr_source": "stored",
                "lr_provenance": "sensor",
                "lr_native_scale": 4,
                "patch_size": 8,
                "batch_size": 1,
                "val_batch_size": 1,
                "num_workers": 0,
                "val_num_workers": 0,
                "pin_memory": False,
                "value_scale": 2047.0,
            },
            "train": {"seed": 42},
            "loss": {},
            "output": {},
        }

    def test_maps_ms_pan_gt_and_normalizes_to_minus_one_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_file(root / "train.h5", count=1)
            dataset = PanCollectionH5Dataset(
                root / "train.h5",
                mode="train",
                scale=4,
                patch_size=None,
                value_scale=2047.0,
                expected_lr_channels=8,
                expected_ref_channels=1,
                expected_hr_channels=8,
            )
            sample = dataset[0]
            self.assertEqual(sample["lr"].shape, (8, 2, 2))
            self.assertEqual(sample["ref"].shape, (1, 8, 8))
            self.assertEqual(sample["hr"].shape, (8, 8, 8))
            self.assertAlmostEqual(float(sample["hr"].max()), 1.0)
            self.assertTrue(float(sample["lr"].min()) > -0.01)

    def test_loader_builds_train_and_validation_from_h5_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_file(root / "train.h5", count=3)
            self._write_file(root / "valid.h5", count=2)
            train_loader, val_loader = build_pancollection_loaders(self._config(root))
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))
            self.assertEqual(set(train_batch), {"lr", "ref", "hr"})
            self.assertEqual(tuple(train_batch["lr"].shape), (1, 8, 2, 2))
            self.assertEqual(tuple(train_batch["ref"].shape), (1, 1, 8, 8))
            self.assertEqual(tuple(val_batch["hr"].shape), (1, 8, 8, 8))

    def test_rejects_non_x4_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("gt", shape=(1, 8, 7, 8), dtype="f4")
                handle.create_dataset("ms", shape=(1, 8, 2, 2), dtype="f4")
                handle.create_dataset("pan", shape=(1, 1, 7, 8), dtype="f4")
            with self.assertRaisesRegex(ValueError, "geometry mismatch"):
                PanCollectionH5Dataset(path, scale=4, patch_size=None)


if __name__ == "__main__":
    unittest.main()
