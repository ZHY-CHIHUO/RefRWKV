"""Regression tests for the aligned four-band Wuhan loader and metrics."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

try:
    import tifffile
except ImportError:  # pragma: no cover - optional dependency in minimal envs
    tifffile = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.refsr.wuhan import WuhanSTFDataset
from metrics.wuhan import wuhan_metric_tensors
from runtime.checkpoint import load_model_weights


@unittest.skipIf(tifffile is None, "tifffile is required for Wuhan tests")
class WuhanDatasetTests(unittest.TestCase):
    def _make_root(self, root: Path) -> None:
        for split in ("train", "val", "test"):
            pair = root / split / "20200101_20210101"
            pair.mkdir(parents=True)
            for kind, date, offset in (
                ("L", "20200101", 1),
                ("L", "20210101", 2),
                ("G", "20200101", 3),
                ("G", "20210101", 4),
            ):
                # C,H,W is the orientation returned by rasterio; the loader
                # must normalize it to H,W,C without dropping a band.
                value = np.arange(4 * 16 * 16, dtype=np.uint16).reshape(4, 16, 16) + offset
                tifffile.imwrite(pair / f"{kind}_{date}.tif", value)

    def test_four_band_aliases_cache_and_crop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_root(root)
            dataset = WuhanSTFDataset(
                root,
                mode="train",
                patch_size=8,
                augment=False,
                return_quadruple=True,
            )
            sample = dataset[0]
            self.assertEqual(sample["lr"].shape, (4, 8, 8))
            self.assertEqual(sample["ref"].shape, (4, 8, 8))
            self.assertEqual(sample["hr"].shape, (4, 8, 8))
            self.assertEqual(len(dataset.path_cache), 4)
            # All temporal tensors have the same crop geometry.
            self.assertEqual({tuple(sample[key].shape[-2:]) for key in ("lr_t1", "lr_t2", "hr_t1", "hr_t2")}, {(8, 8)})

    def test_identity_metrics(self) -> None:
        value = torch.rand(2, 4, 8, 8)
        result = wuhan_metric_tensors(value, value, value_range="zero_one")
        self.assertTrue(torch.allclose(result["rmse"], torch.zeros(2)))
        self.assertTrue(torch.allclose(result["uiqi"], torch.ones(2)))
        self.assertTrue(torch.allclose(result["sam_rad"], torch.zeros(2)))
        self.assertTrue(torch.allclose(result["ergas"], torch.zeros(2)))

    def test_rgb_to_four_band_transfer_keeps_reconstruction_head_new(self) -> None:
        class Boundary(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lr_up = torch.nn.Sequential(torch.nn.Conv2d(4, 8, 3, bias=False))
                self.ref_to_level1 = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 3, bias=False))
                self.output_conv = torch.nn.Conv2d(8, 4, 3, bias=False)

        model = Boundary()
        old_head = model.output_conv.weight.detach().clone()
        report = load_model_weights(
            model,
            {
                "lr_up.0.weight": torch.randn(8, 3, 3, 3),
                "ref_to_level1.0.weight": torch.randn(3, 3, 3, 3),
                "output_conv.weight": torch.randn(3, 8, 3, 3),
            },
            channel_adaptation="rgb_mean",
        )
        self.assertEqual(report["raw_channel_adapted"], 2)
        self.assertEqual(report["raw_shape_mismatch"], 1)
        self.assertTrue(torch.equal(model.output_conv.weight, old_head))


if __name__ == "__main__":
    unittest.main()
