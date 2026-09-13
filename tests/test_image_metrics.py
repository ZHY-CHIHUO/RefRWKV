"""Tests for the native [0, 1] image metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.common import per_image_psnr


class PerBandPSNRTests(unittest.TestCase):
    def test_averages_per_band_scores_not_pooled_mse(self) -> None:
        target = torch.zeros(1, 2, 4, 4)
        pred = torch.zeros(1, 2, 4, 4)
        pred[0, 0] = 0.1
        pred[0, 1] = 0.001
        actual = float(per_image_psnr(pred, target))
        mse = (pred - target).square().mean(dim=(-2, -1))
        band = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))
        pooled = 10.0 * torch.log10(1.0 / (pred - target).square().mean().clamp_min(1e-10))
        self.assertAlmostEqual(actual, float(band.mean()), places=5)
        self.assertGreater(abs(actual - float(pooled)), 1.0)

    def test_identity_is_high(self) -> None:
        value = torch.rand(2, 8, 16, 16)
        score = per_image_psnr(value, value)
        self.assertEqual(tuple(score.shape), (2,))
        self.assertTrue(torch.all(score > 80.0))


if __name__ == "__main__":
    unittest.main()
