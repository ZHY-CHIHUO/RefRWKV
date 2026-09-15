"""Sanity checks for the official pansharpening indices."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.runner import normalize_test_metrics, resolve_eval_tiles
from metrics.pansharpening import d_lambda, d_s, matlab_imresize, q2n, qnr


class PansharpeningMetricTests(unittest.TestCase):
    def test_identity_q2n_is_one(self) -> None:
        value = torch.rand(1, 8, 32, 32)
        score = float(q2n(value, value))
        self.assertGreater(score, 0.99)
        self.assertLessEqual(score, 1.0 + 1.0e-6)

    def test_identity_d_lambda_is_zero(self) -> None:
        value = torch.rand(1, 8, 32, 32)
        score = float(d_lambda(value, value, block_size=16))
        self.assertAlmostEqual(score, 0.0, places=6)

    def test_qnr_components_are_finite_and_d_lambda_vanishes_on_identity(self) -> None:
        ms = torch.rand(1, 4, 32, 32)
        pan = ms.mean(dim=1, keepdim=True)
        scores = qnr(ms, ms, pan, ratio=2, block_size=16)
        self.assertAlmostEqual(float(scores["d_lambda"]), 0.0, places=6)
        self.assertTrue(torch.isfinite(scores["d_s"]).all())
        self.assertTrue(torch.isfinite(scores["qnr"]).all())
        self.assertGreaterEqual(float(scores["qnr"]), 0.0)
        self.assertLessEqual(float(scores["qnr"]), 1.0)

    def test_metric_aliases_include_official_names(self) -> None:
        self.assertEqual(
            normalize_test_metrics(["psnr", "q2n", "sam", "ergas", "d_lambda", "d_s", "qnr"]),
            ["psnr", "q2n", "sam_rad", "sam_deg", "ergas", "d_lambda", "d_s", "qnr"],
        )

    def test_matlab_imresize_preserves_dc_and_output_size(self) -> None:
        ones = matlab_imresize((torch.ones(32, 32) * 0.4).numpy(), 0.25)
        self.assertEqual(ones.shape, (8, 8))
        self.assertTrue(abs(float(ones.mean()) - 0.4) < 1.0e-6)
        checker = ((torch.arange(64).reshape(-1, 1) + torch.arange(64)) % 2).float().numpy()
        resized = matlab_imresize(checker, 0.25)
        self.assertEqual(resized.shape, (16, 16))
        self.assertGreater(float(resized.std()), 0.0)

    def test_ds_uses_matlab_imresize_not_plain_torch_bicubic(self) -> None:
        torch.manual_seed(0)
        ms = torch.rand(1, 4, 64, 64)
        pan = ms.mean(dim=1, keepdim=True)
        matlab = float(d_s(ms, ms, pan, ratio=4, block_size=16))
        pan_lr = torch.nn.functional.interpolate(
            pan, scale_factor=0.25, mode="bicubic", align_corners=False
        )
        from metrics.pansharpening import interp23tap, _uqi_mean
        import numpy as np

        pan_hw = pan[0, 0].numpy().astype(np.float64)
        ms_hwc = ms[0].permute(1, 2, 0).numpy().astype(np.float64)
        pan_filt = interp23tap(pan_lr[0, 0].numpy().astype(np.float64), 4)[..., 0]
        total = 0.0
        for band in range(ms_hwc.shape[-1]):
            total += abs(
                _uqi_mean(ms_hwc[..., band], pan_hw, 16)
                - _uqi_mean(ms_hwc[..., band], pan_filt, 16)
            )
        torch_ds = total / ms_hwc.shape[-1]
        self.assertGreater(abs(matlab - torch_ds), 1.0e-3)
        self.assertTrue(np.isfinite(matlab))

    def test_both_pan_models_tile_at_training_crop(self) -> None:
        for name in ("rdm_pan", "fusion_mamba"):
            tile, overlap = resolve_eval_tiles(
                name,
                {"patch_size": 64, "eval_tile_size": None, "eval_tile_overlap": 0},
                scale=4,
                wuhan_run=False,
            )
            self.assertEqual((tile, overlap), (16, 8), msg=name)


if __name__ == "__main__":
    unittest.main()
