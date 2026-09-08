"""Coverage for bounded-memory aligned tiled inference."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.tiling import tiled_forward


class TiledForwardTests(unittest.TestCase):
    def test_aligned_two_input_output_matches_full_pointwise_model(self) -> None:
        lr = torch.randn(1, 4, 23, 29)
        ref = torch.randn(1, 4, 23, 29)

        def model(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return left * 0.7 + right * 0.3

        expected = model(lr, ref)
        actual = tiled_forward(model, lr, ref, scale=1, tile_size=8, overlap=2)
        self.assertTrue(torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6))

    def test_integer_scale_output_is_stitched_at_output_grid(self) -> None:
        value = torch.randn(1, 2, 17, 19)

        def model(x: torch.Tensor) -> torch.Tensor:
            return F.interpolate(x, scale_factor=2, mode="nearest")

        expected = model(value)
        actual = tiled_forward(model, value, scale=2, tile_size=7, overlap=2)
        self.assertEqual(actual.shape, (1, 2, 34, 38))
        self.assertTrue(torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6))

    def test_rejects_misaligned_ref(self) -> None:
        with self.assertRaisesRegex(ValueError, "share batch and spatial geometry"):
            tiled_forward(
                lambda lr, ref: lr,
                torch.zeros(1, 4, 16, 16),
                torch.zeros(1, 4, 32, 32),
                tile_size=8,
            )


if __name__ == "__main__":
    unittest.main()
