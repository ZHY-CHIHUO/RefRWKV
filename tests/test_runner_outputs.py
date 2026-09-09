from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluation.runner import _save_predictions


def test_save_predictions_writes_png_for_grayscale_and_rgb(tmp_path: Path) -> None:
    grayscale = torch.tensor(
        [[[[0.0, 0.5], [1.0, 0.25]]]], dtype=torch.float32
    )
    rgb = torch.tensor(
        [
            [
                [[0.0, 0.5], [1.0, 0.25]],
                [[1.0, 0.5], [0.0, 0.75]],
                [[0.25, 0.75], [0.5, 1.0]],
            ]
        ],
        dtype=torch.float32,
    )
    _save_predictions(grayscale, tmp_path, ["gray"])
    _save_predictions(rgb, tmp_path, ["rgb"])

    assert Image.open(tmp_path / "gray.png").mode == "L"
    assert Image.open(tmp_path / "rgb.png").mode == "RGB"


def test_save_predictions_keeps_multiband_float_values_in_tiff(tmp_path: Path) -> None:
    values = torch.tensor(
        [[[[0.1, 0.2]], [[0.3, 0.4]], [[0.5, 0.6]], [[0.7, 0.8]]]],
        dtype=torch.float32,
    )
    _save_predictions(values, tmp_path, ["multiband"])

    import tifffile

    saved = tifffile.imread(tmp_path / "multiband.tif")
    np.testing.assert_allclose(saved, values[0].permute(1, 2, 0).numpy(), rtol=0, atol=1e-7)
