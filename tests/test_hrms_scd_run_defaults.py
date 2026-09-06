"""Regression coverage for comparable HRMS-SCD training defaults."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.config import load_config, validate_config  # noqa: E402


RUN_CONFIGS = (
    "configs/runs/sr/swinir/hrms_scd_x4.yaml",
    "configs/runs/refsrwkv/hrms_scd_sr_x4.yaml",
    "configs/runs/refsrwkv/hrms_scd_trefsr_x4.yaml",
)


class HRMSSCDRunDefaultsTests(unittest.TestCase):
    def test_comparable_training_defaults(self) -> None:
        expected_train = {
            "learning_rate": 1.0e-4,
            "lr_scheduler": "plateau",
            "val_check_interval": 1.0,
            "check_val_every_n_epoch": 1,
        }
        for path in RUN_CONFIGS:
            with self.subTest(path=path):
                config = load_config(path, prefer_existing=False)
                validate_config(config)
                self.assertEqual(
                    {key: config["train"][key] for key in expected_train},
                    expected_train,
                )
                self.assertEqual(config["loss"]["name"], "l1")
                if config["task"] == "refsr":
                    self.assertEqual(config["loss"].get("ssim_weight"), 0.0)
                    self.assertEqual(config["loss"].get("fft_weight"), 0.0)


if __name__ == "__main__":
    unittest.main()
