"""FusionMamba-style periodic checkpoints and pan val metrics."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engines.refsr.trainer import RefSRTrainer
from runtime.callbacks import IntervalEpochCheckpoint, build_checkpoint_callbacks
from runtime.config import load_config, validate_config


class _DummyPan(nn.Module):
    def forward(self, lr, ref):
        return F.interpolate(lr, scale_factor=4, mode="nearest")


def _pan_trainer_config() -> dict:
    return {
        "task": "refsr",
        "model": {"name": "rdm_pan", "use_reference": True},
        "data": {
            "scale": 4,
            "reference_mode": "paired",
            "lr_key": "lr",
            "hr_key": "hr",
            "ref_key": "ref",
        },
        "train": {"learning_rate": 1.0e-4, "use_ema": False},
        "loss": {"name": "l1"},
    }


class CheckpointProtocolTests(unittest.TestCase):
    def test_rdm_pan_yaml_follows_fusion_mamba_ckpt_protocol(self) -> None:
        config = load_config(
            "configs/runs/rdm_refsr/pancollection_wv3_pan_x4.yaml",
            prefer_existing=False,
        )
        validate_config(config, require_data=False)
        self.assertEqual(config["loss"]["name"], "l1")
        self.assertEqual(int(config["train"]["ckpt_every_n_epochs"]), 20)
        self.assertTrue(config["train"]["log_pan_metrics"])
        self.assertEqual(int(config["train"]["check_val_every_n_epoch"]), 10)
        self.assertIsNone(config["train"].get("early_stopping_patience"))

    def test_official_fusion_mamba_yaml_keeps_periodic_ckpts(self) -> None:
        config = load_config(
            "configs/runs/refsr/fusion_mamba_pancollection_wv3_official.yaml",
            prefer_existing=False,
        )
        validate_config(config, require_data=False)
        self.assertEqual(config["loss"]["name"], "l1")
        self.assertEqual(int(config["train"]["ckpt_every_n_epochs"]), 20)
        self.assertTrue(config["train"]["log_pan_metrics"])

    def test_periodic_callbacks_do_not_monitor_val_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            callbacks = build_checkpoint_callbacks(
                {"ckpt_every_n_epochs": 20, "save_top_k": 3},
                tmp,
            )
        interval = [item for item in callbacks if isinstance(item, IntervalEpochCheckpoint)]
        self.assertEqual(len(interval), 1)
        self.assertEqual(interval[0].every_n_epochs, 20)
        monitored = [
            item
            for item in callbacks
            if getattr(item, "monitor", None) == "val/loss"
        ]
        self.assertEqual(monitored, [])
        self.assertTrue(any(getattr(item, "save_last", False) for item in callbacks))

    def test_legacy_top_k_still_watches_val_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            callbacks = build_checkpoint_callbacks({"save_top_k": 3}, tmp)
        self.assertEqual(len(callbacks), 1)
        self.assertEqual(callbacks[0].monitor, "val/loss")
        self.assertEqual(callbacks[0].save_top_k, 3)

    def test_interval_checkpoint_uses_one_based_epoch_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            callback = IntervalEpochCheckpoint(tmp, every_n_epochs=20)
            saved: list[str] = []
            trainer = SimpleNamespace(
                current_epoch=19,
                sanity_checking=False,
                save_checkpoint=lambda path: saved.append(path),
            )
            callback.on_train_epoch_end(trainer, None)
            self.assertEqual(saved, [str(Path(tmp) / "20.ckpt")])
            trainer.current_epoch = 20
            callback.on_train_epoch_end(trainer, None)
            self.assertEqual(saved, [str(Path(tmp) / "20.ckpt")])

    def test_pan_val_metrics_include_paper_indices(self) -> None:
        trainer = RefSRTrainer(_DummyPan(), _pan_trainer_config())
        self.assertTrue(trainer.log_pan_metrics)
        torch.manual_seed(0)
        image = torch.rand(2, 8, 32, 32)
        metrics = trainer.benchmark_image_metrics(image, image)
        self.assertGreater(float(metrics["q2n"]), 0.99)
        self.assertAlmostEqual(float(metrics["sam_deg"]), 0.0, places=4)
        self.assertAlmostEqual(float(metrics["ergas"]), 0.0, places=5)
        self.assertTrue(torch.isfinite(metrics["psnr"]))


if __name__ == "__main__":
    unittest.main()
