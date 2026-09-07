"""CPU regression coverage for the added SR and direct-RefSR baselines."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.refsr import build_model as build_refsr_model  # noqa: E402
from models.refsr import list_models as list_refsr_models  # noqa: E402
from models.sr import build_model as build_sr_model  # noqa: E402
from models.sr import list_models as list_sr_models  # noqa: E402
from runtime.config import load_config, validate_config  # noqa: E402


SR_RUNS = {
    "bicubic": "configs/runs/sr/bicubic/hrms_scd_x4.yaml",
    "edsr": "configs/runs/sr/edsr/hrms_scd_x4.yaml",
    "rcan": "configs/runs/sr/rcan/hrms_scd_x4.yaml",
    "hat": "configs/runs/sr/hat/hrms_scd_x4.yaml",
    "mambairv2": "configs/runs/sr/mambairv2/hrms_scd_x4.yaml",
}
REFSR_RUNS = {
    "ttsr": "configs/runs/refsr/ttsr/hrms_scd_x4.yaml",
    "masa_sr": "configs/runs/refsr/masa_sr/hrms_scd_x4.yaml",
    "datsr": "configs/runs/refsr/datsr/hrms_scd_x4.yaml",
}


def _compact_sr_model(name: str, model: dict) -> dict:
    """Shrink a profile only for CPU geometry coverage."""
    result = copy.deepcopy(model)
    if name == "edsr":
        result.update(dim=8, num_blocks=1)
    elif name == "rcan":
        result.update(dim=8, num_groups=1, blocks_per_group=1, reduction=4)
    elif name == "hat":
        result.update(dim=8, num_blocks=1, reduction=4)
    elif name == "mambairv2":
        result.update(dim=8, num_blocks=1, expansion=1)
    return result


def _compact_refsr_model(name: str, model: dict) -> dict:
    """Shrink a profile only for CPU geometry coverage."""
    result = copy.deepcopy(model)
    result.update(dim=8, match_window=3, match_dim=4)
    if name == "ttsr":
        result.update(num_blocks=1, texture_blocks=1)
    elif name == "masa_sr":
        result.update(num_blocks=1, refine_blocks=1)
    elif name == "datsr":
        result.update(num_blocks=1)
    return result


class BaselineRegistryTests(unittest.TestCase):
    def test_registries_expose_all_added_models(self) -> None:
        self.assertTrue(set(SR_RUNS).issubset(list_sr_models()))
        self.assertTrue(set(REFSR_RUNS).issubset(list_refsr_models()))

    def test_sr_profiles_have_finite_x2_outputs(self) -> None:
        lr = torch.randn(1, 3, 4, 5)
        with torch.inference_mode():
            for name, path in SR_RUNS.items():
                with self.subTest(name=name):
                    config = load_config(path, prefer_existing=False)
                    validate_config(config, require_data=False)
                    model = build_sr_model(_compact_sr_model(name, config["model"]), scale=2).eval()
                    output = model(lr)
                    self.assertEqual(tuple(output.shape), (1, 3, 8, 10))
                    self.assertTrue(torch.isfinite(output).all())
                    if name == "bicubic":
                        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 0)

    def test_refsr_profiles_have_finite_x2_outputs(self) -> None:
        lr = torch.randn(1, 3, 4, 5)
        ref = torch.randn(1, 3, 8, 10)
        with torch.inference_mode():
            for name, path in REFSR_RUNS.items():
                with self.subTest(name=name):
                    config = load_config(path, prefer_existing=False)
                    validate_config(config, require_data=False)
                    self.assertEqual(config["data"]["reference_mode"], "paired")
                    model = build_refsr_model(_compact_refsr_model(name, config["model"]), scale=2).eval()
                    output = model(lr, ref)
                    self.assertEqual(tuple(output.shape), (1, 3, 8, 10))
                    self.assertTrue(torch.isfinite(output).all())

    def test_direct_refsr_profiles_reject_lr_derived_references(self) -> None:
        for name, path in REFSR_RUNS.items():
            with self.subTest(name=name):
                config = load_config(path, prefer_existing=False)
                config["data"]["reference_mode"] = "lr_up"
                with self.assertRaisesRegex(ValueError, "paired"):
                    validate_config(config, require_data=False)


if __name__ == "__main__":
    unittest.main()
