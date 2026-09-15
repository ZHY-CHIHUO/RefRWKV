"""4-band STFMamba adapter coverage on the Wuhan STF contract."""

from __future__ import annotations

import unittest

import torch

from models.refsr import list_models
from models.refsr.stf_mamba.cross_attention import Cross_MultiAttention
from runtime.config import load_config, validate_config


class STFMambaAdapterTests(unittest.TestCase):
    def test_registry_and_wuhan_config(self) -> None:
        self.assertIn("stf_mamba", list_models())
        self.assertIn("rdm_stf", list_models())
        for path in (
            "configs/runs/refsr/stf_mamba_wuhan.yaml",
            "configs/runs/rdm_refsr/wuhan_x1.yaml",
        ):
            config = load_config(path, prefer_existing=False)
            validate_config(config, require_data=False)
            self.assertEqual(int(config["data"]["scale"]), 1)
            self.assertTrue(config["data"]["return_quadruple"])
            self.assertTrue(config["data"]["use_c0"])
            self.assertEqual(config["data"]["c0_key"], "lr_t1")
            self.assertEqual(int(config["model"]["inp_channels"]), 4)
            self.assertEqual(int(config["model"].get("in_chans", 4)), 4)
            self.assertAlmostEqual(float(config["data"]["value_scale"]), 11848.0)
            self.assertEqual(int(config["train"]["max_epochs"]), 200)
            self.assertAlmostEqual(float(config["train"]["learning_rate"]), 5e-4)
            self.assertEqual(config["train"]["optimizer"], "adam")
            self.assertEqual(config["train"]["lr_scheduler"], "cosine")
            self.assertEqual(config["train"]["lr_interval"], "epoch")
            self.assertEqual(int(config["train"]["lr_t_max"]), 200)
            self.assertAlmostEqual(float(config["train"]["lr_min"]), 1e-5)
            self.assertEqual(int(config["train"]["seed"]), 2021)
            self.assertEqual(config["loss"]["name"], "charbonnier")
            self.assertAlmostEqual(float(config["loss"]["eps"]), 1e-3)
            self.assertAlmostEqual(float(config["loss"]["ssim_weight"]), 1.0)
            self.assertEqual(int(config["data"]["num_patches"]), 12000)
            self.assertIsNone(config["data"].get("val_num_patches"))
            self.assertIsNone(config["data"].get("test_num_patches"))
            if config["model"]["name"] == "rdm_stf":
                self.assertEqual(int(config["data"]["batch_size"]), 10)
            else:
                self.assertEqual(int(config["data"]["batch_size"]), 2)

    def test_cross_attention_accepts_four_bands_and_non_128(self) -> None:
        module = Cross_MultiAttention(in_channels=4, emb_dim=16, num_heads=4, block_size=16)
        for size in (32, 64, 128):
            left = torch.randn(2, 4, size, size)
            right = torch.randn(2, 4, size, size)
            output = module(left, right)
            self.assertEqual(tuple(output.shape), (2, 4, size, size))
            self.assertTrue(torch.isfinite(output).all())

    def test_wuhan_trainer_requires_and_uses_c0(self) -> None:
        import torch.nn as nn

        from engines.refsr.trainer import RefSRTrainer

        config = load_config("configs/runs/rdm_refsr/wuhan_x1.yaml", prefer_existing=False)
        validate_config(config, require_data=False)

        class DummySTF(nn.Module):
            def forward(self, lr, ref, c0=None, return_aux=False):
                output = lr if c0 is None else lr + 0.01 * c0
                if return_aux:
                    return output, {}
                return output

        trainer = RefSRTrainer(DummySTF(), config)
        self.assertTrue(trainer.use_c0)
        self.assertTrue(trainer.require_c0)
        batch = {
            "lr": torch.rand(1, 4, 8, 8),
            "hr": torch.rand(1, 4, 8, 8),
            "ref": torch.rand(1, 4, 8, 8),
            "lr_t1": torch.zeros(1, 4, 8, 8),
        }
        c0 = trainer._c0(batch)
        self.assertIsNotNone(c0)
        prediction = trainer._forward_model(batch["lr"], batch["ref"], c0)
        self.assertEqual(tuple(prediction.shape), (1, 4, 8, 8))
        with self.assertRaises(KeyError):
            trainer._c0({"lr": batch["lr"], "hr": batch["hr"], "ref": batch["ref"]})

    def test_wuhan_optimizer_matches_official_stfmamba(self) -> None:
        import torch.nn as nn

        from engines.refsr.trainer import RefSRTrainer

        config = load_config("configs/runs/rdm_refsr/wuhan_x1.yaml", prefer_existing=False)
        validate_config(config, require_data=False)

        class DummySTF(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Conv2d(4, 4, 1)

            def forward(self, lr, ref, c0=None, return_aux=False):
                output = self.proj(lr)
                if return_aux:
                    return output, {}
                return output

        module = RefSRTrainer(DummySTF(), config)
        optimizers, schedulers = module.configure_optimizers()
        optimizer = optimizers[0]
        scheduler_cfg = schedulers[0]
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertNotIsInstance(optimizer, torch.optim.AdamW)
        self.assertAlmostEqual(optimizer.defaults["lr"], 5e-4)
        self.assertEqual(scheduler_cfg["interval"], "epoch")
        self.assertEqual(int(scheduler_cfg["scheduler"].T_max), 200)
        self.assertAlmostEqual(float(scheduler_cfg["scheduler"].eta_min), 1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "STFMamba VSS scan needs CUDA")
    def test_four_band_quadruple_forward(self) -> None:
        from models.refsr.stf_mamba.adapter import STFMambaRefSR

        model = STFMambaRefSR(in_chans=4).cuda().eval()
        c1 = torch.rand(1, 4, 64, 64, device="cuda")
        f0 = torch.rand(1, 4, 64, 64, device="cuda")
        c0 = torch.rand(1, 4, 64, 64, device="cuda")
        with torch.no_grad():
            output, aux = model(c1, f0, c0, return_aux=True)
        self.assertEqual(tuple(output.shape), (1, 4, 64, 64))
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)
        for key in ("sr_c0", "sr_c1", "from_f0", "from_c1"):
            self.assertEqual(tuple(aux[key].shape), (1, 4, 64, 64))
            self.assertTrue(torch.isfinite(aux[key]).all())


if __name__ == "__main__":
    unittest.main()
