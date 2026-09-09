"""CPU coverage for the RefSRWKV paper-ablation controls."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.refsr.refsrwkv.model import (  # noqa: E402
    GatedFusion,
    RefSRWKV,
    normalize_fusion_match_config,
)
import models.refsr.refsrwkv.model as refsrwkv_module  # noqa: E402
from runtime.config import load_config, validate_config  # noqa: E402


class FusionMatchConfigTests(unittest.TestCase):
    def test_yaml_defaults_and_overrides_are_materialized(self) -> None:
        config = load_config(
            "configs/runs/refsrwkv/aid_x4.yaml",
            overrides=[
                "model.fusion_match.enabled=false",
                "model.fusion_match.window=3",
                "model.fusion_match.conf=false",
                "model.fusion_match.quality=false",
                "model.decoder_refusion=false",
                "model.global_latent_blocks=0",
                "model.ref_encoder=shallow",
            ],
        )
        model = config["model"]
        self.assertFalse(model["fusion_match"]["enabled"])
        self.assertEqual(model["fusion_match"]["window"], 3)
        self.assertFalse(model["fusion_match"]["conf"])
        self.assertFalse(model["fusion_match"]["quality"])
        self.assertFalse(model["decoder_refusion"])
        self.assertEqual(model["global_latent_blocks"], 0)
        self.assertEqual(model["ref_encoder"], "shallow")

    def test_defaults_and_compact_window_forms(self) -> None:
        default = normalize_fusion_match_config()
        self.assertTrue(default["enabled"])
        self.assertTrue(default["conf"])
        self.assertTrue(default["quality"])
        self.assertEqual(default["window"]["enc1"], 7)
        self.assertEqual(default["window"]["latent"], 3)

        disabled = normalize_fusion_match_config(False)
        self.assertFalse(disabled["enabled"])
        self.assertEqual(disabled["window"], default["window"])

        uniform = normalize_fusion_match_config({"window": 3})
        self.assertEqual(set(uniform["window"].values()), {3})

        staged = normalize_fusion_match_config({"window": {"enc1": 9}, "conf": False})
        self.assertEqual(staged["window"]["enc1"], 9)
        self.assertEqual(staged["window"]["enc2"], default["window"]["enc2"])
        self.assertFalse(staged["conf"])

    def test_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            normalize_fusion_match_config({"enabled": 1})
        with self.assertRaises(ValueError):
            normalize_fusion_match_config({"window": 4})
        with self.assertRaises(ValueError):
            normalize_fusion_match_config({"window": [3, 3]})
        with self.assertRaises(ValueError):
            normalize_fusion_match_config({"window": {"unknown": 3}})


class GatedFusionAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.lr = torch.randn(2, 4, 5, 6)
        self.ref = torch.randn(2, 4, 5, 6)

    def test_disabled_match_uses_positional_cosine_path(self) -> None:
        fusion = GatedFusion(4, window_size=7, match_enabled=False)
        actual = fusion(self.lr, self.ref)
        direct = fusion.norm(fusion.fuse_conv(torch.cat([self.lr, self.ref], dim=1)))
        confidence = (torch.nn.functional.cosine_similarity(self.lr, self.ref, dim=1) + 1.0) / 2.0
        expected = self.lr + fusion.gate(direct) * confidence.unsqueeze(1) * direct
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))
        self.assertFalse(hasattr(fusion, "query"))

    def test_confidence_and_quality_switches_are_independent(self) -> None:
        no_conf = GatedFusion(4, window_size=3, conf_enabled=False)
        no_quality = GatedFusion(4, window_size=3, quality_enabled=False)
        self.assertFalse(no_conf.conf_enabled)
        self.assertTrue(no_conf.quality_enabled)
        self.assertTrue(no_quality.conf_enabled)
        self.assertFalse(no_quality.quality_enabled)
        self.assertIsInstance(no_quality.quality, nn.Identity)
        for fusion in (no_conf, no_quality):
            output = fusion(self.lr, self.ref)
            self.assertEqual(output.shape, self.lr.shape)
            self.assertTrue(torch.isfinite(output).all())


class RefSRWKVStructureTests(unittest.TestCase):
    @staticmethod
    def build(**kwargs) -> RefSRWKV:
        return RefSRWKV(
            dim=16,
            num_blocks=(1, 1, 1, 1),
            num_refinement_blocks=0,
            scale=2,
            **kwargs,
        )

    def test_global_latent_block_count(self) -> None:
        for count in (0, 1, 2):
            model = self.build(global_latent_blocks=count)
            self.assertEqual(len(model.global_latent), count)
            self.assertEqual(model.global_latent_blocks, count)

    def test_decoder_refusion_can_be_removed(self) -> None:
        model = self.build(decoder_refusion=False)
        self.assertFalse(model.decoder_refusion)
        self.assertIsInstance(model.decoder_fuse1, nn.Identity)
        self.assertIsInstance(model.decoder_fuse2, nn.Identity)
        self.assertIsInstance(model.decoder_fuse3, nn.Identity)

    def test_reference_encoder_depth(self) -> None:
        shallow = self.build(ref_encoder="shallow")
        deep = self.build(ref_encoder="deep")
        shallow_convs = [m for m in shallow.ref_to_level1 if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3)]
        deep_convs = [m for m in deep.ref_to_level1 if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3)]
        self.assertEqual(len(shallow_convs), 1)
        self.assertEqual(len(deep_convs), 2)
        self.assertEqual(shallow.ref_encoder, "shallow")
        self.assertEqual(deep.ref_encoder, "deep")

    def test_fusion_options_reach_all_fusion_sites(self) -> None:
        model = self.build(
            fusion_match={"enabled": False, "window": 3, "conf": False, "quality": False}
        )
        fusions = (
            model.fuse1,
            model.fuse2,
            model.fuse3,
            model.fuse4,
            model.decoder_fuse1,
            model.decoder_fuse2,
            model.decoder_fuse3,
        )
        self.assertTrue(all(not fusion.match_enabled for fusion in fusions))
        self.assertTrue(all(not fusion.conf_enabled for fusion in fusions))
        self.assertTrue(all(not fusion.quality_enabled for fusion in fusions))
        self.assertEqual({fusion.window_size for fusion in fusions}, {3})

    def test_unequal_reference_channels_keep_lr_output_channels(self) -> None:
        model = self.build(inp_channels=4, ref_channels=1, out_channels=4)
        self.assertEqual(model.inp_channels, 4)
        self.assertEqual(model.ref_channels, 1)
        self.assertEqual(model.out_channels, 4)
        self.assertEqual(model.lr_up[0].in_channels, 4)
        self.assertEqual(model.ref_to_level1[0].in_channels, 1)
        self.assertEqual(model.output_conv.out_channels, 4)

        ref = torch.randn(2, 1, 8, 10)
        target = torch.randn(2, 4, 8, 10)
        matched = model._match_color(ref, target)
        self.assertEqual(matched.shape, ref.shape)
        self.assertTrue(torch.isfinite(matched).all())

    def test_channel_contract_rejects_invalid_output_or_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, "ref_channels"):
            self.build(inp_channels=3, ref_channels=4, out_channels=3)
        with self.assertRaisesRegex(ValueError, "out_channels"):
            self.build(inp_channels=4, ref_channels=1, out_channels=3)

    def test_same_channel_color_matching_remains_per_band(self) -> None:
        model = self.build(inp_channels=3, ref_channels=3, out_channels=3)
        ref = torch.randn(2, 3, 8, 10)
        target = torch.randn(2, 3, 8, 10)
        matched = model._match_color(ref, target)
        self.assertTrue(torch.allclose(matched.mean(dim=(2, 3)), target.mean(dim=(2, 3)), atol=1e-5))

    def test_forward_crops_padded_non_multiple_lr_geometry(self) -> None:
        model = self.build(inp_channels=4, ref_channels=1, out_channels=4)
        # The production WKV operator is CUDA-only. Returning its value keeps
        # this regression test focused on tensor geometry and channel flow.
        with patch.object(refsrwkv_module, "RUN_CUDA", side_effect=lambda w, u, k, v: v):
            with torch.no_grad():
                output = model(
                    torch.randn(1, 4, 5, 6),
                    torch.randn(1, 1, 10, 12),
                )
        self.assertEqual(output.shape, (1, 4, 10, 12))
        self.assertTrue(torch.isfinite(output).all())


class RefSRWKVConfigChannelTests(unittest.TestCase):
    @staticmethod
    def _config() -> dict:
        return {
            "model": {
                "name": "RefSRWKV",
                "inp_channels": 4,
                "out_channels": 4,
                "ref_channels": 1,
            },
            "data": {"root": "/tmp/data", "scale": 2},
            "train": {},
            "loss": {},
            "output": {},
        }

    def test_accepts_lr_channels_greater_than_reference_channels(self) -> None:
        validate_config(self._config(), require_data=False)

    def test_allows_derived_channel_fields_to_be_null(self) -> None:
        config = self._config()
        config["model"]["out_channels"] = None
        config["model"]["ref_channels"] = None
        validate_config(config, require_data=False)

    def test_rejects_invalid_channel_order(self) -> None:
        config = self._config()
        config["model"]["ref_channels"] = 5
        with self.assertRaisesRegex(ValueError, "ref_channels"):
            validate_config(config, require_data=False)
        config = self._config()
        config["model"]["out_channels"] = 3
        with self.assertRaisesRegex(ValueError, "out_channels"):
            validate_config(config, require_data=False)


if __name__ == "__main__":
    unittest.main()
