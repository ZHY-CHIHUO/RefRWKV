"""Regression tests for the pure RWKV+Mamba RDMRefSR architecture."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engines.refsr import RDMRefSRTrainer  # noqa: E402
from models.refsr import build_model, list_models  # noqa: E402
from models.refsr.rdm_refsr.rdm_pan import RDMPan  # noqa: E402
from models.refsr.rdm_refsr.rdm_refsr import (  # noqa: E402
    RDMMhf,
    RDMRefSR,
    SharedDirectionalRWKV,
    TrueMambaScan,
    _reference_biwkv,
    haar_dwt2d,
    haar_idwt2d,
    normalize_reference_kind,
)
from models.refsr.rdm_refsr.rdm_stf import RDMStf  # noqa: E402
from runtime.config import load_config, validate_config  # noqa: E402
from runtime.tiling import tiled_forward  # noqa: E402


def _compact_model(**updates) -> RDMRefSR:
    options = dict(
        inp_channels=3,
        ref_channels=3,
        out_channels=3,
        target_channels=3,
        dim=16,
        depths=(1, 1, 1, 1),
        decoder_depths=(1, 1, 1),
        scale=2,
        reference_kind="temporal",
        mamba_d_state=2,
        mamba_d_conv=2,
        mamba_expand=1,
        match_window=3,
        match_dim=4,
        shuffle_prob=0.0,
    )
    options.update(updates)
    return RDMRefSR(**options)


def _pan_model(**updates) -> RDMPan:
    options = dict(
        inp_channels=8,
        ref_channels=1,
        out_channels=8,
        dim=16,
        scale=2,
        mamba_d_state=2,
        mamba_d_conv=2,
        mamba_expand=1,
        allow_cpu_mamba=True,
    )
    options.update(updates)
    return RDMPan(**options)


class RDMRefSRTests(unittest.TestCase):
    def test_haar_round_trip_odd_geometry(self) -> None:
        value = torch.randn(2, 5, 7, 9)
        low, detail, size = haar_dwt2d(value)
        restored = haar_idwt2d(low, detail, size)
        self.assertEqual(tuple(restored.shape), tuple(value.shape))
        self.assertLess(float((restored - value).abs().max()), 2.0e-6)

    def test_three_reference_modes_and_independent_channels(self) -> None:
        cases = (
            ("stf", 3, 3, 3),
            ("temporal", 3, 3, 3),
            ("mhf", 31, 4, 31),
            ("hsi_msi", 31, 4, 31),
        )
        for kind, inp, ref_channels, out_channels in cases:
            with self.subTest(kind=kind):
                model = _compact_model(
                    inp_channels=inp,
                    ref_channels=ref_channels,
                    out_channels=out_channels,
                    target_channels=out_channels,
                    reference_kind=kind,
                )
                lr = torch.randn(1, inp, 3, 5, requires_grad=True)
                ref = torch.randn(1, ref_channels, 6, 10)
                output = model(lr, ref)
                self.assertEqual(tuple(output.shape), (1, out_channels, 6, 10))
                self.assertTrue(torch.isfinite(output).all())

        pan = _pan_model()
        lr = torch.rand(1, 8, 4, 5, requires_grad=True)
        ref = torch.rand(1, 1, 8, 10)
        output = pan(lr, ref)
        self.assertEqual(tuple(output.shape), (1, 8, 8, 10))
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(hasattr(pan, "reliability"))
        self.assertFalse(hasattr(pan, "detail_matcher"))

    def test_cpu_backward_reaches_query_and_mamba_fallback(self) -> None:
        model = _compact_model()
        lr = torch.randn(1, 3, 3, 4, requires_grad=True)
        ref = torch.randn(1, 3, 6, 8)
        loss = model(lr, ref).square().mean()
        loss.backward()
        self.assertIsNotNone(model.ms_stem[0].weight.grad)
        self.assertGreater(float(model.ms_stem[0].weight.grad.abs().sum()), 0.0)
        fallback_grads = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "mamba.scan.reference" in name and parameter.grad is not None
        ]
        self.assertTrue(fallback_grads)
        self.assertGreater(float(sum(gradient.abs().sum() for gradient in fallback_grads)), 0.0)

    def test_rdm_refsr_rejects_pansharpening_kind(self) -> None:
        with self.assertRaisesRegex(ValueError, "RDMPan"):
            _compact_model(reference_kind="pan")
        with self.assertRaisesRegex(ValueError, "rdm_pan"):
            build_model(
                {
                    "name": "rdm_refsr",
                    "inp_channels": 8,
                    "ref_channels": 1,
                    "out_channels": 8,
                    "dim": 16,
                    "depths": [1, 1, 1, 1],
                    "decoder_depths": [1, 1, 1],
                    "reference_kind": "pan",
                    "mamba_d_state": 2,
                    "mamba_d_conv": 2,
                    "mamba_expand": 1,
                    "match_window": 3,
                    "match_dim": 4,
                },
                scale=4,
            )

    def test_long_biwkv_fallback_matches_distance_formula(self) -> None:
        torch.manual_seed(11)
        length, channels = 33, 16
        key = torch.randn(1, length, channels, dtype=torch.float64) * 0.2
        value = torch.randn(1, length, channels, dtype=torch.float64)
        decay = torch.rand(channels, dtype=torch.float64) * 0.2 + 0.05
        first = torch.randn(channels, dtype=torch.float64) * 0.1
        actual = _reference_biwkv(decay, first, key, value)
        expected = []
        for index in range(length):
            weights = torch.exp(
                torch.where(
                    torch.arange(length)[:, None] == index,
                    first[None, :] + key[0, index][None, :],
                    key[0] - decay[None, :] * (torch.arange(length)[:, None] - index).abs(),
                )
            )
            expected.append(
                (weights * value[0]).sum(dim=0) / (weights.sum(dim=0) + 1.0e-6)
            )
        expected = torch.stack(expected).unsqueeze(0)
        self.assertTrue(torch.allclose(actual, expected, atol=2.0e-6, rtol=2.0e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the Bi-WKV kernel test")
    def test_cuda_biwkv_is_finite(self) -> None:
        from kernels.wkv import RUN_CUDA

        torch.manual_seed(7)
        for length in (16, 32, 33, 64, 65, 96):
            with self.subTest(length=length):
                key = torch.randn(2, length, 16, device="cuda", requires_grad=True)
                value = torch.randn(2, length, 16, device="cuda", requires_grad=True)
                decay = (torch.rand(16, device="cuda") * 0.2 + 0.05).requires_grad_(True)
                first = (torch.randn(16, device="cuda") * 0.1).requires_grad_(True)
                output = RUN_CUDA(decay, first, key, value)
                output.square().mean().backward()
                self.assertEqual(tuple(output.shape), (2, length, 16))
                self.assertTrue(torch.isfinite(output).all(), msg=f"T={length} output")
                self.assertTrue(torch.isfinite(key.grad).all(), msg=f"T={length} gk")
                self.assertTrue(torch.isfinite(value.grad).all(), msg=f"T={length} gv")
                self.assertTrue(torch.isfinite(decay.grad).all(), msg=f"T={length} gw")
                self.assertTrue(torch.isfinite(first.grad).all(), msg=f"T={length} gu")
                if length <= 32:
                    reference = _reference_biwkv(
                        decay.detach().cpu(),
                        first.detach().cpu(),
                        key.detach().cpu(),
                        value.detach().cpu(),
                    )
                    # Official 32-way kernel matches the exclusive formula at t=0;
                    # later tokens keep the original segmented reduction, not the
                    # Python recurrence.
                    self.assertTrue(
                        torch.allclose(
                            output.detach().cpu()[:, 0],
                            reference[:, 0],
                            atol=2.0e-4,
                            rtol=2.0e-4,
                        ),
                        msg=f"T={length} token0",
                    )

    def test_initial_prediction_is_bicubic_plus_small_residual(self) -> None:
        torch.manual_seed(3)
        model = _compact_model()
        lr = torch.rand(1, 3, 4, 5)
        ref = torch.rand(1, 3, 8, 10)
        expected = F.interpolate(lr, size=(8, 10), mode="bicubic", align_corners=False).clamp(0, 1)
        output = model(lr, ref)
        self.assertLess(float((output - expected).detach().abs().mean()), 5.0e-3)

    def test_registry_and_config_profile(self) -> None:
        self.assertIn("rdm_refsr", list_models())
        self.assertIn("rdm_pan", list_models())
        self.assertIn("rdm_stf", list_models())
        self.assertIn("rdm_mhf", list_models())
        self.assertIn("stf_mamba", list_models())
        config = load_config("configs/runs/rdm_refsr/hrms_scd_x4.yaml", prefer_existing=False)
        validate_config(config, require_data=False)
        compact = copy.deepcopy(config["model"])
        compact.update(
            dim=16,
            depths=[1, 1, 1, 1],
            decoder_depths=[1, 1, 1],
            mamba_d_state=2,
            mamba_d_conv=2,
            mamba_expand=1,
            match_window=3,
            match_dim=4,
        )
        model = build_model(compact, scale=2)
        self.assertIsInstance(model, RDMRefSR)
        self.assertEqual(
            model.reference_condition_stages,
            frozenset({"enc2", "latent", "dec1", "coeff"}),
        )
        self.assertEqual(model.detail_injection_stages, frozenset({"dec1", "coeff"}))

    def test_tiled_forward_accepts_hr_reference_grid(self) -> None:
        model = _compact_model()
        model.eval()
        lr = torch.randn(1, 3, 5, 7)
        ref = torch.randn(1, 3, 10, 14)
        with torch.no_grad():
            output = tiled_forward(
                model,
                lr,
                ref,
                scale=2,
                tile_size=3,
                overlap=1,
                input_scales=(1, 2),
            )
        self.assertEqual(tuple(output.shape), (1, 3, 10, 14))
        self.assertTrue(torch.isfinite(output).all())

    def test_alignment_offset_is_scaled_to_haar_coefficient_grid(self) -> None:
        model = _compact_model(alignment=True, max_offset=3.0)
        # Replace the learned offset predictor with a constant one-pixel HR
        # translation.  The coefficient warp should receive half a pixel.
        class ConstantOffset(torch.nn.Module):
            def forward(self, value):
                return value.new_zeros(value.shape).add_(torch.atanh(value.new_tensor(1.0 / 3.0)))

        model.offset_net = ConstantOffset()
        captured = []
        original = model._warp

        def capture(image, offsets):
            captured.append(offsets.detach().clone())
            return original(image, offsets)

        model._warp = capture
        lr = torch.randn(1, 3, 3, 4)
        ref = torch.randn(1, 3, 6, 8)
        model(lr, ref)
        self.assertEqual(len(captured), 2)
        self.assertTrue(torch.allclose(captured[0], torch.full_like(captured[0], 0.5), atol=1e-5))

    def test_trainer_physical_loss_is_finite(self) -> None:
        config = load_config("configs/runs/rdm_refsr/hrms_scd_x4.yaml", prefer_existing=False)
        config["data"]["scale"] = 2
        config["model"].update(
            dim=16,
            depths=[1, 1, 1, 1],
            decoder_depths=[1, 1, 1],
            mamba_d_state=2,
            mamba_d_conv=2,
            mamba_expand=1,
            match_window=3,
            match_dim=4,
        )
        config["loss"].update(
            sam_weight=0.01,
            wavelet_weight=0.01,
            consistency_weight=0.01,
            change_weight=0.01,
        )
        trainer = RDMRefSRTrainer.from_config(config)
        lr = torch.randn(1, 3, 3, 4)
        hr = torch.randn(1, 3, 6, 8)
        ref = torch.randn(1, 3, 6, 8)
        value = trainer._train_step({"lr": lr, "hr": hr, "ref": ref}, 0)
        self.assertTrue(torch.isfinite(value))

    def test_sam_boundary_has_finite_gradient(self) -> None:
        # Parallel spectra hit cosine=1 exactly; this used to make acos'
        # derivative infinite and poison bf16 training.
        prediction = torch.ones(2, 3, 4, 4, requires_grad=True)
        target = prediction.detach().clone()
        value = RDMRefSRTrainer._sam(prediction, target)
        self.assertTrue(torch.isfinite(value))
        value.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the official backend test")
    def test_cuda_official_mamba_and_biwkv_backward(self) -> None:
        scan = TrueMambaScan(16, d_state=2, d_conv=2, expand=1, allow_cpu_reference=False).cuda()
        sequence = torch.randn(2, 7, 16, device="cuda", requires_grad=True)
        output = scan(sequence)
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertGreater(float(sequence.grad.abs().sum()), 0.0)

        rwkv = SharedDirectionalRWKV(16, shuffle_prob=0.0).cuda()
        image = torch.randn(1, 16, 4, 5, device="cuda", requires_grad=True)
        value = rwkv(image)
        self.assertTrue(torch.isfinite(value).all())
        value.square().mean().backward()
        self.assertGreater(float(image.grad.abs().sum()), 0.0)

    def test_channel_multipliers_change_stage_width(self) -> None:
        model = _compact_model(channel_multipliers=(1, 1, 2, 4), high_order=False)
        self.assertEqual(model.channel_multipliers, (1, 1, 2, 4))
        self.assertEqual(model.enc0.blocks[0].channels, 16)
        self.assertEqual(model.enc1.blocks[0].channels, 16)
        self.assertEqual(model.enc2.blocks[0].channels, 32)
        self.assertEqual(model.latent.blocks[0].channels, 64)
        self.assertFalse(model.latent.blocks[0].rwkv.high_order)

    def test_invalid_channel_multipliers_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _compact_model(channel_multipliers=(1, 2, 4))
        with self.assertRaises(ValueError):
            _compact_model(channel_multipliers=(1, 2, 0, 4))
        with self.assertRaises(ValueError):
            _compact_model(channel_multipliers=(1, 2, 4, 8, 8))

    def test_reference_kind_aliases_map_to_pan_stf_mhf(self) -> None:
        self.assertEqual(normalize_reference_kind("temporal"), "stf")
        self.assertEqual(normalize_reference_kind("hrms"), "stf")
        self.assertEqual(normalize_reference_kind("hsi_msi"), "mhf")
        self.assertEqual(normalize_reference_kind("pansharpening"), "pan")
        self.assertIsInstance(RDMPan(inp_channels=8, ref_channels=1, out_channels=8, dim=16, depths=(1,1,1,1), decoder_depths=(1,1,1), mamba_d_state=2, mamba_d_conv=2, mamba_expand=1, match_window=3, match_dim=4, shuffle_prob=0.0), RDMPan)
        self.assertEqual(RDMStf(dim=16, depths=(1,1,1,1), decoder_depths=(1,1,1), mamba_d_state=2, mamba_d_conv=2, mamba_expand=1, match_window=3, match_dim=4, shuffle_prob=0.0).reference_kind, "stf")
        self.assertEqual(RDMMhf(inp_channels=8, ref_channels=4, out_channels=8, dim=16, depths=(1,1,1,1), decoder_depths=(1,1,1), mamba_d_state=2, mamba_d_conv=2, mamba_expand=1, match_window=3, match_dim=4, shuffle_prob=0.0).reference_kind, "mhf")
        with self.assertRaises(ValueError):
            RDMPan(reference_kind="stf")

    def test_stf_uses_c0_and_has_no_matcher(self) -> None:
        torch.manual_seed(0)
        model = RDMStf(inp_channels=4, ref_channels=4, out_channels=4, dim=16, scale=1, mamba_d_state=2, mamba_d_conv=2, mamba_expand=1)
        self.assertFalse(hasattr(model, "reliability"))
        self.assertFalse(hasattr(model, "detail_matcher"))
        lr = torch.rand(1, 4, 8, 8)
        ref = torch.rand(1, 4, 8, 8)
        c0 = torch.rand(1, 4, 8, 8)
        two = model(lr, ref)
        three = model(lr, ref, c0)
        self.assertEqual(tuple(two.shape), (1, 4, 8, 8))
        self.assertEqual(tuple(three.shape), (1, 4, 8, 8))
        self.assertTrue(torch.isfinite(two).all())
        self.assertTrue(torch.isfinite(three).all())
        self.assertGreater(float((two - three).detach().abs().mean()), 0.0)
        three.square().mean().backward()
        self.assertGreater(float(model.raise_fine[0].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.raise_coarse[0].weight.grad.abs().sum()), 0.0)

    def test_pan_has_no_reliability_gate_unlike_stf(self) -> None:
        temporal = _compact_model()
        pan = _pan_model()
        self.assertFalse(hasattr(pan, "reliability"))
        self.assertFalse(hasattr(pan, "detail_matcher"))
        self.assertFalse(hasattr(pan, "inject_dec1"))
        self.assertAlmostEqual(temporal.inject_dec1.alpha.detach().mean().item(), 0.05, places=5)
        self.assertAlmostEqual(temporal.synthesis.reference_scale.detach().mean().item(), 0.10, places=5)
        self.assertAlmostEqual(temporal.reliability.net[-1].bias.detach().item(), -1.0, places=5)
        self.assertLess(float(pan.to_hrms[-1].weight.detach().abs().max()), 1.1e-4)
        self.assertEqual(float(pan.to_hrms[-1].bias.detach().abs().max()), 0.0)

    def test_pan_dual_stream_starts_from_bicubic_and_uses_pan(self) -> None:
        torch.manual_seed(0)
        model = _pan_model()
        self.assertFalse(hasattr(model, "reliability"))
        self.assertFalse(hasattr(model, "detail_matcher"))
        self.assertEqual(model.channel_multipliers, (1, 2, 4))
        lr = torch.rand(1, 8, 4, 4)
        ref = torch.rand(1, 1, 8, 8)
        expected = F.interpolate(lr, size=(8, 8), mode="bicubic", align_corners=False)
        output = model(lr, ref)
        self.assertEqual(tuple(output.shape), (1, 8, 8, 8))
        self.assertLess(float((output - expected).detach().abs().mean()), 5.0e-3)
        output.square().mean().backward()
        self.assertIsNotNone(model.raise_pan[0].weight.grad)
        self.assertGreater(float(model.raise_pan[0].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.raise_ms[0].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.stage0.pan_from_ms.weight.grad.abs().sum()), 0.0)

    def test_pan_slim_config_matches_fusion_mamba_capacity(self) -> None:
        from models.refsr.fusion_mamba.adapter import FusionMambaRefSR

        config = load_config(
            "configs/runs/rdm_refsr/pancollection_wv3_pan_x4.yaml",
            prefer_existing=False,
        )
        validate_config(config, require_data=False)
        self.assertEqual(config["model"]["name"], "rdm_pan")
        model = build_model(config["model"], scale=4)
        self.assertIsInstance(model, RDMPan)
        self.assertEqual(model.reference_kind, "pan")
        self.assertEqual(model.channel_multipliers, (1, 2, 4))
        self.assertEqual(model.dim, 32)
        self.assertFalse(hasattr(model, "detail_matcher"))
        self.assertFalse(hasattr(model, "reliability"))
        pan_params = sum(parameter.numel() for parameter in model.parameters())
        fusion = FusionMambaRefSR(dim=32, pan_dim=1, ms_dim=8, H=64, W=64, scale=4)
        fusion_params = sum(parameter.numel() for parameter in fusion.parameters())
        self.assertLess(pan_params, 1_800_000)
        self.assertGreater(pan_params / fusion_params, 0.5)
        self.assertLess(pan_params / fusion_params, 3.0)


if __name__ == "__main__":
    unittest.main()
