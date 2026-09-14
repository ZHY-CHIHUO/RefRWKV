import unittest

import torch

from models.refsr.fusion_mamba.adapter import FusionMambaRefSR
from models.refsr.fusion_mamba.fusion_mamba import FusionMamba


class FusionMambaFullImageTests(unittest.TestCase):
    def test_bind_spatial_overrides_constructor_hw(self) -> None:
        block = FusionMamba(32, 64, 64)
        self.assertEqual(block.spa_mamba_layers[0].block.input_h, 64)
        block._bind_spatial(256, 256)
        self.assertEqual(block.spa_mamba_layers[0].block.input_h, 256)
        self.assertEqual(block.spa_cross_mamba.block.input_w, 256)

    @unittest.skipUnless(torch.cuda.is_available(), "FusionMamba 2D scan is too slow on CPU")
    def test_crop_built_model_accepts_reduced_wv3_full_image(self) -> None:
        model = FusionMambaRefSR(dim=32, pan_dim=1, ms_dim=8, H=64, W=64, scale=4).cuda().eval()
        lr = torch.rand(1, 8, 64, 64, device="cuda")
        pan = torch.rand(1, 1, 256, 256, device="cuda")
        with torch.no_grad():
            output = model(lr, pan)
        self.assertEqual(tuple(output.shape), (1, 8, 256, 256))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
