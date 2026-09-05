"""RefDiffRWKV model package.

The lightweight RWKV feature model is importable without diffusion extras.
Stable-Diffusion, OpenCLIP and LPIPS dependencies are loaded only when the
corresponding class is requested by a training or inference entry point.
"""

from .RefDiffRWKV import RefDiffRWKV

__all__ = ["RefDiffRWKV", "SD2RefGenerator", "SD2RefDiscriminator", "SD2RefGANSystem"]


def __getattr__(name):
    if name == "SD2RefGenerator":
        from .sd2_ref_generator import SD2RefGenerator

        return SD2RefGenerator
    if name == "SD2RefDiscriminator":
        from .sd2_ref_discriminator import SD2RefDiscriminator

        return SD2RefDiscriminator
    if name == "SD2RefGANSystem":
        from .sd2_ref_gan_system import SD2RefGANSystem

        return SD2RefGANSystem
    raise AttributeError(name)
