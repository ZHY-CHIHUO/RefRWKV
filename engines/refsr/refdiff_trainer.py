"""RefDiffRWKV training engine entry point.

The diffusion model already implements its specialized manual G/D optimization
loop.  This module gives that implementation a stable engine-facing name while
keeping it physically under the RefSR model family.
"""

from models.refsr.RefDiffRWKV.sd2_ref_gan_system import SD2RefGANSystem


class RefDiffRWKVTrainer(SD2RefGANSystem):
    """Specialized RefDiffRWKV engine with the shared experiment layout."""


__all__ = ["RefDiffRWKVTrainer"]
