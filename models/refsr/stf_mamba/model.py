"""Official STFMamba body with a configurable band count.

The upstream constructor is locked to 6 Landsat/MODIS bands and a CUDA
device handle.  Wuhan's published table is 4-band GF/Landsat, so ``in_chans``
is threaded through SR, encoder, decoder and the final cross-attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .MambaFusion import STFMamba
from .MambaSR import MambaSR
from .cross_attention import Cross_MultiAttention


_DEFAULT_VSS = dict(
    pretrained=None,
    patch_size=4,
    num_classes=1000,
    depths=[2, 2, 9, 2],
    dims=96,
    ssm_d_state=16,
    ssm_ratio=2.0,
    ssm_rank_ratio=2.0,
    ssm_dt_rank=2,
    ssm_act_layer="silu",
    ssm_conv=3,
    ssm_conv_bias=True,
    ssm_drop_rate=0.0,
    ssm_init="v0",
    forward_type="v2",
    mlp_ratio=4.0,
    mlp_act_layer="gelu",
    mlp_drop_rate=0.0,
    drop_path_rate=0.1,
    patch_norm=True,
    norm_layer="ln",
    downsample_version="v2",
    patchembed_version="v2",
    gmlp=False,
    use_checkpoint=False,
)


class model_STF(nn.Module):
    """Two-step STFMamba: SR on C0/C1, then dual-U fusion with F0."""

    def __init__(self, in_chans: int = 4, **overrides):
        super().__init__()
        self.in_chans = int(in_chans)
        kwargs = dict(_DEFAULT_VSS)
        kwargs.update(overrides)
        kwargs["in_chans"] = self.in_chans
        self.model_SR = MambaSR(**kwargs)
        self.model_fusion = STFMamba(**kwargs)
        self.cross_fusion = Cross_MultiAttention(in_channels=self.in_chans, emb_dim=16, num_heads=4)

    def forward(self, ref_lr, data, ref_target, def_device=None):
        if def_device is None:
            def_device = data.device
        super_resolved_c0 = self.model_SR(ref_lr)
        super_resolved_c1 = self.model_SR(data)
        from_f0, from_c1 = self.model_fusion(
            super_resolved_c0, super_resolved_c1, ref_target, def_device
        )
        fused = self.cross_fusion(from_f0, from_c1)
        return super_resolved_c0, super_resolved_c1, from_f0, from_c1, fused
