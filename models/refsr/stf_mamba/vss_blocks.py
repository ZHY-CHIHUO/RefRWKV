"""NCHW-safe dual VSS scans for the 4-band STFMamba port.

The upstream blocks write a spatial VSS in NHWC and then a second VSS whose
``hidden_dim`` does not match the tensor.  ``LayerNorm(hidden_dim)`` therefore
crashes before any Wuhan 4-band tensor can move through the dual-U.  These
helpers keep the published hidden dims and insert 1x1 projections plus permutes
so each scan sees the layout it was written for.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .model_VSS.vmamba import Permute, VSSBlock


def vss_scan(
    hidden_dim: int,
    *,
    channel_first: bool,
    norm_layer: Any,
    ssm_act_layer: Any,
    mlp_act_layer: Any,
    kwargs: dict[str, Any],
) -> list[nn.Module]:
    """One VSS scan that accepts and returns NCHW when ``channel_first`` is false."""
    return [
        Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
        VSSBlock(
            hidden_dim=hidden_dim,
            drop_path=0.1,
            norm_layer=norm_layer,
            channel_first=channel_first,
            ssm_d_state=kwargs["ssm_d_state"],
            ssm_ratio=kwargs["ssm_ratio"],
            ssm_dt_rank=kwargs["ssm_dt_rank"],
            ssm_act_layer=ssm_act_layer,
            ssm_conv=kwargs["ssm_conv"],
            ssm_conv_bias=kwargs["ssm_conv_bias"],
            ssm_drop_rate=kwargs["ssm_drop_rate"],
            ssm_init=kwargs["ssm_init"],
            forward_type=kwargs["forward_type"],
            mlp_ratio=kwargs["mlp_ratio"],
            mlp_act_layer=mlp_act_layer,
            mlp_drop_rate=kwargs["mlp_drop_rate"],
            gmlp=kwargs["gmlp"],
            use_checkpoint=kwargs["use_checkpoint"],
        ),
        Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
    ]


def dual_vss_block(
    spatial_dim: int,
    spectral_dim: int,
    *,
    channel_first: bool,
    norm_layer: Any,
    ssm_act_layer: Any,
    mlp_act_layer: Any,
    kwargs: dict[str, Any],
    in_channels: int | None = None,
    output_dim: int | None = None,
) -> nn.Sequential:
    """Spatial VSS then spectral VSS, NCHW in and out.

    ``in_channels`` optionally projects into ``spatial_dim``.  After the
    spectral scan the tensor is projected to ``output_dim`` (default
    ``spatial_dim``) so residual U-Net paths stay aligned.  The SR head
    passes ``output_dim=12`` so it returns to the 12-channel residual.
    """
    spatial_dim = int(spatial_dim)
    spectral_dim = int(spectral_dim)
    output_dim = spatial_dim if output_dim is None else int(output_dim)
    layers: list[nn.Module] = []
    if in_channels is not None:
        layers.append(nn.Conv2d(int(in_channels), spatial_dim, kernel_size=1))
    layers.extend(
        vss_scan(
            spatial_dim,
            channel_first=channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            kwargs=kwargs,
        )
    )
    if spectral_dim != spatial_dim:
        layers.append(nn.Conv2d(spatial_dim, spectral_dim, kernel_size=1))
    layers.extend(
        vss_scan(
            spectral_dim,
            channel_first=channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            kwargs=kwargs,
        )
    )
    if spectral_dim != output_dim:
        layers.append(nn.Conv2d(spectral_dim, output_dim, kernel_size=1))
    return nn.Sequential(*layers)


__all__ = ["dual_vss_block", "vss_scan"]
