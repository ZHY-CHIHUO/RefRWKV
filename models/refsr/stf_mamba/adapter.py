"""Project adapter for official STFMamba on the [0, 1] STF contract.

Python call is ``forward(lr, ref, c0) -> f1`` with ``lr=C1``, ``ref=F0``,
``c0=C0``.  That is the classic STF map ``(C0, F0, C1) -> F1``.  Internally
the upstream network still sees signed tensors in ``[-1, 1]`` because its
last layer is ``tanh``; the adapter converts both ways so the rest of the
repository never leaves ``[0, 1]``.

The public GitHub ``PatchSet`` is the LGC/DX 6-band ``.npy`` loader
(``[-1, 1]``, ``len=12000``).  Wuhan is not that pipeline: this adapter is
the 4-band table from the paper, fed by this repo's TIFF loader
(``/11848 -> [0, 1]``).

The VMamba body is imported lazily so ``list_models()`` does not compile
selective-scan CUDA extensions.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from ..registry import RefSRModelAdapter, register_adapter


def _to_signed(value: torch.Tensor) -> torch.Tensor:
    return value.float().clamp(0.0, 1.0).mul(2.0).sub(1.0)


def _to_unit(value: torch.Tensor) -> torch.Tensor:
    return value.float().add(1.0).mul(0.5).clamp(0.0, 1.0)


class STFMambaRefSR(nn.Module):
    """Official STFMamba with the repository's three-input STF contract."""

    def __init__(self, in_chans: int = 4, **kwargs: Any) -> None:
        super().__init__()
        from .model import model_STF

        self.in_chans = int(in_chans)
        self.inp_channels = self.in_chans
        self.out_channels = self.in_chans
        self.ref_channels = self.in_chans
        self.net = model_STF(in_chans=self.in_chans, **kwargs)

    def forward(
        self,
        lr: torch.Tensor,
        ref: torch.Tensor,
        c0: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if lr.ndim != 4 or ref.ndim != 4 or c0.ndim != 4:
            raise ValueError("STFMamba expects NCHW C1/F0/C0 tensors")
        if lr.shape[1] != self.in_chans or ref.shape[1] != self.in_chans or c0.shape[1] != self.in_chans:
            raise ValueError(
                f"STFMamba is built for {self.in_chans} bands, got "
                f"C1={tuple(lr.shape)} F0={tuple(ref.shape)} C0={tuple(c0.shape)}"
            )
        if tuple(lr.shape[-2:]) != tuple(ref.shape[-2:]) or tuple(lr.shape[-2:]) != tuple(c0.shape[-2:]):
            raise ValueError(
                "STFMamba Wuhan path expects C0/F0/C1 on one grid; "
                f"got C1={tuple(lr.shape[-2:])} F0={tuple(ref.shape[-2:])} C0={tuple(c0.shape[-2:])}"
            )
        coarse_0 = _to_signed(c0)
        coarse_1 = _to_signed(lr)
        fine_0 = _to_signed(ref)
        sr_c0, sr_c1, from_f0, from_c1, fused = self.net(
            coarse_0, coarse_1, fine_0, def_device=lr.device
        )
        output = _to_unit(fused)
        if not return_aux:
            return output
        return output, {
            "sr_c0": _to_unit(sr_c0),
            "sr_c1": _to_unit(sr_c1),
            "from_f0": _to_unit(from_f0),
            "from_c1": _to_unit(from_c1),
        }


class STFMambaAdapter(RefSRModelAdapter):
    name = "stf_mamba"

    def build(self, model_config: Mapping[str, Any], *, scale: int) -> STFMambaRefSR:
        in_chans = int(
            model_config.get(
                "in_chans",
                model_config.get("inp_channels", model_config.get("out_channels", 4)),
            )
        )
        return STFMambaRefSR(in_chans=in_chans)

    def describe(self, model_config: Mapping[str, Any], *, scale: int) -> dict[str, Any]:
        result = super().describe(model_config, scale=scale)
        result.update(
            {
                "implementation": "official_stfmamba",
                "family": "stf",
                "task": "spatio_temporal_fusion",
                "inputs": "(c0, f0, c1)",
                "call": "forward(lr=C1, ref=F0, c0=C0)",
                "range": "[0, 1]",
                "bands": "4-band Wuhan; not the 6-band LGC/DX npy loader",
            }
        )
        return result


register_adapter(STFMambaAdapter())

__all__ = ["STFMambaAdapter", "STFMambaRefSR"]
