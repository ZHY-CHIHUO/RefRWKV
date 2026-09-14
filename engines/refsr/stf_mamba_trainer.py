"""Official STFMamba training loss on the [0, 1] STF contract.

The public train loop supervises the fused image plus the two decoder
hypotheses and the C1 super-resolution head.  VGG is omitted because Wuhan
is 4-band (NIR is not an ImageNet RGB channel).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .trainer import RefSRTrainer


class STFMambaTrainer(RefSRTrainer):
    """Charbonnier + (1-SSIM) on fusion, with the official auxiliary heads."""

    def _stf_head_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        residual = prediction - target
        value = (residual.square() + self.charbonnier_eps**2).sqrt().mean()
        if self.ssim_weight:
            from runtime.common import gaussian_ssim

            value = value + self.ssim_weight * (1.0 - gaussian_ssim(prediction, target).mean())
        return value

    def _train_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        lr, hr, ref = self._unpack(batch)
        reference = self._apply_reference_dropout(self._reference(lr, hr, ref), lr)
        c0 = self._c0(batch)
        prediction, aux = self.model(lr, reference, c0, return_aux=True)
        loss = self._stf_head_loss(prediction, hr)
        loss = loss + self._stf_head_loss(aux["from_f0"], hr)
        loss = loss + self._stf_head_loss(aux["from_c1"], hr)
        loss = loss + self._stf_head_loss(aux["sr_c1"], hr)
        loss = loss + self._stf_head_loss(aux["sr_c0"], reference)
        return loss

    def _eval_step(self, batch: Any, batch_idx: int, *, stage: str) -> dict[str, torch.Tensor]:
        lr, hr, ref = self._unpack(batch)
        reference = self._reference(lr, hr, ref)
        prediction = self.predict_for_eval(lr, reference, self._c0(batch))
        return {"loss": self._loss(prediction, hr), **self.benchmark_image_metrics(prediction, hr)}


__all__ = ["STFMambaTrainer"]
