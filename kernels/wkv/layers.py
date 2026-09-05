"""Shared neural-network layers used by WKV-based RefSR models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OmniShift(nn.Module):
    """Learnable depthwise spatial shift with an inference reparameterization."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.conv1x1 = nn.Conv2d(dim, dim, 1, groups=dim, bias=False)
        self.conv3x3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.conv5x5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.alpha = nn.Parameter(torch.ones(4) * 0.25)
        self.gate = nn.Parameter(torch.zeros(1))
        self.register_buffer("conv5x5_reparam_weight", torch.zeros(dim, 1, 5, 5))
        self._reparam_done = False

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.softmax(self.alpha, dim=0)
        shifted = (
            alpha[0] * x
            + alpha[1] * self.conv1x1(x)
            + alpha[2] * self.conv3x3(x)
            + alpha[3] * self.conv5x5(x)
        )
        return x + torch.tanh(self.gate) * (shifted - x)

    def reparam_5x5(self) -> None:
        if self._reparam_done:
            return
        with torch.no_grad():
            gate = torch.tanh(self.gate)
            alpha = torch.softmax(self.alpha, dim=0)
            weight = self.conv1x1.weight
            identity = torch.zeros(
                self.dim, 1, 5, 5, device=weight.device, dtype=weight.dtype
            )
            identity[:, :, 2, 2] = 1.0
            weight_1, weight_3, weight_5 = (
                F.pad(weight, (2, 2, 2, 2)),
                F.pad(self.conv3x3.weight, (1, 1, 1, 1)),
                self.conv5x5.weight,
            )
            self.conv5x5_reparam_weight.copy_(
                (1.0 - gate) * identity
                + gate
                * (
                    alpha[0] * identity
                    + alpha[1] * weight_1
                    + alpha[2] * weight_3
                    + alpha[3] * weight_5
                )
            )
        self._reparam_done = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._reparam_done = False
            return self.forward_train(x)
        if not self._reparam_done:
            self.reparam_5x5()
        return F.conv2d(x, self.conv5x5_reparam_weight, padding=2, groups=self.dim)


__all__ = ["OmniShift"]
