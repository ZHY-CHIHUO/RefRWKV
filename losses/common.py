"""Small loss helpers shared by SR engines."""

from __future__ import annotations

import torch


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor, name: str = "l1", eps: float = 1e-3) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shapes differ: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    normalized = str(name).lower()
    if normalized == "l2":
        normalized = "mse"
    if normalized == "l1":
        return (prediction - target).abs().mean()
    if normalized == "mse":
        return (prediction - target).square().mean()
    if normalized == "charbonnier":
        return ((prediction - target).square() + float(eps) ** 2).sqrt().mean()
    raise ValueError(f"unsupported reconstruction loss: {name}")

