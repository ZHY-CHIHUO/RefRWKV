"""Shared configuration, data, metrics, and checkpoint helpers."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(value: str | os.PathLike[str], *, prefer_cwd: bool = False) -> Path:
    """Resolve relative paths consistently from the repository root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_path = Path.cwd() / path
    repo_path = PROJECT_ROOT / path
    if prefer_cwd and cwd_path.exists():
        return cwd_path.resolve()
    if cwd_path.exists() and not repo_path.exists():
        return cwd_path.resolve()
    return repo_path.resolve()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def gaussian_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image RGB SSIM for tensors in the [-1, 1] range."""
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError(f"SSIM 输入形状不一致: {pred.shape} vs {target.shape}")
    channels = pred.shape[1]
    size = min(11, int(pred.shape[-2]), int(pred.shape[-1]))
    if size % 2 == 0:
        size -= 1
    if size < 3:
        return torch.ones(pred.shape[0], device=pred.device, dtype=torch.float32)
    sigma = 1.5
    coords = torch.arange(size, dtype=torch.float32, device=pred.device)
    gaussian = torch.exp(-((coords - size // 2) ** 2) / (2 * sigma**2))
    window = torch.outer(gaussian, gaussian)
    window = (window / window.sum()).view(1, 1, size, size).expand(channels, 1, size, size)
    pad = size // 2
    pred_f, target_f = pred.float(), target.float()
    mu_p = F.conv2d(pred_f, window, padding=pad, groups=channels)
    mu_t = F.conv2d(target_f, window, padding=pad, groups=channels)
    var_p = (F.conv2d(pred_f.square(), window, padding=pad, groups=channels) - mu_p.square()).clamp_min(0)
    var_t = (F.conv2d(target_f.square(), window, padding=pad, groups=channels) - mu_t.square()).clamp_min(0)
    cov = F.conv2d(pred_f * target_f, window, padding=pad, groups=channels) - mu_p * mu_t
    c1, c2 = (0.01 * 2.0) ** 2, (0.03 * 2.0) ** 2
    numerator = (2 * mu_p * mu_t + c1) * (2 * cov + c2)
    denominator = (mu_p.square() + mu_t.square() + c1) * (var_p + var_t + c2)
    score = numerator / denominator.clamp_min(1e-12)
    return score.clamp(-1, 1).mean(dim=(1, 2, 3))


def per_image_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (pred.float() - target.float()).square().mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(4.0 / mse.clamp_min(1e-10))


class EMA:
    """Small, model-agnostic EMA used by all compatible training runners."""

    def __init__(self, decay: float = 0.999):
        self.decay = float(decay)
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("EMA decay 必须位于 [0, 1)")
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        self.initialized = False
        self.applied = False

    def _init(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name not in self.shadow:
                self.shadow[name] = param.detach().float().clone()
        self.initialized = True

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self._init(model)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                self.shadow[name].mul_(self.decay).add_(
                    param.detach().float(), alpha=1 - self.decay
                )

    @torch.no_grad()
    def apply(self, model: torch.nn.Module) -> None:
        if self.applied:
            return
        self._init(model)
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))
        self.applied = True

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        if not self.applied:
            return
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].to(device=param.device, dtype=param.dtype))
        self.backup = {}
        self.applied = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": {name: value.detach().clone() for name, value in self.shadow.items()},
            "initialized": self.initialized,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("EMA state 必须是 mapping")
        self.decay = float(state.get("decay", self.decay))
        self.shadow = {
            name: value.detach().float().clone()
            for name, value in state.get("shadow", {}).items()
            if isinstance(name, str) and torch.is_tensor(value)
        }
        self.initialized = bool(state.get("initialized", bool(self.shadow)))
        self.backup, self.applied = {}, False


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(json_safe(value), file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
