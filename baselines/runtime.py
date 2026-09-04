"""Checkpoint loading helpers shared by baseline training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


_OUTER_PREFIXES = (
    "module.",
    "model.",
    "model_sr.",
    "generator.",
    "state_dict.",
)


def load_checkpoint(path: str | Path) -> Any:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _find_tensor_mapping(value: Any) -> dict[str, torch.Tensor] | None:
    if not isinstance(value, Mapping):
        return None
    direct = {
        str(key): tensor
        for key, tensor in value.items()
        if isinstance(key, str) and torch.is_tensor(tensor)
    }
    if direct:
        return direct
    for key in (
        "state_dict",
        "model_state_dict",
        "model",
        "params_ema",
        "params",
        "weights",
        "generator",
    ):
        if key in value:
            found = _find_tensor_mapping(value[key])
            if found:
                return found
    return None


def _strip_outer_prefixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in _OUTER_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
                break
    return key


def _matching_state(
    target_state: Mapping[str, torch.Tensor], source_state: Mapping[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Match Lightning-wrapped and raw architecture state dictionaries."""
    matched: dict[str, torch.Tensor] = {}
    missing_source, shape_mismatch = 0, 0
    for source_key, value in source_state.items():
        if not torch.is_tensor(value):
            continue
        key = _strip_outer_prefixes(str(source_key))
        candidates = [key]
        # A raw official SwinIR state dict has ``conv_first.*`` while this
        # repository wrapper stores it under ``net.conv_first.*``.
        if not key.startswith("net."):
            candidates.append(f"net.{key}")
        target_key = next((candidate for candidate in candidates if candidate in target_state), None)
        if target_key is None:
            missing_source += 1
            continue
        if tuple(value.shape) != tuple(target_state[target_key].shape):
            shape_mismatch += 1
            continue
        matched[target_key] = value
    return matched, missing_source, shape_mismatch


def load_model_weights(
    model: torch.nn.Module, checkpoint: Any, *, prefer_ema: bool = True
) -> dict[str, int | str | bool]:
    """Load raw state plus optional EMA parameters into a baseline model.

    The raw state loads buffers such as SwinIR position indexes.  EMA shadows
    then overwrite matching trainable parameters, which preserves the usual
    EMA evaluation behavior without requiring EMA to duplicate buffers.
    """
    target_state = model.state_dict()
    raw_source = _find_tensor_mapping(checkpoint)
    if not raw_source:
        raise ValueError("checkpoint does not contain a tensor state dictionary")
    raw_matched, raw_unused, raw_mismatch = _matching_state(target_state, raw_source)
    if not raw_matched:
        raise RuntimeError("checkpoint has no parameter compatible with the selected baseline")
    model.load_state_dict(raw_matched, strict=False)

    report: dict[str, int | str | bool] = {
        "raw_matched": len(raw_matched),
        "raw_unused": raw_unused,
        "raw_shape_mismatch": raw_mismatch,
        "ema_applied": False,
        "ema_matched": 0,
    }
    if not prefer_ema or not isinstance(checkpoint, Mapping):
        return report
    ema_state = checkpoint.get("baseline_ema_state") or checkpoint.get("ema_state_dict")
    shadow = ema_state.get("shadow") if isinstance(ema_state, Mapping) else None
    if not isinstance(shadow, Mapping):
        return report
    ema_matched, _unused, _mismatch = _matching_state(target_state, shadow)
    if not ema_matched:
        return report
    model.load_state_dict(ema_matched, strict=False)
    report["ema_applied"] = True
    report["ema_matched"] = len(ema_matched)
    return report


def checkpoint_config(checkpoint: Any) -> dict[str, Any] | None:
    """Return the materialized run configuration saved by this runner."""
    if not isinstance(checkpoint, Mapping):
        return None
    config = checkpoint.get("baseline_config")
    return dict(config) if isinstance(config, Mapping) else None


def checkpoint_signature(checkpoint: Any) -> dict[str, Any] | None:
    if not isinstance(checkpoint, Mapping):
        return None
    signature = checkpoint.get("baseline_experiment_signature")
    return dict(signature) if isinstance(signature, Mapping) else None
