"""Checkpoint loading helpers shared by training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


_OUTER_PREFIXES = (
    "module.",
    "model.",
    "model_sr.",
    "model_diff.",
    "model_enhance.",
    "system.",
    "generator.",
    "sr_model.",
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


_RGB_MULTIBAND_BOUNDARY_SUFFIXES = (
    "lr_up.0.weight",
    "ref_to_level1.0.weight",
    "ref_to_level1.2.weight",
)


def _resize_rgb_axis(value: torch.Tensor, target_size: int, axis: int) -> torch.Tensor | None:
    """Copy RGB weights and initialize an added band with their mean."""
    source_size = int(value.shape[axis])
    if source_size == target_size:
        return value
    if {source_size, int(target_size)} != {3, 4}:
        return None
    result_shape = list(value.shape)
    result_shape[axis] = int(target_size)
    result = value.new_empty(result_shape)
    common = min(source_size, int(target_size))
    result.narrow(axis, 0, common).copy_(value.narrow(axis, 0, common))
    if target_size > source_size:
        mean = value.mean(dim=axis, keepdim=True)
        result.narrow(axis, source_size, target_size - source_size).copy_(mean.expand_as(result.narrow(axis, source_size, target_size - source_size)))
    return result


def _adapt_rgb_multiband_boundary(
    key: str, source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor | None:
    """Adapt safe RefSR RGB boundary tensors to four spectral bands.

    This intentionally does *not* try to adapt scale-dependent PixelUnshuffle
    layers, the reconstruction head, or a task/fusion head with a different
    reconstruction scale. Those components remain newly initialized for
    HRMS-SCD x4 -> Wuhan x1 transfer.
    """
    if not key.endswith(_RGB_MULTIBAND_BOUNDARY_SUFFIXES):
        return None
    if source.ndim != target.ndim or source.ndim not in {1, 4}:
        return None
    if source.ndim == 1:
        return _resize_rgb_axis(source, int(target.shape[0]), 0)
    if source.shape[2:] != target.shape[2:]:
        return None
    value = _resize_rgb_axis(source, int(target.shape[0]), 0)
    if value is None:
        return None
    value = _resize_rgb_axis(value, int(target.shape[1]), 1)
    if value is None or tuple(value.shape) != tuple(target.shape):
        return None
    return value


def _matching_state(
    target_state: Mapping[str, torch.Tensor],
    source_state: Mapping[str, torch.Tensor],
    *,
    channel_adaptation: str = "none",
) -> tuple[dict[str, torch.Tensor], int, int, int]:
    """Match Lightning-wrapped and raw architecture state dictionaries."""
    matched: dict[str, torch.Tensor] = {}
    missing_source, shape_mismatch, adapted = 0, 0, 0
    for source_key, value in source_state.items():
        if not torch.is_tensor(value):
            continue
        original_key = str(source_key)
        key = _strip_outer_prefixes(original_key)
        # 先匹配原始键名，以保留系统 checkpoint 的 generator/sr_model 命名空间。
        candidates = [original_key]
        if key != original_key:
            candidates.append(key)
        # 通过 net. 前缀兼容 SwinIR 的封装参数名。
        if not key.startswith("net."):
            candidates.append(f"net.{key}")
        target_key = next(
            (candidate for candidate in candidates if candidate in target_state), None
        )
        if target_key is None:
            missing_source += 1
            continue
        target = target_state[target_key]
        if tuple(value.shape) != tuple(target.shape):
            adapted_value = (
                _adapt_rgb_multiband_boundary(target_key, value, target)
                if channel_adaptation == "rgb_mean"
                else None
            )
            if adapted_value is None:
                shape_mismatch += 1
                continue
            matched[target_key] = adapted_value
            adapted += 1
            continue
        matched[target_key] = value
    return matched, missing_source, shape_mismatch, adapted


def load_model_weights(
    model: torch.nn.Module,
    checkpoint: Any,
    *,
    prefer_ema: bool = True,
    channel_adaptation: str = "none",
) -> dict[str, int | str | bool]:
    """Load raw state plus optional EMA parameters into a selected model.

    The raw state loads buffers such as SwinIR position indexes.  EMA shadows
    then overwrite matching trainable parameters, which preserves the usual
    EMA evaluation behavior without requiring EMA to duplicate buffers.
    """
    channel_adaptation = str(channel_adaptation).strip().lower()
    if channel_adaptation not in {"none", "rgb_mean"}:
        raise ValueError("channel_adaptation must be none or rgb_mean")
    target_state = model.state_dict()
    raw_source = _find_tensor_mapping(checkpoint)
    if not raw_source:
        raise ValueError("checkpoint does not contain a tensor state dictionary")
    raw_matched, raw_unused, raw_mismatch, raw_adapted = _matching_state(
        target_state, raw_source, channel_adaptation=channel_adaptation
    )
    if not raw_matched:
        raise RuntimeError("checkpoint has no parameter compatible with the selected model")
    model.load_state_dict(raw_matched, strict=False)

    report: dict[str, int | str | bool] = {
        "raw_matched": len(raw_matched),
        "raw_unused": raw_unused,
        "raw_shape_mismatch": raw_mismatch,
        "raw_channel_adapted": raw_adapted,
        "channel_adaptation": channel_adaptation,
        "ema_applied": False,
        "ema_matched": 0,
        "ema_channel_adapted": 0,
    }
    if not prefer_ema or not isinstance(checkpoint, Mapping):
        return report
    ema_state = (
        checkpoint.get("ema_state")
        or checkpoint.get("ema_state_dict")
        or checkpoint.get("model_ema_state")
        # Check the additional EMA field.
        or checkpoint.get("baseline_ema_state")
        or checkpoint.get("ema")
    )
    shadow = ema_state.get("shadow") if isinstance(ema_state, Mapping) else None
    if not isinstance(shadow, Mapping):
        return report
    ema_matched, _unused, _mismatch, ema_adapted = _matching_state(
        target_state, shadow, channel_adaptation=channel_adaptation
    )
    if not ema_matched:
        return report
    model.load_state_dict(ema_matched, strict=False)
    report["ema_applied"] = True
    report["ema_matched"] = len(ema_matched)
    report["ema_channel_adapted"] = ema_adapted
    return report


def checkpoint_config(checkpoint: Any) -> dict[str, Any] | None:
    """Return the materialized run configuration saved by this runner."""
    if not isinstance(checkpoint, Mapping):
        return None
    config = (
        checkpoint.get("trainer_config")
        or checkpoint.get("config")
        # Check the additional configuration field.
        or checkpoint.get("baseline_config")
    )
    return dict(config) if isinstance(config, Mapping) else None


def checkpoint_signature(checkpoint: Any) -> dict[str, Any] | None:
    if not isinstance(checkpoint, Mapping):
        return None
    signature = (
        checkpoint.get("experiment_signature")
        or checkpoint.get("trainer_signature")
        # Check the additional experiment signature field.
        or checkpoint.get("baseline_experiment_signature")
    )
    return dict(signature) if isinstance(signature, Mapping) else None
