"""SwinIR adapter under the repository-wide ``[-1, 1]`` SISR contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from .registry import BaselineAdapter, register_adapter
from .swinir_network import SwinIR


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"model.{name} must be a positive integer, got {value!r}")
    return int(value)


def _int_sequence(value: Any, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"model.{name} must be a non-empty integer list")
    return [_positive_int(item, f"{name}[{index}]") for index, item in enumerate(value)]


class SwinIRWrapper(nn.Module):
    """Map repository tensors to the original SwinIR RGB value convention."""

    def __init__(self, *, scale: int, **kwargs: Any) -> None:
        super().__init__()
        self.scale = _positive_int(scale, "scale")
        self.net = SwinIR(upscale=self.scale, **kwargs)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        if lr.ndim != 4 or lr.shape[1] != 3:
            raise ValueError(f"SwinIR expects RGB NCHW input, got {tuple(lr.shape)}")
        height, width = lr.shape[-2:]
        # The copied SwinIR implementation expects [0, 1] RGB because it
        # subtracts the official RGB mean internally.  All project loaders
        # remain in [-1, 1], including targets and metric code.
        output = self.net((lr + 1.0) * 0.5) * 2.0 - 1.0
        expected = (height * self.scale, width * self.scale)
        if output.shape[-2:] != expected:
            raise RuntimeError(
                f"SwinIR output geometry mismatch: {tuple(output.shape[-2:])} vs {expected}"
            )
        return output


class SwinIRAdapter(BaselineAdapter):
    name = "swinir"

    def build(self, model_config: Mapping[str, Any], *, scale: int) -> nn.Module:
        depths = _int_sequence(model_config.get("depths", [6, 6, 6, 6, 6, 6]), "depths")
        heads = _int_sequence(model_config.get("num_heads", [6] * len(depths)), "num_heads")
        if len(depths) != len(heads):
            raise ValueError("model.depths and model.num_heads must have the same length")

        embed_dim = _positive_int(model_config.get("embed_dim", 180), "embed_dim")
        for index, head_count in enumerate(heads):
            if embed_dim % head_count:
                raise ValueError(
                    f"model.num_heads[{index}] ({head_count}) must divide "
                    f"model.embed_dim ({embed_dim})"
                )
        window_size = _positive_int(model_config.get("window_size", 8), "window_size")
        img_size = _positive_int(model_config.get("img_size", 48), "img_size")
        upsampler = str(model_config.get("upsampler", "pixelshuffle")).lower()
        if upsampler not in {"pixelshuffle", "pixelshuffledirect", "nearest+conv"}:
            raise ValueError(f"unsupported SwinIR upsampler: {upsampler}")
        resi_connection = str(model_config.get("resi_connection", "1conv")).lower()
        if resi_connection not in {"1conv", "3conv"}:
            raise ValueError(f"unsupported SwinIR resi_connection: {resi_connection}")

        return SwinIRWrapper(
            scale=scale,
            img_size=img_size,
            patch_size=1,
            in_chans=3,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=heads,
            window_size=window_size,
            mlp_ratio=float(model_config.get("mlp_ratio", 2.0)),
            qkv_bias=bool(model_config.get("qkv_bias", True)),
            qk_scale=None,
            drop_rate=float(model_config.get("drop_rate", 0.0)),
            attn_drop_rate=float(model_config.get("attn_drop_rate", 0.0)),
            drop_path_rate=float(model_config.get("drop_path_rate", 0.0)),
            ape=bool(model_config.get("ape", False)),
            patch_norm=bool(model_config.get("patch_norm", True)),
            use_checkpoint=bool(model_config.get("use_checkpoint", False)),
            img_range=1.0,
            upsampler=upsampler,
            resi_connection=resi_connection,
        )

    def describe(self, model_config: Mapping[str, Any], *, scale: int) -> dict[str, Any]:
        return {
            "name": self.name,
            "variant": str(model_config.get("variant", "custom")),
            "scale": int(scale),
            "img_size": int(model_config.get("img_size", 48)),
            "window_size": int(model_config.get("window_size", 8)),
            "embed_dim": int(model_config.get("embed_dim", 180)),
            "depths": list(model_config.get("depths", [6, 6, 6, 6, 6, 6])),
            "num_heads": list(model_config.get("num_heads", [6, 6, 6, 6, 6, 6])),
            "mlp_ratio": float(model_config.get("mlp_ratio", 2.0)),
            "upsampler": str(model_config.get("upsampler", "pixelshuffle")),
            "resi_connection": str(model_config.get("resi_connection", "1conv")),
        }


register_adapter(SwinIRAdapter())
