"""Memory-bounded tiled inference for aligned image-to-image models.

Wuhan's TIFFs are 1000×1000 on the model grid.  RefSRWKV's local matching
uses an unfolded feature tensor, so evaluating the whole grid at once is far
more memory hungry than a training crop.  This module reconstructs the full
output from overlapping input tiles without changing the supervised grid.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def _starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    final = length - tile_size
    values = list(range(0, final + 1, stride))
    if values[-1] != final:
        values.append(final)
    return values


def _blend_window(
    height: int,
    width: int,
    overlap: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a non-zero feathering window for stable overlap averaging."""
    def axis(length: int) -> torch.Tensor:
        values = torch.ones(length, device=device, dtype=dtype)
        edge = min(int(overlap), length // 2)
        if edge:
            # Keep the endpoint non-zero: a border covered by one tile must
            # retain its exact prediction rather than divide by a tiny number.
            ramp = torch.linspace(0.1, 1.0, edge, device=device, dtype=dtype)
            values[:edge] = ramp
            values[-edge:] = torch.flip(ramp, dims=(0,))
        return values

    return axis(height)[:, None].mul(axis(width)[None, :]).view(1, 1, height, width)


def tiled_forward(
    forward: Callable[..., torch.Tensor],
    *inputs: torch.Tensor,
    scale: int = 1,
    tile_size: int | None = None,
    overlap: int = 0,
) -> torch.Tensor:
    """Apply ``forward`` to aligned NCHW tiles and blend its full output.

    Inputs must share batch and spatial geometry.  The output is required to
    have the same batch size and an integer ``scale`` times the tile height and
    width; this covers both SISR and scale-one aligned sensor fusion.
    """
    if not inputs:
        raise ValueError("tiled_forward requires at least one input tensor")
    if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be a positive integer")
    if tile_size is None:
        return forward(*inputs)
    if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError("tile_size must be a positive integer or None")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be an integer in [0, tile_size)")
    if any(not torch.is_tensor(value) or value.ndim != 4 for value in inputs):
        raise TypeError("tiled_forward inputs must be NCHW tensors")

    primary = inputs[0]
    batch, _channels, height, width = primary.shape
    if any(value.shape[0] != batch or value.shape[-2:] != (height, width) for value in inputs[1:]):
        raise ValueError("all tiled_forward inputs must share batch and spatial geometry")
    if height <= tile_size and width <= tile_size:
        return forward(*inputs)

    stride = tile_size - overlap
    y_starts, x_starts = _starts(height, tile_size, stride), _starts(width, tile_size, stride)
    output_sum: torch.Tensor | None = None
    weight_sum: torch.Tensor | None = None
    for y in y_starts:
        tile_height = min(tile_size, height - y)
        for x in x_starts:
            tile_width = min(tile_size, width - x)
            tile_inputs = tuple(value[..., y : y + tile_height, x : x + tile_width] for value in inputs)
            tile_output = forward(*tile_inputs)
            expected = (tile_height * scale, tile_width * scale)
            if (
                not torch.is_tensor(tile_output)
                or tile_output.ndim != 4
                or tile_output.shape[0] != batch
                or tuple(tile_output.shape[-2:]) != expected
            ):
                actual = tuple(tile_output.shape) if torch.is_tensor(tile_output) else type(tile_output).__name__
                raise ValueError(
                    "tiled forward output must be NCHW with spatial size "
                    f"tile * x{scale}; got {actual}, expected spatial {expected}"
                )
            if output_sum is None:
                output_sum = tile_output.new_zeros(
                    (batch, tile_output.shape[1], height * scale, width * scale)
                )
                weight_sum = tile_output.new_zeros((1, 1, height * scale, width * scale))
            assert weight_sum is not None
            window = _blend_window(
                tile_output.shape[-2],
                tile_output.shape[-1],
                overlap * scale,
                device=tile_output.device,
                dtype=tile_output.dtype,
            )
            y_out, x_out = y * scale, x * scale
            output_sum[..., y_out : y_out + expected[0], x_out : x_out + expected[1]].add_(tile_output * window)
            weight_sum[..., y_out : y_out + expected[0], x_out : x_out + expected[1]].add_(window)
    assert output_sum is not None and weight_sum is not None
    return output_sum / weight_sum.clamp_min(torch.finfo(output_sum.dtype).eps)


__all__ = ["tiled_forward"]
