"""Lazy Python binding for the shared Bi-WKV CUDA extension.

The C++/CUDA sources intentionally live next to this module rather than in a
specific model directory.  Importing this module never compiles the extension;
the first CUDA WKV call does.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_CUDA_DIR = Path(__file__).resolve().parent
_wkv_cuda = None
_wkv_load_error: Exception | None = None


def _get_wkv_cuda():
    """Compile and return the process-local Bi-WKV extension on demand."""
    global _wkv_cuda, _wkv_load_error
    if _wkv_cuda is not None:
        return _wkv_cuda
    if _wkv_load_error is not None:
        raise RuntimeError("Bi-WKV CUDA extension failed to load") from _wkv_load_error
    if not torch.cuda.is_available():
        raise RuntimeError("Bi-WKV requires a CUDA environment")

    cpp_source = _CUDA_DIR / "bi_wkv.cpp"
    cuda_source = _CUDA_DIR / "bi_wkv_kernel.cu"
    if not cpp_source.is_file() or not cuda_source.is_file():
        raise FileNotFoundError(f"Bi-WKV CUDA sources are missing from {_CUDA_DIR}")

    capability = torch.cuda.get_device_capability()
    arch = f"compute_{capability[0]}{capability[1]}"
    sm = f"sm_{capability[0]}{capability[1]}"
    try:
        _wkv_cuda = load(
            name="bi_wkv",
            sources=[str(cpp_source), str(cuda_source)],
            verbose=True,
            extra_cuda_cflags=[
                "-res-usage",
                "--maxrregcount 60",
                "--use_fast_math",
                "-O3",
                "-Xptxas -O3",
                f"-gencode arch={arch},code={sm}",
                f"-gencode arch={arch},code={arch}",
            ],
        )
    except Exception as exc:
        _wkv_load_error = exc
        raise RuntimeError(
            f"Bi-WKV CUDA extension failed to compile/load (sm_{capability[0]}{capability[1]})"
        ) from exc
    return _wkv_cuda


try:
    _compiler_disable = torch.compiler.disable
except AttributeError:

    def _compiler_disable(fn=None, **kwargs):
        return fn if fn is not None else (lambda function: function)


class WKV(torch.autograd.Function):
    """Autograd wrapper around the shared Bi-WKV CUDA operator."""

    @staticmethod
    def forward(ctx, w, u, k, v):
        half_mode = w.dtype == torch.half
        bf_mode = w.dtype == torch.bfloat16
        ctx.save_for_backward(w, u, k, v)
        y = _get_wkv_cuda().bi_wkv_forward(
            w.float().contiguous(),
            u.float().contiguous(),
            k.float().contiguous(),
            v.float().contiguous(),
        )
        return y.half() if half_mode else (y.bfloat16() if bf_mode else y)

    @staticmethod
    def backward(ctx, grad_output):
        w, u, k, v = ctx.saved_tensors
        half_mode = w.dtype == torch.half
        bf_mode = w.dtype == torch.bfloat16
        grad_w, grad_u, grad_k, grad_v = _get_wkv_cuda().bi_wkv_backward(
            w.float().contiguous(),
            u.float().contiguous(),
            k.float().contiguous(),
            v.float().contiguous(),
            grad_output.float().contiguous(),
        )
        if half_mode:
            return grad_w.half(), grad_u.half(), grad_k.half(), grad_v.half()
        if bf_mode:
            return (
                grad_w.bfloat16(),
                grad_u.bfloat16(),
                grad_k.bfloat16(),
                grad_v.bfloat16(),
            )
        return grad_w, grad_u, grad_k, grad_v


@_compiler_disable()
def RUN_CUDA(w, u, k, v):
    """Run the Bi-WKV operator with the legacy float32 call contract."""
    return WKV.apply(w.float(), u.float(), k.float(), v.float())


__all__ = ["RUN_CUDA", "WKV"]
