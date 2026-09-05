#!/usr/bin/env python3
"""CUDA smoke test for native-LR RefSRWKV geometry and Bi-WKV backpropagation."""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.refsr.refsrwkv.model import RMSNorm2d, RefSRWKV


def check_rmsnorm_size_independence():
    norm = RMSNorm2d(3).cuda().eval()
    small = torch.randn(1, 3, 5, 7, device="cuda")
    large = torch.cat((small, torch.randn(1, 3, 5, 11, device="cuda")), dim=3)
    if not torch.equal(norm(small), norm(large)[..., : small.shape[-1]]):
        raise AssertionError("RMSNorm2d must not depend on neighboring spatial pixels")


def check_case(scale, upsampler, lr_height, lr_width, backward):
    model = RefSRWKV(
        # C=16 is the smallest channel count accepted by the CUDA kernel and
        # keeps this development smoke test quick. Production runs use C=48.
        dim=16,
        num_blocks=(1, 1, 1, 1),
        num_refinement_blocks=1,
        scale=scale,
        upsampler=upsampler,
    ).cuda()
    model.train()
    # The production residual starts at bicubic. Make this smoke-test head
    # nonzero so gradients traverse the complete RWKV path.
    with torch.no_grad():
        model.output_conv.weight.normal_(mean=0.0, std=1e-3)
    lr = torch.randn(1, 3, lr_height, lr_width, device="cuda") * 0.1
    ref = torch.randn(
        1, 3, lr_height * scale, lr_width * scale, device="cuda"
    ) * 0.1
    output = model(lr, ref)
    expected = (1, 3, lr_height * scale, lr_width * scale)
    if tuple(output.shape) != expected:
        raise AssertionError(f"x{scale} output {tuple(output.shape)} != {expected}")
    if not torch.isfinite(output).all():
        raise AssertionError(f"x{scale} forward produced non-finite output")
    if backward:
        output.square().mean().backward()
        grad = model.encoder_level1[0].att.key.weight.grad
        if grad is None or not torch.isfinite(grad).all():
            raise AssertionError(f"x{scale} CUDA WKV backward gradient is invalid")
    try:
        model(lr, ref[..., :-1, :])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched Ref geometry must raise ValueError")
    torch.cuda.synchronize()
    print(
        f"x{scale} {upsampler}: LR {lr_height}x{lr_width} -> "
        f"HR {output.shape[2]}x{output.shape[3]} {'forward+backward' if backward else 'forward'} OK",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("x2", "x3", "x4", "x10"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires the CUDA Bi-WKV implementation")
    check_rmsnorm_size_independence()
    # Non-multiples of eight exercise synchronized LR/Ref padding and crop.
    cases = {
        "x2": (2, "progressive", 47, 51, False),
        "x3": (3, "progressive", 49, 50, False),
        "x4": (4, "progressive", 53, 48, True),
        "x10": (10, "direct", 48, 48, False),
    }
    case_names = (args.only,) if args.only else tuple(cases)
    for name in case_names:
        check_case(*cases[name])
    print("native geometry CUDA smoke test passed", flush=True)


if __name__ == "__main__":
    main()
