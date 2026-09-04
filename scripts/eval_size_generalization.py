#!/usr/bin/env python3
"""Measure one native-LR RefSRWKV checkpoint across crop and full-image sizes."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from RefSR_data.RefDataset import RefPNGDataset
from SR_data.SRDataset import SRPNGDataset
from models.RefSRWKV import RefSRWKV
from eval_four_settings import checkpoint_model_options, gaussian_ssim, load_weights


def build_model(options, raw):
    model = RefSRWKV(
        inp_channels=options.get("inp_channels", 3),
        out_channels=options.get("out_channels", 3),
        ref_channels=options.get("ref_channels", 3),
        dim=options.get("dim", 48),
        num_blocks=tuple(options.get("num_blocks", [4, 6, 6, 8])),
        num_refinement_blocks=options.get("num_refinement_blocks", 4),
        scale=int(options["scale"]),
        upsampler=options.get("upsampler", "progressive"),
        color_match=options.get("color_match", "global"),
        hidden_rate=options.get("hidden_rate", 4),
        windows=options.get("windows"),
    )
    load_weights(model, options["checkpoint"], use_ema=not raw)
    return model.prepare_for_inference().cuda()


def usable_names(dataset, hr_patch, limit):
    if hr_patch is None:
        return dataset.filenames[:limit]
    names = []
    for name in dataset.filenames:
        with Image.open(dataset.hr_dir / f"{name}.png") as image:
            width, height = image.size
        if height >= hr_patch and width >= hr_patch:
            names.append(name)
            if len(names) >= limit:
                break
    return names


@torch.no_grad()
def evaluate(model, data_root, split, scale, lr_size, n, workers, reference):
    hr_patch = None if lr_size is None else lr_size * scale
    dataset_class = RefPNGDataset if reference == "dataset" else SRPNGDataset
    dataset_kwargs = dict(
        data_dir=data_root,
        mode=split,
        scale=scale,
        patch_size=hr_patch,
        augment=False,
        max_samples=(None, None, n),
    )
    if dataset_class is RefPNGDataset:
        dataset_kwargs["augment_ref"] = False
    dataset = dataset_class(**dataset_kwargs)
    names = usable_names(dataset, hr_patch, n)
    if not names:
        return None, "source images are smaller than the requested HR crop"
    dataset.filenames = names
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers)
    psnr, ssim = [], []
    for batch in loader:
        lr = batch["lr"].cuda(non_blocking=True)
        hr = batch["hr"].cuda(non_blocking=True)
        expected = (lr.shape[2] * scale, lr.shape[3] * scale)
        if hr.shape[2:] != expected:
            raise ValueError(
                "dataset geometry is not native LR x scale: "
                f"LR={tuple(lr.shape[2:])}, HR={tuple(hr.shape[2:])}, scale=x{scale}"
            )
        if reference == "sisr":
            ref = F.interpolate(lr, size=expected, mode="bicubic", align_corners=False)
        else:
            ref = batch["ref"].cuda(non_blocking=True)
        output = model(lr, ref).clamp(-1, 1)
        mse = (output.float() - hr.float()).square().mean()
        psnr.append((10.0 * torch.log10(4.0 / mse.clamp_min(1e-10))).item())
        ssim.append(gaussian_ssim(output, hr).item())
    return (float(np.mean(psnr)), float(np.mean(ssim)), len(psnr)), None


def parse_lr_size(value):
    if value.lower() == "full":
        return None
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("LR size must be positive or 'full'")
    return number


def main():
    parser = argparse.ArgumentParser(
        description="Native-LR crop/full-image PSNR and SSIM self-check"
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--reference", choices=("sisr", "dataset"), default="sisr")
    parser.add_argument("--color-match", choices=("global", "none"), default=None)
    parser.add_argument(
        "--lr-sizes",
        nargs="+",
        type=parse_lr_size,
        default=[48, 64, 96, 150, None],
        help="LR crop sizes followed by optional 'full'",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("RefSRWKV size self-check requires CUDA WKV")
    if args.n < 1:
        raise ValueError("--n must be positive")

    options = checkpoint_model_options(args.ckpt)
    options["checkpoint"] = args.ckpt
    if args.color_match is not None:
        options["color_match"] = args.color_match
    scale = int(options["scale"])
    model = build_model(options, args.raw)
    print(
        f"checkpoint={args.ckpt} | scale=x{scale} | head={model.upsampler} | "
        f"color_match={model.color_match} | reference={args.reference}",
        flush=True,
    )
    print("| LR input | HR target | samples | PSNR | SSIM |", flush=True)
    print("|---:|---:|---:|---:|---:|", flush=True)
    for lr_size in args.lr_sizes:
        result, error = evaluate(
            model,
            args.data,
            args.split,
            scale,
            lr_size,
            args.n,
            args.num_workers,
            args.reference,
        )
        label = "full" if lr_size is None else str(lr_size)
        hr_label = "full" if lr_size is None else str(lr_size * scale)
        if error is not None:
            print(f"| {label} | {hr_label} | 0 | N/A | N/A | {error}", flush=True)
            continue
        value_psnr, value_ssim, samples = result
        print(
            f"| {label} | {hr_label} | {samples} | {value_psnr:.4f} | {value_ssim:.5f} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
