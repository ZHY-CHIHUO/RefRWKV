#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RefSRWKV 评测：bicubic、SISR Ref、数据集 Ref 和 HR 参考上限。"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.RefSRWKV import RefSRWKV
from RefSR_data.RefDataset import RefPNGDataset

PREFIXES = ("model_sr.", "model.", "generator.sr_model.", "sr_model.", "module.")
# EMA shadow 不含 buffer，这些键缺失属正常
BUFFER_SUFFIXES = ("conv5x5_reparam_weight", ".scale")
SETTINGS = ["bicubic", "sisr_ref", "dataset_ref", "perfect_ref"]


def gaussian_ssim(pred, target):
    """Per-image RGB SSIM on [-1, 1] tensors using an 11x11 Gaussian window."""
    if pred.shape != target.shape:
        raise ValueError(
            f"SSIM 输入形状不一致: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    channels, window_size, sigma = pred.shape[1], 11, 1.5
    pred_f, target_f = pred.float(), target.float()
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
    gaussian = torch.exp(-((coords - window_size // 2) ** 2) / (2.0 * sigma**2))
    window_2d = torch.outer(gaussian, gaussian)
    window = (window_2d / window_2d.sum()).view(1, 1, window_size, window_size)
    window = window.expand(channels, 1, -1, -1).contiguous()
    pad = window_size // 2
    mu_p = F.conv2d(pred_f, window, padding=pad, groups=channels)
    mu_t = F.conv2d(target_f, window, padding=pad, groups=channels)
    sigma_p_sq = (
        F.conv2d(pred_f.square(), window, padding=pad, groups=channels) - mu_p.square()
    ).clamp_min(0.0)
    sigma_t_sq = (
        F.conv2d(target_f.square(), window, padding=pad, groups=channels)
        - mu_t.square()
    ).clamp_min(0.0)
    sigma_pt = (
        F.conv2d(pred_f * target_f, window, padding=pad, groups=channels) - mu_p * mu_t
    )
    c1, c2 = (0.01 * 2.0) ** 2, (0.03 * 2.0) ** 2
    numerator = (2.0 * mu_p * mu_t + c1) * (2.0 * sigma_pt + c2)
    denominator = (mu_p.square() + mu_t.square() + c1) * (sigma_p_sq + sigma_t_sq + c2)
    return (
        (numerator / denominator.clamp_min(1e-12)).clamp(-1.0, 1.0).mean(dim=(1, 2, 3))
    )


def strip_prefix(sd):
    out = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        for p in PREFIXES:
            if k.startswith(p):
                k = k[len(p) :]
                break
        out[k] = v
    return out


def load_weights(model, ckpt_path, use_ema):
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    ema_sd = ckpt.get("ema_state_dict") if isinstance(ckpt, dict) else None
    if use_ema and isinstance(ema_sd, dict) and ema_sd.get("shadow"):
        sd, tag = strip_prefix(ema_sd["shadow"]), "ema"
    else:
        sd, tag = strip_prefix(ckpt.get("state_dict", ckpt)), "raw"

    missing, unexpected = model.load_state_dict(sd, strict=False)
    buf_miss = [k for k in missing if k.endswith(BUFFER_SUFFIXES)]
    real_miss = [k for k in missing if not k.endswith(BUFFER_SUFFIXES)]
    epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    print(f"[load] {ckpt_path}")
    print(
        f"       weights={tag} | epoch={epoch} | "
        f"missing={len(missing)} (buffer {len(buf_miss)} / 参数 {len(real_miss)}) "
        f"| unexpected={len(unexpected)}"
    )
    if unexpected:
        print(f"       unexpected 示例: {list(unexpected)[:5]}")
    if real_miss:
        print(
            f"[FATAL] {len(real_miss)} 个真实参数未加载（非 buffer），结果必然错误，中止。"
        )
        print(f"        示例: {real_miss[:10]}")
        sys.exit(1)


def checkpoint_model_options(ckpt_path):
    """Read the native-LR model contract saved with a checkpoint."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    signature = (
        ckpt.get("refsrwkv_experiment_signature") if isinstance(ckpt, dict) else None
    )
    options = {}
    if isinstance(signature, dict):
        for key in (
            "scale",
            "inp_channels",
            "out_channels",
            "dim",
            "hidden_rate",
            "ref_channels",
            "num_blocks",
            "num_refinement_blocks",
            "upsampler",
            "color_match",
            "windows",
        ):
            if signature.get(key) is not None:
                options[key] = signature[key]

    # ``train_config.yaml`` is retained beside checkpoints as a readable
    # fallback, but native-LR checkpoints write the same contract in-signature.
    config_path = Path(ckpt_path).parent / "train_config.yaml"
    if config_path.is_file():
        try:
            with config_path.open("r", encoding="utf-8-sig") as file_obj:
                train_cfg = yaml.safe_load(file_obj)
        except (OSError, yaml.YAMLError):
            train_cfg = None
        if isinstance(train_cfg, dict):
            data_cfg = train_cfg.get("data")
            model_cfg = train_cfg.get("model")
            if isinstance(data_cfg, dict):
                if "scale" not in options and data_cfg.get("scale") is not None:
                    options["scale"] = int(data_cfg["scale"])
            if isinstance(model_cfg, dict):
                for key in (
                    "inp_channels",
                    "out_channels",
                    "ref_channels",
                    "dim",
                    "num_blocks",
                    "num_refinement_blocks",
                    "hidden_rate",
                    "upsampler",
                    "color_match",
                    "windows",
                ):
                    if key not in options and model_cfg.get(key) is not None:
                        options[key] = model_cfg[key]
    if "scale" not in options:
        raise ValueError("checkpoint 未记录原生 LR scale，无法安全评测")
    return options


@torch.no_grad()
def run_split(model, split, args, device):
    ds = RefPNGDataset(
        data_dir=args.data,
        mode=split,
        scale=args.scale,
        patch_size=args.patch,
        augment=False,
        augment_ref=False,
    )
    n = min(args.n, len(ds))
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    acc = {s: {"psnr": [], "ssim": []} for s in args.settings}
    t0 = time.time()
    tag = f"patch{args.patch}" if args.patch else "full"
    print(f"\n>>> {split} [{tag}] 开始（{n} 张，batch={args.batch_size}）", flush=True)
    seen = 0
    last_log = 0
    for batch in loader:
        remaining = n - seen
        if remaining <= 0:
            break
        lr = batch["lr"][:remaining].to(device, non_blocking=True)
        hr = batch["hr"][:remaining].to(device, non_blocking=True)
        # The SISR setting is defined by the runtime bicubic LR upsample,
        # independent of interpolation details or stale Ref files.
        H, W = hr.shape[2:]
        if (H, W) != (lr.shape[2] * args.scale, lr.shape[3] * args.scale):
            raise ValueError(
                "数据的 HR 尺寸必须严格等于 LR x scale: "
                f"LR={tuple(lr.shape[2:])}, HR={(H, W)}, scale=x{args.scale}"
            )
        lr_up = F.interpolate(lr, size=(H, W), mode="bicubic", align_corners=False)
        outs = {"bicubic": lr_up}
        if "sisr_ref" in args.settings:
            outs["sisr_ref"] = model(lr, lr_up).clamp(-1, 1)
        if "dataset_ref" in args.settings:
            ref = batch["ref"][:remaining].to(device, non_blocking=True)
            if ref.shape != hr.shape:
                raise ValueError(
                    "dataset_ref requires paired Ref and HR with identical shapes: "
                    f"{tuple(ref.shape)} vs {tuple(hr.shape)}"
                )
            outs["dataset_ref"] = model(lr, ref).clamp(-1, 1)
        if "perfect_ref" in args.settings:
            outs["perfect_ref"] = model(lr, hr).clamp(-1, 1)
        for s, o in outs.items():
            if s not in acc:
                continue
            mse = (o.float() - hr.float()).square().mean(dim=(1, 2, 3))
            acc[s]["psnr"].extend(
                (10.0 * torch.log10(4.0 / mse.clamp_min(1e-10))).cpu().tolist()
            )
            acc[s]["ssim"].extend(gaussian_ssim(o, hr).cpu().tolist())
        seen += lr.size(0)
        if seen - last_log >= 50 or seen == n:
            run = "  ".join(
                f"{s}=PSNR {np.mean(acc[s]['psnr']):.2f} / SSIM {np.mean(acc[s]['ssim']):.4f}"
                for s in args.settings
            )
            print(
                f"  [{split}] {seen}/{n}  {(time.time() - t0) / 60:.1f}min  {run}",
                flush=True,
            )
            last_log = seen
    return {
        s: {metric: float(np.mean(values)) for metric, values in metrics.items()}
        for s, metrics in acc.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="RefSR_data/HRMS_SCD")
    ap.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="默认优先使用 test；若数据集没有 test，则使用 test_easy/test_hard",
    )
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument(
        "--settings",
        nargs="+",
        choices=SETTINGS,
        default=SETTINGS,
        help="要评测的设置；默认 bicubic、sisr_ref、dataset_ref、perfect_ref 全部执行",
    )
    ap.add_argument(
        "--scale",
        type=int,
        default=None,
        help="SR scale; omit to read the checkpoint signature",
    )
    ap.add_argument(
        "--patch",
        type=int,
        default=None,
        help="评测时的 HR 裁剪边长；默认 None 表示全图",
    )
    ap.add_argument("--raw", action="store_true", help="用原始权重而非 EMA")
    ap.add_argument("--window-size", type=int, default=None)
    ap.add_argument("--shift-size", type=int, default=None)
    ap.add_argument("--shift-cycle", type=int, default=None)
    ap.add_argument("--upsampler", choices=("progressive", "direct"), default=None)
    ap.add_argument("--color-match", choices=("global", "none"), default=None)
    ap.add_argument("--window-phase-mode", choices=("local", "global"), default=None)
    args = ap.parse_args()

    if args.splits is None:
        available = {
            name
            for name in ("test", "test_easy", "test_hard")
            if os.path.isdir(os.path.join(args.data, name))
        }
        args.splits = (
            ["test"]
            if "test" in available
            else [name for name in ("test_easy", "test_hard") if name in available]
        )
        if not args.splits:
            raise FileNotFoundError(
                f"未找到可用测试 split: {args.data}/test 或 test_easy/test_hard"
            )

    ckpt_options = checkpoint_model_options(args.ckpt)
    if args.scale is None:
        args.scale = ckpt_options.get("scale", 4)
    windows = ckpt_options.get("windows")
    if (
        args.window_size is not None
        or args.shift_size is not None
        or args.shift_cycle is not None
        or args.window_phase_mode is not None
    ):
        windows = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RefSRWKV(
        inp_channels=ckpt_options.get("inp_channels", 3),
        out_channels=ckpt_options.get("out_channels", 3),
        ref_channels=ckpt_options.get("ref_channels", 3),
        dim=ckpt_options.get("dim", 48),
        num_blocks=tuple(ckpt_options.get("num_blocks", [4, 6, 6, 8])),
        num_refinement_blocks=ckpt_options.get("num_refinement_blocks", 4),
        scale=args.scale,
        upsampler=(
            args.upsampler
            if args.upsampler is not None
            else ckpt_options.get("upsampler", "progressive")
        ),
        color_match=(
            args.color_match
            if args.color_match is not None
            else ckpt_options.get("color_match", "global")
        ),
        drop_path_rate=0.1,
        hidden_rate=ckpt_options.get("hidden_rate", 4),
        windows=windows,
        window_size=args.window_size if args.window_size is not None else 8,
        shift_size=args.shift_size if args.shift_size is not None else 3,
        shift_cycle=args.shift_cycle if args.shift_cycle is not None else 3,
        window_phase_mode=args.window_phase_mode,
    )
    print(
        "[model] "
        f"native LR grid | scale=x{args.scale} | "
        f"ref/output=x{model.ref_down_factor} | head={model.upsampler} | "
        f"color_match={model.color_match}",
        flush=True,
    )
    load_weights(model, args.ckpt, use_ema=not args.raw)
    model.prepare_for_inference().to(device)

    results = {}
    for split in args.splits:
        results[split] = run_split(model, split, args, device)

    print("\n===== 汇总（Δ vs bicubic）=====")
    for sp, r in results.items():
        b = r.get("bicubic")
        parts = []
        for setting, metric in r.items():
            text = f"{setting} PSNR {metric['psnr']:.2f} / SSIM {metric['ssim']:.4f}"
            if b is not None and setting != "bicubic":
                text += f" (ΔPSNR {metric['psnr'] - b['psnr']:+.2f})"
            parts.append(text)
        print(f"  {sp}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
