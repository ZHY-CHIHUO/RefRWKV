#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RefSRWKV 四设定评测：bicubic、无参考、真实参考、理想参考。"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())

from models.RefSRWKV import RefSRWKV
from RefSR_data.RefDataset import RefPNGDataset

PREFIXES = ("model_sr.", "model.", "generator.sr_model.", "sr_model.", "module.")
# EMA shadow 不含 buffer，这些键缺失属正常
BUFFER_SUFFIXES = ("conv5x5_reparam_weight", ".scale")
SETTINGS = ["bicubic", "no_ref", "real_ref", "perfect_ref"]

def strip_prefix(sd):
    out = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        for p in PREFIXES:
            if k.startswith(p):
                k = k[len(p):]
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
    print(f"       weights={tag} | epoch={epoch} | "
          f"missing={len(missing)} (buffer {len(buf_miss)} / 参数 {len(real_miss)}) "
          f"| unexpected={len(unexpected)}")
    if unexpected:
        print(f"       unexpected 示例: {list(unexpected)[:5]}")
    if real_miss:
        print(f"[FATAL] {len(real_miss)} 个真实参数未加载（非 buffer），结果必然错误，中止。")
        print(f"        示例: {real_miss[:10]}")
        sys.exit(1)

@torch.no_grad()
def run_split(model, split, args, device):
    ds = RefPNGDataset(data_dir=args.data, mode=split, scale=args.scale,
                       patch_size=args.patch, augment=False, augment_ref=False)
    n = min(args.n, len(ds))
    acc = {s: [] for s in SETTINGS}
    t0 = time.time()
    tag = f"patch{args.patch}" if args.patch else "full"
    print(f"\n>>> {split} [{tag}] 开始（{n} 张）", flush=True)
    for i in range(n):
        it = ds[i]
        lr = it["lr"].unsqueeze(0).to(device)
        hr = it["hr"].unsqueeze(0).to(device)
        ref = it["ref"].unsqueeze(0).to(device)
        H, W = hr.shape[2:]
        lr_up = F.interpolate(lr, size=(H, W), mode="bicubic", align_corners=False)
        outs = {
            "bicubic": lr_up,
            "no_ref": model(lr, lr_up).clamp(-1, 1),
            "real_ref": model(lr, ref).clamp(-1, 1),
            "perfect_ref": model(lr, hr).clamp(-1, 1),
        }
        for s, o in outs.items():
            mse = F.mse_loss(o.float(), hr.float()).item()
            acc[s].append(10.0 * np.log10(4.0 / max(mse, 1e-10)))
        if (i + 1) % 50 == 0 or (i + 1) == n:
            run = "  ".join(f"{s}={np.mean(acc[s]):.2f}" for s in SETTINGS)
            print(f"  [{split}] {i + 1}/{n}  {(time.time() - t0) / 60:.1f}min  {run}",
                  flush=True)
    return {s: float(np.mean(acc[s])) for s in SETTINGS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/refrwkv_sr_4/last.ckpt")
    ap.add_argument("--data", default="RefSR_data/HRMS_SCD")
    ap.add_argument("--splits", nargs="+", default=["test_easy", "test_hard"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--hr_size", type=int, default=512,
                    help="SR checkpoint 训练时的 HR patch 边长；sr_prior_10 使用 480")
    ap.add_argument("--patch", type=int, default=None,
                    help="评测时的 HR 裁剪边长；默认 None 表示全图")
    ap.add_argument("--raw", action="store_true", help="用原始权重而非 EMA")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RefSRWKV(inp_channels=3, out_channels=3, dim=48,
                     num_blocks=(4, 6, 6, 8), num_refinement_blocks=4,
                     scale=args.scale, hr_size=args.hr_size,
                     drop_path_rate=0.1, hidden_rate=4)
    load_weights(model, args.ckpt, use_ema=not args.raw)
    model.prepare_for_inference().to(device)

    results = {}
    for split in args.splits:
        results[split] = run_split(model, split, args, device)

    print("\n===== 汇总（Δ vs bicubic）=====")
    for sp, r in results.items():
        b = r["bicubic"]
        print(f"  {sp}: bicubic {b:.2f} | no_ref {r['no_ref']:.2f} ({r['no_ref'] - b:+.2f}) "
              f"| real_ref {r['real_ref']:.2f} ({r['real_ref'] - b:+.2f}) "
              f"| perfect_ref {r['perfect_ref']:.2f} ({r['perfect_ref'] - b:+.2f})")
if __name__ == "__main__":
    main()
