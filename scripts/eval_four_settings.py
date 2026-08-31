#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RefSRWKV 四设定评测（修复版）
=============================
设定: bicubic / no_ref / real_ref / perfect_ref  ×  test_easy / test_hard

为什么有这个版本（16.6 dB 异常根因）:
1. EMA shadow 只存 parameter 不存 buffer。加载时 missing=132 全部是 buffer
   （88 个 OmniShift.conv5x5_reparam_weight + 44 个 SpatialMix.scale），无害。
   本脚本分类校验：buffer 缺失放行；若有任何真实 parameter 缺失会直接中止，
   不再静默产出假指标。
2. 10b39af 之前的旧代码 reparam_5x5 漏乘 gate 因子，而 eval 模式下 OmniShift
   会自动走重参数化路径 → 输出全毁（训练 val 走训练前向，不受影响）。
   本脚本统一把 OmniShift 的评测前向绑定为 forward_train，绕过重参数化，
   在任意代码版本下结果都正确（代价是该算子约 3 倍耗时，评测可接受）。
3. 训练 val 口径 = HR patch 128 随机裁剪；加 --patch 128 可同口径对比。

用法（WSL）:
    cp /mnt/c/Users/ZHY/Desktop/RefRWKV/scripts/eval_four_settings.py ~/PROJECT/RefRWKV/
    cd ~/PROJECT/RefRWKV
    python3 eval_four_settings.py                      # 全图 × 两个 split × 各500张
    python3 eval_four_settings.py --patch 128          # 与训练 val 同口径（随机裁剪）
    python3 eval_four_settings.py --n 100              # 快测
    python3 eval_four_settings.py --raw                # 原始权重（默认 EMA）
    python3 eval_four_settings.py --splits test_easy   # 只测一个 split
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())

from models.RefSRWKV import RefSRWKV
try:
    from models.RefSRWKV import OmniShift
except ImportError:  # 极老版本没有该类名导出
    OmniShift = None
from RefSR_data.RefDataset import RefPNGDataset

PREFIXES = ("model_sr.", "model.", "generator.sr_model.", "sr_model.", "module.")
# EMA shadow 不含 buffer，这些键缺失属正常
BUFFER_SUFFIXES = ("conv5x5_reparam_weight", ".scale")
SETTINGS = ["bicubic", "no_ref", "real_ref", "perfect_ref"]


def is_omnishift(m):
    if OmniShift is not None and isinstance(m, OmniShift):
        return True
    return type(m).__name__ == "OmniShift"


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


def reparam_selfcheck(model):
    """诊断：训练前向 vs 重参数化前向是否一致（检测旧代码 reparam 漏 gate）。"""
    torch.manual_seed(0)
    checked, bad, max_err = 0, 0, 0.0
    with torch.no_grad():
        for m in model.modules():
            if is_omnishift(m) and hasattr(m, "forward_train"):
                x = torch.randn(1, m.dim, 16, 16)
                y_train = m.forward_train(x)
                m._reparam_done = False
                m.reparam_5x5()
                y_rep = F.conv2d(x, m.conv5x5_reparam_weight, padding=2, groups=m.dim)
                err = (y_train - y_rep).abs().max().item()
                max_err = max(max_err, err)
                checked += 1
                if err > 1e-4:
                    bad += 1
    if checked == 0:
        print("[selfcheck] 未找到 OmniShift 模块，跳过")
        return
    if bad == 0:
        print(f"[selfcheck] reparam 一致性 OK（{checked} 个模块，max_err={max_err:.2e}）"
              " → 代码已是 10b39af 修复版")
    else:
        print(f"[selfcheck] reparam 不一致：{bad}/{checked} 个模块，max_err={max_err:.4f}"
              " → 当前代码是 10b39af 之前的旧版（重参数化漏乘 gate）")


def bind_train_forward(model):
    """把 OmniShift 的评测前向统一绑定为训练前向，绕过重参数化路径。"""
    n = 0
    for m in model.modules():
        if is_omnishift(m) and hasattr(m, "forward_train"):
            m.forward = m.forward_train
            n += 1
    print(f"[fix] 已将 {n} 个 OmniShift 绑定为训练前向（绕过重参数化，结果与代码版本无关）")


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
    ap.add_argument("--patch", type=int, default=None,
                    help="HR patch 边长；训练 val 口径用 128。默认 None=全图")
    ap.add_argument("--raw", action="store_true", help="用原始权重而非 EMA")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RefSRWKV(inp_channels=3, out_channels=3, dim=48,
                     num_blocks=(4, 6, 6, 8), num_refinement_blocks=4,
                     scale=args.scale, drop_path_rate=0.1, hidden_rate=4)
    load_weights(model, args.ckpt, use_ema=not args.raw)
    reparam_selfcheck(model)
    bind_train_forward(model)
    model.eval().to(device)

    results = {}
    for split in args.splits:
        results[split] = run_split(model, split, args, device)

    print("\n===== 汇总（Δ vs bicubic）=====")
    for sp, r in results.items():
        b = r["bicubic"]
        print(f"  {sp}: bicubic {b:.2f} | no_ref {r['no_ref']:.2f} ({r['no_ref'] - b:+.2f}) "
              f"| real_ref {r['real_ref']:.2f} ({r['real_ref'] - b:+.2f}) "
              f"| perfect_ref {r['perfect_ref']:.2f} ({r['perfect_ref'] - b:+.2f})")
    print("\n旧锚点（修复前消融）: test_easy 23.15/23.09/23.21/23.53 | "
          "test_hard 23.05/22.95/23.08/23.42")
    print("Phase A 验收线: no_ref 设定 test_easy >= 23.65 dB（bicubic + 0.5）")


if __name__ == "__main__":
    main()
