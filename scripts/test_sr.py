#!/usr/bin/env python
"""
RefSRWKV 测试脚本 — 对比两种前缀剥离逻辑，确认 SR prior 加载问题。
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RefRWKV.models.RefSRWKV import RefSRWKV
from RefRWKV.RefSR_data.RefSR_dataset import RefLMDBDataset
from RefRWKV.evaluation.eval_pyiqa import IQAEngine


def _build_raw_model():
    return RefSRWKV(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=(4, 6, 6, 8), num_refinement_blocks=4,
        scale=10, drop_path_rate=0.1, hidden_rate=4,
    )


def load_model_old(ckpt_path: str, device: torch.device):
    """旧版剥离逻辑：只处理 module. 和 model.，不处理 model_sr.。
       这就是训练脚本 build_sr_model 原来的逻辑。"""
    model = _build_raw_model()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    clean = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        if k.startswith("model."):
            k = k[len("model."):]
        clean[k] = v

    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"  [OLD] Missing: {len(missing)}  |  Unexpected: {len(unexpected)}")
    if missing:
        print(f"        First missing: {missing[:3]}")
    model.to(device).eval()
    return model


def load_model_new(ckpt_path: str, device: torch.device):
    """新版剥离逻辑：model_sr. → model. → module.，顺序正确。"""
    model = _build_raw_model()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    clean = {}
    for k, v in state_dict.items():
        if k.startswith("model_sr."):
            k = k[len("model_sr."):]
        elif k.startswith("model."):
            k = k[len("model."):]
        k = k.replace("module.", "")
        clean[k] = v

    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"  [NEW] Missing: {len(missing)}  |  Unexpected: {len(unexpected)}")
    if missing:
        print(f"        First missing: {missing[:3]}")
    model.to(device).eval()
    return model


@torch.no_grad()
def evaluate(model, loader, iqa, device):
    agg = {}
    count = 0
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        if isinstance(batch, dict):
            lr = batch["lr"].to(device).float()
            hr = batch["hr"].to(device).float()
            ref = batch["ref"].to(device).float()
        else:
            lr, hr, ref = [x.to(device).float() for x in batch]

        sr = model(lr, ref)
        sr_01 = torch.clamp((sr + 1.0) / 2.0, 0.0, 1.0)
        hr_01 = torch.clamp((hr + 1.0) / 2.0, 0.0, 1.0)

        m = iqa.evaluate_single(sr_01[0].cpu().numpy(), hr_01[0].cpu().numpy())
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v
        count += 1
    return {k: v / count for k, v in agg.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="/home/zhy/PROJECT/RefRWKV/checkpoints/refrwkv_sr/last.ckpt")
    parser.add_argument("--data_root", type=str,
                        default="/home/zhy/PROJECT/RefRWKV/RefSR_data/ALL_2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results/test_sr")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    iqa = IQAEngine(device=str(device), nr_metrics=[],
                    fr_metrics=["psnr", "ssim", "lpips", "dists"],
                    use_y_channel=True, verbose=False)

    for mode in ["val"]:   # 只跑 val 就行，省时间
        loader = DataLoader(
            RefLMDBDataset(mode=mode, data_dir=args.data_root, patch_size=480, scale=10,
                           augment=False, augment_ref=False,
                           ref_aug_strengths=[0.12]*4, ref_aug_probs=[0.5]*4,
                           ref_gray_prob=0.0, max_samples=(None,None,None),
                           sample_seed=42, lr_key="lr", hr_key="hr", ref_key="ref"),
            batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        print(f"\n{'='*60}")
        print(f"  Mode: {mode}  ({len(loader.dataset)} samples)")
        print(f"{'='*60}")

        # 1. 旧版剥离逻辑（build_sr_model 原来的逻辑）
        print("\n--- OLD stripping (no model_sr.) ---")
        model_old = load_model_old(args.ckpt, device)
        res_old = evaluate(model_old, loader, iqa, device)
        for k, v in res_old.items():
            print(f"  [OLD] {k}: {v:.5f}")

        # 2. 新版剥离逻辑（正确版本）
        print("\n--- NEW stripping (model_sr. → model. → module.) ---")
        model_new = load_model_new(args.ckpt, device)
        res_new = evaluate(model_new, loader, iqa, device)
        for k, v in res_new.items():
            print(f"  [NEW] {k}: {v:.5f}")

        # 3. 对比
        print("\n--- Diff ---")
        for k in res_old:
            diff = res_new[k] - res_old[k]
            print(f"  Δ {k}: {diff:+.5f}  {'⚠️ 新>>旧，确认权重加载不全' if diff > 1 else ''}")

    print(f"\n{'='*60}")
    print("  ✅ Done")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
