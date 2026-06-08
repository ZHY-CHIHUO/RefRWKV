#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估 Restore_RWKV_Ref 模型在测试集上的性能（基于 pyiqa）
用法:
    python eval.py --weight best_restore_rwkv_ref.pth --data_root /path/to/ALL_2 [--save_images]
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from Restore_RWKV import Restore_RWKV_Ref
from RWKV.RefSR_data.RefSR_dataset import RefPNGDataset
from RWKV.evaluation.eval_pyiqa import IQAEngine   # 修正导入路径

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Restore_RWKV_Ref on test set")
    parser.add_argument("--weight", type=str, required=True, help="Path to the trained model weights (.pth)")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--scale", type=int, default=10, help="Super-resolution scale factor")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of test samples")
    parser.add_argument("--save_images", action="store_true", help="Save predicted HR images to disk")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save results")
    parser.add_argument("--nr_metrics", nargs="+", default=["clipiqa", "musiq", "niqe"],
                        help="No-reference metrics to compute")
    parser.add_argument("--no_nr", action="store_true", help="Skip all no-reference metrics")
    return parser.parse_args()

def collate_fn(batch):
    lr_list, hr_list, ref_list = zip(*batch)
    lr = torch.stack(lr_list, 0)
    hr = torch.stack(hr_list, 0)
    ref = torch.stack(ref_list, 0)
    return lr, ref, lr, hr

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 数据集
    test_dataset = RefPNGDataset(
        data_dir=args.data_root, mode="test", patch_size=None, scale=args.scale,
        augment=False, max_samples=(None, None, args.max_samples), sample_seed=42
    )
    print(f"Test samples: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    # 模型
    model = Restore_RWKV_Ref(inp_channels=3, out_channels=3, scale=args.scale).to(device)
    checkpoint = torch.load(args.weight, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Model loaded from {args.weight}")

    # ---------- 使用 IQAEngine 代替原来的指标初始化 ----------
    nr_metrics_list = [] if args.no_nr else args.nr_metrics
    fr_metrics_list = ['psnr', 'ssim', 'lpips', 'dists']
    engine = IQAEngine(device=args.device, nr_metrics=nr_metrics_list, fr_metrics=fr_metrics_list,
                       use_y_channel=True, verbose=False)

    total_results = {k: 0.0 for k in nr_metrics_list + fr_metrics_list}
    total_samples = 0

    # ---------- 推理 ----------
    with torch.no_grad():
        for batch_idx, (lr1, ref, lr2, hr_gt) in enumerate(tqdm(test_loader, desc="Testing")):
            lr1 = lr1.to(device)
            ref = ref.to(device)
            lr2 = lr2.to(device)
            hr_gt = hr_gt.to(device)

            pred = model(lr1, ref, lr2)

            for i in range(pred.size(0)):
                pred_i = pred[i:i+1]
                gt_i = hr_gt[i:i+1]

                # 单张评估，得到指标字典
                res = engine.evaluate_single(pred_i, gt_i)
                for k, v in res.items():
                    total_results[k] += v
                total_samples += 1

                # 保存图像
                if args.save_images:
                    img_np = pred_i.squeeze(0).cpu().numpy().transpose(1,2,0)
                    img_np = (img_np * 255).clip(0,255).astype(np.uint8)
                    save_path = os.path.join(args.output_dir, f"pred_{batch_idx}_{i}.png")
                    from PIL import Image
                    Image.fromarray(img_np).save(save_path)

    # 平均并输出
    avg_results = {k: v / total_samples for k, v in total_results.items()}
    print("\n" + "=" * 60)
    print(f"Evaluation Results on {total_samples} test images")
    print("=" * 60)
    for k, v in avg_results.items():
        print(f"{k:10s}: {v:.5f}")
    print("=" * 60)

    result_txt = os.path.join(args.output_dir, "metrics.txt")
    with open(result_txt, "w") as f:
        for k, v in avg_results.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"Metrics saved to {result_txt}")
    if args.save_images:
        print(f"Predicted images saved to {args.output_dir}")

if __name__ == "__main__":
    main()