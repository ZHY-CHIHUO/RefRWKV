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
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pyiqa

# 确保项目根目录在 sys.path 中，以便导入自定义模块
# 根据你的脚本所在位置调整以下路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from Restore_RWKV import Restore_RWKV_Ref
from RWKV.RefSR_data.RefSR_dataset import RefPNGDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Restore_RWKV_Ref on test set"
    )
    parser.add_argument(
        "--weight",
        type=str,
        required=True,
        help="Path to the trained model weights (.pth)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of the dataset (e.g., /path/to/ALL_2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cuda:1, cpu)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for inference"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=10,
        help="Super-resolution scale factor (must match model)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of test samples (None = all)",
    )
    parser.add_argument(
        "--save_images", action="store_true", help="Save predicted HR images to disk"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save predicted images and metrics.txt",
    )
    parser.add_argument(
        "--nr_metrics",
        nargs="+",
        default=["clipiqa", "musiq", "niqe"],
        help="No-reference metrics to compute (default: clipiqa, musiq, niqe)",
    )
    parser.add_argument(
        "--no_nr", action="store_true", help="Skip all no-reference metrics"
    )
    return parser.parse_args()


def collate_fn(batch):
    """重组批次数据为 (lr, ref, lr, hr) 以匹配模型输入"""
    lr_list, hr_list, ref_list = zip(*batch)
    lr = torch.stack(lr_list, 0)  # [B,3,H_lr,W_lr]
    hr = torch.stack(hr_list, 0)  # [B,3,H_hr,W_hr]
    ref = torch.stack(ref_list, 0)  # [B,3,H_hr,W_hr]
    # 模型前向需要: lr1, hr1(参考), lr2
    return lr, ref, lr, hr


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- 1. 构建测试数据集 ----------
    test_dataset = RefPNGDataset(
        data_dir=args.data_root,
        mode="test",
        patch_size=None,  # 测试时不裁剪
        scale=args.scale,
        augment=False,
        max_samples=(None, None, args.max_samples),  # 控制测试数量
        sample_seed=42,
    )
    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ---------- 2. 加载模型 ----------
    model = Restore_RWKV_Ref(inp_channels=3, out_channels=3, scale=args.scale).to(
        device
    )
    checkpoint = torch.load(args.weight, map_location=device)

    # 兼容多种保存格式
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:  # Lightning .ckpt
        state_dict = checkpoint["state_dict"]
        # 移除 'model.' 前缀
        state_dict = {
            k.replace("model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Model loaded from {args.weight}")

    # ---------- 3. 初始化 pyiqa 指标 ----------
    fr_metrics = {
        "PSNR": pyiqa.create_metric(
            "psnr", test_y_channel=True, color_space="ycbcr"
        ).to(device),
        "SSIM": pyiqa.create_metric(
            "ssim", test_y_channel=True, color_space="ycbcr"
        ).to(device),
        "LPIPS": pyiqa.create_metric("lpips").to(device),
        "DISTS": pyiqa.create_metric("dists").to(device),
    }

    nr_metrics = {}
    if not args.no_nr:
        for name in args.nr_metrics:
            try:
                nr_metrics[name] = pyiqa.create_metric(name).to(device)
            except KeyError:
                print(f'Warning: Metric "{name}" not found in pyiqa, skipping.')

    # 累加器
    total_results = {k: 0.0 for k in list(fr_metrics.keys()) + list(nr_metrics.keys())}
    total_samples = 0

    # ---------- 4. 推理并评估 ----------
    with torch.no_grad():
        for batch_idx, (lr1, ref, lr2, hr_gt) in enumerate(
            tqdm(test_loader, desc="Testing")
        ):
            lr1 = lr1.to(device)
            ref = ref.to(device)
            lr2 = lr2.to(device)
            hr_gt = hr_gt.to(device)

            # 前向推理
            pred = model(lr1, ref, lr2)  # [B,3,H,W], 值域 [0,1]

            # 确保尺寸与 GT 一致（理论上应该一致）
            if pred.shape[-2:] != hr_gt.shape[-2:]:
                pred = F.interpolate(
                    pred, size=hr_gt.shape[-2:], mode="bicubic", align_corners=False
                )

            # 逐样本计算指标（pyiqa 支持批量但为简单起见逐样本累加）
            for i in range(pred.size(0)):
                pred_i = pred[i : i + 1]  # 保持 [1,3,H,W]
                gt_i = hr_gt[i : i + 1]

                # 全参考指标
                for name, metric in fr_metrics.items():
                    total_results[name] += metric(pred_i, gt_i).item()

                # 无参考指标（只评估预测图）
                for name, metric in nr_metrics.items():
                    total_results[name] += metric(pred_i).item()

                total_samples += 1

                # 保存预测图像（可选）
                if args.save_images:
                    # 转换为 numpy 并保存
                    img_np = pred_i.squeeze(0).cpu().numpy()  # [C,H,W]
                    img_np = np.transpose(img_np, (1, 2, 0))  # HWC
                    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
                    save_path = os.path.join(
                        args.output_dir, f"pred_{batch_idx}_{i}.png"
                    )
                    from PIL import Image

                    Image.fromarray(img_np).save(save_path)

    # ---------- 5. 计算平均指标并输出 ----------
    avg_results = {k: v / total_samples for k, v in total_results.items() if v != 0.0}

    print("\n" + "=" * 60)
    print(f"Evaluation Results on {total_samples} test images")
    print("=" * 60)
    for k, v in avg_results.items():
        if k == "FID":  # 这里我们没有算FID，但保留格式
            print(f"{k:10s}: {v:.2f}")
        else:
            print(f"{k:10s}: {v:.5f}")
    print("=" * 60)

    # 保存指标到文件
    result_txt = os.path.join(args.output_dir, "metrics.txt")
    with open(result_txt, "w") as f:
        for k, v in avg_results.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"Metrics saved to {result_txt}")

    # 可选：如果保存了图像，给出提示
    if args.save_images:
        print(f"Predicted images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
