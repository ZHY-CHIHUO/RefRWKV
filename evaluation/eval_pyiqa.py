#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 pyiqa 的全能图像质量评估工具
支持: NumPy 数组 / 单张图片 / 文件夹 / .npy 潜变量
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import pyiqa
from tqdm import tqdm

# --------------------- 图像读取工具 (不依赖外部 img_utils) ---------------------
def imread(path: Union[str, Path], chn: str = 'rgb', dtype: str = 'float32') -> np.ndarray:
    """用 PIL 读取图像，返回 HWC numpy 数组，归一化到 [0,1]"""
    img = Image.open(str(path))
    if chn.lower() == 'rgb':
        img = img.convert('RGB')
    elif chn.lower() == 'gray':
        img = img.convert('L')
    else:
        raise ValueError(f"chn must be 'rgb' or 'gray', got {chn}")
    img = np.array(img).astype(np.float32) / 255.0
    if chn.lower() == 'gray' and img.ndim == 2:
        img = np.expand_dims(img, axis=2)  # H W -> H W 1
    return img

def img2tensor(img: np.ndarray, out_type=torch.float32) -> torch.Tensor:
    """HWC numpy -> 1 C H W tensor"""
    if img.ndim == 2:
        img = img[None, None, ...]  # H W -> 1 1 H W
    elif img.ndim == 3:
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # H W C -> 1 C H W
    else:
        raise ValueError(f"Expected 2D or 3D array, got {img.ndim}D")
    return img.to(out_type)

def load_npy_as_tensor(path: str, shape_hint: str = 'hwc') -> torch.Tensor:
    """
    加载 .npy 文件并转为 1 C H W tensor。
    shape_hint: 'hwc' 表示存储格式为 HWC, 'chw' 为 CHW
    """
    arr = np.load(path)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0 if shape_hint == 'chw' else 2)
    if shape_hint == 'hwc':
        arr = arr.transpose(2, 0, 1)
    elif shape_hint == 'chw':
        pass
    else:
        raise ValueError("shape_hint must be 'hwc' or 'chw'")
    return torch.from_numpy(arr).unsqueeze(0).float()

def generate_patch_dataset(img_dir, output_dir):
    """
    将 img_dir 中每张图像切出 n_patches 个随机块，保存到 output_dir。
    """
    patch_size=128
    n_patches=20
    seed=42
    import random
    random.seed(seed)
    img_dir = Path(img_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(img_dir.glob("*.[jpJP][pnPN]*[gG]")) +
                       list(img_dir.glob("*.npy")))

    count = 0
    for img_path in tqdm(img_files, desc=f"Generating patches from {img_dir.name}"):
        # 读取图像（兼容你已有的 imread）
        img_np = imread(str(img_path))  # 返回 HWC, [0,1]
        H, W = img_np.shape[:2]

        # 若图像尺寸小于 patch_size，跳过或缩放（这里跳过）
        if H < patch_size or W < patch_size:
            continue

        for _ in range(n_patches):
            y = random.randint(0, H - patch_size)
            x = random.randint(0, W - patch_size)
            patch = img_np[y:y+patch_size, x:x+patch_size, :]
            # 保存为 PNG
            patch_img = Image.fromarray((patch * 255).astype(np.uint8))
            patch_img.save(output_dir / f"{count:07d}.png")
            count += 1

    print(f"Generated {count} patches in {output_dir}")

# --------------------- 评估引擎 ---------------------
class IQAEngine:
    def __init__(self, device='cuda', nr_metrics=None, fr_metrics=None, 
                 use_y_channel=True, verbose=True):
        """
        nr_metrics: 无参考指标列表，默认 ['clipiqa', 'musiq', 'niqe', 'maniqa']
        fr_metrics: 全参考指标列表，默认 ['psnr', 'ssim', 'lpips', 'dists']
        use_y_channel: 是否仅在亮度通道计算 PSNR/SSIM（仅对 RGB 有效）
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose

        # 默认指标
        if nr_metrics is None:
            nr_metrics = ['clipiqa', 'musiq', 'niqe', 'maniqa']
        if fr_metrics is None:
            fr_metrics = ['psnr', 'ssim', 'lpips', 'dists']

        self.nr_metrics = {}
        for name in nr_metrics:
            self.nr_metrics[name] = pyiqa.create_metric(name).to(self.device)

        self.fr_metrics = {}
        for name in fr_metrics:
            if name in ['psnr', 'ssim'] and use_y_channel:
                metric = pyiqa.create_metric(name, test_y_channel=True, color_space='ycbcr')
            else:
                metric = pyiqa.create_metric(name)
            self.fr_metrics[name] = metric.to(self.device)

        self.fid_metric = None  # lazy init

    def _prepare_tensor(self, data, name='image') -> torch.Tensor:
        """
        将各种输入统一转为 (1, C, H, W) tensor on device。
        支持: numpy (HWC/CHW), PIL Image, str/Path (图片或.npy)
        """
        if isinstance(data, (str, Path)):
            data = str(data)
            if data.endswith('.npy'):
                tensor = load_npy_as_tensor(data, shape_hint='hwc')  # 默认 HWC
            else:
                img_np = imread(data)
                tensor = img2tensor(img_np)
        elif isinstance(data, np.ndarray):
            # 尝试猜测维度顺序
            if data.ndim == 3:
                # 假设 C,H,W 如果第一维 <= 4 且第二/三维 > 4，否则 H,W,C
                if data.shape[0] <= 4 and (data.shape[1] > 4 or data.shape[2] > 4):
                    # CHW
                    tensor = torch.from_numpy(data).unsqueeze(0)
                else:
                    # HWC
                    tensor = torch.from_numpy(data).permute(2,0,1).unsqueeze(0)
            elif data.ndim == 2:
                tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
            else:
                raise ValueError(f"Unsupported numpy shape {data.shape}")
            tensor = tensor.float()
        elif isinstance(data, Image.Image):
            data = np.array(data).astype(np.float32) / 255.0
            tensor = img2tensor(data)
        else:
            raise TypeError(f"Unsupported type for {name}: {type(data)}")
        return tensor.to(self.device)

    def evaluate_single(self, pred, gt=None) -> Dict[str, float]:
        """单张评估，pred 可以是 tensor/numpy/路径，gt 可选"""
        pred_tensor = self._prepare_tensor(pred, 'pred')
        results = {}

        # 无参考指标
        for name, metric in self.nr_metrics.items():
            with torch.no_grad():
                results[name] = float(metric(pred_tensor).item())

        # 全参考指标
        if gt is not None:
            gt_tensor = self._prepare_tensor(gt, 'gt')
            # 尺寸对齐
            if pred_tensor.shape[-2:] != gt_tensor.shape[-2:]:
                if self.verbose:
                    print(f"Resizing pred from {pred_tensor.shape[-2:]} to {gt_tensor.shape[-2:]}")
                pred_tensor = F.interpolate(pred_tensor, size=gt_tensor.shape[-2:],
                                            mode='bicubic', align_corners=False)
            for name, metric in self.fr_metrics.items():
                with torch.no_grad():
                    results[name] = float(metric(pred_tensor, gt_tensor).item())
        return results

    def evaluate_folder(self, pred_dir, gt_dir=None, ntest=None, out_path=None) -> Dict[str, float]:
        """
        批量评估文件夹。
        pred_dir: 预测图像文件夹
        gt_dir: 参考图像文件夹（可选）
        ntest: 限制评估前 n 张图
        out_path: 结果保存路径（可选）
        """
        pred_path = Path(pred_dir)
        assert pred_path.is_dir(), f"{pred_dir} is not a directory"

        pred_files = sorted(list(pred_path.glob("*.[jpJP][pnPN]*[gG]")) + 
                            list(pred_path.glob("*.npy")))
        if ntest:
            pred_files = pred_files[:ntest]

        gt_files = None
        if gt_dir is not None:
            gt_path = Path(gt_dir)
            gt_files = sorted(list(gt_path.glob("*.[jpJP][pnPN]*[gG]")) + 
                              list(gt_path.glob("*.npy")))
            if ntest:
                gt_files = gt_files[:ntest]
            if len(gt_files) != len(pred_files):
                raise ValueError(f"Number of images in pred ({len(pred_files)}) and gt ({len(gt_files)}) must match")

        print(f"Found {len(pred_files)} images in {pred_dir}")

        total_results = {k: 0.0 for k in list(self.nr_metrics.keys()) + list(self.fr_metrics.keys())}
        count = 0

        for i, pred_file in enumerate(tqdm(pred_files, desc="Evaluating")):
            try:
                gt_file = gt_files[i] if gt_files else None
                res = self.evaluate_single(str(pred_file), str(gt_file) if gt_file else None)
                for k, v in res.items():
                    total_results[k] += v
                count += 1
            except Exception as e:
                print(f"Error processing {pred_file}: {e}")
                continue

        # 平均
        avg_results = {k: v / count for k, v in total_results.items()}

        # FID (如果图片数>=1000 且提供 gt_dir)
        if gt_dir and count >= 1000:
            if self.fid_metric is None:
                self.fid_metric = pyiqa.create_metric('fid')
            avg_results['fid'] = float(self.fid_metric(str(pred_dir), str(gt_dir)))

            import tempfile
            with tempfile.TemporaryDirectory() as tmp_pred, tempfile.TemporaryDirectory() as tmp_gt:
                print("Extracting patches for pFID...")
                generate_patch_dataset(pred_dir, tmp_pred)
                generate_patch_dataset(gt_dir, tmp_gt)

                pfid_value = float(self.fid_metric(tmp_pred, tmp_gt))
                avg_results['pfid'] = pfid_value

        # 打印与保存
        self._print_and_save(avg_results, out_path, count)
        return avg_results

    def _print_and_save(self, results, out_path, total_images):
        lines = []
        for k, v in results.items():
            if k == 'fid':
                line = f"{k}: {v:.2f}"
            else:
                line = f"{k}: {v:.5f}"
            print(line)
            lines.append(line)

        if out_path:
            os.makedirs(out_path, exist_ok=True)
            with open(os.path.join(out_path, "results.txt"), 'w') as f:
                f.write('\n'.join(lines))

# --------------------- 命令行接口 ---------------------
def main():
    parser = argparse.ArgumentParser(description='PyIQA-based Image Quality Assessment')
    parser.add_argument('--pred', '-p', type=str, required=True,
                        help='Prediction: image path, numpy array path, or folder')
    parser.add_argument('--gt', '-g', type=str, default=None,
                        help='Ground truth (optional): image path, numpy array path, or folder')
    parser.add_argument('--nr_metrics', nargs='+', default=['clipiqa', 'musiq', 'niqe', 'maniqa'],
                        help='No-reference metrics to compute')
    parser.add_argument('--fr_metrics', nargs='+', default=['psnr', 'ssim', 'lpips', 'dists'],
                        help='Full-reference metrics to compute')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--ntest', type=int, default=None,
                        help='Limit evaluation to first N images')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory to save results.txt')
    parser.add_argument('--no_y_channel', action='store_true',
                        help='Do not use Y channel only for PSNR/SSIM')
    args = parser.parse_args()

    engine = IQAEngine(
        device=args.device,
        nr_metrics=args.nr_metrics,
        fr_metrics=args.fr_metrics,
        use_y_channel=not args.no_y_channel
    )

    # 判断输入类型：文件夹 或 单文件/数组
    pred_path = Path(args.pred)
    if pred_path.is_dir():
        engine.evaluate_folder(args.pred, args.gt, args.ntest, args.output)
    else:
        # 单张评估
        gt = args.gt if args.gt else None
        res = engine.evaluate_single(args.pred, gt)
        print("\n".join([f"{k}: {v:.5f}" for k, v in res.items()]))
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            with open(os.path.join(args.output, "results.txt"), 'w') as f:
                f.write('\n'.join([f"{k}: {v:.5f}" for k, v in res.items()]))

if __name__ == '__main__':
    main()