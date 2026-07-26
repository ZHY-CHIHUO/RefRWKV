#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 pyiqa 的全能图像质量评估工具
支持: NumPy 数组 / 单张图片 / 文件夹 / .npy 潜变量
"""

import os
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import pyiqa
from tqdm import tqdm

# 支持的图像扩展名
_IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


# --------------------- 图像读取工具 ---------------------
def imread(
    path: Union[str, Path], chn: str = "rgb", dtype: str = "float32"
) -> np.ndarray:
    """用 PIL 读取图像，返回 HWC numpy 数组，归一化到 [0,1]"""
    img = Image.open(str(path))
    if chn.lower() == "rgb":
        img = img.convert("RGB")
    elif chn.lower() == "gray":
        img = img.convert("L")
    else:
        raise ValueError(f"chn must be 'rgb' or 'gray', got {chn}")
    img = np.array(img).astype(np.float32) / 255.0
    if chn.lower() == "gray" and img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img


def img2tensor(img: np.ndarray, out_type=torch.float32) -> torch.Tensor:
    """HWC numpy -> 1 C H W tensor"""
    if img.ndim == 2:
        img = img[None, None, ...]
    elif img.ndim == 3:
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError(f"Expected 2D or 3D array, got {img.ndim}D")
    return img.to(out_type)


def load_npy_as_tensor(path: str, shape_hint: str = "hwc") -> torch.Tensor:
    """加载 .npy 文件并转为 1 C H W tensor。"""
    arr = np.load(path)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0 if shape_hint == "chw" else 2)
    if shape_hint == "hwc":
        arr = arr.transpose(2, 0, 1)
    elif shape_hint == "chw":
        pass
    else:
        raise ValueError("shape_hint must be 'hwc' or 'chw'")
    return torch.from_numpy(arr).unsqueeze(0).float()


def _collect_images(directory: Path) -> List[Path]:
    """收集目录下所有支持格式的图像文件，按文件名排序。"""
    files = []
    for ext in _IMG_EXTS:
        files.extend(directory.glob(ext))
    files.extend(directory.glob("*.npy"))
    return sorted(files)


def _match_by_stem(
    pred_files: List[Path],
    gt_files: List[Path],
    pred_suffixes=("_pred", "_sr", "_output", "_result"),
):
    """按文件名 stem 建立 pred → gt 映射。

    返回:
        pairs: List[(pred_path, gt_path)]
        unmatched_pred: List[Path]  无匹配 GT 的 pred 文件
        unmatched_gt:   List[Path]  无匹配 pred 的 gt 文件
    """
    # 建立 GT 映射（检测重复）
    gt_map = {}
    for gf in gt_files:
        stem = gf.stem
        if stem in gt_map:
            raise ValueError(f"GT 文件名重复: {stem} ({gf})")
        gt_map[stem] = gf

    pairs = []
    unmatched_pred = []
    matched_gt_stems = set()

    for pf in pred_files:
        stem = pf.stem
        # 尝试去掉常见后缀匹配 GT
        for suffix in pred_suffixes:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        if stem in gt_map:
            pairs.append((pf, gt_map[stem]))
            matched_gt_stems.add(stem)
        else:
            unmatched_pred.append(pf)

    unmatched_gt = [gf for gf in gt_files if gf.stem not in matched_gt_stems]

    return pairs, unmatched_pred, unmatched_gt


def generate_patch_dataset(img_dir, output_dir, patch_size=128, n_patches=20, seed=42):
    """将 img_dir 中每张图像切出 n_patches 个随机块，保存到 output_dir。"""
    import random

    random.seed(seed)
    img_dir = Path(img_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_files = _collect_images(img_dir)
    count = 0
    for img_path in tqdm(img_files, desc=f"Generating patches from {img_dir.name}"):
        img_np = imread(str(img_path))
        H, W = img_np.shape[:2]
        if H < patch_size or W < patch_size:
            continue
        for _ in range(n_patches):
            y = random.randint(0, H - patch_size)
            x = random.randint(0, W - patch_size)
            patch = img_np[y : y + patch_size, x : x + patch_size, :]
            patch_img = Image.fromarray((patch * 255).astype(np.uint8))
            patch_img.save(output_dir / f"{count:07d}.png")
            count += 1
    print(f"Generated {count} patches in {output_dir}")


# --------------------- 评估引擎 ---------------------
class IQAEngine:
    def __init__(
        self,
        device="cuda",
        nr_metrics=None,
        fr_metrics=None,
        use_y_channel=True,
        verbose=True,
        allow_resize=False,
    ):
        """
        Args:
            nr_metrics:   无参考指标列表
            fr_metrics:   全参考指标列表
            use_y_channel: 是否仅在亮度通道计算 PSNR/SSIM
            verbose:      是否打印详细信息
            allow_resize: 尺寸不匹配时是否允许 bicubic resize（默认 False = 报错）
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.allow_resize = allow_resize

        if nr_metrics is None:
            nr_metrics = ["clipiqa", "musiq", "niqe", "maniqa"]
        if fr_metrics is None:
            fr_metrics = ["psnr", "ssim", "lpips", "dists"]

        self.nr_metrics = {}
        for name in nr_metrics:
            self.nr_metrics[name] = pyiqa.create_metric(name).to(self.device)

        self.fr_metrics = {}
        for name in fr_metrics:
            if name in ["psnr", "ssim"] and use_y_channel:
                metric = pyiqa.create_metric(
                    name, test_y_channel=True, color_space="ycbcr"
                )
            else:
                metric = pyiqa.create_metric(name)
            self.fr_metrics[name] = metric.to(self.device)

        self.fid_metric = None

    def _prepare_tensor(self, data, name="image") -> torch.Tensor:
        """将各种输入统一转为 (1, C, H, W) tensor on device。"""
        if isinstance(data, (str, Path)):
            data = str(data)
            if data.endswith(".npy"):
                tensor = load_npy_as_tensor(data, shape_hint="hwc")
            else:
                img_np = imread(data)
                tensor = img2tensor(img_np)
        elif isinstance(data, np.ndarray):
            if data.ndim == 3:
                if data.shape[-1] <= 4:
                    # HWC (H, W, C)
                    tensor = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0)
                elif data.shape[0] <= 4:
                    # CHW (C, H, W)
                    tensor = torch.from_numpy(data).unsqueeze(0)
                else:
                    raise ValueError(
                        f"无法判断 numpy 数组维度顺序: shape={data.shape}。"
                        f"请确保输入为 HWC 或 CHW 格式。"
                    )
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
        """单张评估。"""
        pred_tensor = self._prepare_tensor(pred, "pred")
        results = {}

        for name, metric in self.nr_metrics.items():
            with torch.no_grad():
                results[name] = float(metric(pred_tensor).item())

        if gt is not None:
            gt_tensor = self._prepare_tensor(gt, "gt")

            # 尺寸校验
            if pred_tensor.shape[-2:] != gt_tensor.shape[-2:]:
                if self.allow_resize:
                    if self.verbose:
                        print(
                            f"[allow_resize] Resizing pred from "
                            f"{list(pred_tensor.shape[-2:])} to "
                            f"{list(gt_tensor.shape[-2:])}"
                        )
                    pred_tensor = F.interpolate(
                        pred_tensor,
                        size=gt_tensor.shape[-2:],
                        mode="bicubic",
                        align_corners=False,
                    )
                else:
                    raise ValueError(
                        f"尺寸不匹配: pred={list(pred_tensor.shape[-2:])}, "
                        f"gt={list(gt_tensor.shape[-2:])}。"
                        f"请检查输出是否正确，或设置 allow_resize=True 显式启用缩放。"
                    )

            for name, metric in self.fr_metrics.items():
                with torch.no_grad():
                    results[name] = float(metric(pred_tensor, gt_tensor).item())

        return results

    def evaluate_folder(
        self, pred_dir, gt_dir=None, ntest=None, out_path=None
    ) -> Dict[str, float]:
        """
        批量评估文件夹。按文件名 stem 配对，不按位置。
        """
        pred_path = Path(pred_dir)
        assert pred_path.is_dir(), f"{pred_dir} is not a directory"

        pred_files = _collect_images(pred_path)
        if ntest:
            pred_files = pred_files[:ntest]

        pairs = None
        if gt_dir is not None:
            gt_path = Path(gt_dir)
            assert gt_path.is_dir(), f"{gt_dir} is not a directory"
            gt_files = _collect_images(gt_path)
            if ntest:
                gt_files = gt_files[:ntest]

            pairs, unmatched_pred, unmatched_gt = _match_by_stem(pred_files, gt_files)

            if unmatched_pred:
                print(f"警告: {len(unmatched_pred)} 个 pred 文件无匹配 GT:")
                for f in unmatched_pred[:5]:
                    print(f"  - {f.name}")
                if len(unmatched_pred) > 5:
                    print(f"  ... 及其他 {len(unmatched_pred) - 5} 个")

            if unmatched_gt:
                print(f"警告: {len(unmatched_gt)} 个 GT 文件无匹配 pred:")
                for f in unmatched_gt[:5]:
                    print(f"  - {f.name}")

            if len(pairs) == 0:
                raise RuntimeError(
                    f"无任何匹配的 pred-GT 对！"
                    f"pred={len(pred_files)} 个, gt={len(gt_files)} 个。"
                    f"请检查文件命名是否一致。"
                )

            print(
                f"匹配到 {len(pairs)} 对图像 (pred={len(pred_files)}, gt={len(gt_files)})"
            )
        else:
            print(f"Found {len(pred_files)} images in {pred_dir} (无参考模式)")

        total_results = {
            k: 0.0 for k in list(self.nr_metrics.keys()) + list(self.fr_metrics.keys())
        }
        count = 0
        failures = 0

        if pairs is not None:
            # 有 GT：按配对评估
            iterator = tqdm(pairs, desc="Evaluating")
            for pred_file, gt_file in iterator:
                try:
                    res = self.evaluate_single(str(pred_file), str(gt_file))
                    for k, v in res.items():
                        total_results[k] += v
                    count += 1
                except Exception as e:
                    failures += 1
                    print(f"\n错误: {pred_file.name} ↔ {gt_file.name}: {e}")
                    if self.verbose:
                        traceback.print_exc()
        else:
            # 无 GT：仅无参考指标
            iterator = tqdm(pred_files, desc="Evaluating (NR only)")
            for pred_file in iterator:
                try:
                    res = self.evaluate_single(str(pred_file))
                    for k, v in res.items():
                        total_results[k] += v
                    count += 1
                except Exception as e:
                    failures += 1
                    print(f"\n错误: {pred_file.name}: {e}")
                    if self.verbose:
                        traceback.print_exc()

        # 除零保护
        if count == 0:
            raise RuntimeError(
                f"所有图像评估均失败 ({failures} failures)，无有效结果。"
                f"请检查图像格式和路径。"
            )

        if failures > 0:
            print(
                f"\n评估完成: {count} 成功, {failures} 失败 "
                f"(失败率 {failures / (count + failures) * 100:.1f}%)"
            )

        avg_results = {k: v / count for k, v in total_results.items()}

        # FID
        if gt_dir and count >= 1000:
            if self.fid_metric is None:
                self.fid_metric = pyiqa.create_metric("fid")
            avg_results["fid"] = float(self.fid_metric(str(pred_dir), str(gt_dir)))

            import tempfile

            with tempfile.TemporaryDirectory() as tmp_pred, tempfile.TemporaryDirectory() as tmp_gt:
                print("Extracting patches for pFID...")
                generate_patch_dataset(pred_dir, tmp_pred)
                generate_patch_dataset(gt_dir, tmp_gt)
                avg_results["pfid"] = float(self.fid_metric(tmp_pred, tmp_gt))

        self._print_and_save(avg_results, out_path, count)
        return avg_results

    def _print_and_save(self, results, out_path, total_images):
        lines = [f"Total images: {total_images}", ""]
        for k, v in results.items():
            if k == "fid":
                line = f"{k}: {v:.2f}"
            else:
                line = f"{k}: {v:.5f}"
            print(line)
            lines.append(line)

        if out_path:
            os.makedirs(out_path, exist_ok=True)
            with open(os.path.join(out_path, "results.txt"), "w") as f:
                f.write("\n".join(lines))
            print(f"\n结果已保存到 {out_path}/results.txt")


# --------------------- 命令行接口 ---------------------
def main():
    parser = argparse.ArgumentParser(description="PyIQA-based Image Quality Assessment")
    parser.add_argument(
        "--pred",
        "-p",
        type=str,
        required=True,
        help="Prediction: image path, numpy array path, or folder",
    )
    parser.add_argument(
        "--gt",
        "-g",
        type=str,
        default=None,
        help="Ground truth (optional): image path, numpy array path, or folder",
    )
    parser.add_argument(
        "--nr_metrics",
        nargs="+",
        default=["clipiqa", "musiq", "niqe", "maniqa"],
        help="No-reference metrics to compute",
    )
    parser.add_argument(
        "--fr_metrics",
        nargs="+",
        default=["psnr", "ssim", "lpips", "dists"],
        help="Full-reference metrics to compute",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--ntest", type=int, default=None, help="Limit evaluation to first N images"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory to save results.txt",
    )
    parser.add_argument(
        "--no_y_channel",
        action="store_true",
        help="Do not use Y channel only for PSNR/SSIM",
    )
    parser.add_argument(
        "--allow-resize",
        action="store_true",
        help="Allow bicubic resize when pred/gt sizes mismatch "
        "(default: raise error)",
    )
    args = parser.parse_args()

    engine = IQAEngine(
        device=args.device,
        nr_metrics=args.nr_metrics,
        fr_metrics=args.fr_metrics,
        use_y_channel=not args.no_y_channel,
        allow_resize=args.allow_resize,
    )

    pred_path = Path(args.pred)
    if pred_path.is_dir():
        engine.evaluate_folder(args.pred, args.gt, args.ntest, args.output)
    else:
        gt = args.gt if args.gt else None
        res = engine.evaluate_single(args.pred, gt)
        print("\n".join([f"{k}: {v:.5f}" for k, v in res.items()]))
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            with open(os.path.join(args.output, "results.txt"), "w") as f:
                f.write("\n".join([f"{k}: {v:.5f}" for k, v in res.items()]))


if __name__ == "__main__":
    main()
