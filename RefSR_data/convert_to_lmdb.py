#!/usr/bin/env python3
"""PNG 参考超分三目录数据集 → LMDB 转换（命令行工具）。

用法:
    python RefSR_data/convert_to_lmdb.py --data_dir RefSR_data/ALL_2
    python RefSR_data/convert_to_lmdb.py --data_dir RefSR_data/ALL_2 \
        --splits train val --ext .png

约定（与 RefPNGDataset 一致）:
    <data_dir>/<split>/{HR,LR,Ref}/*.png
    每个 split 按文件名（去扩展名）对齐，写入 <data_dir>/<split>.lmdb。
    每个样本写入 3 个键: {idx:06d}_hr / _lr / _ref，值为 uint8 RGB bytes。
"""

import argparse
import os

import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_image_paths(split, ext):
    """返回 (hr_path, lr_path, ref_path) 列表，按文件名排序对齐。"""
    hr_dir = os.path.join(split, "HR")
    lr_dir = os.path.join(split, "LR")
    ref_dir = os.path.join(split, "Ref")

    names = sorted(
        f for f in os.listdir(hr_dir) if f.endswith(ext)
    )
    paths = []
    for name in names:
        hr_path = os.path.join(hr_dir, name)
        lr_path = os.path.join(lr_dir, name)
        ref_path = os.path.join(ref_dir, name)
        if not (os.path.exists(lr_path) and os.path.exists(ref_path)):
            print(f"警告：缺失对应文件 {name}，跳过")
            continue
        paths.append((hr_path, lr_path, ref_path))
    return paths


def load_rgb(path):
    """统一转为 RGB 的 uint8 ndarray，避免灰度/RGBA 写入不一致。"""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def write_split_to_lmdb(data_dir, split, ext, max_bytes, out_dir):
    paths = get_image_paths(os.path.join(data_dir, split), ext)
    num_samples = len(paths)
    print(f"[{split}] 找到 {num_samples} 对图像")

    if num_samples == 0:
        print(f"[{split}] 无样本，跳过")
        return

    # 每个样本 3 张图，2 倍安全余量
    map_size = num_samples * max_bytes * 3 * 2
    db_path = os.path.join(out_dir, f"{split}.lmdb")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    env = lmdb.open(db_path, map_size=map_size, writemap=True)
    try:
        with env.begin(write=True) as txn:
            for idx, (hr_p, lr_p, ref_p) in enumerate(tqdm(paths, desc=f"写入 {split}")):
                hr = load_rgb(hr_p)
                lr = load_rgb(lr_p)
                ref = load_rgb(ref_p)

                # 形状校验：LR 与 HR 的尺寸比应一致，HR 与 Ref 应同尺寸
                if hr.shape != ref.shape:
                    raise ValueError(
                        f"[{split}] 样本 {idx} HR {hr.shape} 与 Ref {ref.shape} 尺寸不一致: "
                        f"{hr_p} vs {ref_p}"
                    )
                if hr.shape[0] % lr.shape[0] != 0 or hr.shape[1] % lr.shape[1] != 0:
                    print(f"警告：[{split}] 样本 {idx} HR {hr.shape} 非 LR {lr.shape} 整数倍")

                txn.put(f"{idx:06d}_hr".encode(), hr.tobytes())
                txn.put(f"{idx:06d}_lr".encode(), lr.tobytes())
                txn.put(f"{idx:06d}_ref".encode(), ref.tobytes())
    finally:
        env.close()
    print(f"[{split}] LMDB 保存完毕：{db_path}")


def main():
    parser = argparse.ArgumentParser(description="PNG → LMDB 转换（RefSR_data/ALL_2 布局）")
    parser.add_argument("--data_dir", required=True, help="含 train/val/test 子目录的数据根目录")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="要转换的 split（默认 train val test）")
    parser.add_argument("--ext", default=".png", help="图像扩展名（默认 .png）")
    parser.add_argument("--max_bytes", type=int, default=700_000,
                        help="单张图最大字节数估计，用于 LMDB map_size（默认 700000）")
    parser.add_argument("--out_dir", default=None,
                        help="LMDB 输出目录（默认与 data_dir 相同）")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else data_dir

    for split in args.splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"警告：split 目录不存在，跳过: {split_dir}")
            continue
        write_split_to_lmdb(data_dir, split, args.ext, args.max_bytes, out_dir)


if __name__ == "__main__":
    main()
