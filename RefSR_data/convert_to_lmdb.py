import os
import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------- 配置 ----------
DATA_ROOT = "/home/zhy/PROJECT/RefRWKV/RefSR_data/ALL_2"
SPLITS = ["train", "val", "test"]
SUBFOLDERS = ["HR", "LR", "Ref"]          # 对应三组图像
IMAGE_EXT = ".png"                        # 假设都是 png
# 估算每张图最大字节数 (留余量，如 480x480x3 = 691200)
MAX_BYTES_PER_IMG = 700_000

def get_image_paths(split):
    """返回 (hr_path, lr_path, ref_path) 列表，按文件名排序对齐"""
    hr_dir = os.path.join(DATA_ROOT, split, "HR")
    lr_dir = os.path.join(DATA_ROOT, split, "LR")
    ref_dir = os.path.join(DATA_ROOT, split, "Ref")

    # 获取所有 .png 文件名（不含扩展名）
    names = sorted([f for f in os.listdir(hr_dir) if f.endswith(IMAGE_EXT)])
    paths = []
    for name in names:
        hr_path = os.path.join(hr_dir, name)
        lr_path = os.path.join(lr_dir, name)   # 假设 LR 和 Ref 里文件名完全一致
        ref_path = os.path.join(ref_dir, name)
        if not (os.path.exists(lr_path) and os.path.exists(ref_path)):
            print(f"警告：缺失对应文件 {name}，跳过")
            continue
        paths.append((hr_path, lr_path, ref_path))
    return paths

def write_split_to_lmdb(split):
    paths = get_image_paths(split)
    num_samples = len(paths)
    print(f"[{split}] 找到 {num_samples} 对图像")

    # 每个样本存 3 张图 → 3 个键值对
    map_size = num_samples * MAX_BYTES_PER_IMG * 3 * 2  # 2 倍安全余量
    db_path = os.path.join(DATA_ROOT, f"{split}.lmdb")

    env = lmdb.open(db_path, map_size=map_size, writemap=True)
    with env.begin(write=True) as txn:
        for idx, (hr_p, lr_p, ref_p) in enumerate(tqdm(paths, desc=f"写入 {split}")):
            # 读取图像并转为 uint8 数组
            hr = np.array(Image.open(hr_p))
            lr = np.array(Image.open(lr_p))
            ref = np.array(Image.open(ref_p))

            # 键名格式：索引_类型，如 "000001_hr"
            key_hr = f"{idx:06d}_hr".encode()
            key_lr = f"{idx:06d}_lr".encode()
            key_ref = f"{idx:06d}_ref".encode()

            txn.put(key_hr, hr.tobytes())
            txn.put(key_lr, lr.tobytes())
            txn.put(key_ref, ref.tobytes())

    env.close()
    print(f"[{split}] LMDB 保存完毕：{db_path}")

if __name__ == "__main__":
    for split in SPLITS:
        write_split_to_lmdb(split)