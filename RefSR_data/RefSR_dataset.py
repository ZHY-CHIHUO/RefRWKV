import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class RefPNGDataset(Dataset):
    """
    PNG 参考超分配对数据集（仅文件夹模式）。

    目录结构必须为:
        data_dir/
            train/
                HR/   (480*480)
                LR/   (48*48)
                Ref/  (480*480)
            val/   ...
            test/  ...

    Args:
        data_dir: 数据根目录
        mode: 'train' / 'val' / 'test'
        patch_size: HR/Ref 随机裁剪尺寸，None 表示全图
        scale: HR 到 LR 的缩放倍数（默认 10）
        augment: 是否数据增强（仅 train 生效）
        max_samples: 三元组 (train_num, val_num, test_num)，
                     指定每个模式最多随机选取的样本数，None 表示全取
        sample_seed: 随机选取样本的种子（默认 42）
    """

    def __init__(
        self,
        data_dir: str,
        mode: str = "train",
        patch_size: int = None,
        scale: int = 10,
        augment: bool = False,
        max_samples: tuple = (None, None, None),
        sample_seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment and (mode == "train")
        self.lr_patch_size = None if patch_size is None else patch_size // scale

        # 确定子文件夹
        self.lr_dir = self.data_dir / mode / "LR"
        self.hr_dir = self.data_dir / mode / "HR"
        self.ref_dir = self.data_dir / mode / "Ref"

        if not self.lr_dir.exists():
            raise FileNotFoundError(f"LR directory not found: {self.lr_dir}")

        # 所有可用文件名（不含扩展名）
        all_names = sorted([p.stem for p in self.lr_dir.glob("*.png")])

        # 根据 mode 确定本次需要取多少样本
        mode_index = {"train": 0, "val": 1, "test": 2}[mode]
        num_to_sample = max_samples[mode_index]

        if num_to_sample is not None:
            if num_to_sample > len(all_names):
                print(f"Warning: requested {num_to_sample} samples but only {len(all_names)} available, using all.")
                num_to_sample = len(all_names)
            rng = random.Random(sample_seed)           # 固定种子，保证同模式重复创建时结果一致
            self.filenames = rng.sample(all_names, num_to_sample)
        else:
            self.filenames = all_names

        # 校验 HR 和 Ref 是否存在
        for name in self.filenames:
            if not (self.hr_dir / f"{name}.png").exists():
                raise FileNotFoundError(f"Missing HR: {self.hr_dir / f'{name}.png'}")
            if not (self.ref_dir / f"{name}.png").exists():
                raise FileNotFoundError(f"Missing Ref: {self.ref_dir / f'{name}.png'}")

        print(f"RefPNGDataset [{mode}]: {len(self.filenames)} paired samples")
        if patch_size is not None:
            print(f"  Random crop: HR {patch_size}×{patch_size} → LR {self.lr_patch_size}×{self.lr_patch_size}")
        if self.augment:
            print("  Augmentation: ON (random flip & rot90)")

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.float32) / 255.0

    def _random_crop(self, lr, hr, ref):
        # 1. 计算 LR 裁剪尺寸（向下取整，与现有逻辑一致）
        th_lr = self.patch_size // self.scale
        tw_lr = self.patch_size // self.scale

        H_lr, W_lr = lr.shape[:2]
        H_hr, W_hr = hr.shape[:2]

        # 2. 检查尺寸是否足够（用 HR 尺寸保证 patch 不超出；同时检查 LR 尺寸）
        if H_hr < self.patch_size or W_hr < self.patch_size:
            raise ValueError(f"HR image {H_hr}×{W_hr} smaller than patch {self.patch_size}×{self.patch_size}")
        if H_lr < th_lr or W_lr < tw_lr:
            raise ValueError(f"LR image {H_lr}×{W_lr} smaller than required LR patch {th_lr}×{tw_lr}")

        # 3. 在 LR 上随机确定合法位置
        y_lr = random.randint(0, H_lr - th_lr)
        x_lr = random.randint(0, W_lr - tw_lr)

        # 4. 反推 HR 坐标（直接乘 scale，无需边界检查）
        y_hr = y_lr * self.scale
        x_hr = x_lr * self.scale

        # 5. 裁剪三张图
        lr_crop = lr[y_lr:y_lr + th_lr, x_lr:x_lr + tw_lr]
        hr_crop = hr[y_hr:y_hr + self.patch_size, x_hr:x_hr + self.patch_size]
        ref_crop = ref[y_hr:y_hr + self.patch_size, x_hr:x_hr + self.patch_size]

        return lr_crop, hr_crop, ref_crop

    def _augment(self, lr, hr, ref):
        """注意：输入为 (C, H, W) 格式"""
        if random.random() > 0.5:
            lr = np.flip(lr, axis=2)   # 水平翻转 (左右)
            hr = np.flip(hr, axis=2)
            ref = np.flip(ref, axis=2)

        if random.random() > 0.5:
            lr = np.flip(lr, axis=1)   # 垂直翻转 (上下)
            hr = np.flip(hr, axis=1)
            ref = np.flip(ref, axis=1)

        k = random.randint(0, 3)
        if k > 0:
            lr = np.rot90(lr, k, axes=(1, 2))   # 只在 H 和 W 上旋转
            hr = np.rot90(hr, k, axes=(1, 2))
            ref = np.rot90(ref, k, axes=(1, 2))

        return lr, hr, ref

    def __getitem__(self, idx):
        name = self.filenames[idx]
        lr = self._load_image(self.lr_dir / f"{name}.png")
        hr = self._load_image(self.hr_dir / f"{name}.png")
        ref = self._load_image(self.ref_dir / f"{name}.png")
        if self.patch_size is not None:
            lr, hr, ref = self._random_crop(lr, hr, ref)
        lr = np.transpose(lr, (2,0,1))
        hr = np.transpose(hr, (2,0,1))
        ref = np.transpose(ref, (2,0,1))
        if self.augment:
            lr, hr, ref = self._augment(lr, hr, ref)
            lr = np.ascontiguousarray(lr)
            hr = np.ascontiguousarray(hr)
            ref = np.ascontiguousarray(ref)
        return torch.from_numpy(lr), torch.from_numpy(hr), torch.from_numpy(ref)

    def __len__(self):
        return len(self.filenames)

def main():
    from torch.utils.data import DataLoader

    # 数据集路径与参数
    data_root = r"/home/zhy/PROJECT/RWKV/RefSR_data/ALL_2"
    max_samples = (1000, 200, None)  # train:1000, val:200, test:全部
    patch_size = 128
    batch_size = 4

    # 创建三个数据集
    train_ds = RefPNGDataset(
        data_dir=data_root,
        mode="train",
        max_samples=max_samples,
        sample_seed=42,
        patch_size=patch_size,
        augment=True,
    )
    val_ds = RefPNGDataset(
        data_dir=data_root,
        mode="val",
        max_samples=max_samples,
        sample_seed=42,
        patch_size=None,       # 验证时用全图，方便评估
        augment=False,
    )
    test_ds = RefPNGDataset(
        data_dir=data_root,
        mode="test",
        max_samples=max_samples,
        sample_seed=42,
        patch_size=None,
        augment=False,
    )

    print(f"\n数据集大小 —— Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}\n")

    # 构建 DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=0)

    # 检查训练集一个 batch
    print("===== 训练集 batch 测试 =====")
    for lr, hr, ref in train_loader:
        print(f"LR  shape: {lr.shape}   min/max: {lr.min():.3f} / {lr.max():.3f}")
        print(f"HR  shape: {hr.shape}   min/max: {hr.min():.3f} / {hr.max():.3f}")
        print(f"Ref shape: {ref.shape}  min/max: {ref.min():.3f} / {ref.max():.3f}")

        # 简单检查 HR 和 Ref 是否相同（在此任务中，Ref 可能与 HR 内容不同）
        # 以及 LR 尺寸是否为 HR 的 1/scale
        assert hr.shape[-2] == hr.shape[-1] == patch_size, "HR patch size mismatch"
        assert lr.shape[-2] == lr.shape[-1] == patch_size // 10, "LR patch size mismatch"
        break

    # 检查验证集一个样本
    print("\n===== 验证集样本测试 =====")
    for lr, hr, ref in val_loader:
        print(f"LR  shape: {lr.shape}")    # 应为 [1,3,48,48] 或全图尺寸
        print(f"HR  shape: {hr.shape}")    # 应为 [1,3,480,480]
        print(f"Ref shape: {ref.shape}")
        break

    print("\n所有基本测试通过！")

if __name__ == "__main__":    main()