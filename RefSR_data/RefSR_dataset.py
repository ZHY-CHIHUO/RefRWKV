import random, os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import lmdb
import torchvision.transforms.functional as TF


class RefLMDBDataset(Dataset):
    """
    LMDB 版参考超分数据集（免解码，极速读取）。
    全部增强均已改为 PyTorch 张量操作，避免 PIL/NumPy 中间开销。
    """

    def __init__(
        self,
        data_dir: str,
        mode: str = "train",
        patch_size: int = None,
        scale: int = 10,
        augment: bool = False,
        augment_ref: bool = False,
        ref_aug_strengths: list = None,
        ref_aug_probs: list = None,
        ref_gray_prob: float = 0.2,
        max_samples: tuple = (None, None, None),
        sample_seed: int = 42,
    ):
        if ref_aug_strengths is None:
            ref_aug_strengths = [0.15, 0.15, 0.15, 0.03]
        if ref_aug_probs is None:
            ref_aug_probs = [0.5, 0.5, 0.5, 0.5]

        self.mode = mode
        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment and (mode == "train")
        self.augment_ref = augment_ref and (mode == "train")
        self.lr_patch_size = None if patch_size is None else patch_size // scale

        self.ref_aug_strengths = ref_aug_strengths
        self.ref_aug_probs = ref_aug_probs
        self.ref_gray_prob = ref_gray_prob

        # 打开 LMDB（只读、无锁、无自动读缓冲）
        lmdb_path = os.path.join(data_dir, f"{self.mode}.lmdb")
        self.env = lmdb.open(
            str(lmdb_path), readonly=True, lock=False,
            readahead=False, meminit=False
        )

        with self.env.begin() as txn:
            total_entries = txn.stat()['entries']
            self.total_samples = total_entries // 3

        # 子采样逻辑
        mode_index = {"train": 0, "val": 1, "test": 2}[mode]
        num_to_sample = max_samples[mode_index]
        if num_to_sample is not None:
            if num_to_sample > self.total_samples:
                print(f"Warning: requested {num_to_sample} but only {self.total_samples} available.")
                num_to_sample = self.total_samples
            rng = random.Random(sample_seed)
            self.indices = rng.sample(range(self.total_samples), num_to_sample)
        else:
            self.indices = list(range(self.total_samples))

        print(f"RefLMDBDataset [{mode}]: {len(self.indices)} / {self.total_samples} samples")
        if patch_size is not None:
            print(f"  Random crop: HR {patch_size}×{patch_size} → LR {self.lr_patch_size}×{self.lr_patch_size}")
        if self.augment:
            print("  Spatial Augmentation: ON (Torch flips/rot90)")
        if self.augment_ref:
            print("  Ref Style Augmentation: ON (Torch color jitter/gray)")

    # ---------- 从 LMDB 读取像素 ----------
    def _read_image_bytes(self, idx):
        with self.env.begin() as txn:
            hr_bytes = txn.get(f"{idx:06d}_hr".encode())
            lr_bytes = txn.get(f"{idx:06d}_lr".encode())
            ref_bytes = txn.get(f"{idx:06d}_ref".encode())
        hr = np.frombuffer(hr_bytes, dtype=np.uint8).reshape(480, 480, 3).copy()
        lr = np.frombuffer(lr_bytes, dtype=np.uint8).reshape(48, 48, 3).copy()
        ref = np.frombuffer(ref_bytes, dtype=np.uint8).reshape(480, 480, 3).copy()
        return lr, hr, ref

    # ---------- Torch 增强操作 ----------
    def _random_crop_tensor(self, lr, hr, ref):
        """在 uint8 张量上随机裁剪，输入 (C,H,W)"""
        _, H_hr, W_hr = hr.shape
        y_hr = torch.randint(0, H_hr - self.patch_size + 1, (1,)).item()
        x_hr = torch.randint(0, W_hr - self.patch_size + 1, (1,)).item()
        y_lr = y_hr // self.scale
        x_lr = x_hr // self.scale

        lr_crop = lr[:, y_lr:y_lr+self.lr_patch_size, x_lr:x_lr+self.lr_patch_size]
        hr_crop = hr[:, y_hr:y_hr+self.patch_size, x_hr:x_hr+self.patch_size]
        ref_crop = ref[:, y_hr:y_hr+self.patch_size, x_hr:x_hr+self.patch_size]
        return lr_crop, hr_crop, ref_crop

    def _augment_tensor(self, lr, hr, ref):
        """空间增强：翻转 + 旋转 (C,H,W) uint8 张量"""
        if torch.rand(1) > 0.5:
            lr = torch.flip(lr, dims=[2])
            hr = torch.flip(hr, dims=[2])
            ref = torch.flip(ref, dims=[2])
        if torch.rand(1) > 0.5:
            lr = torch.flip(lr, dims=[1])
            hr = torch.flip(hr, dims=[1])
            ref = torch.flip(ref, dims=[1])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            lr = torch.rot90(lr, k, dims=[1, 2])
            hr = torch.rot90(hr, k, dims=[1, 2])
            ref = torch.rot90(ref, k, dims=[1, 2])
        # 返回连续内存张量
        return lr.contiguous(), hr.contiguous(), ref.contiguous()

    def _augment_ref_tensor(self, ref_uint8):
        """
        Ref 色彩增强，直接在 float 张量上操作。
        输入: (C,H,W) uint8，范围 0-255
        返回: (C,H,W) float32，范围 [-1, 1]
        """
        # 转为 float [0, 1]
        ref = ref_uint8.float() / 255.0

        # 随机灰度
        if torch.rand(1).item() < self.ref_gray_prob:
            ref = TF.rgb_to_grayscale(ref, num_output_channels=1)
            ref = ref.repeat(3, 1, 1)      # 恢复 3 通道
        else:
            # 依次判断亮度、对比度、饱和度、色调
            for idx, (strength, prob) in enumerate(zip(self.ref_aug_strengths, self.ref_aug_probs)):
                if torch.rand(1).item() > prob:
                    continue
                if idx == 0:   # 亮度
                    factor = 1.0 + (torch.rand(1).item() * 2 - 1) * strength
                    ref = TF.adjust_brightness(ref, factor)
                elif idx == 1: # 对比度
                    factor = 1.0 + (torch.rand(1).item() * 2 - 1) * strength
                    ref = TF.adjust_contrast(ref, factor)
                elif idx == 2: # 饱和度
                    factor = 1.0 + (torch.rand(1).item() * 2 - 1) * strength
                    ref = TF.adjust_saturation(ref, factor)
                elif idx == 3: # 色调
                    if strength > 0:
                        hue_shift = (torch.rand(1).item() * 2 - 1) * strength
                        ref = TF.adjust_hue(ref, hue_shift)

        # 归一化到 [-1, 1]
        return (ref * 2.0) - 1.0

    # ---------- 主入口 ----------
    def __getitem__(self, idx):
        true_idx = self.indices[idx]
        lr_np, hr_np, ref_np = self._read_image_bytes(true_idx)
        lr = torch.from_numpy(lr_np).permute(2, 0, 1)
        hr = torch.from_numpy(hr_np).permute(2, 0, 1)
        ref = torch.from_numpy(ref_np).permute(2, 0, 1)

        # 随机裁剪（uint8 上直接切片）
        if self.patch_size is not None:
            lr, hr, ref = self._random_crop_tensor(lr, hr, ref)

        # 空间增强（翻转/旋转，仍在 uint8 上操作）
        if self.augment:
            lr, hr, ref = self._augment_tensor(lr, hr, ref)

        # Ref 处理：色彩增强 → 归一化到 [-1,1]
        if self.augment_ref:
            ref = self._augment_ref_tensor(ref)
        else:
            ref = (ref.float() / 127.5) - 1.0

        # LR / HR 归一化到 [-1,1]
        lr = (lr.float() / 127.5) - 1.0
        hr = (hr.float() / 127.5) - 1.0

        return lr, hr, ref

    def __len__(self):
        return len(self.indices)

