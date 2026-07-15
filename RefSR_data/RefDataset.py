import random, os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image, ImageEnhance
import lmdb
import torchvision.transforms.functional as TF


class RefPNGDataset(Dataset):
    """
    PNG 参考超分配对数据集（仅文件夹模式）。

    目录结构:
        data_dir/
            train/
                HR/   (480x480)
                LR/   (48x48)
                Ref/  (480x480)
            val/   ...
            test/  ...

    Args:
        data_dir: 数据根目录
        mode: 'train' / 'val' / 'test'
        patch_size: HR/Ref 随机裁剪尺寸，None 表示全图
        scale: HR 到 LR 的缩放倍数（默认 10）
        augment: 是否空间增强（随机翻转、旋转，仅 train 生效）
        augment_ref: 是否对参考图像进行风格增强（仅 train 生效）
        ref_aug_strengths: 对应增强的扰动幅度
        ref_aug_probs: 对应增强的触发概率
        ref_gray_prob: 转为灰度的概率
        max_samples: 三元组 (train_num, val_num, test_num)，指定各模式最多样本数
        sample_seed: 随机选取样本的种子
    """

    def __init__(
        self,
        data_dir: str,
        mode: str = "train",
        patch_size: int = None,
        scale: int = 10,
        augment: bool = False,
        augment_ref: bool = False,
        # 两个列表：分别控制 [亮度, 对比度, 饱和度, 色调] 的强度和触发概率
        ref_aug_strengths: list = [0.12, 0.12, 0.12, 0.03],
        ref_aug_probs: list = [0.5, 0.5, 0.5, 0.5],
        ref_gray_prob: float = 0.2,
        max_samples: tuple = (None, None, None),
        sample_seed: int = 42,
        lr_key: str = "lr",
        hr_key: str = "hr",
        ref_key: str = "ref",
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment and (mode == "train")
        self.augment_ref = augment_ref and (mode == "train")
        self.lr_patch_size = None if patch_size is None else patch_size // scale

        # 风格增强参数
        self.ref_aug_strengths = ref_aug_strengths
        self.ref_aug_probs = ref_aug_probs
        self.ref_gray_prob = ref_gray_prob

        # 子文件夹
        self.lr_dir = self.data_dir / mode / "LR"
        self.hr_dir = self.data_dir / mode / "HR"
        self.ref_dir = self.data_dir / mode / "Ref"

        self.lr_key = lr_key
        self.hr_key = hr_key
        self.ref_key = ref_key

        if not self.lr_dir.exists():
            raise FileNotFoundError(f"LR directory not found: {self.lr_dir}")

        # 可用文件名（不含扩展名）
        all_names = sorted([p.stem for p in self.lr_dir.glob("*.png")])

        mode_index = {"train": 0, "val": 1, "test": 2}[mode]
        num_to_sample = max_samples[mode_index]

        if num_to_sample is not None:
            if num_to_sample > len(all_names):
                print(
                    f"Warning: requested {num_to_sample} samples but only {len(all_names)} available, using all."
                )
                num_to_sample = len(all_names)
            rng = random.Random(sample_seed)
            self.filenames = rng.sample(all_names, num_to_sample)
        else:
            self.filenames = all_names

        # 校验 HR 和 Ref 存在
        for name in self.filenames:
            if not (self.hr_dir / f"{name}.png").exists():
                raise FileNotFoundError(f"Missing HR: {self.hr_dir / f'{name}.png'}")
            if not (self.ref_dir / f"{name}.png").exists():
                raise FileNotFoundError(f"Missing Ref: {self.ref_dir / f'{name}.png'}")

        print(f"RefPNGDataset [{mode}]: {len(self.filenames)} paired samples")
        if patch_size is not None:
            print(
                f"  Random crop: HR {patch_size}×{patch_size} → LR {self.lr_patch_size}×{self.lr_patch_size}"
            )
        if self.augment:
            print("  Spatial Augmentation: ON (flip & rot90)")
        if self.augment_ref:
            print(
                "  Ref Style Augmentation: ON (brightness/contrast/saturation/hue/gray)"
            )

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        return (np.array(img, dtype=np.float32) / 127.5) - 1.0

    # ---------- 参考图风格增强 ----------
    def _augment_ref(self, ref_img: Image.Image) -> Image.Image:
        """对 PIL 图像进行随机亮度、对比度、饱和度、色调、灰度变换"""
        # 1. 随机灰度
        if random.random() < self.ref_gray_prob:
            ref_img = ref_img.convert("L").convert("RGB")
        else:
            # 亮度、对比度、饱和度、色调循环
            # 索引 0:亮度, 1:对比度, 2:饱和度, 3:色调
            for idx, (strength, prob) in enumerate(
                zip(self.ref_aug_strengths, self.ref_aug_probs)
            ):
                if random.random() > prob:
                    continue
                if idx == 0:  # 亮度
                    factor = 1.0 + (random.random() * 2 - 1) * strength
                    ref_img = ImageEnhance.Brightness(ref_img).enhance(factor)
                elif idx == 1:  # 对比度
                    factor = 1.0 + (random.random() * 2 - 1) * strength
                    ref_img = ImageEnhance.Contrast(ref_img).enhance(factor)
                elif idx == 2:  # 饱和度
                    factor = 1.0 + (random.random() * 2 - 1) * strength
                    ref_img = ImageEnhance.Color(ref_img).enhance(factor)
                elif idx == 3:  # 色调
                    if strength > 0:
                        hue_shift = random.uniform(-strength, strength)
                        ref_img = self._hue_rotate(ref_img, hue_shift)
        return ref_img

    @staticmethod
    def _hue_rotate(img: Image.Image, delta: float) -> Image.Image:
        """色调旋转，delta 为旋转量，范围 [0,1] 对应 [0,360] 度"""
        if delta == 0:
            return img
        # 转为 HSV 模式，H 通道值范围为 0-255 (代表 0-360 度)
        img_hsv = img.convert("HSV")
        h, s, v = img_hsv.split()
        # H 通道增加 delta * 255
        h_arr = np.array(h, dtype=np.int32)
        h_arr = (h_arr + int(delta * 255)) % 256
        h = Image.fromarray(h_arr.astype(np.uint8), mode="L")
        img_hsv = Image.merge("HSV", (h, s, v))
        return img_hsv.convert("RGB")

    # ---------- 随机裁剪 ----------
    def _random_crop(self, lr, hr, ref):
        th_lr = self.patch_size // self.scale
        tw_lr = self.patch_size // self.scale

        H_lr, W_lr = lr.shape[:2]
        H_hr, W_hr = hr.shape[:2]

        if H_hr < self.patch_size or W_hr < self.patch_size:
            raise ValueError(
                f"HR image {H_hr}×{W_hr} smaller than patch {self.patch_size}"
            )
        if H_lr < th_lr or W_lr < tw_lr:
            raise ValueError(
                f"LR image {H_lr}×{W_lr} smaller than required LR patch {th_lr}×{tw_lr}"
            )

        y_lr = random.randint(0, H_lr - th_lr)
        x_lr = random.randint(0, W_lr - tw_lr)

        y_hr = y_lr * self.scale
        x_hr = x_lr * self.scale

        lr_crop = lr[y_lr : y_lr + th_lr, x_lr : x_lr + tw_lr]
        hr_crop = hr[y_hr : y_hr + self.patch_size, x_hr : x_hr + self.patch_size]
        ref_crop = ref[y_hr : y_hr + self.patch_size, x_hr : x_hr + self.patch_size]

        return lr_crop, hr_crop, ref_crop

    # ---------- 空间增强（翻转、旋转）----------
    def _augment(self, lr, hr, ref):
        """输入为 (C, H, W) 格式的 numpy 数组"""
        if random.random() > 0.5:
            lr = np.flip(lr, axis=2)  # 水平翻转
            hr = np.flip(hr, axis=2)
            ref = np.flip(ref, axis=2)

        if random.random() > 0.5:
            lr = np.flip(lr, axis=1)  # 垂直翻转
            hr = np.flip(hr, axis=1)
            ref = np.flip(ref, axis=1)

        k = random.randint(0, 3)
        if k > 0:
            lr = np.rot90(lr, k, axes=(1, 2))
            hr = np.rot90(hr, k, axes=(1, 2))
            ref = np.rot90(ref, k, axes=(1, 2))

        return lr, hr, ref

    # ---------- 主入口 ----------
    def __getitem__(self, idx):
        name = self.filenames[idx]

        lr = self._load_image(self.lr_dir / f"{name}.png")
        hr = self._load_image(self.hr_dir / f"{name}.png")
        # Ref 读取流程：先作为 PIL 打开，进行风格增强后再转 numpy
        ref_pil = Image.open(self.ref_dir / f"{name}.png").convert("RGB")
        if self.augment_ref:
            ref_pil = self._augment_ref(ref_pil)
        ref = (np.array(ref_pil, dtype=np.float32) / 127.5) - 1.0

        # 随机裁剪（若需要）
        if self.patch_size is not None:
            lr, hr, ref = self._random_crop(lr, hr, ref)

        # 转为 (C, H, W)
        lr = np.transpose(lr, (2, 0, 1))
        hr = np.transpose(hr, (2, 0, 1))
        ref = np.transpose(ref, (2, 0, 1))

        # 空间增强（翻转、旋转）
        if self.augment:
            lr, hr, ref = self._augment(lr, hr, ref)
            lr = np.ascontiguousarray(lr)
            hr = np.ascontiguousarray(hr)
            ref = np.ascontiguousarray(ref)

        return {
            self.lr_key: torch.from_numpy(lr),
            self.hr_key: torch.from_numpy(hr),
            self.ref_key: torch.from_numpy(ref),
        }

    def __len__(self):
        return len(self.filenames)


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
        lr_key: str = "lr",
        hr_key: str = "hr",
        ref_key: str = "ref",
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

        self.lr_key = lr_key
        self.hr_key = hr_key
        self.ref_key = ref_key

        # 打开 LMDB（只读、无锁、无自动读缓冲）
        lmdb_path = os.path.join(data_dir, f"{self.mode}.lmdb")
        self.env = lmdb.open(
            str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False
        )

        with self.env.begin() as txn:
            total_entries = txn.stat()["entries"]
            self.total_samples = total_entries // 3

        # 子采样逻辑
        mode_index = {"train": 0, "val": 1, "test": 2}[mode]
        num_to_sample = max_samples[mode_index]
        if num_to_sample is not None:
            if num_to_sample > self.total_samples:
                print(
                    f"Warning: requested {num_to_sample} but only {self.total_samples} available."
                )
                num_to_sample = self.total_samples
            rng = random.Random(sample_seed)
            self.indices = rng.sample(range(self.total_samples), num_to_sample)
        else:
            self.indices = list(range(self.total_samples))

        print(
            f"RefLMDBDataset [{mode}]: {len(self.indices)} / {self.total_samples} samples"
        )
        if patch_size is not None:
            print(
                f"  Random crop: HR {patch_size}×{patch_size} → LR {self.lr_patch_size}×{self.lr_patch_size}"
            )
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

        lr_crop = lr[
            :, y_lr : y_lr + self.lr_patch_size, x_lr : x_lr + self.lr_patch_size
        ]
        hr_crop = hr[:, y_hr : y_hr + self.patch_size, x_hr : x_hr + self.patch_size]
        ref_crop = ref[:, y_hr : y_hr + self.patch_size, x_hr : x_hr + self.patch_size]
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
            ref = ref.repeat(3, 1, 1)  # 恢复 3 通道
        else:
            # 依次判断亮度、对比度、饱和度、色调
            for idx, (strength, prob) in enumerate(
                zip(self.ref_aug_strengths, self.ref_aug_probs)
            ):
                if torch.rand(1).item() > prob:
                    continue
                if idx == 0:  # 亮度
                    factor = 1.0 + (torch.rand(1).item() * 2 - 1) * strength
                    ref = TF.adjust_brightness(ref, factor)
                elif idx == 1:  # 对比度
                    factor = 1.0 + (torch.rand(1).item() * 2 - 1) * strength
                    ref = TF.adjust_contrast(ref, factor)
                elif idx == 2:  # 饱和度
                    factor = 1.0 + (torch.rand(1).item() * 2 - 1) * strength
                    ref = TF.adjust_saturation(ref, factor)
                elif idx == 3:  # 色调
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

        return {
            self.lr_key: lr,
            self.hr_key: hr,
            self.ref_key: ref,
        }

    def __len__(self):
        return len(self.indices)
