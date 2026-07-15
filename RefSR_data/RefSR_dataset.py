import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image, ImageEnhance


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


