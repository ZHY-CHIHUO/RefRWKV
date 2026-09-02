import random
import os
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
            ref_aug_strengths = [0.12, 0.12, 0.12, 0.03]
        if ref_aug_probs is None:
            ref_aug_probs = [0.5, 0.5, 0.5, 0.5]

        self.data_dir = Path(data_dir)
        self.mode = mode
        self.patch_size = patch_size
        self.scale = scale
        self.sample_seed = sample_seed
        self.augment = augment and (mode == "train")
        self.augment_ref = augment_ref and (mode == "train")
        self.lr_patch_size = None if patch_size is None else patch_size // scale

        self.ref_aug_strengths = ref_aug_strengths
        self.ref_aug_probs = ref_aug_probs
        self.ref_gray_prob = ref_gray_prob

        self.lr_dir = self.data_dir / mode / "LR"
        self.hr_dir = self.data_dir / mode / "HR"
        self.ref_dir = self.data_dir / mode / "Ref"

        self.lr_key = lr_key
        self.hr_key = hr_key
        self.ref_key = ref_key

        if not self.lr_dir.exists():
            raise FileNotFoundError(f"LR directory not found: {self.lr_dir}")

        all_names = sorted(
            [
                os.path.splitext(f)[0]
                for f in os.listdir(self.lr_dir)
                if f.endswith(".png")
            ]
        )

        if mode not in {"train", "val", "test", "test_easy", "test_hard"}:
            raise ValueError(f"Unknown mode: {mode}")
        mode_index = {"train": 0, "val": 1}.get(mode, 2)
        num_to_sample = max_samples[mode_index]

        if num_to_sample is not None:
            if num_to_sample > len(all_names):
                print(
                    f"Warning: requested {num_to_sample} samples but only "
                    f"{len(all_names)} available, using all."
                )
                num_to_sample = len(all_names)
            rng = random.Random(sample_seed)
            self.filenames = rng.sample(all_names, num_to_sample)
        else:
            self.filenames = all_names

        for name in self.filenames:
            if not (self.hr_dir / f"{name}.png").exists():
                raise FileNotFoundError(f"Missing HR: {self.hr_dir / f'{name}.png'}")
            if not (self.ref_dir / f"{name}.png").exists():
                raise FileNotFoundError(f"Missing Ref: {self.ref_dir / f'{name}.png'}")

        print(f"RefPNGDataset [{mode}]: {len(self.filenames)} paired samples")
        if patch_size is not None:
            print(
                f"  Random crop: HR {patch_size}x{patch_size} "
                f"-> LR {self.lr_patch_size}x{self.lr_patch_size}"
            )
        if self.augment:
            print("  Spatial Augmentation: ON (flip & rot90)")
        if self.augment_ref:
            print("  Ref Style Augmentation: ON")

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        return (np.array(img, dtype=np.float32) / 127.5) - 1.0

    def _augment_ref(self, ref_img: Image.Image) -> Image.Image:
        if random.random() < self.ref_gray_prob:
            ref_img = ref_img.convert("L").convert("RGB")
        else:
            for idx, (strength, prob) in enumerate(
                zip(self.ref_aug_strengths, self.ref_aug_probs)
            ):
                if random.random() > prob:
                    continue
                if idx == 0:
                    factor = 1.0 + (random.random() * 2 - 1) * strength
                    ref_img = ImageEnhance.Brightness(ref_img).enhance(factor)
                elif idx == 1:
                    factor = 1.0 + (random.random() * 2 - 1) * strength
                    ref_img = ImageEnhance.Contrast(ref_img).enhance(factor)
                elif idx == 2:
                    factor = 1.0 + (random.random() * 2 - 1) * strength
                    ref_img = ImageEnhance.Color(ref_img).enhance(factor)
                elif idx == 3:
                    if strength > 0:
                        hue_shift = random.uniform(-strength, strength)
                        ref_img = self._hue_rotate(ref_img, hue_shift)
        return ref_img

    @staticmethod
    def _hue_rotate(img: Image.Image, delta: float) -> Image.Image:
        if delta == 0:
            return img
        img_hsv = img.convert("HSV")
        h, s, v = img_hsv.split()
        h_arr = np.array(h, dtype=np.int32)
        h_arr = (h_arr + int(delta * 255)) % 256
        h = Image.fromarray(h_arr.astype(np.uint8), mode="L")
        img_hsv = Image.merge("HSV", (h, s, v))
        return img_hsv.convert("RGB")

    def _random_crop(self, lr, hr, ref, rng=random):
        """Sample LR coordinates, then map them to aligned HR/ref pixels."""
        th_lr = self.patch_size // self.scale
        tw_lr = self.patch_size // self.scale

        H_lr, W_lr = lr.shape[:2]
        H_hr, W_hr = hr.shape[:2]

        if H_hr < self.patch_size or W_hr < self.patch_size:
            raise ValueError(
                f"HR image {H_hr}x{W_hr} smaller than patch {self.patch_size}"
            )
        if H_lr < th_lr or W_lr < tw_lr:
            raise ValueError(
                f"LR image {H_lr}x{W_lr} smaller than LR patch {th_lr}x{tw_lr}"
            )

        # ★ 先采 LR 坐标，再乘 scale 映射到 HR（保证整除对齐）
        y_lr = rng.randint(0, H_lr - th_lr)
        x_lr = rng.randint(0, W_lr - tw_lr)
        y_hr = y_lr * self.scale
        x_hr = x_lr * self.scale

        lr_crop = lr[y_lr : y_lr + th_lr, x_lr : x_lr + tw_lr]
        hr_crop = hr[y_hr : y_hr + self.patch_size, x_hr : x_hr + self.patch_size]
        ref_crop = ref[y_hr : y_hr + self.patch_size, x_hr : x_hr + self.patch_size]

        return lr_crop, hr_crop, ref_crop

    def _augment(self, lr, hr, ref):
        """输入为 (C, H, W) 格式的 numpy 数组。"""
        if random.random() > 0.5:
            lr = np.flip(lr, axis=2).copy()
            hr = np.flip(hr, axis=2).copy()
            ref = np.flip(ref, axis=2).copy()

        if random.random() > 0.5:
            lr = np.flip(lr, axis=1).copy()
            hr = np.flip(hr, axis=1).copy()
            ref = np.flip(ref, axis=1).copy()

        k = random.randint(0, 3)
        if k > 0:
            lr = np.ascontiguousarray(np.rot90(lr, k, axes=(1, 2)))
            hr = np.ascontiguousarray(np.rot90(hr, k, axes=(1, 2)))
            ref = np.ascontiguousarray(np.rot90(ref, k, axes=(1, 2)))

        return lr, hr, ref

    def __getitem__(self, idx):
        name = self.filenames[idx]

        lr = self._load_image(self.lr_dir / f"{name}.png")
        hr = self._load_image(self.hr_dir / f"{name}.png")
        ref_pil = Image.open(self.ref_dir / f"{name}.png").convert("RGB")
        if self.augment_ref:
            ref_pil = self._augment_ref(ref_pil)
        ref = (np.array(ref_pil, dtype=np.float32) / 127.5) - 1.0

        if self.patch_size is not None:
            # Validation/test crops must be repeatable: val_loss then measures
            # model changes rather than a different random region each pass.
            crop_rng = random if self.mode == "train" else random.Random(self.sample_seed + idx)
            lr, hr, ref = self._random_crop(lr, hr, ref, rng=crop_rng)

        lr = np.ascontiguousarray(np.transpose(lr, (2, 0, 1)))
        hr = np.ascontiguousarray(np.transpose(hr, (2, 0, 1)))
        ref = np.ascontiguousarray(np.transpose(ref, (2, 0, 1)))

        if self.augment:
            lr, hr, ref = self._augment(lr, hr, ref)

        return {
            self.lr_key: torch.from_numpy(lr),
            self.hr_key: torch.from_numpy(hr),
            self.ref_key: torch.from_numpy(ref),
        }

    def __len__(self):
        return len(self.filenames)
