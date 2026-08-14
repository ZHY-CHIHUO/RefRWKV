import random
import os
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

        mode_index = {"train": 0, "val": 1, "test": 2}[mode]
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

    def _random_crop(self, lr, hr, ref):
        """先采 LR 整数坐标，再映射 HR/ref，保证像素严格对齐。"""
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
        y_lr = random.randint(0, H_lr - th_lr)
        x_lr = random.randint(0, W_lr - tw_lr)
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
            lr, hr, ref = self._random_crop(lr, hr, ref)

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


class RefLMDBDataset(Dataset):
    """
    LMDB 版参考超分数据集。

    约定:
    1. crop 对齐：先采 LR 坐标再映射 HR（与 PNG 版一致）
    2. worker 安全：env 不序列化，worker 首次访问时延迟打开
    3. 读取校验：key 缺失 / bytes 长度错误时报错而非崩溃
    4. shape 可配置：不再硬编码 480/48
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
        hr_size: int = 480,
        lr_size: int = 48,
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

        # ★ shape 可配置
        self.hr_size = hr_size
        self.lr_size = lr_size
        self._hr_bytes_len = hr_size * hr_size * 3
        self._lr_bytes_len = lr_size * lr_size * 3

        # ★ 保存路径，不在此处打开 env（worker 安全）
        self._lmdb_path = os.path.join(data_dir, f"{self.mode}.lmdb")
        if not os.path.exists(self._lmdb_path):
            raise FileNotFoundError(f"LMDB not found: {self._lmdb_path}")

        self.env = None  # 延迟打开

        # 在主进程中临时打开以获取样本数
        env_tmp = lmdb.open(
            self._lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env_tmp.begin() as txn:
            total_entries = txn.stat()["entries"]
        env_tmp.close()

        self.total_samples = total_entries // 3
        if self.total_samples == 0:
            raise ValueError(f"LMDB is empty: {self._lmdb_path}")

        mode_index = {"train": 0, "val": 1, "test": 2}[mode]
        num_to_sample = max_samples[mode_index]
        if num_to_sample is not None:
            if num_to_sample > self.total_samples:
                print(
                    f"Warning: requested {num_to_sample} but only "
                    f"{self.total_samples} available."
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
                f"  Random crop: HR {patch_size}x{patch_size} "
                f"-> LR {self.lr_patch_size}x{self.lr_patch_size}"
            )

    # ★ worker 安全：env 不序列化，worker 首次访问时延迟打开
    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"] = None  # 不序列化 env
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.env = None  # worker 中延迟打开

    def _ensure_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self._lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

    def _read_image_bytes(self, idx):
        self._ensure_env()
        with self.env.begin() as txn:
            hr_key = f"{idx:06d}_hr".encode()
            lr_key = f"{idx:06d}_lr".encode()
            ref_key = f"{idx:06d}_ref".encode()

            hr_bytes = txn.get(hr_key)
            lr_bytes = txn.get(lr_key)
            ref_bytes = txn.get(ref_key)

        # ★ 读取校验
        if hr_bytes is None or lr_bytes is None or ref_bytes is None:
            missing = []
            if hr_bytes is None:
                missing.append("hr")
            if lr_bytes is None:
                missing.append("lr")
            if ref_bytes is None:
                missing.append("ref")
            raise KeyError(f"LMDB key missing for idx={idx}: {missing}")

        if len(hr_bytes) != self._hr_bytes_len:
            raise ValueError(
                f"HR bytes length mismatch: expected {self._hr_bytes_len}, "
                f"got {len(hr_bytes)} (idx={idx})"
            )
        if len(lr_bytes) != self._lr_bytes_len:
            raise ValueError(
                f"LR bytes length mismatch: expected {self._lr_bytes_len}, "
                f"got {len(lr_bytes)} (idx={idx})"
            )
        if len(ref_bytes) != self._hr_bytes_len:
            raise ValueError(
                f"Ref bytes length mismatch: expected {self._hr_bytes_len}, "
                f"got {len(ref_bytes)} (idx={idx})"
            )

        hr = (
            np.frombuffer(hr_bytes, dtype=np.uint8)
            .reshape(self.hr_size, self.hr_size, 3)
            .copy()
        )
        lr = (
            np.frombuffer(lr_bytes, dtype=np.uint8)
            .reshape(self.lr_size, self.lr_size, 3)
            .copy()
        )
        ref = (
            np.frombuffer(ref_bytes, dtype=np.uint8)
            .reshape(self.hr_size, self.hr_size, 3)
            .copy()
        )
        return lr, hr, ref

    def _random_crop_tensor(self, lr, hr, ref):
        """★ 先采 LR 坐标，再映射 HR（保证像素严格对齐）。"""
        _, H_lr, W_lr = lr.shape
        th_lr = self.lr_patch_size
        tw_lr = self.lr_patch_size

        # 先采 LR 整数坐标
        y_lr = torch.randint(0, H_lr - th_lr + 1, (1,)).item()
        x_lr = torch.randint(0, W_lr - tw_lr + 1, (1,)).item()

        # 再乘 scale 映射到 HR
        y_hr = y_lr * self.scale
        x_hr = x_lr * self.scale

        lr_crop = lr[:, y_lr : y_lr + th_lr, x_lr : x_lr + tw_lr]
        hr_crop = hr[:, y_hr : y_hr + self.patch_size, x_hr : x_hr + self.patch_size]
        ref_crop = ref[:, y_hr : y_hr + self.patch_size, x_hr : x_hr + self.patch_size]
        return lr_crop, hr_crop, ref_crop

    def _augment_tensor(self, lr, hr, ref):
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
        return lr.contiguous(), hr.contiguous(), ref.contiguous()

    def _augment_ref_tensor(self, ref_uint8):
        ref = ref_uint8.float() / 255.0

        if torch.rand(1).item() < self.ref_gray_prob:
            ref = TF.rgb_to_grayscale(ref, num_output_channels=1)
            ref = ref.repeat(3, 1, 1)
        else:
            for idx, (strength, prob) in enumerate(
                zip(self.ref_aug_strengths, self.ref_aug_probs)
            ):
                if torch.rand(1).item() > prob:
                    continue
                if idx == 0:
                    factor = 1.0 + (torch.rand(1).item() * 2 - 1) * strength
                    ref = TF.adjust_brightness(ref, factor)
                elif idx == 1:
                    factor = 1.0 + (torch.rand(1).item() * 2 - 1) * strength
                    ref = TF.adjust_contrast(ref, factor)
                elif idx == 2:
                    factor = 1.0 + (torch.rand(1).item() * 2 - 1) * strength
                    ref = TF.adjust_saturation(ref, factor)
                elif idx == 3:
                    if strength > 0:
                        hue_shift = (torch.rand(1).item() * 2 - 1) * strength
                        ref = TF.adjust_hue(ref, hue_shift)

        return (ref * 2.0) - 1.0

    def __getitem__(self, idx):
        true_idx = self.indices[idx]
        lr_np, hr_np, ref_np = self._read_image_bytes(true_idx)
        lr = torch.from_numpy(lr_np).permute(2, 0, 1)
        hr = torch.from_numpy(hr_np).permute(2, 0, 1)
        ref = torch.from_numpy(ref_np).permute(2, 0, 1)

        if self.patch_size is not None:
            lr, hr, ref = self._random_crop_tensor(lr, hr, ref)

        if self.augment:
            lr, hr, ref = self._augment_tensor(lr, hr, ref)

        if self.augment_ref:
            ref = self._augment_ref_tensor(ref)
        else:
            ref = (ref.float() / 127.5) - 1.0

        lr = (lr.float() / 127.5) - 1.0
        hr = (hr.float() / 127.5) - 1.0

        return {
            self.lr_key: lr,
            self.hr_key: hr,
            self.ref_key: ref,
        }

    def __len__(self):
        return len(self.indices)
