"""PNG loader for single-image super-resolution datasets.

This module deliberately owns the ``HR/LR``-only contract.  Reference-based
super-resolution data belongs to :mod:`RefSR_data.RefDataset` instead.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SRPNGDataset(Dataset):
    """Load aligned RGB PNG pairs from ``<root>/<split>/{HR,LR}``.

    Images are returned in the common ``[-1, 1]`` floating-point range.  A
    training crop is specified on the HR grid; LR coordinates are sampled
    first so every crop remains exactly aligned at arbitrary integer scales.
    """

    _VALID_MODES = {"train", "val", "test", "test_easy", "test_hard"}

    def __init__(
        self,
        data_dir: str | Path,
        mode: str = "train",
        patch_size: int | None = None,
        scale: int = 4,
        augment: bool = False,
        max_samples: tuple[int | None, int | None, int | None] = (None, None, None),
        sample_seed: int = 42,
        lr_key: str = "lr",
        hr_key: str = "hr",
    ) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(f"Unknown mode: {mode}")
        if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
            raise ValueError(f"scale must be a positive integer, got {scale!r}")
        if patch_size is not None:
            if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size < 1:
                raise ValueError(f"patch_size must be a positive integer or None, got {patch_size!r}")
            if patch_size % scale:
                raise ValueError(
                    f"patch_size ({patch_size}) must be divisible by scale ({scale})"
                )
        if not isinstance(max_samples, (tuple, list)) or len(max_samples) != 3:
            raise ValueError("max_samples must be a (train, val, test) tuple")
        if not isinstance(sample_seed, int):
            raise ValueError("sample_seed must be an integer")
        for name, value in (("lr_key", lr_key), ("hr_key", hr_key)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")

        self.data_dir = Path(data_dir)
        self.mode = mode
        self.patch_size = patch_size
        self.scale = scale
        self.sample_seed = sample_seed
        self.augment = bool(augment) and mode == "train"
        self.lr_patch_size = None if patch_size is None else patch_size // scale
        self.lr_key, self.hr_key = lr_key, hr_key

        self.lr_dir = self.data_dir / mode / "LR"
        self.hr_dir = self.data_dir / mode / "HR"
        if not self.lr_dir.is_dir():
            raise FileNotFoundError(f"LR directory not found: {self.lr_dir}")
        if not self.hr_dir.is_dir():
            raise FileNotFoundError(f"HR directory not found: {self.hr_dir}")

        all_names = sorted(path.stem for path in self.lr_dir.glob("*.png"))
        mode_index = {"train": 0, "val": 1}.get(mode, 2)
        requested = max_samples[mode_index]
        if requested is not None:
            if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
                raise ValueError("max_samples entries must be non-negative integers or None")
            if requested > len(all_names):
                print(
                    f"Warning: requested {requested} samples but only {len(all_names)} are available; using all."
                )
                requested = len(all_names)
            all_names = sorted(random.Random(sample_seed).sample(all_names, requested))
        self.filenames = all_names

        for name in self.filenames:
            hr_path = self.hr_dir / f"{name}.png"
            if not hr_path.is_file():
                raise FileNotFoundError(f"Missing HR for LR sample {name!r}: {hr_path}")

        print(f"SRPNGDataset [{mode}]: {len(self.filenames)} LR/HR pairs")
        if patch_size is not None:
            print(
                f"  Random crop: HR {patch_size}x{patch_size} -> "
                f"LR {self.lr_patch_size}x{self.lr_patch_size}"
            )
        if self.augment:
            print("  Spatial augmentation: ON (flip & rot90)")

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return (np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5) - 1.0

    def _validate_geometry(self, lr: np.ndarray, hr: np.ndarray, name: str) -> None:
        lr_height, lr_width = lr.shape[:2]
        hr_height, hr_width = hr.shape[:2]
        expected = (lr_height * self.scale, lr_width * self.scale)
        if (hr_height, hr_width) != expected:
            raise ValueError(
                f"Unaligned LR/HR pair {name!r}: LR={lr_width}x{lr_height}, "
                f"HR={hr_width}x{hr_height}, expected HR={expected[1]}x{expected[0]} for x{self.scale}"
            )

    def _random_crop(self, lr: np.ndarray, hr: np.ndarray, rng=random) -> tuple[np.ndarray, np.ndarray]:
        assert self.patch_size is not None
        assert self.lr_patch_size is not None
        lr_height, lr_width = lr.shape[:2]
        if lr_height < self.lr_patch_size or lr_width < self.lr_patch_size:
            raise ValueError(
                f"LR image {lr_width}x{lr_height} is smaller than LR crop "
                f"{self.lr_patch_size}x{self.lr_patch_size}"
            )

        y_lr = rng.randint(0, lr_height - self.lr_patch_size)
        x_lr = rng.randint(0, lr_width - self.lr_patch_size)
        y_hr, x_hr = y_lr * self.scale, x_lr * self.scale
        return (
            lr[y_lr : y_lr + self.lr_patch_size, x_lr : x_lr + self.lr_patch_size],
            hr[y_hr : y_hr + self.patch_size, x_hr : x_hr + self.patch_size],
        )

    @staticmethod
    def _augment(lr: np.ndarray, hr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() > 0.5:
            lr = np.flip(lr, axis=2).copy()
            hr = np.flip(hr, axis=2).copy()
        if random.random() > 0.5:
            lr = np.flip(lr, axis=1).copy()
            hr = np.flip(hr, axis=1).copy()
        rotations = random.randint(0, 3)
        if rotations:
            lr = np.ascontiguousarray(np.rot90(lr, rotations, axes=(1, 2)))
            hr = np.ascontiguousarray(np.rot90(hr, rotations, axes=(1, 2)))
        return lr, hr

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        name = self.filenames[idx]
        lr = self._load_image(self.lr_dir / f"{name}.png")
        hr = self._load_image(self.hr_dir / f"{name}.png")
        self._validate_geometry(lr, hr, name)

        if self.patch_size is not None:
            crop_rng = random if self.mode == "train" else random.Random(self.sample_seed + idx)
            lr, hr = self._random_crop(lr, hr, rng=crop_rng)

        lr = np.ascontiguousarray(np.transpose(lr, (2, 0, 1)))
        hr = np.ascontiguousarray(np.transpose(hr, (2, 0, 1)))
        if self.augment:
            lr, hr = self._augment(lr, hr)
        return {self.lr_key: torch.from_numpy(lr), self.hr_key: torch.from_numpy(hr)}

    def __len__(self) -> int:
        return len(self.filenames)
