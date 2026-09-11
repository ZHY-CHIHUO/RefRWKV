"""PanCollection H5 loader for reduced-resolution pansharpening data.

PanCollection stores each split as one HDF5 file with NCHW datasets named
``gt``, ``ms``, ``pan`` and (for convenience in the upstream toolbox) ``lms``.
The RefSR contract used here deliberately maps ``ms`` to the low-resolution
input and ``pan`` to the high-resolution reference; ``lms`` is an upsampled
baseline and is not used as the LR tensor.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PanCollectionH5Dataset(Dataset):
    """Read one PanCollection H5 split as ``lr/ref/hr`` tensors.

    The H5 handle is opened lazily in each worker process, which avoids sharing
    an inherited file descriptor with a forked DataLoader worker.  Values are
    converted from PanCollection's DN/[0, 2047] representation to the
    repository's ``[-1, 1]`` range.
    """

    _VALID_MODES = {"train", "val", "test", "test_easy", "test_hard"}

    def __init__(
        self,
        file_path: str | Path,
        *,
        mode: str = "train",
        scale: int = 4,
        patch_size: int | None = 64,
        augment: bool = False,
        max_samples: int | tuple[int | None, int | None, int | None] | None = None,
        sample_seed: int = 42,
        lr_key: str = "lr",
        hr_key: str = "hr",
        ref_key: str = "ref",
        h5_lr_key: str = "ms",
        h5_hr_key: str = "gt",
        h5_ref_key: str = "pan",
        value_scale: float = 2047.0,
        clip_range: bool = True,
        expected_lr_channels: int | None = None,
        expected_ref_channels: int | None = None,
        expected_hr_channels: int | None = None,
        return_sample_id: bool = False,
    ) -> None:
        mode = str(mode).strip().lower()
        if mode not in self._VALID_MODES:
            raise ValueError(f"Unknown PanCollection split: {mode}")
        if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
            raise ValueError("scale must be a positive integer")
        if patch_size is not None:
            if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size < 1:
                raise ValueError("patch_size must be a positive integer or None")
            if patch_size % scale:
                raise ValueError(f"patch_size ({patch_size}) must be divisible by scale ({scale})")
        if isinstance(sample_seed, bool) or not isinstance(sample_seed, int):
            raise ValueError("sample_seed must be an integer")
        if not np.isfinite(float(value_scale)) or float(value_scale) <= 0:
            raise ValueError("value_scale must be a positive finite number")
        for name, value in (
            ("lr_key", lr_key),
            ("hr_key", hr_key),
            ("ref_key", ref_key),
            ("h5_lr_key", h5_lr_key),
            ("h5_hr_key", h5_hr_key),
            ("h5_ref_key", h5_ref_key),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name, value in (
            ("expected_lr_channels", expected_lr_channels),
            ("expected_ref_channels", expected_ref_channels),
            ("expected_hr_channels", expected_hr_channels),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or None")

        self.file_path = Path(file_path).expanduser().resolve()
        if not self.file_path.is_file():
            raise FileNotFoundError(f"PanCollection H5 file not found: {self.file_path}")
        self.mode = mode
        self.scale = scale
        self.patch_size = patch_size
        self.lr_patch_size = None if patch_size is None else patch_size // scale
        self.augment = bool(augment) and mode == "train"
        self.sample_seed = sample_seed
        self.lr_key, self.hr_key, self.ref_key = lr_key, hr_key, ref_key
        self.h5_lr_key, self.h5_hr_key, self.h5_ref_key = h5_lr_key, h5_hr_key, h5_ref_key
        self.value_scale = float(value_scale)
        self.clip_range = bool(clip_range)
        self.expected_lr_channels = expected_lr_channels
        self.expected_ref_channels = expected_ref_channels
        self.expected_hr_channels = expected_hr_channels
        self.return_sample_id = bool(return_sample_id)
        self._h5 = None
        self._h5_pid: int | None = None

        shapes = self._inspect_file()
        self._shapes = shapes
        self.length = int(shapes[self.h5_hr_key][0])
        requested = self._requested_count(max_samples, mode)
        if requested is not None:
            if requested < 0:
                raise ValueError("max_samples must be non-negative")
            if requested > self.length:
                requested = self.length
            self.indices = sorted(random.Random(sample_seed).sample(range(self.length), requested))
        else:
            self.indices = list(range(self.length))
        self.filenames = [f"{self.file_path.stem}_{index:06d}" for index in self.indices]
        print(
            f"PanCollectionH5Dataset [{mode}]: {len(self.indices)} samples "
            f"lr={self._shapes[self.h5_lr_key][1:]} "
            f"ref={self._shapes[self.h5_ref_key][1:]} "
            f"hr={self._shapes[self.h5_hr_key][1:]} scale=x{scale}"
        )
        if patch_size is not None:
            print(f"  Synchronized crop: HR {patch_size}x{patch_size} -> LR {self.lr_patch_size}x{self.lr_patch_size}")
        if self.augment:
            print("  Spatial augmentation: ON (synchronized flip & rot90)")

    @staticmethod
    def _requested_count(
        value: int | tuple[int | None, int | None, int | None] | None,
        mode: str,
    ) -> int | None:
        if value is None:
            return None
        if isinstance(value, (tuple, list)):
            if len(value) != 3:
                raise ValueError("max_samples tuple must be (train, val, test)")
            value = value[0 if mode == "train" else 1 if mode == "val" else 2]
            if value is None:
                return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("max_samples must be an integer, tuple, or None")
        return int(value)

    def _inspect_file(self) -> dict[str, tuple[int, ...]]:
        try:
            import h5py  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("PanCollection H5 loading requires h5py") from exc
        with h5py.File(self.file_path, "r") as handle:
            shapes = {key: tuple(int(dim) for dim in handle[key].shape) for key in handle.keys()}
        required = (self.h5_lr_key, self.h5_ref_key, self.h5_hr_key)
        missing = [key for key in required if key not in shapes]
        if missing:
            raise KeyError(f"PanCollection H5 is missing datasets: {', '.join(missing)}")
        for key in required:
            shape = shapes[key]
            if len(shape) != 4:
                raise ValueError(f"PanCollection dataset {key!r} must be NCHW rank-4, got {shape}")
            if shape[0] != shapes[self.h5_hr_key][0]:
                raise ValueError("PanCollection gt/ms/pan sample counts do not match")
            if shape[1] < 1 or shape[2] < 1 or shape[3] < 1:
                raise ValueError(f"PanCollection dataset {key!r} has invalid shape {shape}")
        lr_shape = shapes[self.h5_lr_key]
        ref_shape = shapes[self.h5_ref_key]
        hr_shape = shapes[self.h5_hr_key]
        expected_hr = (lr_shape[2] * self.scale, lr_shape[3] * self.scale)
        if hr_shape[2:] != expected_hr or ref_shape[2:] != expected_hr:
            raise ValueError(
                "PanCollection geometry mismatch: "
                f"LR={lr_shape[2:]}, Ref={ref_shape[2:]}, HR={hr_shape[2:]}, scale=x{self.scale}"
            )
        for name, expected, actual in (
            ("LR", self.expected_lr_channels, lr_shape[1]),
            ("Ref", self.expected_ref_channels, ref_shape[1]),
            ("HR", self.expected_hr_channels, hr_shape[1]),
        ):
            if expected is not None and actual != expected:
                raise ValueError(f"PanCollection {name} channels: expected {expected}, got {actual}")
        return shapes

    def _handle(self):
        if self._h5 is None or self._h5_pid != os.getpid():
            if self._h5 is not None:
                try:
                    self._h5.close()
                except Exception:
                    pass
            import h5py  # type: ignore

            self._h5 = h5py.File(self.file_path, "r")
            self._h5_pid = os.getpid()
        return self._h5

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_h5"] = None
        state["_h5_pid"] = None
        return state

    @staticmethod
    def _normalize(array: np.ndarray, value_scale: float, clip_range: bool) -> np.ndarray:
        values = np.asarray(array, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("PanCollection sample contains non-finite values")
        minimum, maximum = float(values.min()), float(values.max())
        if minimum >= -1.0 and maximum <= 1.0 and value_scale <= 1.0:
            normalized = values if minimum < 0.0 else values * 2.0 - 1.0
        else:
            normalized = values / value_scale * 2.0 - 1.0
        return np.clip(normalized, -1.0, 1.0) if clip_range else normalized

    @staticmethod
    def _spatial_transform(images: dict[str, np.ndarray], *, augment: bool) -> dict[str, np.ndarray]:
        result = images
        if augment and random.random() > 0.5:
            result = {key: np.flip(value, axis=2).copy() for key, value in result.items()}
        if augment and random.random() > 0.5:
            result = {key: np.flip(value, axis=1).copy() for key, value in result.items()}
        rotations = random.randint(0, 3) if augment else 0
        if rotations:
            result = {
                key: np.ascontiguousarray(np.rot90(value, rotations, axes=(1, 2)))
                for key, value in result.items()
            }
        return result

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if index < 0 or index >= len(self.indices):
            raise IndexError(index)
        source_index = self.indices[index]
        handle = self._handle()
        images = {
            "lr": self._normalize(handle[self.h5_lr_key][source_index], self.value_scale, self.clip_range),
            "ref": self._normalize(handle[self.h5_ref_key][source_index], self.value_scale, self.clip_range),
            "hr": self._normalize(handle[self.h5_hr_key][source_index], self.value_scale, self.clip_range),
        }
        if self.patch_size is not None:
            _, lr_height, lr_width = images["lr"].shape
            if lr_height < self.lr_patch_size or lr_width < self.lr_patch_size:
                raise ValueError(
                    f"LR sample is smaller than crop {self.lr_patch_size}x{self.lr_patch_size}: "
                    f"{lr_height}x{lr_width}"
                )
            rng = random if self.mode == "train" else random.Random(self.sample_seed + index)
            y = rng.randint(0, lr_height - self.lr_patch_size)
            x = rng.randint(0, lr_width - self.lr_patch_size)
            images["lr"] = images["lr"][:, y : y + self.lr_patch_size, x : x + self.lr_patch_size]
            y_hr, x_hr = y * self.scale, x * self.scale
            images["ref"] = images["ref"][:, y_hr : y_hr + self.patch_size, x_hr : x_hr + self.patch_size]
            images["hr"] = images["hr"][:, y_hr : y_hr + self.patch_size, x_hr : x_hr + self.patch_size]
        images = self._spatial_transform(images, augment=self.augment)
        result: dict[str, torch.Tensor | str] = {
            self.lr_key: torch.from_numpy(np.ascontiguousarray(images["lr"])),
            self.ref_key: torch.from_numpy(np.ascontiguousarray(images["ref"])),
            self.hr_key: torch.from_numpy(np.ascontiguousarray(images["hr"])),
        }
        if self.return_sample_id:
            result["sample_id"] = self.filenames[index]
        return result

    def __len__(self) -> int:
        return len(self.indices)


PanCollectionWV3Dataset = PanCollectionH5Dataset

__all__ = ["PanCollectionH5Dataset", "PanCollectionWV3Dataset"]
