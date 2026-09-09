"""Unified PNG dataset for SISR and reference-based super-resolution.

The physical dataset contract is intentionally small:

    <root>/<split>/HR/*.png
    <root>/<split>/LR/*.png       (optional for bicubic data)
    <root>/<split>/Ref/*.png      (optional unless a stored reference is used)

``SuperResolutionDataset`` decides which tensors to return through
``return_items``.  It can therefore be used for ordinary SISR (``lr, hr``),
paired RefSR (``lr, hr, ref``), or HR-only preparation runs where LR is
generated from HR in memory.  Task-specific entry points in
:mod:`data.sr.dataset` and :mod:`data.refsr.dataset` expose the same loader
with task-oriented defaults.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


class SuperResolutionDataset(Dataset):
    """Load aligned HR/LR/Ref PNGs with an explicit physical LR contract.

    Args:
        data_dir: Dataset root containing ``train``/``val``/``test`` splits.
        return_items: Ordered subset of ``("lr", "hr", "ref")``. ``hr`` is
            required because this dataset is supervised. The default returns
            ``lr`` and ``hr``.
        lr_source: ``auto`` uses a matching stored LR and otherwise generates
            bicubic LR from HR; ``stored`` requires an exact stored LR;
            ``from_hr`` always generates LR from HR.
        lr_native_scale: Physical scale of the stored LR representation. It
            is informational for bicubic data and mandatory for sensor data.
        lr_provenance: ``bicubic`` permits HR-derived scale changes;
            ``sensor`` forbids them and requires the stored LR at its native
            scale.
        reference_source: ``stored`` reads ``Ref/``; ``lr_up`` creates a
            bicubic HR-resolution reference from LR; ``none`` disables refs.
    """

    _VALID_MODES = {"train", "val", "test", "test_easy", "test_hard"}
    _VALID_ITEMS = ("lr", "hr", "ref")
    _VALID_LR_SOURCES = {"auto", "stored", "from_hr"}
    _VALID_LR_PROVENANCE = {"bicubic", "sensor"}
    _VALID_REFERENCE_SOURCES = {"none", "stored", "lr_up"}

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
        ref_key: str = "ref",
        return_items: Sequence[str] | str = ("lr", "hr"),
        return_ref: bool | None = None,
        lr_source: str = "auto",
        lr_native_scale: int | None = None,
        lr_provenance: str = "bicubic",
        reference_source: str | None = None,
        augment_ref: bool = False,
        ref_aug_strengths: Sequence[float] | None = None,
        ref_aug_probs: Sequence[float] | None = None,
        ref_gray_prob: float = 0.2,
        return_sample_id: bool = False,
    ) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(f"Unknown mode: {mode}")
        if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
            raise ValueError(f"scale must be a positive integer, got {scale!r}")
        if patch_size is not None:
            if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size < 1:
                raise ValueError(f"patch_size must be a positive integer or None, got {patch_size!r}")
            if patch_size % scale:
                raise ValueError(f"patch_size ({patch_size}) must be divisible by scale ({scale})")
        if not isinstance(max_samples, (tuple, list)) or len(max_samples) != 3:
            raise ValueError("max_samples must be a (train, val, test) tuple")
        if isinstance(sample_seed, bool) or not isinstance(sample_seed, int):
            raise ValueError("sample_seed must be an integer")
        for name, value in (("lr_key", lr_key), ("hr_key", hr_key), ("ref_key", ref_key)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")

        items = self._normalize_items(return_items)
        if return_ref is not None:
            if not isinstance(return_ref, bool):
                raise ValueError("return_ref must be a boolean or None")
            items = tuple(item for item in items if item != "ref")
            if return_ref:
                items = (*items, "ref")
        if "hr" not in items:
            raise ValueError("return_items must contain 'hr'")

        lr_source = str(lr_source).strip().lower()
        if lr_source not in self._VALID_LR_SOURCES:
            options = ", ".join(sorted(self._VALID_LR_SOURCES))
            raise ValueError(f"lr_source must be one of {options}, got {lr_source!r}")
        lr_provenance = str(lr_provenance).strip().lower()
        if lr_provenance not in self._VALID_LR_PROVENANCE:
            options = ", ".join(sorted(self._VALID_LR_PROVENANCE))
            raise ValueError(
                f"lr_provenance must be one of {options}, got {lr_provenance!r}"
            )
        if lr_native_scale is not None:
            if isinstance(lr_native_scale, bool) or not isinstance(lr_native_scale, int) or lr_native_scale < 1:
                raise ValueError("lr_native_scale must be a positive integer or None")
        if lr_provenance == "sensor" and lr_source == "from_hr":
            raise ValueError("sensor LR cannot use lr_source='from_hr'")

        if reference_source is None:
            reference_source = "stored" if "ref" in items else "none"
        reference_source = str(reference_source).strip().lower()
        if reference_source not in self._VALID_REFERENCE_SOURCES:
            options = ", ".join(sorted(self._VALID_REFERENCE_SOURCES))
            raise ValueError(
                f"reference_source must be one of {options}, got {reference_source!r}"
            )
        if "ref" in items and reference_source == "none":
            raise ValueError("return_items contains 'ref' but reference_source='none'")
        if "ref" not in items and reference_source == "stored":
            # Reading a physical Ref is opt-in. This avoids requiring Ref when
            # SwinIR intentionally ignores it on a RefSR directory.
            reference_source = "none"

        if ref_aug_strengths is None:
            ref_aug_strengths = (0.12, 0.12, 0.12, 0.03)
        if ref_aug_probs is None:
            ref_aug_probs = (0.5, 0.5, 0.5, 0.5)
        if len(ref_aug_strengths) != 4 or len(ref_aug_probs) != 4:
            raise ValueError("ref_aug_strengths and ref_aug_probs must contain four values")
        if not 0.0 <= float(ref_gray_prob) <= 1.0:
            raise ValueError("ref_gray_prob must be in [0, 1]")

        self.data_dir = Path(data_dir)
        self.mode = mode
        self.patch_size = patch_size
        self.scale = scale
        self.lr_patch_size = None if patch_size is None else patch_size // scale
        self.sample_seed = sample_seed
        self.augment = bool(augment) and mode == "train"
        self.augment_ref = bool(augment_ref) and mode == "train" and "ref" in items
        self.lr_key, self.hr_key, self.ref_key = lr_key, hr_key, ref_key
        self.return_items = items
        self.lr_source = lr_source
        self.lr_native_scale = lr_native_scale if lr_native_scale is not None else scale
        self.lr_provenance = lr_provenance
        self.reference_source = reference_source
        self.ref_aug_strengths = tuple(float(value) for value in ref_aug_strengths)
        self.ref_aug_probs = tuple(float(value) for value in ref_aug_probs)
        self.ref_gray_prob = float(ref_gray_prob)
        self.return_sample_id = bool(return_sample_id)

        split_dir = self.data_dir / mode
        self.lr_dir = split_dir / "LR"
        self.hr_dir = split_dir / "HR"
        self.ref_dir = split_dir / "Ref"
        if not self.hr_dir.is_dir():
            raise FileNotFoundError(f"HR directory not found: {self.hr_dir}")

        self.needs_lr = "lr" in items or reference_source == "lr_up"
        self.needs_ref = "ref" in items and reference_source == "stored"
        if self.needs_lr and lr_source == "stored" and not self.lr_dir.is_dir():
            raise FileNotFoundError(f"LR directory not found: {self.lr_dir}")
        if self.needs_lr and lr_provenance == "sensor" and not self.lr_dir.is_dir():
            raise FileNotFoundError(
                f"sensor LR requires a stored LR directory: {self.lr_dir}"
            )
        if self.needs_ref and not self.ref_dir.is_dir():
            raise FileNotFoundError(f"Ref directory not found: {self.ref_dir}")

        all_names = sorted(path.stem for path in self.hr_dir.glob("*.png"))
        if not all_names:
            raise FileNotFoundError(f"no HR PNG files found in {self.hr_dir}")
        if self.needs_lr and self.lr_source != "from_hr" and self.lr_dir.is_dir():
            for name in all_names:
                lr_path = self.lr_dir / f"{name}.png"
                if not lr_path.is_file() and (self.lr_source == "stored" or lr_provenance == "sensor"):
                    raise FileNotFoundError(f"Missing LR for HR sample {name!r}: {lr_path}")
        if self.needs_ref:
            for name in all_names:
                ref_path = self.ref_dir / f"{name}.png"
                if not ref_path.is_file():
                    raise FileNotFoundError(f"Missing Ref for HR sample {name!r}: {ref_path}")

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

        print(
            f"SuperResolutionDataset [{mode}]: {len(self.filenames)} samples "
            f"return={','.join(self.return_items)} lr={self.lr_provenance}/x{self.lr_native_scale}"
        )
        if patch_size is not None:
            print(
                f"  Random crop: HR {patch_size}x{patch_size} -> "
                f"LR {self.lr_patch_size}x{self.lr_patch_size}"
            )
        if self.augment:
            print("  Spatial augmentation: ON (flip & rot90)")
        if self.augment_ref:
            print("  Ref style augmentation: ON")

    @classmethod
    def _normalize_items(cls, value: Sequence[str] | str) -> tuple[str, ...]:
        if isinstance(value, str):
            raw: Iterable[str] = value.replace(",", " ").split()
        else:
            raw = value
        try:
            items = tuple(str(item).strip().lower() for item in raw)
        except TypeError as exc:
            raise ValueError("return_items must be a sequence of lr/hr/ref names") from exc
        if not items or len(set(items)) != len(items) or any(item not in cls._VALID_ITEMS for item in items):
            raise ValueError("return_items must be a non-empty subset of unique lr/hr/ref names")
        return items

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0

    @staticmethod
    def _resize_normalized(image: np.ndarray, height: int, width: int) -> np.ndarray:
        if image.shape[:2] == (height, width):
            return image
        pil = Image.fromarray(np.clip((image + 1.0) * 127.5, 0, 255).astype(np.uint8))
        resized = pil.resize((width, height), Image.Resampling.BICUBIC)
        return np.asarray(resized, dtype=np.float32) / 127.5 - 1.0

    def _stored_lr(self, name: str) -> np.ndarray | None:
        if self.lr_source == "from_hr":
            return None
        path = self.lr_dir / f"{name}.png"
        if not path.is_file():
            if self.lr_source == "stored" or self.lr_provenance == "sensor":
                raise FileNotFoundError(f"Missing LR for sample {name!r}: {path}")
            return None
        return self._load_image(path)

    def _lr_for_scale(self, name: str, hr: np.ndarray) -> np.ndarray:
        target_height, target_width = hr.shape[0] // self.scale, hr.shape[1] // self.scale
        if self.lr_provenance == "sensor" and self.scale != self.lr_native_scale:
            raise ValueError(
                f"sensor LR only supports its native scale x{self.lr_native_scale}; "
                f"requested x{self.scale} for {name!r}"
            )
        stored = self._stored_lr(name)
        if stored is not None and stored.shape[:2] == (target_height, target_width):
            return stored
        if self.lr_provenance == "sensor":
            raise ValueError(
                f"sensor LR geometry or scale mismatch for {name!r}: "
                f"stored={None if stored is None else stored.shape[:2]}, "
                f"requested={(target_height, target_width)} at x{self.scale}; "
                f"native scale is x{self.lr_native_scale}"
            )
        if self.lr_source == "stored" and stored is not None:
            raise ValueError(
                f"stored LR geometry mismatch for {name!r}: "
                f"stored={stored.shape[:2]}, requested={(target_height, target_width)}"
            )
        return self._resize_normalized(hr, target_height, target_width)

    def _validate_geometry(self, lr: np.ndarray, hr: np.ndarray, name: str) -> None:
        expected = (lr.shape[0] * self.scale, lr.shape[1] * self.scale)
        if hr.shape[:2] != expected:
            raise ValueError(
                f"Unaligned LR/HR pair {name!r}: LR={lr.shape[1]}x{lr.shape[0]}, "
                f"HR={hr.shape[1]}x{hr.shape[0]}, expected HR={expected[1]}x{expected[0]} "
                f"for x{self.scale}"
            )

    def _load_reference(self, name: str, lr: np.ndarray | None, hr: np.ndarray) -> np.ndarray:
        if self.reference_source == "stored":
            ref = self._load_image(self.ref_dir / f"{name}.png")
            if self.augment_ref:
                with Image.open(self.ref_dir / f"{name}.png") as image:
                    ref_image = image.convert("RGB")
                ref_image = self._augment_ref(ref_image)
                ref = np.asarray(ref_image, dtype=np.float32) / 127.5 - 1.0
        elif self.reference_source == "lr_up":
            if lr is None:
                raise ValueError("reference_source='lr_up' requires an LR tensor")
            ref = self._resize_normalized(lr, hr.shape[0], hr.shape[1])
        else:
            raise ValueError("a reference was requested but reference_source='none'")
        if ref.shape[:2] != hr.shape[:2]:
            raise ValueError(
                f"Ref/HR size mismatch for {name!r}: Ref={ref.shape[:2]}, HR={hr.shape[:2]}"
            )
        return ref

    def _augment_ref(self, image: Image.Image) -> Image.Image:
        if random.random() < self.ref_gray_prob:
            return image.convert("L").convert("RGB")
        for index, (strength, probability) in enumerate(
            zip(self.ref_aug_strengths, self.ref_aug_probs)
        ):
            if random.random() > probability:
                continue
            if index == 0:
                image = ImageEnhance.Brightness(image).enhance(
                    1.0 + (random.random() * 2.0 - 1.0) * strength
                )
            elif index == 1:
                image = ImageEnhance.Contrast(image).enhance(
                    1.0 + (random.random() * 2.0 - 1.0) * strength
                )
            elif index == 2:
                image = ImageEnhance.Color(image).enhance(
                    1.0 + (random.random() * 2.0 - 1.0) * strength
                )
            elif index == 3 and strength > 0:
                image = self._hue_rotate(image, random.uniform(-strength, strength))
        return image

    @staticmethod
    def _hue_rotate(image: Image.Image, delta: float) -> Image.Image:
        if delta == 0:
            return image
        hsv = image.convert("HSV")
        hue, saturation, value = hsv.split()
        hue_array = (np.asarray(hue, dtype=np.int32) + int(delta * 255)) % 256
        return Image.merge(
            "HSV", (Image.fromarray(hue_array.astype(np.uint8), mode="L"), saturation, value)
        ).convert("RGB")

    def _random_crop(
        self,
        images: dict[str, np.ndarray],
        rng: Any,
    ) -> dict[str, np.ndarray]:
        assert self.patch_size is not None
        hr = images["hr"]
        height, width = hr.shape[:2]
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"HR image {width}x{height} is smaller than crop {self.patch_size}x{self.patch_size}"
            )
        lr = images.get("lr")
        if lr is not None:
            assert self.lr_patch_size is not None
            lr_height, lr_width = lr.shape[:2]
            if lr_height < self.lr_patch_size or lr_width < self.lr_patch_size:
                raise ValueError(
                    f"LR image {lr_width}x{lr_height} is smaller than crop "
                    f"{self.lr_patch_size}x{self.lr_patch_size}"
                )
            y_lr = rng.randint(0, lr_height - self.lr_patch_size)
            x_lr = rng.randint(0, lr_width - self.lr_patch_size)
            y_hr, x_hr = y_lr * self.scale, x_lr * self.scale
            result = dict(images)
            result["lr"] = lr[y_lr : y_lr + self.lr_patch_size, x_lr : x_lr + self.lr_patch_size]
            result["hr"] = hr[y_hr : y_hr + self.patch_size, x_hr : x_hr + self.patch_size]
            if "ref" in result:
                result["ref"] = result["ref"][y_hr : y_hr + self.patch_size, x_hr : x_hr + self.patch_size]
            return result

        y_hr = rng.randint(0, height - self.patch_size)
        x_hr = rng.randint(0, width - self.patch_size)
        result = dict(images)
        result["hr"] = hr[y_hr : y_hr + self.patch_size, x_hr : x_hr + self.patch_size]
        if "ref" in result:
            result["ref"] = result["ref"][y_hr : y_hr + self.patch_size, x_hr : x_hr + self.patch_size]
        return result

    @staticmethod
    def _augment(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result = dict(images)
        if random.random() > 0.5:
            result = {key: np.flip(value, axis=1).copy() for key, value in result.items()}
        if random.random() > 0.5:
            result = {key: np.flip(value, axis=0).copy() for key, value in result.items()}
        rotations = random.randint(0, 3)
        if rotations:
            result = {
                key: np.ascontiguousarray(np.rot90(value, rotations, axes=(0, 1)))
                for key, value in result.items()
            }
        return result

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        name = self.filenames[idx]
        hr = self._load_image(self.hr_dir / f"{name}.png")
        if hr.shape[0] % self.scale or hr.shape[1] % self.scale:
            raise ValueError(
                f"HR image {name!r} size {hr.shape[1]}x{hr.shape[0]} cannot be represented at x{self.scale}"
            )

        images: dict[str, np.ndarray] = {"hr": hr}
        lr: np.ndarray | None = None
        if self.needs_lr:
            lr = self._lr_for_scale(name, hr)
            self._validate_geometry(lr, hr, name)
            images["lr"] = lr
        if "ref" in self.return_items:
            images["ref"] = self._load_reference(name, lr, hr)

        if self.patch_size is not None:
            crop_rng = random if self.mode == "train" else random.Random(self.sample_seed + idx)
            images = self._random_crop(images, rng=crop_rng)
        if self.augment:
            images = self._augment(images)

        result: dict[str, torch.Tensor] = {}
        key_map = {"lr": self.lr_key, "hr": self.hr_key, "ref": self.ref_key}
        for item in self.return_items:
            array = np.ascontiguousarray(np.transpose(images[item], (2, 0, 1)))
            result[key_map[item]] = torch.from_numpy(array)
        if self.return_sample_id:
            result["sample_id"] = name
        return result

    def __len__(self) -> int:
        return len(self.filenames)


UnifiedSRDataset = SuperResolutionDataset

__all__ = ["SuperResolutionDataset", "UnifiedSRDataset"]
