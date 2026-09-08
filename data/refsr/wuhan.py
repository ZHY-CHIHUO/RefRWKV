"""Wuhan spatiotemporal-fusion dataset loader.

The public Wuhan dataset is organised as temporal-pair directories rather
than the flat ``HR/LR/Ref`` PNG layout used by the other RefSR datasets::

    <root>/<split>/<date1>_<date2>/
        G_<date1>.tif  G_<date2>.tif
        L_<date1>.tif  L_<date2>.tif

``G`` is the 8 m Gaofen image and ``L`` is the 30 m Landsat image.  The
released files have already been co-registered and resampled to the same
1000x1000 pixel grid.  Consequently this loader deliberately treats the
tensor geometry as ``scale=1``; the physical 30/8 resolution ratio belongs
to ERGAS, not to a pixel-shuffle upsampler.

Each item exposes the usual direct-RefSR aliases (``lr``, ``ref``, ``hr``)
for the ``L_t2, G_t1 -> G_t2`` prediction contract.  ``return_quadruple``
can additionally retain all four temporal images for a future STF model.
TIFF arrays are cached by absolute path in each dataset process, and one crop
and one spatial augmentation are shared by all four images.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


_NAME_RE = re.compile(r"^(?P<kind>[GL])_(?P<date>[^.]+)\.(?P<ext>tif|tiff)$", re.IGNORECASE)


class WuhanSTFDataset(Dataset):
    """Load aligned four-band Wuhan temporal pairs.

    Parameters are intentionally independent of the generic PNG dataset.
    ``patch_size`` is measured on the already aligned grid, so no divisibility
    relation with the physical 30/8 ratio is required.
    """

    _VALID_MODES = {"train", "val", "test", "test_easy", "test_hard"}

    def __init__(
        self,
        data_dir: str | Path,
        mode: str = "train",
        patch_size: int | None = 128,
        scale: int = 1,
        augment: bool = False,
        max_samples: int | tuple[int | None, int | None, int | None] | None = None,
        sample_seed: int = 42,
        lr_key: str = "lr",
        hr_key: str = "hr",
        ref_key: str = "ref",
        return_quadruple: bool = False,
        target_time: str = "t2",
        reference_time: str = "t1",
        value_scale: float = 11848.0,
        clip_range: bool = True,
        cache: bool = True,
        cache_size: int | None = None,
        lr_source: str = "stored",
        lr_native_scale: int | None = 1,
        lr_provenance: str = "sensor",
    ) -> None:
        mode = str(mode).strip().lower()
        if mode not in self._VALID_MODES:
            raise ValueError(f"Unknown Wuhan split: {mode}")
        if patch_size is not None:
            if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size < 1:
                raise ValueError("patch_size must be a positive integer or None")
        if isinstance(scale, bool) or not isinstance(scale, int) or scale != 1:
            raise ValueError(
                "Wuhan files are already aligned on one tensor grid; scale must be 1"
            )
        if str(lr_source).strip().lower() == "from_hr":
            raise ValueError("Wuhan sensor LR must be read from stored TIFF files")
        if lr_native_scale is not None and (
            isinstance(lr_native_scale, bool)
            or not isinstance(lr_native_scale, int)
            or lr_native_scale != 1
        ):
            raise ValueError("Wuhan lr_native_scale must be 1")
        if str(lr_provenance).strip().lower() not in {"sensor", "real", "measured"}:
            raise ValueError("Wuhan LR provenance must be sensor")
        if isinstance(sample_seed, bool) or not isinstance(sample_seed, int):
            raise ValueError("sample_seed must be an integer")
        for name, value in (("lr_key", lr_key), ("hr_key", hr_key), ("ref_key", ref_key)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        target_time = str(target_time).strip().lower()
        reference_time = str(reference_time).strip().lower()
        if target_time not in {"t1", "t2"} or reference_time not in {"t1", "t2"}:
            raise ValueError("target_time and reference_time must be t1 or t2")
        if target_time == reference_time:
            raise ValueError("target_time and reference_time must differ")
        if not np.isfinite(float(value_scale)) or float(value_scale) <= 0:
            raise ValueError("value_scale must be a positive finite number")
        if cache_size is not None:
            if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 1:
                raise ValueError("cache_size must be a positive integer or None")

        self.data_dir = Path(data_dir).expanduser().resolve()
        self.mode = mode
        self.patch_size = patch_size
        self.sample_seed = sample_seed
        self.augment = bool(augment) and mode == "train"
        self.lr_key, self.hr_key, self.ref_key = lr_key, hr_key, ref_key
        self.return_quadruple = bool(return_quadruple)
        self.target_time, self.reference_time = target_time, reference_time
        self.value_scale = float(value_scale)
        self.clip_range = bool(clip_range)
        self.cache_enabled = bool(cache)
        self.cache_size = cache_size
        # The explicit path-keyed dictionary is useful for both performance and
        # diagnostics (and is intentionally process-local when workers > 0).
        self._cache: dict[str, np.ndarray] = {}
        self.path_cache = self._cache

        split_dir = self.data_dir / mode
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Wuhan split directory not found: {split_dir}")
        self.pairs = self._discover_pairs(split_dir)
        if not self.pairs:
            raise FileNotFoundError(f"no Wuhan temporal pairs found in {split_dir}")

        requested = self._requested_count(max_samples, mode)
        if requested is not None:
            if requested < 0:
                raise ValueError("max_samples must be non-negative")
            if requested > len(self.pairs):
                requested = len(self.pairs)
            indices = sorted(random.Random(sample_seed).sample(range(len(self.pairs)), requested))
            self.pairs = [self.pairs[index] for index in indices]

        self.filenames = [pair["name"] for pair in self.pairs]
        print(
            f"WuhanSTFDataset [{mode}]: {len(self.pairs)} temporal pairs, "
            f"channels=4, tensor_scale=x1, return_quadruple={self.return_quadruple}"
        )
        if patch_size is not None:
            print(f"  Synchronized crop: {patch_size}x{patch_size}")
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
            index = 0 if mode == "train" else 1 if mode == "val" else 2
            value = value[index]
            if value is None:
                return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("max_samples must be an integer, tuple, or None")
        return int(value)

    @classmethod
    def _discover_pairs(cls, split_dir: Path) -> list[dict[str, Any]]:
        pairs: list[dict[str, Any]] = []
        for pair_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            dates = pair_dir.name.split("_")
            if len(dates) != 2 or not all(dates):
                # Ignore unrelated metadata directories, but fail if a
                # directory looks like a pair and is incomplete below.
                continue
            date1, date2 = dates
            files: dict[tuple[str, str], Path] = {}
            for path in pair_dir.iterdir():
                if not path.is_file():
                    continue
                match = _NAME_RE.match(path.name)
                if match is not None:
                    files[(match.group("kind").upper(), match.group("date"))] = path
            required = {
                ("G", date1): "g_t1",
                ("G", date2): "g_t2",
                ("L", date1): "l_t1",
                ("L", date2): "l_t2",
            }
            missing = [label for key, label in required.items() if key not in files]
            if missing:
                raise FileNotFoundError(
                    f"incomplete Wuhan pair {pair_dir.name!r}; missing {', '.join(missing)}"
                )
            pairs.append(
                {
                    "name": pair_dir.name,
                    "date1": date1,
                    "date2": date2,
                    "l_t1": files[("L", date1)],
                    "l_t2": files[("L", date2)],
                    "g_t1": files[("G", date1)],
                    "g_t2": files[("G", date2)],
                }
            )
        return pairs

    @staticmethod
    def _as_hwc(array: np.ndarray, path: Path) -> np.ndarray:
        array = np.asarray(array)
        while array.ndim > 3 and 1 in array.shape:
            array = np.squeeze(array)
        if array.ndim == 2:
            array = array[..., None]
        if array.ndim != 3:
            raise ValueError(f"Wuhan TIFF must be 2D/3D, got {array.shape} for {path}")
        # tifffile/rasterio commonly return (C,H,W), while PIL returns (H,W,C).
        if array.shape[0] == 4 and array.shape[-1] != 4:
            array = np.transpose(array, (1, 2, 0))
        if array.shape[-1] != 4:
            raise ValueError(
                f"Wuhan TIFF must have exactly 4 channels, got {array.shape} for {path}"
            )
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"Wuhan TIFF must contain numeric data: {path}")
        return np.ascontiguousarray(array)

    @classmethod
    def _read_tiff(cls, path: Path) -> np.ndarray:
        errors: list[Exception] = []
        try:
            import tifffile  # type: ignore

            return cls._as_hwc(tifffile.imread(str(path)), path)
        except Exception as exc:  # pragma: no cover - fallback depends on env
            errors.append(exc)
        try:
            import rasterio  # type: ignore

            with rasterio.open(path) as source:
                return cls._as_hwc(source.read(), path)
        except Exception as exc:  # pragma: no cover - fallback depends on env
            errors.append(exc)
        try:
            from PIL import Image

            with Image.open(path) as image:
                return cls._as_hwc(np.asarray(image), path)
        except Exception as exc:  # pragma: no cover - fallback depends on env
            errors.append(exc)
        detail = "; ".join(f"{type(error).__name__}: {error}" for error in errors[-3:])
        raise RuntimeError(
            f"Unable to read Wuhan TIFF {path}. Install tifffile (recommended) "
            f"or rasterio/Pillow. Details: {detail}"
        )

    def _load_cached(self, path: Path) -> np.ndarray:
        key = str(path.resolve())
        if self.cache_enabled and key in self._cache:
            return self._cache[key]
        array = self._read_tiff(path)
        if self.cache_enabled:
            if self.cache_size is not None and len(self._cache) >= self.cache_size:
                # FIFO eviction keeps the implementation deterministic and
                # avoids an unbounded resident set for repeated experiments.
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = array
        return array

    def _load_pair(self, pair: dict[str, Any]) -> dict[str, np.ndarray]:
        images = {
            key: self._load_cached(pair[key])
            for key in ("l_t1", "l_t2", "g_t1", "g_t2")
        }
        shape = images["l_t1"].shape
        for key, image in images.items():
            if image.shape != shape:
                raise ValueError(
                    f"unaligned Wuhan pair {pair['name']!r}: {key}={image.shape}, expected {shape}"
                )
        return images

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        value = image.astype(np.float32, copy=False) / self.value_scale
        if self.clip_range:
            value = np.clip(value, 0.0, 1.0)
        return value * 2.0 - 1.0

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert an HWC raw image to the repository's NCHW value range."""
        return torch.from_numpy(
            np.ascontiguousarray(np.transpose(self._normalize(image), (2, 0, 1)))
        )

    def _crop_and_augment(
        self, images: dict[str, np.ndarray], rng: Any
    ) -> dict[str, np.ndarray]:
        if self.patch_size is not None:
            height, width = images["l_t1"].shape[:2]
            if height < self.patch_size or width < self.patch_size:
                raise ValueError(
                    f"Wuhan image {width}x{height} is smaller than crop "
                    f"{self.patch_size}x{self.patch_size}"
                )
            y = rng.randint(0, height - self.patch_size)
            x = rng.randint(0, width - self.patch_size)
            images = {
                key: image[y : y + self.patch_size, x : x + self.patch_size]
                for key, image in images.items()
            }
        if self.augment:
            if random.random() > 0.5:
                images = {key: np.flip(image, axis=1).copy() for key, image in images.items()}
            if random.random() > 0.5:
                images = {key: np.flip(image, axis=0).copy() for key, image in images.items()}
            rotations = random.randint(0, 3)
            if rotations:
                images = {
                    key: np.ascontiguousarray(np.rot90(image, rotations, axes=(0, 1)))
                    for key, image in images.items()
                }
        return images

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pair = self.pairs[index]
        images = self._load_pair(pair)
        rng = random if self.mode == "train" else random.Random(self.sample_seed + int(index))
        images = self._crop_and_augment(images, rng)

        # Direct RefSR contract: L at target time + G at reference time -> G
        # at target time.  The temporal aliases remain available on request.
        target_lr = images[f"l_{self.target_time}"]
        target_hr = images[f"g_{self.target_time}"]
        reference = images[f"g_{self.reference_time}"]
        values = {
            self.lr_key: target_lr,
            self.hr_key: target_hr,
            self.ref_key: reference,
        }
        if self.return_quadruple:
            values.update(
                {
                    "lr_t1": images["l_t1"],
                    "lr_t2": images["l_t2"],
                    "hr_t1": images["g_t1"],
                    "hr_t2": images["g_t2"],
                }
            )
        return {
            key: self._to_tensor(value)
            for key, value in values.items()
        }

    def __len__(self) -> int:
        return len(self.pairs)


WuhanDataset = WuhanSTFDataset
WuhanTemporalDataset = WuhanSTFDataset

__all__ = ["WuhanDataset", "WuhanSTFDataset", "WuhanTemporalDataset"]
