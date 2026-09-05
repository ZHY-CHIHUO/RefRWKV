"""DataLoader factories for the two physically separate data contracts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torch.utils.data import ConcatDataset, DataLoader, Dataset

from data.refsr import RefPNGDataset
from data.sr import SRPNGDataset
from runtime.common import resolve_path
from runtime.config import normalize_reference_mode, validate_refsr_reference_contract


def _max_samples(data: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (
        data.get("max_samples_train"),
        data.get("max_samples_val"),
        data.get("max_samples_test"),
    )


def _configured_roots(data: Mapping[str, Any]) -> list[Path]:
    """Return explicit data roots, allowing one root or a list of roots.

    ``data.root`` remains the normal single-dataset setting.  ``data.roots``
    is useful when a run intentionally combines selected datasets.  A root can
    also be the task container (for example ``data/sr``), in which case its
    direct dataset children are discovered below.
    """
    # An explicit YAML ``roots: null`` should retain the normal single-root
    # fallback instead of turning a valid ``root`` configuration into an error.
    raw_roots = data.get("roots") or data.get("root")
    if isinstance(raw_roots, (str, Path)):
        raw_roots = [raw_roots]
    if not isinstance(raw_roots, (list, tuple)) or not raw_roots:
        raise ValueError("data.root or non-empty data.roots must be configured")

    roots: list[Path] = []
    for raw_root in raw_roots:
        if not isinstance(raw_root, (str, Path)):
            raise ValueError("data.roots entries must be paths")
        root = resolve_path(raw_root)
        if not root.is_dir():
            raise FileNotFoundError(f"data root not found: {root}")
        if root not in roots:
            roots.append(root)
    return roots


def _has_any_split(root: Path) -> bool:
    return any((root / split).is_dir() for split in ("train", "val", "test", "test_easy", "test_hard"))


def _dataset_roots(
    data: Mapping[str, Any],
    *,
    required_splits: tuple[str, ...],
    require_reference: bool,
) -> list[Path]:
    """Resolve dataset roots for one task contract.

    A task container is intentionally discovered only one level deep.  This
    keeps ``data/sr`` and ``data/refsr`` physically separate and prevents raw
    archives or arbitrary nested directories from being treated as datasets.
    Every discovered dataset must contain all splits requested by the caller;
    silently skipping an incomplete dataset would make an aggregate run hard
    to reproduce.
    """
    roots: list[Path] = []
    expected_dirs = ("HR", "LR", "Ref") if require_reference else ("HR", "LR")
    for configured in _configured_roots(data):
        candidates = [configured] if _has_any_split(configured) else sorted(
            child for child in configured.iterdir() if child.is_dir() and _has_any_split(child)
        )
        if not candidates:
            raise FileNotFoundError(
                f"no dataset roots found below {configured}; expected <dataset>/<split>/{'{HR,LR,Ref}' if require_reference else '{HR,LR}'}"
            )
        for candidate in candidates:
            missing = [
                str(candidate / split / directory)
                for split in required_splits
                for directory in expected_dirs
                if not (candidate / split / directory).is_dir()
            ]
            if missing:
                joined = ", ".join(missing)
                raise FileNotFoundError(f"incomplete dataset root {candidate}: missing {joined}")
            if candidate not in roots:
                roots.append(candidate)
    return roots


def _combine(datasets: list[Dataset]) -> Dataset:
    if not datasets:
        raise ValueError("no datasets were constructed")
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def _loader(dataset, data: Mapping[str, Any], *, train: bool) -> DataLoader:
    workers = int(data.get("num_workers" if train else "val_num_workers", 4 if train else 2))
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(data.get("batch_size" if train else "val_batch_size", 4 if train else 1)),
        "shuffle": train,
        "num_workers": workers,
        "pin_memory": bool(data.get("pin_memory", True)),
    }
    if train:
        kwargs["drop_last"] = bool(data.get("drop_last", True)) and len(dataset) >= kwargs["batch_size"]
    if workers > 0:
        kwargs["persistent_workers"] = bool(data.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(data.get("prefetch_factor", 2))
    return DataLoader(**kwargs)


def build_sr_loaders(config: Mapping[str, Any]):
    """Build HR/LR-only loaders for one or more single-image SR datasets."""
    data = config["data"]
    common = {
        "scale": int(data["scale"]),
        "max_samples": _max_samples(data),
        "sample_seed": int(data.get("sample_seed", config.get("train", {}).get("seed", 42))),
        "lr_key": data.get("lr_key", "lr"),
        "hr_key": data.get("hr_key", "hr"),
    }
    roots = _dataset_roots(data, required_splits=("train", "val"), require_reference=False)
    train = _combine([
        SRPNGDataset(
            data_dir=root,
            mode="train",
            patch_size=data.get("patch_size"),
            augment=bool(data.get("augment", True)),
            **common,
        )
        for root in roots
    ])
    val = _combine([
        SRPNGDataset(
            data_dir=root,
            mode="val",
            patch_size=data.get("val_patch_size"),
            augment=False,
            **common,
        )
        for root in roots
    ])
    if not len(train) or not len(val):
        raise ValueError("training and validation splits must be non-empty")
    return _loader(train, data, train=True), _loader(val, data, train=False)


def build_refsr_loaders(config: Mapping[str, Any]):
    """Build loaders for one or more paired or LR-derived RefSR datasets."""
    validate_refsr_reference_contract(config)
    data = config["data"]
    reference_mode = normalize_reference_mode(data.get("reference_mode", "paired"))
    dataset_cls = SRPNGDataset if reference_mode == "lr_up" else RefPNGDataset
    common = {
        "scale": int(data["scale"]),
        "max_samples": _max_samples(data),
        "sample_seed": int(data.get("sample_seed", config.get("train", {}).get("seed", 42))),
        "lr_key": data.get("lr_key", "lr"),
        "hr_key": data.get("hr_key", "hr"),
    }
    extra: dict[str, Any] = {}
    if dataset_cls is RefPNGDataset:
        extra = {
            "ref_key": data.get("ref_key", "ref"),
            "ref_aug_strengths": data.get("ref_aug_strengths", [0.12, 0.12, 0.12, 0.03]),
            "ref_aug_probs": data.get("ref_aug_probs", [0.5, 0.5, 0.5, 0.5]),
            "ref_gray_prob": float(data.get("ref_gray_prob", 0.2)),
        }
    roots = _dataset_roots(
        data,
        required_splits=("train", "val"),
        require_reference=dataset_cls is RefPNGDataset,
    )
    train_datasets: list[Dataset] = []
    val_datasets: list[Dataset] = []
    for root in roots:
        train_kwargs = dict(
            data_dir=root,
            mode="train",
            patch_size=data.get("patch_size"),
            augment=bool(data.get("augment", True)),
            **common,
            **extra,
        )
        val_kwargs = dict(
            data_dir=root,
            mode="val",
            patch_size=data.get("val_patch_size"),
            augment=False,
            **common,
            **extra,
        )
        if dataset_cls is RefPNGDataset:
            train_kwargs["augment_ref"] = bool(data.get("augment_ref", False))
            val_kwargs["augment_ref"] = False
        train_datasets.append(dataset_cls(**train_kwargs))
        val_datasets.append(dataset_cls(**val_kwargs))
    train, val = _combine(train_datasets), _combine(val_datasets)
    if not len(train) or not len(val):
        raise ValueError("training and validation splits must be non-empty")
    return _loader(train, data, train=True), _loader(val, data, train=False)


def _test_dataset_kwargs(config: Mapping[str, Any], split: str) -> dict[str, Any]:
    """Build the common immutable part of a test dataset configuration.

    Test loaders intentionally do not inherit the training crop.  Unless a
    caller explicitly sets ``data.test_patch_size``, each sample is evaluated
    at its native resolution and can therefore be written back to a PNG
    without any stitching or padding policy.
    """
    data = config["data"]
    split = str(split).strip()
    if split not in {"test", "test_easy", "test_hard"}:
        raise ValueError("split must be one of test, test_easy, or test_hard")
    max_samples = _max_samples(data)
    return {
        "mode": split,
        "patch_size": data.get("test_patch_size"),
        "scale": int(data["scale"]),
        "augment": False,
        "max_samples": max_samples,
        "sample_seed": int(data.get("sample_seed", config.get("train", {}).get("seed", 42))),
        "lr_key": data.get("lr_key", "lr"),
        "hr_key": data.get("hr_key", "hr"),
    }


def _test_loader(dataset, data: Mapping[str, Any], *, batch_size: int | None = None) -> DataLoader:
    options = dict(data)
    if batch_size is not None:
        options["val_batch_size"] = int(batch_size)
    return _loader(dataset, options, train=False)


def build_sr_test_loader(
    config: Mapping[str, Any], *, split: str = "test", batch_size: int | None = None
) -> DataLoader:
    """Build a native-resolution HR/LR test loader for an SR run."""
    data = config["data"]
    roots = _dataset_roots(data, required_splits=(split,), require_reference=False)
    dataset = _combine([
        SRPNGDataset(data_dir=root, **_test_dataset_kwargs(config, split)) for root in roots
    ])
    if not len(dataset):
        raise ValueError(f"test split {split!r} is empty")
    return _test_loader(dataset, data, batch_size=batch_size)


def build_refsr_test_loader(
    config: Mapping[str, Any], *, split: str = "test", batch_size: int | None = None
) -> DataLoader:
    """Build a native-resolution RefSR test loader.

    ``reference_mode=lr_up`` remains supported for SISR datasets used to
    train/evaluate RefSRWKV without a real reference image.
    """
    validate_refsr_reference_contract(config)
    data = config["data"]
    reference_mode = normalize_reference_mode(data.get("reference_mode", "paired"))
    dataset_cls = SRPNGDataset if reference_mode == "lr_up" else RefPNGDataset
    roots = _dataset_roots(
        data,
        required_splits=(split,),
        require_reference=dataset_cls is RefPNGDataset,
    )
    datasets: list[Dataset] = []
    for root in roots:
        kwargs = dict(data_dir=root, **_test_dataset_kwargs(config, split))
        if dataset_cls is RefPNGDataset:
            kwargs.update(
                ref_key=data.get("ref_key", "ref"),
                ref_aug_strengths=data.get("ref_aug_strengths", [0.12, 0.12, 0.12, 0.03]),
                ref_aug_probs=data.get("ref_aug_probs", [0.5, 0.5, 0.5, 0.5]),
                ref_gray_prob=float(data.get("ref_gray_prob", 0.2)),
                augment_ref=False,
            )
        datasets.append(dataset_cls(**kwargs))
    dataset = _combine(datasets)
    if not len(dataset):
        raise ValueError(f"test split {split!r} is empty")
    return _test_loader(dataset, data, batch_size=batch_size)


__all__ = [
    "build_sr_loaders",
    "build_refsr_loaders",
    "build_sr_test_loader",
    "build_refsr_test_loader",
]
