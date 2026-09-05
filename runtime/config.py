"""Configuration loading for the task/model/data split.

Run YAML files are intentionally small.  They may inherit one or more base
files, refer to a dataset description, and override any value from the
command line with ``section.key=value``.  The loader returns one materialized
mapping consumed by both training and evaluation entry points.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from .common import PROJECT_ROOT, deep_merge, resolve_path


def load_yaml_file(path: str | os.PathLike[str], stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read a YAML file and recursively merge its ``base`` references."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved) if (Path.cwd() / resolved).is_file() else PROJECT_ROOT / resolved
    resolved = resolved.resolve()
    if resolved in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved))
        raise ValueError(f"配置 base 循环引用: {chain}")
    if not resolved.is_file():
        raise FileNotFoundError(f"配置文件不存在: {resolved}")
    with resolved.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"配置文件顶层必须是 mapping: {resolved}")
    value = copy.deepcopy(value)
    bases = value.pop("base", None)
    if bases is None:
        return value
    if isinstance(bases, (str, os.PathLike)):
        bases = [bases]
    if not isinstance(bases, (list, tuple)):
        raise ValueError(f"配置 base 必须是路径或路径列表: {resolved}")
    merged: dict[str, Any] = {}
    for reference in bases:
        base_path = Path(reference).expanduser()
        if not base_path.is_absolute():
            base_path = resolved.parent / base_path
        merged = deep_merge(merged, load_yaml_file(base_path, (*stack, resolved)))
    return deep_merge(merged, value)


def apply_overrides(config: dict[str, Any], overrides: list[str] | None = None) -> dict[str, Any]:
    """Apply dotted ``key=value`` overrides using YAML scalar parsing."""
    result = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--overrides 项缺少 = : {item!r}")
        key, raw = item.split("=", 1)
        parts = [part.strip() for part in key.split(".")]
        if not parts or any(not part for part in parts):
            raise ValueError(f"--overrides 键无效: {item!r}")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        node: dict[str, Any] = result
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
    return result


def _dataset_spec(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    raw = config.get("dataset", {})
    if isinstance(raw, str):
        raw = {"id": raw}
    if not isinstance(raw, dict):
        raise ValueError("dataset 必须是 mapping、名称或配置路径")

    reference = raw.get("config") or raw.get("ref")
    if reference:
        ref_path = Path(reference).expanduser()
        if not ref_path.is_absolute():
            ref_path = config_path.parent / ref_path
        referenced = load_yaml_file(ref_path)
        referenced = referenced.get("dataset", referenced)
        if not isinstance(referenced, dict):
            raise ValueError(f"数据集配置没有 dataset mapping: {ref_path}")
        local = {key: value for key, value in raw.items() if key not in {"config", "ref"}}
        return deep_merge(referenced, local)

    dataset_id = raw.get("id")
    if dataset_id and not raw.get("root"):
        candidates = [
            PROJECT_ROOT / "configs" / "datasets" / f"{dataset_id}.yaml",
            PROJECT_ROOT / "configs" / "datasets" / "sr" / f"{dataset_id}.yaml",
            PROJECT_ROOT / "configs" / "datasets" / "refsr" / f"{dataset_id}.yaml",
        ]
        for candidate in candidates:
            if candidate.is_file() and candidate.resolve() != config_path.resolve():
                referenced = load_yaml_file(candidate).get("dataset", {})
                if isinstance(referenced, dict):
                    return deep_merge(referenced, raw)
    return raw


def _slug(value: Any, fallback: str) -> str:
    text = str(value or fallback).strip().replace("\\", "_").replace("/", "_")
    return text or fallback


def materialize_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Fill the common task/data/model/output fields without model imports."""
    result = copy.deepcopy(config)
    dataset = _dataset_spec(result, config_path)
    for section_name in ("run", "data", "model", "train", "loss", "output"):
        section = result.setdefault(section_name, {})
        if not isinstance(section, dict):
            raise ValueError(f"{section_name} 必须是 mapping")

    model = result["model"]
    data = result["data"]
    run = result["run"]
    train = result["train"]
    output = result["output"]
    dataset_id = _slug(dataset.get("id"), "dataset")
    model_name = _slug(model.get("name") or model.get("id"), config_path.parent.name)

    task = str(result.get("task") or model.get("task") or "").lower().strip()
    if task not in {"sr", "refsr"}:
        task = "refsr" if model_name.lower() in {"refsrwkv", "refdiffrwkv"} else "sr"
    result["task"] = task
    model["name"] = model_name

    root = data.get("root") or run.get("data_root") or dataset.get("root")
    if root:
        data["root"] = root
    scale = run.get("scale", data.get("scale", dataset.get("default_scale")))
    if scale is not None:
        if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
            raise ValueError(f"scale 必须是正整数，得到 {scale!r}")
        data["scale"] = int(scale)
        run["scale"] = int(scale)

    lr_patch = run.get("lr_patch", data.get("train_lr_patch"))
    if lr_patch is not None:
        if isinstance(lr_patch, bool) or not isinstance(lr_patch, int) or lr_patch < 1:
            raise ValueError(f"run.lr_patch 必须是正整数，得到 {lr_patch!r}")
        data["train_lr_patch"] = int(lr_patch)
        if scale is not None:
            data["train_hr_patch"] = int(lr_patch) * int(scale)
            data.setdefault("patch_size", data["train_hr_patch"])

    if task == "refsr" and "reference_mode" not in data:
        data["reference_mode"] = "paired" if dataset.get("kind") == "paired_reference" else "lr_up"
    data.setdefault("lr_key", model.get("lr_key", "lr"))
    data.setdefault("hr_key", model.get("hr_key", "hr"))
    data.setdefault("ref_key", model.get("ref_key", "ref"))
    data.setdefault("max_samples_train", None)
    data.setdefault("max_samples_val", None)
    data.setdefault("max_samples_test", None)
    data.setdefault("pin_memory", True)

    run_name = _slug(run.get("name") or output.get("experiment_name"), f"{dataset_id}_x{scale or 'native'}")
    run["name"] = run_name
    output.setdefault("root", "experiments")
    output.setdefault("train_root", str(Path(output["root"]) / "train"))
    output.setdefault("test_root", str(Path(output["root"]) / "test"))
    output.setdefault("experiment_name", run_name)
    output.setdefault("checkpoint_dir", str(Path(output["train_root"]) / task / model_name / dataset_id / f"x{scale or 'native'}" / run_name / "checkpoints"))
    output.setdefault("log_dir", str(Path(output["train_root"]) / task / model_name / dataset_id / f"x{scale or 'native'}" / run_name / "logs"))
    output.setdefault("test_dir", str(Path(output["test_root"]) / task / model_name / dataset_id / f"x{scale or 'native'}" / run_name))
    result["dataset"] = dataset
    result["_config_path"] = str(config_path.resolve())
    return result


def load_config(path: str | os.PathLike[str], overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = resolve_path(path, prefer_cwd=True)
    return materialize_config(apply_overrides(load_yaml_file(config_path), overrides), config_path)


def validate_config(config: dict[str, Any], *, require_data: bool = True) -> None:
    """Validate fields shared by all native loaders and Lightning runners."""
    for section in ("data", "model", "train", "loss", "output"):
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"{section} 必须是 mapping")
    data, model = config.get("data", {}), config.get("model", {})
    if require_data:
        if not data.get("root"):
            raise ValueError("data.root 未设置")
        if not isinstance(data.get("scale"), int) or data["scale"] < 1:
            raise ValueError("data.scale 必须是正整数")
    if not str(model.get("name", "")).strip():
        raise ValueError("model.name 必须是非空字符串")
    for key, default in (("batch_size", 1), ("val_batch_size", 1), ("num_workers", 0), ("val_num_workers", 0)):
        value = data.get(key, default)
        minimum = 0 if "workers" in key else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"data.{key} 必须是 >= {minimum} 的整数")
    train = config.get("train", {})
    for key in ("learning_rate", "weight_decay", "lr_factor", "lr_min", "ema_decay"):
        if key in train and not isinstance(train[key], (int, float)):
            raise ValueError(f"train.{key} 必须是数值")


__all__ = [
    "apply_overrides",
    "load_config",
    "load_yaml_file",
    "materialize_config",
    "validate_config",
]
