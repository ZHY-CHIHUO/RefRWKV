"""Shared configuration, data, metrics, and checkpoint helpers for baselines."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(value: str | os.PathLike[str], *, prefer_cwd: bool = False) -> Path:
    """Resolve relative paths consistently from the repository root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_path = Path.cwd() / path
    repo_path = PROJECT_ROOT / path
    if prefer_cwd and cwd_path.exists():
        return cwd_path.resolve()
    if cwd_path.exists() and not repo_path.exists():
        return cwd_path.resolve()
    return repo_path.resolve()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml_file(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML file and its relative ``base`` chain."""
    path = path.expanduser().resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"配置 base 循环引用: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8-sig") as file_obj:
        config = yaml.safe_load(file_obj)
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError(f"配置文件顶层必须是 mapping: {path}")
    config = copy.deepcopy(config)
    base_ref = config.pop("base", None)
    if base_ref is None:
        return config
    refs = base_ref if isinstance(base_ref, (list, tuple)) else [base_ref]
    merged: dict[str, Any] = {}
    for ref in refs:
        if not isinstance(ref, (str, os.PathLike)) or not str(ref):
            raise ValueError(f"配置 base 必须是非空路径: {path}")
        base_path = Path(ref).expanduser()
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = deep_merge(merged, load_yaml_file(base_path, (*stack, path)))
    return deep_merge(merged, config)


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply dotted ``key=value`` command-line overrides."""
    config = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--overrides 项缺少 = : {item!r}")
        key, raw_value = item.split("=", 1)
        parts = [part.strip() for part in key.split(".")]
        if not parts or any(not part for part in parts):
            raise ValueError(f"--overrides 键无效: {item!r}")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            value = raw_value
        node: dict[str, Any] = config
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value
    return config


def _load_dataset_spec(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    raw = config.get("dataset", {})
    if isinstance(raw, str):
        raw = {"id": raw}
    if not isinstance(raw, dict):
        raise ValueError("dataset 必须是 mapping 或数据集配置路径")

    reference = raw.get("config") or raw.get("ref")
    if reference:
        ref_path = Path(reference).expanduser()
        if not ref_path.is_absolute():
            ref_path = config_path.parent / ref_path
        referenced = load_yaml_file(ref_path)
        referenced_spec = referenced.get("dataset", referenced)
        if not isinstance(referenced_spec, dict):
            raise ValueError(f"数据集配置没有 dataset mapping: {ref_path}")
        local = {key: value for key, value in raw.items() if key not in {"config", "ref"}}
        raw = deep_merge(referenced_spec, local)
    elif raw.get("id") and not raw.get("root"):
        # ``dataset: aid`` is a convenient shorthand for future runs.
        ref_path = PROJECT_ROOT / "configs" / "datasets" / f"{raw['id']}.yaml"
        if ref_path.is_file():
            referenced = load_yaml_file(ref_path)
            referenced_spec = referenced.get("dataset", referenced)
            if isinstance(referenced_spec, dict):
                raw = deep_merge(referenced_spec, raw)
    return raw


def materialize_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Expand dataset/run fields into the common SISR training contract."""
    config = copy.deepcopy(config)
    dataset = _load_dataset_spec(config, config_path)
    run = config.setdefault("run", {})
    data = config.setdefault("data", {})
    model = config.setdefault("model", {})
    output = config.setdefault("output", {})
    for name, section in (("run", run), ("data", data), ("model", model), ("output", output)):
        if not isinstance(section, dict):
            raise ValueError(f"{name} 必须是 mapping")

    dataset_id = str(dataset.get("id", "dataset")).strip()
    if not dataset_id:
        raise ValueError("dataset.id 必须是非空字符串")
    root = data.get("root") or run.get("data_root") or dataset.get("root")
    if not root:
        raise ValueError("必须提供 data.root、run.data_root 或 dataset.root")
    scale = run.get("scale", data.get("scale", dataset.get("default_scale")))
    if scale is None:
        raise ValueError("run.scale 未设置，且数据集没有 default_scale")
    lr_patch = run.get("lr_patch", data.get("train_lr_patch"))
    if lr_patch is None:
        raise ValueError("原生尺度训练必须显式设置 run.lr_patch")
    if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
        raise ValueError(f"run.scale 必须是正整数，得到 {scale!r}")
    if isinstance(lr_patch, bool) or not isinstance(lr_patch, int) or lr_patch < 1:
        raise ValueError(f"run.lr_patch 必须是正整数，得到 {lr_patch!r}")

    data["root"] = root
    data["scale"] = int(scale)
    data["train_lr_patch"] = int(lr_patch)
    data["train_hr_patch"] = int(lr_patch) * int(scale)
    data["patch_size"] = data["train_hr_patch"]
    data["val_patch_size"] = run.get("val_hr_patch", data.get("val_patch_size"))
    config["dataset"] = dataset

    run_name = str(run.get("name") or f"{dataset_id}_x{scale}_{model.get('name', 'model')}").strip()
    if not run_name:
        raise ValueError("run.name 必须是非空字符串")
    run["name"] = run_name
    output.setdefault("experiment_name", run_name)
    output.setdefault("checkpoint_root", "checkpoints/baselines")
    output.setdefault("log_root", "logs/baselines")
    output.setdefault("experiment_prefix", "baseline")
    if not output.get("checkpoint_dir"):
        output["checkpoint_dir"] = str(Path(output["checkpoint_root"]) / run_name)
    if not output.get("log_dir"):
        output["log_dir"] = str(Path(output["log_root"]))

    model_name = model.get("name") or model.get("id")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model.name 必须是非空字符串")
    model["name"] = model_name.strip()
    config.setdefault("loss", {"name": "l1"})
    if not isinstance(config["loss"], dict):
        raise ValueError("loss 必须是 mapping")
    return config


def load_config(path: str, overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = resolve_path(path, prefer_cwd=True)
    config = load_yaml_file(config_path)
    config = apply_overrides(config, overrides)
    return materialize_config(config, config_path)


def validate_config(config: dict[str, Any]) -> None:
    for key in ("data", "model", "train", "loss"):
        if key in config and not isinstance(config[key], dict):
            raise ValueError(f"{key} 必须是 mapping")
    data, model = config["data"], config["model"]
    train, loss = config.get("train", {}), config.get("loss", {})
    scale = data.get("scale")
    lr_patch = data.get("train_lr_patch")
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("data.scale 必须是正整数")
    if not isinstance(lr_patch, int) or lr_patch < 1:
        raise ValueError("data.train_lr_patch 必须是正整数")
    if data.get("patch_size") != lr_patch * scale:
        raise ValueError("data.patch_size 必须等于 train_lr_patch * scale")
    for key, default in (("batch_size", 4), ("val_batch_size", 1), ("num_workers", 4), ("val_num_workers", 2)):
        value = data.get(key, default)
        minimum = 0 if "workers" in key else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"data.{key} 必须是 >= {minimum} 的整数")
    if not isinstance(model.get("name"), str) or not model["name"].strip():
        raise ValueError("model.name 必须是非空字符串")
    loss_name = str(loss.get("name", "l1")).lower()
    if loss_name not in {"l1", "charbonnier", "mse", "l2"}:
        raise ValueError("loss.name 只能是 l1、charbonnier 或 mse")
    loss["name"] = loss_name
    interval = train.get("val_check_interval", 1.0)
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) or float(interval) <= 0:
        raise ValueError("train.val_check_interval 必须为正数")
    check_every = train.get("check_val_every_n_epoch", 1)
    if isinstance(check_every, bool) or not isinstance(check_every, int) or check_every < 1:
        raise ValueError("train.check_val_every_n_epoch 必须是正整数")
    if check_every > 1 and not (isinstance(interval, float) and float(interval) == 1.0):
        raise ValueError("按 epoch 间隔验证时，val_check_interval 必须为 1.0")
    max_epochs = train.get("max_epochs", 200)
    max_steps = train.get("max_steps", -1)
    if max_epochs == -1 and max_steps == -1:
        raise ValueError("train.max_epochs 和 train.max_steps 不能同时为 -1")
    for key in ("learning_rate", "lr_factor", "lr_min", "lr_threshold", "ema_decay"):
        if key in train and not isinstance(train[key], (int, float)):
            raise ValueError(f"train.{key} 必须是数值")


def make_dataset(config: dict[str, Any], mode: str):
    # Import lazily so configuration tools remain usable without data-loader
    # side effects.  Baselines use the HR/LR-only SISR contract by design.
    from SR_data.SRDataset import SRPNGDataset

    data = config["data"]
    train_mode = mode == "train"
    max_samples = (
        data.get("max_samples_train"),
        data.get("max_samples_val"),
        data.get("max_samples_test"),
    )
    return SRPNGDataset(
        data_dir=str(resolve_path(data["root"])),
        mode=mode,
        patch_size=data.get("patch_size") if train_mode else data.get("val_patch_size"),
        scale=int(data["scale"]),
        augment=bool(data.get("augment", True)) if train_mode else False,
        max_samples=max_samples,
        sample_seed=int(data.get("sample_seed", config.get("train", {}).get("seed", 42))),
        lr_key=data.get("lr_key", "lr"),
        hr_key=data.get("hr_key", "hr"),
    )


def build_dataloaders(config: dict[str, Any]):
    validate_config(config)
    data = config["data"]
    train_ds = make_dataset(config, "train")
    val_ds = make_dataset(config, "val")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("训练集和验证集都必须非空")
    common = {
        "pin_memory": bool(data.get("pin_memory", True)),
        "persistent_workers": bool(data.get("persistent_workers", True)),
    }
    train_workers = int(data.get("num_workers", 4))
    val_workers = int(data.get("val_num_workers", 2))
    train_kwargs = dict(
        dataset=train_ds,
        batch_size=int(data.get("batch_size", 4)),
        shuffle=True,
        num_workers=train_workers,
        drop_last=bool(data.get("drop_last", True)),
        pin_memory=common["pin_memory"],
    )
    val_kwargs = dict(
        dataset=val_ds,
        batch_size=int(data.get("val_batch_size", 1)),
        shuffle=False,
        num_workers=val_workers,
        pin_memory=common["pin_memory"],
    )
    prefetch = int(data.get("prefetch_factor", 2))
    if train_workers > 0:
        train_kwargs["persistent_workers"] = common["persistent_workers"]
        train_kwargs["prefetch_factor"] = prefetch
    if val_workers > 0:
        val_kwargs["persistent_workers"] = common["persistent_workers"]
        val_kwargs["prefetch_factor"] = prefetch
    if train_kwargs["drop_last"] and len(train_ds) < train_kwargs["batch_size"]:
        train_kwargs["drop_last"] = False
    return DataLoader(**train_kwargs), DataLoader(**val_kwargs)


def gaussian_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image RGB SSIM for tensors in the [-1, 1] range."""
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError(f"SSIM 输入形状不一致: {pred.shape} vs {target.shape}")
    channels = pred.shape[1]
    size = min(11, int(pred.shape[-2]), int(pred.shape[-1]))
    if size % 2 == 0:
        size -= 1
    if size < 3:
        return torch.ones(pred.shape[0], device=pred.device, dtype=torch.float32)
    sigma = 1.5
    coords = torch.arange(size, dtype=torch.float32, device=pred.device)
    gaussian = torch.exp(-((coords - size // 2) ** 2) / (2 * sigma**2))
    window = torch.outer(gaussian, gaussian)
    window = (window / window.sum()).view(1, 1, size, size).expand(channels, 1, size, size)
    pad = size // 2
    pred_f, target_f = pred.float(), target.float()
    mu_p = F.conv2d(pred_f, window, padding=pad, groups=channels)
    mu_t = F.conv2d(target_f, window, padding=pad, groups=channels)
    var_p = (F.conv2d(pred_f.square(), window, padding=pad, groups=channels) - mu_p.square()).clamp_min(0)
    var_t = (F.conv2d(target_f.square(), window, padding=pad, groups=channels) - mu_t.square()).clamp_min(0)
    cov = F.conv2d(pred_f * target_f, window, padding=pad, groups=channels) - mu_p * mu_t
    c1, c2 = (0.01 * 2.0) ** 2, (0.03 * 2.0) ** 2
    numerator = (2 * mu_p * mu_t + c1) * (2 * cov + c2)
    denominator = (mu_p.square() + mu_t.square() + c1) * (var_p + var_t + c2)
    score = numerator / denominator.clamp_min(1e-12)
    return score.clamp(-1, 1).mean(dim=(1, 2, 3))


def per_image_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (pred.float() - target.float()).square().mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(4.0 / mse.clamp_min(1e-10))


class EMA:
    """Small, model-agnostic EMA used by all baseline runners."""

    def __init__(self, decay: float = 0.999):
        self.decay = float(decay)
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("EMA decay 必须位于 [0, 1)")
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        self.initialized = False
        self.applied = False

    def _init(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name not in self.shadow:
                self.shadow[name] = param.detach().float().clone()
        self.initialized = True

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self._init(model)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                self.shadow[name].mul_(self.decay).add_(
                    param.detach().float(), alpha=1 - self.decay
                )

    @torch.no_grad()
    def apply(self, model: torch.nn.Module) -> None:
        if self.applied:
            return
        self._init(model)
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))
        self.applied = True

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        if not self.applied:
            return
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].to(device=param.device, dtype=param.dtype))
        self.backup = {}
        self.applied = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": {name: value.detach().clone() for name, value in self.shadow.items()},
            "initialized": self.initialized,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("EMA state 必须是 mapping")
        self.decay = float(state.get("decay", self.decay))
        self.shadow = {
            name: value.detach().float().clone()
            for name, value in state.get("shadow", {}).items()
            if isinstance(name, str) and torch.is_tensor(value)
        }
        self.initialized = bool(state.get("initialized", bool(self.shadow)))
        self.backup, self.applied = {}, False


def experiment_signature(config: dict[str, Any]) -> dict[str, Any]:
    data, model, loss = config["data"], config["model"], config.get("loss", {})
    dataset = config.get("dataset", {})
    return {
        "architecture": "baseline_sisr_v1",
        "model": str(model["name"]),
        "dataset": str(dataset.get("id", "dataset")),
        "data_root": str(resolve_path(data["root"])),
        "scale": int(data["scale"]),
        "train_lr_patch": int(data["train_lr_patch"]),
        "train_hr_patch": int(data["train_hr_patch"]),
        "val_patch_size": data.get("val_patch_size"),
        "loss": str(loss.get("name", "l1")),
        "value_range": "[-1,1]",
        "model_config": copy.deepcopy(model),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(json_safe(value), file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
