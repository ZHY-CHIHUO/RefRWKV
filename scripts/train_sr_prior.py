#!/usr/bin/env python
"""
RefSRWKV SR Prior 训练脚本。

配置文件、数据目录和 checkpoint 路径默认相对于仓库根目录解析，因而从
PowerShell 的任意当前目录启动都得到一致行为。配置文本按 UTF-8（兼容 BOM）读取。
"""

import argparse
import copy
import json
import logging
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from RefRWKV.models.RefSRWKV import (
    LitRefSRWKV,
    RefSRWKV,
    normalize_window_config,
)
from RefRWKV.RefSR_data.RefDataset import RefPNGDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_sr_prior")


def _resolve_path(value, *, prefer_cwd=False) -> Path:
    """Resolve a user path without depending on the PowerShell cwd."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    repo_path = PROJECT_ROOT / path
    if prefer_cwd and cwd_path.exists():
        return cwd_path.resolve()
    if cwd_path.exists() and not repo_path.exists():
        return cwd_path.resolve()
    return repo_path.resolve()


def _apply_overrides(config, overrides):
    """Apply dotted ``key=value`` overrides after loading YAML."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--overrides 项缺少 = : {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--overrides 键不能为空: {item!r}")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            value = raw_value
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            if not part:
                raise ValueError(f"--overrides 键无效: {item!r}")
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        if not parts[-1]:
            raise ValueError(f"--overrides 键无效: {item!r}")
        node[parts[-1]] = value
    return config


def _deep_merge(base, override):
    """Recursively merge mappings while keeping the input configurations intact."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_config_file(path, stack=()):
    """Load one YAML file and its relative ``base`` chain."""
    path = Path(path).expanduser().resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"配置 base 循环引用: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8-sig") as file_obj:
        config = yaml.safe_load(file_obj)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件顶层必须是 mapping: {path}")

    base_ref = config.pop("base", None)
    if base_ref is None:
        return config
    base_refs = base_ref if isinstance(base_ref, (list, tuple)) else [base_ref]
    merged = {}
    for ref in base_refs:
        if not isinstance(ref, (str, os.PathLike)) or not str(ref):
            raise ValueError(f"配置 base 必须是非空路径: {path}")
        base_path = Path(ref).expanduser()
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = _deep_merge(merged, _load_config_file(base_path, (*stack, path)))
    return _deep_merge(merged, config)


def _materialize_run_config(config):
    """Expand dataset metadata and a native-LR run into loader/model fields."""
    dataset_cfg = config.get("dataset")
    run_cfg = config.get("run")
    if dataset_cfg is None and run_cfg is None:
        # Keep reading older flat configs during the migration period.
        return config
    if not isinstance(dataset_cfg, dict):
        raise ValueError("dataset 必须是 mapping")
    if not isinstance(run_cfg, dict):
        raise ValueError("run 必须是 mapping")

    data_cfg = config.setdefault("data", {})
    model_cfg = config.setdefault("model", {})
    output_cfg = config.setdefault("output", {})
    if not isinstance(data_cfg, dict) or not isinstance(model_cfg, dict):
        raise ValueError("data 和 model 必须是 mapping")
    if not isinstance(output_cfg, dict):
        raise ValueError("output 必须是 mapping")

    dataset_id = dataset_cfg.get("id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset.id 必须是非空字符串")
    dataset_root = dataset_cfg.get("root")
    run_root = run_cfg.get("data_root")
    if data_cfg.get("root") in (None, ""):
        data_cfg["root"] = run_root if run_root not in (None, "") else dataset_root
    if data_cfg.get("root") in (None, ""):
        raise ValueError("数据集配置必须提供 dataset.root 或 run.data_root")

    prepared_cfg = dataset_cfg.get("prepared", {})
    if not isinstance(prepared_cfg, dict):
        prepared_cfg = {}
    scale = run_cfg.get("scale")
    if scale is None:
        scale = data_cfg.get(
            "scale", dataset_cfg.get("default_scale", prepared_cfg.get("scale"))
        )
    if scale is None:
        raise ValueError("run.scale 未设置，且数据集没有 default_scale")
    lr_patch = run_cfg.get("lr_patch", data_cfg.get("train_lr_patch"))
    if lr_patch is None:
        raise ValueError("原生尺度训练必须显式设置 run.lr_patch")
    _require_int(lr_patch, "run.lr_patch")
    _require_int(scale, "run.scale")
    hr_patch = int(lr_patch) * int(scale)

    # ``data.patch_size`` remains the dataset constructor's HR crop argument.
    # The explicit names are saved too, so a checkpoint records the actual
    # native LR grid rather than an opaque HR-side size.
    data_cfg["scale"] = scale
    data_cfg["train_lr_patch"] = int(lr_patch)
    data_cfg["train_hr_patch"] = hr_patch
    data_cfg["patch_size"] = hr_patch
    data_cfg["val_patch_size"] = run_cfg.get(
        "val_hr_patch", data_cfg.get("val_patch_size")
    )

    run_name = run_cfg.get("name")
    if run_name is None:
        run_name = f"{dataset_id}_x{scale}"
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run.name 必须是非空字符串")
    run_name = run_name.strip()
    prefix = (
        str(output_cfg.get("experiment_prefix", "refrwkv_sr")).strip() or "refrwkv_sr"
    )
    output_cfg.setdefault("experiment_name", run_name)
    if not output_cfg.get("checkpoint_dir"):
        checkpoint_root = output_cfg.get("checkpoint_root", "checkpoints")
        output_cfg["checkpoint_dir"] = str(
            Path(checkpoint_root) / f"{prefix}_{run_name}"
        )
    if not output_cfg.get("log_dir"):
        log_root = output_cfg.get("log_root", "logs")
        # TensorBoardLogger adds ``name`` below this root, yielding a stable
        # layout such as logs/refrwkv_sr/aid_x4_l1/version_0.
        output_cfg["log_dir"] = str(Path(log_root) / prefix)

    # Prepared remote-sensing variants record their actual degradation scale.
    # Check it when metadata is present, but keep custom datasets usable.
    metadata_path = _resolve_path(data_cfg["root"]) / "metadata.json"
    if metadata_path.is_file():
        try:
            with metadata_path.open("r", encoding="utf-8-sig") as file_obj:
                metadata = json.load(file_obj)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"无法读取数据集 metadata.json: {metadata_path}: {exc}"
            ) from exc
        actual_scale = metadata.get("scale") if isinstance(metadata, dict) else None
        if actual_scale is not None and int(actual_scale) != int(scale):
            raise ValueError(
                f"数据目录 {data_cfg['root']} 是 {actual_scale}x 配对，"
                f"但 run.scale={scale}；请使用对应的准备数据目录"
            )
    return config


def load_config(path, overrides=None):
    config_path = _resolve_path(path, prefer_cwd=True)
    config = _load_config_file(config_path)
    config = _apply_overrides(config, overrides)
    return _materialize_run_config(config)


def _require_int(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} 必须是 >= {minimum} 的整数，得到 {value!r}")


def _require_number(value, name, *, minimum=None, maximum=None, maximum_inclusive=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数值，得到 {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数值，得到 {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} 不能小于 {minimum}，得到 {value}")
    if maximum is not None and (
        value > maximum or (not maximum_inclusive and value == maximum)
    ):
        suffix = ")" if not maximum_inclusive else "]"
        raise ValueError(f"{name} 必须位于 [{minimum}, {maximum}{suffix}，得到 {value}")


def validate_config(cfg):
    """Validate the native-LR crop, full-image validation, and model contract."""
    if not isinstance(cfg, dict):
        raise ValueError("配置必须是 mapping")
    for section in ("data", "model"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"配置缺少 mapping: {section}")

    dc, mc = cfg["data"], cfg["model"]
    patch_size = dc.get("train_hr_patch", dc.get("patch_size", 480))
    lr_patch = dc.get("train_lr_patch")
    scale = dc.get("scale", 4)
    _require_int(patch_size, "data.patch_size")
    _require_int(scale, "data.scale")
    if lr_patch is None:
        if patch_size % scale:
            raise ValueError(
                f"data.patch_size ({patch_size}) 必须能被 data.scale ({scale}) 整除"
            )
        lr_patch = patch_size // scale
        dc["train_lr_patch"] = lr_patch
    _require_int(lr_patch, "data.train_lr_patch")
    if patch_size != lr_patch * scale:
        raise ValueError(
            "data.train_hr_patch/patch_size 必须等于 "
            "data.train_lr_patch x data.scale: "
            f"{patch_size} vs {lr_patch} x {scale}"
        )
    upsampler = str(mc.get("upsampler", "progressive")).lower()
    if upsampler not in {"progressive", "direct"}:
        raise ValueError("model.upsampler 只能是 progressive 或 direct")
    mc["upsampler"] = upsampler
    color_match = str(mc.get("color_match", "global")).lower()
    if color_match not in {"global", "none"}:
        raise ValueError("model.color_match 只能是 global 或 none")
    mc["color_match"] = color_match
    _require_int(mc.get("inp_channels", 3), "model.inp_channels")
    _require_int(mc.get("out_channels", 3), "model.out_channels")
    ref_channels = mc.get("ref_channels", mc.get("inp_channels", 3))
    _require_int(ref_channels, "model.ref_channels")
    if ref_channels != mc.get("inp_channels", 3):
        raise ValueError(
            "当前颜色对齐路径要求 model.ref_channels == model.inp_channels"
        )

    dim = mc.get("dim", 48)
    _require_int(dim, "model.dim")
    if dim % 16 != 0:
        raise ValueError("model.dim 必须是 16 的倍数，以满足 CUDA WKV 的通道分组约束")
    blocks = mc.get("num_blocks", [4, 6, 6, 8])
    if not isinstance(blocks, (list, tuple)) or len(blocks) != 4:
        raise ValueError("model.num_blocks 必须包含四项")
    for index, count in enumerate(blocks):
        _require_int(count, f"model.num_blocks[{index}]")
    _require_int(
        mc.get("num_refinement_blocks", 4), "model.num_refinement_blocks", minimum=0
    )
    _require_number(
        mc.get("drop_path_rate", 0.1),
        "model.drop_path_rate",
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    _require_number(mc.get("hidden_rate", 4), "model.hidden_rate", minimum=1e-12)
    learning_rate = mc.get("learning_rate", 1e-4)
    _require_number(learning_rate, "model.learning_rate", minimum=1e-12)
    lr_scheduler = str(mc.get("lr_scheduler", "plateau")).lower()
    if lr_scheduler not in {"plateau", "cosine"}:
        raise ValueError("model.lr_scheduler 只能是 plateau 或 cosine")
    _require_int(mc.get("lr_patience", 2), "model.lr_patience", minimum=0)
    _require_number(
        mc.get("lr_factor", 0.5),
        "model.lr_factor",
        minimum=1e-12,
        maximum=1.0,
        maximum_inclusive=False,
    )
    lr_min = mc.get("lr_min", 1e-6)
    _require_number(lr_min, "model.lr_min", minimum=0.0)
    if float(lr_min) > float(learning_rate):
        raise ValueError("model.lr_min 不能大于 model.learning_rate")
    _require_number(mc.get("lr_threshold", 1e-4), "model.lr_threshold", minimum=0.0)
    _require_int(mc.get("warmup_steps", 0), "model.warmup_steps", minimum=0)
    _require_number(mc.get("grad_clip_norm", 1.0), "model.grad_clip_norm", minimum=0.0)
    _require_number(
        mc.get("ema_decay", 0.999),
        "model.ema_decay",
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    adam_betas = mc.get("adam_betas", [0.9, 0.999])
    if not isinstance(adam_betas, (list, tuple)) or len(adam_betas) != 2:
        raise ValueError("model.adam_betas 必须包含两个数值")
    for index, beta in enumerate(adam_betas):
        _require_number(
            beta,
            f"model.adam_betas[{index}]",
            minimum=0.0,
            maximum=1.0,
            maximum_inclusive=False,
        )
    _require_number(mc.get("weight_decay", 0.0), "model.weight_decay", minimum=0.0)
    _require_number(mc.get("ssim_weight", 0.0), "model.ssim_weight", minimum=0.0)
    _require_number(mc.get("fft_weight", 0.0), "model.fft_weight", minimum=0.0)
    _require_number(
        mc.get("ref_drop_prob", 0.0), "model.ref_drop_prob", minimum=0.0, maximum=1.0
    )
    reference_mode = str(dc.get("reference_mode", "paired")).lower()
    if reference_mode in {"sisr", "lr", "lr_up", "bicubic_lr"}:
        dc["reference_mode"] = "lr_up"
    elif reference_mode != "paired":
        raise ValueError("data.reference_mode 只能是 paired 或 lr_up")
    # Normalize and validate the stage window schema before constructing the
    # model.  Store the canonical form so train_config.yaml is self-contained.
    mc["windows"] = normalize_window_config(mc.get("windows"))

    root = dc.get("root")
    if not isinstance(root, (str, Path)) or not str(root):
        raise ValueError("data.root 必须是非空路径")
    for name in ("batch_size", "val_batch_size", "num_workers", "val_num_workers"):
        minimum = 0 if "workers" in name else 1
        _require_int(
            dc.get(name, 4 if name == "batch_size" else 1),
            f"data.{name}",
            minimum=minimum,
        )
    val_patch_size = dc.get("val_patch_size")
    if val_patch_size is not None:
        _require_int(val_patch_size, "data.val_patch_size")
        if val_patch_size % scale:
            raise ValueError("data.val_patch_size 必须能被 data.scale 整除")
    _require_int(dc.get("prefetch_factor", 4), "data.prefetch_factor")
    for name in ("max_samples_train", "max_samples_val", "max_samples_test"):
        value = dc.get(name)
        if value is not None:
            _require_int(value, f"data.{name}")
    strengths = dc.get("ref_aug_strengths", [0.12, 0.12, 0.12, 0.03])
    probs = dc.get("ref_aug_probs", [0.5, 0.5, 0.5, 0.5])
    if (
        not isinstance(strengths, (list, tuple))
        or not isinstance(probs, (list, tuple))
        or len(strengths) != len(probs)
    ):
        raise ValueError("data.ref_aug_strengths 与 data.ref_aug_probs 必须是等长列表")
    for index, value in enumerate(strengths):
        _require_number(value, f"data.ref_aug_strengths[{index}]", minimum=0.0)
    for index, value in enumerate(probs):
        _require_number(value, f"data.ref_aug_probs[{index}]", minimum=0.0, maximum=1.0)
    tc = cfg.get("train", {})
    if not isinstance(tc, dict):
        raise ValueError("train 必须是 mapping")
    _require_number(
        dc.get("ref_gray_prob", 0.2), "data.ref_gray_prob", minimum=0.0, maximum=1.0
    )
    _require_int(
        dc.get("sample_seed", tc.get("seed", 42)), "data.sample_seed", minimum=0
    )
    max_epochs = tc.get("max_epochs", 200)
    if max_epochs != -1:
        _require_int(max_epochs, "train.max_epochs")
    _require_int(tc.get("accumulate_grad_batches", 1), "train.accumulate_grad_batches")
    if "max_steps" in tc:
        if (
            isinstance(tc["max_steps"], bool)
            or not isinstance(tc["max_steps"], int)
            or tc["max_steps"] < -1
        ):
            raise ValueError("train.max_steps 必须是 >= -1 的整数")
    if max_epochs == -1 and tc.get("max_steps", -1) == -1:
        raise ValueError("train.max_epochs 和 train.max_steps 不能同时为 -1")
    if "num_sanity_val_steps" in tc:
        _require_int(
            tc["num_sanity_val_steps"], "train.num_sanity_val_steps", minimum=-1
        )
    _require_int(tc.get("log_every_n_steps", 20), "train.log_every_n_steps")
    interval = tc.get("val_check_interval", 1.0)
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not math.isfinite(float(interval))
        or interval <= 0
    ):
        raise ValueError("train.val_check_interval 必须为正数")
    if isinstance(interval, float) and interval > 1.0:
        raise ValueError(
            "train.val_check_interval 为小数时必须位于 (0, 1]；整数批次数请使用整数"
        )
    _require_int(tc.get("save_top_k", 3), "train.save_top_k", minimum=-1)
    early_stopping_patience = tc.get("early_stopping_patience", 30)
    if early_stopping_patience is not None:
        _require_int(
            early_stopping_patience, "train.early_stopping_patience", minimum=0
        )
    if "grad_clip_val" in tc and tc["grad_clip_val"] is not None:
        _require_number(tc["grad_clip_val"], "train.grad_clip_val", minimum=0.0)
    algorithm = str(tc.get("gradient_clip_algorithm", "norm")).lower()
    if algorithm not in {"norm", "value"}:
        raise ValueError("train.gradient_clip_algorithm 只能是 norm 或 value")
    accelerator = str(tc.get("accelerator", "gpu")).lower()
    if accelerator not in {"gpu", "cpu", "auto", "tpu", "mps"}:
        raise ValueError(f"不支持的 train.accelerator: {accelerator}")
    devices = tc.get("devices", 1)
    if isinstance(devices, int):
        _require_int(devices, "train.devices")
    elif isinstance(devices, (list, tuple)) and devices:
        for index, device in enumerate(devices):
            _require_int(device, f"train.devices[{index}]", minimum=0)
    else:
        raise ValueError("train.devices 必须是正整数或非空设备列表")

    output = cfg.get("output", {})
    if not isinstance(output, dict):
        raise ValueError("output 必须是 mapping")

    return cfg


def build_dataloaders(cfg):
    validate_config(cfg)
    dc, tc = cfg["data"], cfg.get("train", {})
    data_root = _resolve_path(dc["root"])
    common = dict(
        data_dir=str(data_root),
        scale=dc.get("scale", 4),
        ref_aug_strengths=dc.get("ref_aug_strengths", [0.12, 0.12, 0.12, 0.03]),
        ref_aug_probs=dc.get("ref_aug_probs", [0.5, 0.5, 0.5, 0.5]),
        ref_gray_prob=dc.get("ref_gray_prob", 0.2),
        max_samples=(
            dc.get("max_samples_train"),
            dc.get("max_samples_val"),
            dc.get("max_samples_test"),
        ),
        sample_seed=dc.get("sample_seed", tc.get("seed", 42)),
        lr_key=dc.get("lr_key", "lr"),
        hr_key=dc.get("hr_key", "hr"),
        ref_key=dc.get("ref_key", "ref"),
    )
    train_ds = RefPNGDataset(
        mode="train",
        patch_size=dc.get("train_hr_patch", dc.get("patch_size", 480)),
        augment=dc.get("augment", True),
        augment_ref=dc.get("augment_ref", True),
        **common,
    )
    val_ds = RefPNGDataset(
        mode="val",
        patch_size=dc.get("val_patch_size"),
        augment=False,
        augment_ref=False,
        **common,
    )
    if len(train_ds) == 0:
        raise ValueError("训练数据集为空")
    if len(val_ds) == 0:
        raise ValueError(
            "验证数据集为空；当前脚本需要 val_loss 进行 checkpoint/early stopping"
        )

    pin_memory = bool(dc.get("pin_memory", False))
    train_workers = dc.get("num_workers", 4)
    val_workers = dc.get("val_num_workers", 2)
    batch_size = dc.get("batch_size", 4)
    val_batch_size = dc.get("val_batch_size", 1)
    drop_last = bool(dc.get("drop_last", True))
    if drop_last and len(train_ds) < batch_size:
        logger.warning(
            "训练样本数 (%d) 小于 batch_size (%d)，自动关闭 drop_last 以避免零 step",
            len(train_ds),
            batch_size,
        )
        drop_last = False

    train_kwargs = dict(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    if train_workers > 0:
        train_kwargs["persistent_workers"] = bool(dc.get("persistent_workers", True))
        train_kwargs["prefetch_factor"] = dc.get("prefetch_factor", 4)
    val_kwargs = dict(
        dataset=val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=pin_memory,
    )
    if val_workers > 0:
        val_kwargs["persistent_workers"] = bool(dc.get("persistent_workers", True))
        val_kwargs["prefetch_factor"] = dc.get("prefetch_factor", 4)
    return DataLoader(**train_kwargs), DataLoader(**val_kwargs)


def build_model(cfg):
    validate_config(cfg)
    mc, dc = cfg["model"], cfg["data"]
    lr_key, hr_key, ref_key = (
        dc.get("lr_key", "lr"),
        dc.get("hr_key", "hr"),
        dc.get("ref_key", "ref"),
    )
    train_lr_patch = dc.get("train_lr_patch")
    train_hr_patch = dc.get("train_hr_patch", dc.get("patch_size", 480))
    data_scale = dc.get("scale", 4)
    model = RefSRWKV(
        inp_channels=mc.get("inp_channels", 3),
        out_channels=mc.get("out_channels", 3),
        ref_channels=mc.get("ref_channels", mc.get("inp_channels", 3)),
        dim=mc.get("dim", 48),
        num_blocks=tuple(mc.get("num_blocks", [4, 6, 6, 8])),
        num_refinement_blocks=mc.get("num_refinement_blocks", 4),
        scale=data_scale,
        upsampler=mc.get("upsampler", "progressive"),
        color_match=mc.get("color_match", "global"),
        drop_path_rate=mc.get("drop_path_rate", 0.1),
        hidden_rate=mc.get("hidden_rate", 4),
        windows=mc.get("windows"),
    )
    logger.info(
        "RefSRWKV 参数量: %.2fM (train LR=%d, train HR=%d, scale=x%d, output=%s)",
        sum(parameter.numel() for parameter in model.parameters()) / 1e6,
        train_lr_patch,
        train_hr_patch,
        data_scale,
        model.upsampler,
    )
    lit_model = LitRefSRWKV(
        model_sr=model,
        learning_rate=mc.get("learning_rate", 1e-4),
        lr_scheduler=mc.get("lr_scheduler", "plateau"),
        lr_patience=mc.get("lr_patience", 2),
        lr_factor=mc.get("lr_factor", 0.5),
        lr_min=mc.get("lr_min", 1e-6),
        lr_threshold=mc.get("lr_threshold", 1e-4),
        warmup_steps=mc.get("warmup_steps", 0),
        grad_clip_norm=mc.get("grad_clip_norm", 1.0),
        ema_decay=mc.get("ema_decay", 0.999),
        use_ema=mc.get("use_ema", True),
        adam_betas=mc.get("adam_betas", [0.9, 0.999]),
        weight_decay=mc.get("weight_decay", 0.0),
        ssim_weight=mc.get("ssim_weight", 0.0),
        fft_weight=mc.get("fft_weight", 0.0),
        ref_drop_prob=mc.get("ref_drop_prob", 0.0),
        reference_mode=dc.get("reference_mode", "paired"),
        lr_key=lr_key,
        hr_key=hr_key,
        ref_key=ref_key,
    )
    # Persist the data/grid contract with new checkpoints so an accidental
    # cross-dataset --resume is rejected before Lightning restores optimizer
    # state. Older checkpoints are checked through their nearby train_config.
    lit_model._experiment_signature = _experiment_signature(cfg)
    return lit_model


_CKPT_PREFIXES = tuple(
    sorted(
        (
            "generator.sr_model.",
            "generator.model_sr.",
            "sr_model.",
            "model_sr.",
            "model.",
            "module.",
        ),
        key=len,
        reverse=True,
    )
)


def _strip_prefix(key: str) -> str:
    if not isinstance(key, str):
        return key
    # Checkpoints from nested Lightning/generator wrappers can contain more
    # than one of these prefixes.
    changed = True
    while changed:
        changed = False
        for prefix in _CKPT_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
                break
    return key


def _load_raw_checkpoint(path):
    path = _resolve_path(path, prefer_cwd=True)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        return torch.load(path, map_location="cpu")


def _find_tensor_mapping(value):
    if isinstance(value, dict):
        direct = {
            key: tensor
            for key, tensor in value.items()
            if isinstance(key, str) and torch.is_tensor(tensor)
        }
        if direct:
            return direct
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "params",
            "weights",
            "generator",
            "sr_model",
        ):
            if key in value:
                found = _find_tensor_mapping(value[key])
                if found:
                    return found
    return None


def _extract_state_dict(checkpoint, source: str) -> dict:
    state_dict = _find_tensor_mapping(checkpoint)
    if not state_dict:
        raise ValueError(f"checkpoint 中没有找到 tensor state_dict: {source}")
    return state_dict


def _load_state_dict(path: str) -> dict:
    return _extract_state_dict(_load_raw_checkpoint(path), path)


def _load_hot_start_state_dict(path: str):
    """Load EMA weights for transfer when the checkpoint provides them."""
    checkpoint = _load_raw_checkpoint(path)
    if isinstance(checkpoint, dict):
        ema_state = checkpoint.get("ema_state_dict")
        shadow = ema_state.get("shadow") if isinstance(ema_state, dict) else None
        if isinstance(shadow, dict) and any(
            torch.is_tensor(value) for value in shadow.values()
        ):
            return shadow, "EMA"
    return _extract_state_dict(checkpoint, path), "raw"


def _normalise_state_dict(state_dict):
    """Strip wrapper prefixes while preferring the shortest original key."""
    normalized = {}
    source_lengths = {}
    for key, value in state_dict.items():
        normalized_key = _strip_prefix(key)
        if not isinstance(normalized_key, str):
            continue
        key_length = len(key)
        if (
            normalized_key not in normalized
            or key_length < source_lengths[normalized_key]
        ):
            normalized[normalized_key] = value
            source_lengths[normalized_key] = key_length
    return normalized


def _resume_values_equal(left, right):
    """Compare checkpoint/config values without treating YAML lists as tuples."""
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        if set(left) != set(right):
            return False
        return all(_resume_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return len(left) == len(right) and all(
            _resume_values_equal(item_left, item_right)
            for item_left, item_right in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return left == right


_RESUME_HPARAM_KEYS = (
    "learning_rate",
    "lr_scheduler",
    "lr_patience",
    "lr_factor",
    "lr_min",
    "lr_threshold",
    "warmup_steps",
    "grad_clip_norm",
    "ema_decay",
    "use_ema",
    "adam_betas",
    "weight_decay",
    "ssim_weight",
    "fft_weight",
    "ref_drop_prob",
    "reference_mode",
    "lr_key",
    "hr_key",
    "ref_key",
)


def _experiment_signature(cfg):
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    data_root = _resolve_path(data_cfg["root"])
    scale = int(data_cfg.get("scale", 4))
    train_lr_patch = int(data_cfg["train_lr_patch"])
    train_hr_patch = int(data_cfg["train_hr_patch"])
    return {
        "architecture": "native_lr_v1",
        "data_root": str(data_root),
        "scale": scale,
        "reference_mode": str(data_cfg.get("reference_mode", "paired")),
        "train_lr_patch": train_lr_patch,
        "train_hr_patch": train_hr_patch,
        "val_patch_size": data_cfg.get("val_patch_size"),
        "inp_channels": int(model_cfg.get("inp_channels", 3)),
        "out_channels": int(model_cfg.get("out_channels", 3)),
        "dim": int(model_cfg.get("dim", 48)),
        "hidden_rate": float(model_cfg.get("hidden_rate", 4)),
        "drop_path_rate": float(model_cfg.get("drop_path_rate", 0.1)),
        "ref_channels": int(
            model_cfg.get("ref_channels", model_cfg.get("inp_channels", 3))
        ),
        "num_blocks": list(model_cfg.get("num_blocks", [4, 6, 6, 8])),
        "num_refinement_blocks": int(model_cfg.get("num_refinement_blocks", 4)),
        "upsampler": str(model_cfg.get("upsampler", "progressive")),
        "color_match": str(model_cfg.get("color_match", "global")),
        "normalization": "rmsnorm2d",
        "windows": copy.deepcopy(model_cfg.get("windows")),
    }


def check_resume_compatible(ckpt_path: str, lit_model):
    try:
        checkpoint = _load_raw_checkpoint(ckpt_path)
        state_dict = _normalise_state_dict(_extract_state_dict(checkpoint, ckpt_path))
    except Exception as exc:
        return False, f"checkpoint 读取失败: {exc}"
    if isinstance(checkpoint, dict) and checkpoint.get("lr_schedulers"):
        hparams = checkpoint.get("hyper_parameters") or {}
        if not hasattr(hparams, "get"):
            hparams = {}
        saved_scheduler = str(hparams.get("lr_scheduler", "cosine")).lower()
        current_scheduler = str(
            getattr(lit_model.hparams, "lr_scheduler", "plateau")
        ).lower()
        if saved_scheduler != current_scheduler:
            return False, (
                f"学习率调度器不兼容: checkpoint={saved_scheduler}, "
                f"当前配置={current_scheduler}；请使用 --load_weights 重新开始优化器状态"
            )
        if saved_scheduler == "plateau":
            # A plateau state is resumable only when its update unit is
            # explicitly recorded as validation.
            scheduler_states = checkpoint.get("lr_schedulers")
            scheduler_state = (
                scheduler_states[0]
                if isinstance(scheduler_states, (list, tuple)) and scheduler_states
                else {}
            )
            validation_updates = checkpoint.get("plateau_step_unit") == "validation"
            validation_updates = validation_updates or (
                "lr_threshold" in hparams
                and isinstance(scheduler_state, dict)
                and scheduler_state.get("threshold_mode") == "abs"
            )
            if not validation_updates:
                return False, (
                    "checkpoint 使用按 epoch 更新的 plateau 状态，无法恢复验证次数；"
                    "请使用 --load_weights 仅加载模型权重"
                )

    saved_hparams = (
        checkpoint.get("hyper_parameters", {}) if isinstance(checkpoint, dict) else {}
    )
    current_hparams = getattr(lit_model, "hparams", {})
    if hasattr(saved_hparams, "get") and hasattr(current_hparams, "get"):
        for key in _RESUME_HPARAM_KEYS:
            # Older checkpoints do not contain the newly added optimizer
            # fields; absent keys retain their constructor defaults.
            if key in saved_hparams and key in current_hparams:
                if not _resume_values_equal(saved_hparams[key], current_hparams[key]):
                    return False, (
                        f"训练参数不兼容: {key}={saved_hparams[key]!r} vs "
                        f"当前={current_hparams[key]!r}；请使用 --load_weights"
                    )

    current_signature = getattr(lit_model, "_experiment_signature", None)
    saved_signature = (
        checkpoint.get("refsrwkv_experiment_signature")
        if isinstance(checkpoint, dict)
        else None
    )
    if not isinstance(current_signature, dict):
        return False, "当前运行缺少原生 LR 实验签名"
    if not isinstance(saved_signature, dict):
        return False, "checkpoint 没有原生 LR 实验签名；请使用 --load_weights 或从头训练"
    if saved_signature.get("architecture") != "native_lr_v1":
        return False, "checkpoint 不是原生 LR 架构；不能 --resume，请使用 --load_weights"
    for key in (
        "data_root",
        "scale",
        "reference_mode",
        "train_lr_patch",
        "train_hr_patch",
        "val_patch_size",
        "inp_channels",
        "out_channels",
        "dim",
        "hidden_rate",
        "drop_path_rate",
        "ref_channels",
        "num_blocks",
        "num_refinement_blocks",
        "upsampler",
        "color_match",
        "normalization",
        "windows",
    ):
        if key not in saved_signature:
            return False, f"checkpoint 签名缺少 {key}；请使用 --load_weights"
        if not _resume_values_equal(saved_signature[key], current_signature[key]):
            return False, (
                f"实验结构不兼容: {key}={saved_signature[key]!r} vs "
                f"当前={current_signature[key]!r}；请使用 --load_weights"
            )

    reference = _normalise_state_dict(
        {
            key: value
            for key, value in lit_model.state_dict().items()
            if torch.is_tensor(value)
        }
    )
    overlap = sorted(set(state_dict).intersection(reference))
    if not overlap:
        return False, "checkpoint 与当前模型没有任何同名参数"
    mismatched = [
        (key, tuple(state_dict[key].shape), tuple(reference[key].shape))
        for key in overlap
        if state_dict[key].shape != reference[key].shape
    ]
    if mismatched:
        key, ckpt_shape, model_shape = mismatched[0]
        return (
            False,
            f"{len(mismatched)} 个参数形状不匹配: {key}: ckpt{ckpt_shape} vs 现模型{model_shape}",
        )
    missing = sorted(set(reference) - set(state_dict))
    if missing:
        return (
            False,
            f"当前模型有 {len(missing)} 个参数在 checkpoint 中缺失: {missing[:3]}",
        )
    extra = sorted(set(state_dict) - set(reference))
    if extra:
        return False, f"checkpoint 有 {len(extra)} 个多余参数: {extra[:3]}"
    return True, f"结构一致（{len(overlap)} 个张量全部匹配）"


def load_weights_filtered(lit_model, ckpt_path: str):
    state_dict, source = _load_hot_start_state_dict(ckpt_path)
    state_dict = _normalise_state_dict(state_dict)
    reference = {
        key: value
        for key, value in lit_model.state_dict().items()
        if torch.is_tensor(value)
    }
    reference_by_normalized = {}
    for key in reference:
        reference_by_normalized.setdefault(_strip_prefix(key), key)

    matched, skipped, absent = {}, [], []
    for normalized_key, value in state_dict.items():
        target_key = reference_by_normalized.get(normalized_key)
        if target_key is None:
            absent.append(normalized_key)
            continue
        if tuple(value.shape) == tuple(reference[target_key].shape):
            matched[target_key] = value
        else:
            skipped.append(normalized_key)
    missing, _unexpected = lit_model.load_state_dict(matched, strict=False)
    logger.info(
        "热启动 %s (%s): 目标匹配 %d/%d | 源权重未使用 %d | 形状不匹配 %d | 目标缺失 %d",
        ckpt_path,
        source,
        len(matched),
        len(reference),
        len(absent),
        len(skipped),
        len(missing),
    )
    if skipped:
        logger.warning("  形状不匹配（保持随机初始化）示例: %s", skipped[:5])
    if absent:
        logger.info("  当前模型未使用的源权重示例: %s", absent[:5])
    if not matched:
        raise RuntimeError(f"热启动失败：checkpoint 没有匹配到任何参数 ({ckpt_path})")


def _checkpoint_path(value):
    return None if value is None else _resolve_path(value, prefer_cwd=True)


def main():
    parser = argparse.ArgumentParser(description="RefSRWKV SR Prior 训练")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="覆盖配置字段，如 data.scale=4 output.experiment_name=aid_x4",
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--load_weights",
        type=str,
        default=None,
        help="仅加载匹配的模型权重，重新初始化优化器",
    )
    checkpoint_group.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复完整 Lightning checkpoint（含 optimizer/EMA）",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    validate_config(cfg)
    # Keep all relative data/output paths anchored at the repository root.
    os.chdir(PROJECT_ROOT)
    tc, mc, oc = cfg.get("train", {}), cfg["model"], cfg.get("output", {})
    accelerator = str(tc.get("accelerator", "gpu")).lower()
    if accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if accelerator != "gpu" or not torch.cuda.is_available():
        raise RuntimeError(
            "RefSRWKV 的空间 WKV 算子只支持 CUDA；请安装 CUDA 版 PyTorch 并使用 train.accelerator=gpu"
        )
    pl.seed_everything(tc.get("seed", 42), workers=True)

    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(
        "训练样本: %d, 验证样本: %d", len(train_loader.dataset), len(val_loader.dataset)
    )
    lit_model = build_model(cfg)

    ckpt_dir = _resolve_path(oc.get("checkpoint_dir", "checkpoints/refrwkv_sr"))
    log_dir = _resolve_path(oc.get("log_dir", "logs/refrwkv_sr"))
    exp_name = str(oc.get("experiment_name", "refrwkv_sr"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    with (ckpt_dir / "train_config.yaml").open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(cfg, file_obj, allow_unicode=True, sort_keys=False)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="{epoch:04d}-{val_loss:.6f}",
            monitor="val_loss",
            save_top_k=tc.get("save_top_k", 3),
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    early_stopping_patience = tc.get("early_stopping_patience", 30)
    if early_stopping_patience is not None:
        callbacks.insert(
            1,
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                mode="min",
                verbose=True,
            ),
        )

    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=tc.get("devices", 1),
        precision=tc.get("precision", "bf16-mixed"),
        max_epochs=tc.get("max_epochs", 200),
        log_every_n_steps=tc.get("log_every_n_steps", 20),
        val_check_interval=tc.get("val_check_interval", 1.0),
        gradient_clip_val=tc.get("grad_clip_val", mc.get("grad_clip_norm", 1.0)),
        gradient_clip_algorithm=str(tc.get("gradient_clip_algorithm", "norm")).lower(),
        callbacks=callbacks,
        logger=TensorBoardLogger(str(log_dir), name=exp_name),
        enable_progress_bar=bool(tc.get("enable_progress_bar", True)),
        accumulate_grad_batches=tc.get("accumulate_grad_batches", 1),
    )
    if "max_steps" in tc:
        trainer_kwargs["max_steps"] = tc["max_steps"]
    if "strategy" in tc:
        trainer_kwargs["strategy"] = tc["strategy"]
    if "num_sanity_val_steps" in tc:
        trainer_kwargs["num_sanity_val_steps"] = tc["num_sanity_val_steps"]
    trainer = pl.Trainer(**trainer_kwargs)

    resume_ckpt = None
    explicit_resume = _checkpoint_path(args.resume)
    explicit_weights = _checkpoint_path(args.load_weights)
    if explicit_weights is not None:
        load_weights_filtered(lit_model, explicit_weights)
    else:
        candidate = explicit_resume
        source = "显式 --resume"
        if candidate is None:
            last_ckpt = ckpt_dir / "last.ckpt"
            if last_ckpt.is_file():
                candidate, source = last_ckpt, "自动检测 last.ckpt"
        if candidate is not None:
            compatible, reason = check_resume_compatible(candidate, lit_model)
            if compatible:
                resume_ckpt = str(candidate)
                logger.info("断点续训 (%s): %s（%s）", source, candidate, reason)
            elif explicit_resume is not None:
                logger.error("--resume 不兼容: %s\n  原因: %s", candidate, reason)
                raise SystemExit(1)
            else:
                # Leave an incompatible last.ckpt untouched; it may still be
                # useful for a later configuration or manual weight loading.
                logger.warning(
                    "%s 不兼容，已忽略，从头训练: %s\n  原因: %s",
                    source,
                    candidate,
                    reason,
                )

    logger.info("=" * 60)
    logger.info("  RefSRWKV SR Prior 训练 (Native LR Grid)")
    logger.info(
        "  数据: %s (train LR=%d, train HR=%d, scale=x%d)",
        cfg["data"]["root"],
        cfg["data"].get("train_lr_patch"),
        cfg["data"].get("train_hr_patch", cfg["data"].get("patch_size", 480)),
        cfg["data"].get("scale", 4),
    )
    logger.info(
        "  验证: %s | scale=x%d | output head=%s",
        "full image" if cfg["data"].get("val_patch_size") is None else f"HR {cfg['data']['val_patch_size']}",
        cfg["data"].get("scale", 4),
        lit_model.model_sr.upsampler,
    )
    window_cfg = lit_model.model_sr.window_config
    window_text = ", ".join(
        f"{name}={spec['size']}/{list(spec['offsets'])}"
        for name, spec in window_cfg["stages"].items()
    )
    logger.info("  Windows: phase=%s | %s", window_cfg["phase_mode"], window_text)
    logger.info(
        "  Batch size: %d × accumulate %d = 等效 %d",
        cfg["data"].get("batch_size", 4),
        tc.get("accumulate_grad_batches", 1),
        cfg["data"].get("batch_size", 4) * tc.get("accumulate_grad_batches", 1),
    )
    scheduler_name = str(mc.get("lr_scheduler", "plateau")).lower()
    if scheduler_name == "plateau":
        logger.info(
            "  LR: %.1e | plateau: every validation, patience=%d, threshold=%.1e, factor=%.3g, min=%.1e",
            mc.get("learning_rate", 1e-4),
            mc.get("lr_patience", 2),
            mc.get("lr_threshold", 1e-4),
            mc.get("lr_factor", 0.5),
            mc.get("lr_min", 1e-6),
        )
    else:
        logger.info(
            "  LR: %.1e | cosine warmup: %d 步",
            mc.get("learning_rate", 1e-4),
            mc.get("warmup_steps", 0),
        )
    logger.info(
        "  Loss: L1 + SSIM(%.2f) + FFT(%.2f) | Adam betas=%s | weight_decay=%.3g | Ref=%s | Ref dropout=%.2f",
        mc.get("ssim_weight", 0.0),
        mc.get("fft_weight", 0.0),
        tuple(mc.get("adam_betas", [0.9, 0.999])),
        mc.get("weight_decay", 0.0),
        cfg["data"].get("reference_mode", "paired"),
        mc.get("ref_drop_prob", 0.0),
    )
    logger.info(
        "  EMA: %s",
        (
            "on (decay=%.4f)" % mc.get("ema_decay", 0.999)
            if mc.get("use_ema", True)
            else "off"
        ),
    )
    if explicit_weights is not None:
        logger.info("  权重来源: 热启动 %s", explicit_weights)
    else:
        logger.info("  恢复: %s", resume_ckpt or "无（从头训练）")
    logger.info("=" * 60)

    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=resume_ckpt)
    logger.info("训练完成！最佳模型: %s", trainer.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
