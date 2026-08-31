#!/usr/bin/env python
"""
RefSRWKV SR Prior 训练脚本。

配置文件、数据目录和 checkpoint 路径默认相对于仓库根目录解析，因而从
PowerShell 的任意当前目录启动都得到一致行为。配置文本按 UTF-8（兼容 BOM）读取。
"""
import argparse
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
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from RefRWKV.models.RefSRWKV import LitRefSRWKV, RefSRWKV
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


def load_config(path):
    path = _resolve_path(path, prefer_cwd=True)
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8-sig") as file_obj:
        config = yaml.safe_load(file_obj)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件顶层必须是 mapping: {path}")
    return config


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
    if maximum is not None and (value > maximum or (not maximum_inclusive and value == maximum)):
        suffix = ")" if not maximum_inclusive else "]"
        raise ValueError(f"{name} 必须位于 [{minimum}, {maximum}{suffix}，得到 {value}")


def validate_config(cfg):
    """Validate assumptions imposed by the fixed HR/4 U-Net grid."""
    if not isinstance(cfg, dict):
        raise ValueError("配置必须是 mapping")
    for section in ("data", "model"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"配置缺少 mapping: {section}")

    dc, mc = cfg["data"], cfg["model"]
    patch_size = dc.get("patch_size", 480)
    scale = dc.get("scale", 4)
    _require_int(patch_size, "data.patch_size")
    _require_int(scale, "data.scale")
    if patch_size % scale != 0:
        raise ValueError(f"data.patch_size ({patch_size}) 必须能被 data.scale ({scale}) 整除")
    if patch_size % 32 != 0:
        raise ValueError("data.patch_size 必须能被 32 整除（HR/4 网格再经过三次 x2 下采样和 8x8 窗口）")

    model_scale = mc.get("scale", scale)
    _require_int(model_scale, "model.scale")
    if model_scale != scale:
        raise ValueError(f"model.scale ({model_scale}) 与 data.scale ({scale}) 不一致")
    _require_int(mc.get("inp_channels", 3), "model.inp_channels")
    _require_int(mc.get("out_channels", 3), "model.out_channels")
    ref_channels = mc.get("ref_channels", mc.get("inp_channels", 3))
    _require_int(ref_channels, "model.ref_channels")
    if ref_channels != mc.get("inp_channels", 3):
        raise ValueError("当前颜色对齐路径要求 model.ref_channels == model.inp_channels")

    dim = mc.get("dim", 48)
    _require_int(dim, "model.dim")
    if dim % 16 != 0:
        raise ValueError("model.dim 必须是 16 的倍数，以满足 CUDA WKV 的通道分组约束")
    blocks = mc.get("num_blocks", [4, 6, 6, 8])
    if not isinstance(blocks, (list, tuple)) or len(blocks) != 4:
        raise ValueError("model.num_blocks 必须包含四项")
    for index, count in enumerate(blocks):
        _require_int(count, f"model.num_blocks[{index}]")
    _require_int(mc.get("num_refinement_blocks", 4), "model.num_refinement_blocks", minimum=0)
    _require_number(mc.get("drop_path_rate", 0.1), "model.drop_path_rate", minimum=0.0, maximum=1.0, maximum_inclusive=False)
    _require_number(mc.get("hidden_rate", 4), "model.hidden_rate", minimum=1e-12)
    learning_rate = mc.get("learning_rate", 1e-4)
    _require_number(learning_rate, "model.learning_rate", minimum=1e-12)
    lr_scheduler = str(mc.get("lr_scheduler", "plateau")).lower()
    if lr_scheduler not in {"plateau", "cosine"}:
        raise ValueError("model.lr_scheduler 只能是 plateau 或 cosine")
    _require_int(mc.get("lr_patience", 1), "model.lr_patience", minimum=0)
    _require_number(mc.get("lr_factor", 0.5), "model.lr_factor", minimum=1e-12, maximum=1.0, maximum_inclusive=False)
    lr_min = mc.get("lr_min", 1e-6)
    _require_number(lr_min, "model.lr_min", minimum=0.0)
    if float(lr_min) > float(learning_rate):
        raise ValueError("model.lr_min 不能大于 model.learning_rate")
    _require_int(mc.get("warmup_steps", 0), "model.warmup_steps", minimum=0)
    _require_number(mc.get("grad_clip_norm", 1.0), "model.grad_clip_norm", minimum=0.0)
    _require_number(mc.get("ema_decay", 0.999), "model.ema_decay", minimum=0.0, maximum=1.0, maximum_inclusive=False)
    _require_number(mc.get("ssim_weight", 0.0), "model.ssim_weight", minimum=0.0)
    _require_number(mc.get("fft_weight", 0.0), "model.fft_weight", minimum=0.0)
    _require_number(mc.get("ref_drop_prob", 0.0), "model.ref_drop_prob", minimum=0.0, maximum=1.0)

    root = dc.get("root")
    if not isinstance(root, (str, Path)) or not str(root):
        raise ValueError("data.root 必须是非空路径")
    for name in ("batch_size", "val_batch_size", "num_workers", "val_num_workers"):
        minimum = 0 if "workers" in name else 1
        _require_int(dc.get(name, 4 if name == "batch_size" else 1), f"data.{name}", minimum=minimum)
    _require_int(dc.get("prefetch_factor", 4), "data.prefetch_factor")
    for name in ("max_samples_train", "max_samples_val", "max_samples_test"):
        value = dc.get(name)
        if value is not None:
            _require_int(value, f"data.{name}")
    strengths = dc.get("ref_aug_strengths", [0.12, 0.12, 0.12, 0.03])
    probs = dc.get("ref_aug_probs", [0.5, 0.5, 0.5, 0.5])
    if not isinstance(strengths, (list, tuple)) or not isinstance(probs, (list, tuple)) or len(strengths) != len(probs):
        raise ValueError("data.ref_aug_strengths 与 data.ref_aug_probs 必须是等长列表")
    for index, value in enumerate(strengths):
        _require_number(value, f"data.ref_aug_strengths[{index}]", minimum=0.0)
    for index, value in enumerate(probs):
        _require_number(value, f"data.ref_aug_probs[{index}]", minimum=0.0, maximum=1.0)
    tc = cfg.get("train", {})
    if not isinstance(tc, dict):
        raise ValueError("train 必须是 mapping")
    _require_number(dc.get("ref_gray_prob", 0.2), "data.ref_gray_prob", minimum=0.0, maximum=1.0)
    _require_int(dc.get("sample_seed", tc.get("seed", 42)), "data.sample_seed", minimum=0)
    _require_int(tc.get("max_epochs", 200), "train.max_epochs")
    _require_int(tc.get("accumulate_grad_batches", 1), "train.accumulate_grad_batches")
    if "max_steps" in tc:
        if isinstance(tc["max_steps"], bool) or not isinstance(tc["max_steps"], int) or tc["max_steps"] < -1:
            raise ValueError("train.max_steps 必须是 >= -1 的整数")
    if "num_sanity_val_steps" in tc:
        _require_int(tc["num_sanity_val_steps"], "train.num_sanity_val_steps", minimum=-1)
    _require_int(tc.get("log_every_n_steps", 20), "train.log_every_n_steps")
    interval = tc.get("val_check_interval", 0.1)
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(float(interval)) or interval <= 0:
        raise ValueError("train.val_check_interval 必须为正数")
    if isinstance(interval, float) and interval > 1.0:
        raise ValueError("train.val_check_interval 为小数时必须位于 (0, 1]；整数批次数请使用整数")
    _require_int(tc.get("save_top_k", 3), "train.save_top_k", minimum=-1)
    _require_int(tc.get("early_stopping_patience", 30), "train.early_stopping_patience", minimum=0)
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
        patch_size=dc.get("patch_size", 480),
        scale=dc.get("scale", 4),
        ref_aug_strengths=dc.get("ref_aug_strengths", [0.12, 0.12, 0.12, 0.03]),
        ref_aug_probs=dc.get("ref_aug_probs", [0.5, 0.5, 0.5, 0.5]),
        ref_gray_prob=dc.get("ref_gray_prob", 0.2),
        max_samples=(dc.get("max_samples_train"), dc.get("max_samples_val"), dc.get("max_samples_test")),
        sample_seed=dc.get("sample_seed", tc.get("seed", 42)),
        lr_key=dc.get("lr_key", "lr"),
        hr_key=dc.get("hr_key", "hr"),
        ref_key=dc.get("ref_key", "ref"),
    )
    train_ds = RefPNGDataset(
        mode="train",
        augment=dc.get("augment", True),
        augment_ref=dc.get("augment_ref", True),
        **common,
    )
    val_ds = RefPNGDataset(mode="val", augment=False, augment_ref=False, **common)
    if len(train_ds) == 0:
        raise ValueError("训练数据集为空")
    if len(val_ds) == 0:
        raise ValueError("验证数据集为空；当前脚本需要 val_loss 进行 checkpoint/early stopping")

    pin_memory = bool(dc.get("pin_memory", False))
    train_workers = dc.get("num_workers", 4)
    val_workers = dc.get("val_num_workers", 2)
    batch_size = dc.get("batch_size", 4)
    val_batch_size = dc.get("val_batch_size", 1)
    drop_last = bool(dc.get("drop_last", True))
    if drop_last and len(train_ds) < batch_size:
        logger.warning("训练样本数 (%d) 小于 batch_size (%d)，自动关闭 drop_last 以避免零 step", len(train_ds), batch_size)
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
    lr_key, hr_key, ref_key = dc.get("lr_key", "lr"), dc.get("hr_key", "hr"), dc.get("ref_key", "ref")
    hr_size = dc.get("patch_size", 480)
    data_scale = dc.get("scale", 4)
    model = RefSRWKV(
        inp_channels=mc.get("inp_channels", 3),
        out_channels=mc.get("out_channels", 3),
        ref_channels=mc.get("ref_channels", mc.get("inp_channels", 3)),
        dim=mc.get("dim", 48),
        num_blocks=tuple(mc.get("num_blocks", [4, 6, 6, 8])),
        num_refinement_blocks=mc.get("num_refinement_blocks", 4),
        scale=data_scale,
        hr_size=hr_size,
        drop_path_rate=mc.get("drop_path_rate", 0.1),
        hidden_rate=mc.get("hidden_rate", 4),
    )
    logger.info(
        "RefSRWKV 参数量: %.2fM (HR=%d, Internal=%d, scale=%d)",
        sum(parameter.numel() for parameter in model.parameters()) / 1e6,
        hr_size,
        model.internal_size,
        data_scale,
    )
    return LitRefSRWKV(
        model_sr=model,
        learning_rate=mc.get("learning_rate", 1e-4),
        lr_scheduler=mc.get("lr_scheduler", "plateau"),
        lr_patience=mc.get("lr_patience", 1),
        lr_factor=mc.get("lr_factor", 0.5),
        lr_min=mc.get("lr_min", 1e-6),
        warmup_steps=mc.get("warmup_steps", 0),
        grad_clip_norm=mc.get("grad_clip_norm", 1.0),
        ema_decay=mc.get("ema_decay", 0.999),
        use_ema=mc.get("use_ema", True),
        ssim_weight=mc.get("ssim_weight", 0.0),
        fft_weight=mc.get("fft_weight", 0.0),
        ref_drop_prob=mc.get("ref_drop_prob", 0.0),
        lr_key=lr_key,
        hr_key=hr_key,
        ref_key=ref_key,
    )


_CKPT_PREFIXES = tuple(
    sorted(
        ("generator.sr_model.", "generator.model_sr.", "sr_model.", "model_sr.", "model.", "module."),
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
                key = key[len(prefix):]
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
        direct = {key: tensor for key, tensor in value.items() if isinstance(key, str) and torch.is_tensor(tensor)}
        if direct:
            return direct
        for key in ("state_dict", "model_state_dict", "model", "params", "weights", "generator", "sr_model"):
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


def _normalise_state_dict(state_dict):
    """Strip wrapper prefixes while preferring the shortest original key."""
    normalized = {}
    source_lengths = {}
    for key, value in state_dict.items():
        normalized_key = _strip_prefix(key)
        if not isinstance(normalized_key, str):
            continue
        key_length = len(key)
        if normalized_key not in normalized or key_length < source_lengths[normalized_key]:
            normalized[normalized_key] = value
            source_lengths[normalized_key] = key_length
    return normalized


def check_resume_compatible(ckpt_path: str, lit_model):
    try:
        checkpoint = _load_raw_checkpoint(ckpt_path)
        state_dict = _normalise_state_dict(_extract_state_dict(checkpoint, ckpt_path))
    except Exception as exc:
        return False, f"checkpoint 读取失败: {exc}"
    if isinstance(checkpoint, dict) and checkpoint.get("lr_schedulers"):
        hparams = checkpoint.get("hyper_parameters", {})
        saved_scheduler = str(hparams.get("lr_scheduler", "cosine")).lower()
        current_scheduler = str(getattr(lit_model.hparams, "lr_scheduler", "plateau")).lower()
        if saved_scheduler != current_scheduler:
            return False, (
                f"学习率调度器不兼容: checkpoint={saved_scheduler}, "
                f"当前配置={current_scheduler}；请使用 --load_weights 重新开始优化器状态"
            )
    reference = _normalise_state_dict({key: value for key, value in lit_model.state_dict().items() if torch.is_tensor(value)})
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
        return False, f"{len(mismatched)} 个参数形状不匹配: {key}: ckpt{ckpt_shape} vs 现模型{model_shape}"
    missing = sorted(set(reference) - set(state_dict))
    if missing:
        return False, f"当前模型有 {len(missing)} 个参数在 checkpoint 中缺失: {missing[:3]}"
    extra = sorted(set(state_dict) - set(reference))
    if extra:
        return False, f"checkpoint 有 {len(extra)} 个多余参数: {extra[:3]}"
    return True, f"结构一致（{len(overlap)} 个张量全部匹配）"


def load_weights_filtered(lit_model, ckpt_path: str):
    state_dict = _normalise_state_dict(_load_state_dict(ckpt_path))
    reference = {key: value for key, value in lit_model.state_dict().items() if torch.is_tensor(value)}
    reference_by_normalized = {}
    for key in reference:
        reference_by_normalized.setdefault(_strip_prefix(key), key)

    matched, skipped = {}, []
    for normalized_key, value in state_dict.items():
        target_key = reference_by_normalized.get(normalized_key)
        if target_key is None:
            continue
        if tuple(value.shape) == tuple(reference[target_key].shape):
            matched[target_key] = value
        else:
            skipped.append(normalized_key)
    missing, _unexpected = lit_model.load_state_dict(matched, strict=False)
    logger.info(
        "热启动 %s: 匹配 %d/%d | 形状不匹配跳过 %d | 缺失 %d",
        ckpt_path,
        len(matched),
        len(reference),
        len(skipped),
        len(missing),
    )
    if skipped:
        logger.warning("  形状不匹配（保持随机初始化）示例: %s", skipped[:5])
    if not matched:
        raise RuntimeError(f"热启动失败：checkpoint 没有匹配到任何参数 ({ckpt_path})")


def _checkpoint_path(value):
    return None if value is None else _resolve_path(value, prefer_cwd=True)


def main():
    parser = argparse.ArgumentParser(description="RefSRWKV SR Prior 训练")
    parser.add_argument("--config", type=str, required=True)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--load_weights", type=str, default=None, help="仅加载匹配的模型权重，重新初始化优化器")
    checkpoint_group.add_argument("--resume", type=str, default=None, help="恢复完整 Lightning checkpoint（含 optimizer/EMA）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)
    # Keep all relative data/output paths anchored at the repository root.
    os.chdir(PROJECT_ROOT)
    tc, mc, oc = cfg.get("train", {}), cfg["model"], cfg.get("output", {})
    accelerator = str(tc.get("accelerator", "gpu")).lower()
    if accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if accelerator != "gpu" or not torch.cuda.is_available():
        raise RuntimeError("RefSRWKV 的空间 WKV 算子只支持 CUDA；请安装 CUDA 版 PyTorch 并使用 train.accelerator=gpu")
    pl.seed_everything(tc.get("seed", 42), workers=True)

    train_loader, val_loader = build_dataloaders(cfg)
    logger.info("训练样本: %d, 验证样本: %d", len(train_loader.dataset), len(val_loader.dataset))
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
        EarlyStopping(
            monitor="val_loss",
            patience=tc.get("early_stopping_patience", 30),
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=tc.get("devices", 1),
        precision=tc.get("precision", "bf16-mixed"),
        max_epochs=tc.get("max_epochs", 200),
        log_every_n_steps=tc.get("log_every_n_steps", 20),
        val_check_interval=tc.get("val_check_interval", 0.1),
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
                logger.warning("%s 不兼容，已忽略，从头训练: %s\n  原因: %s", source, candidate, reason)

    logger.info("=" * 60)
    logger.info("  RefSRWKV SR Prior 训练 (Fixed HR/4 Internal Resolution)")
    logger.info("  数据: %s (patch_size=%d, scale=%d)", cfg["data"]["root"], cfg["data"].get("patch_size", 480), cfg["data"].get("scale", 4))
    logger.info("  Batch size: %d × accumulate %d = 等效 %d", cfg["data"].get("batch_size", 4), tc.get("accumulate_grad_batches", 1), cfg["data"].get("batch_size", 4) * tc.get("accumulate_grad_batches", 1))
    scheduler_name = str(mc.get("lr_scheduler", "plateau")).lower()
    if scheduler_name == "plateau":
        logger.info(
            "  LR: %.1e | plateau: patience=%d epoch(s), factor=%.3g, min=%.1e",
            mc.get("learning_rate", 1e-4),
            mc.get("lr_patience", 1),
            mc.get("lr_factor", 0.5),
            mc.get("lr_min", 1e-6),
        )
    else:
        logger.info("  LR: %.1e | cosine warmup: %d 步", mc.get("learning_rate", 1e-4), mc.get("warmup_steps", 0))
    logger.info("  SSIM weight: %.2f | FFT weight: %.2f | Ref dropout: %.2f", mc.get("ssim_weight", 0.0), mc.get("fft_weight", 0.0), mc.get("ref_drop_prob", 0.0))
    logger.info("  EMA: %s", "on (decay=%.4f)" % mc.get("ema_decay", 0.999) if mc.get("use_ema", True) else "off")
    if explicit_weights is not None:
        logger.info("  权重来源: 热启动 %s", explicit_weights)
    else:
        logger.info("  恢复: %s", resume_ckpt or "无（从头训练）")
    logger.info("=" * 60)

    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=resume_ckpt)
    logger.info("训练完成！最佳模型: %s", trainer.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
