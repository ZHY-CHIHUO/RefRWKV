#!/usr/bin/env python3
"""Train an HR/LR single-image SR baseline from configs/baselines/."""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from baselines.common import (
    build_dataloaders,
    experiment_signature,
    load_config,
    resolve_path,
    validate_config,
)
from baselines.lit import LitSISRBaseline
from baselines.registry import build_model, get_adapter
from baselines.runtime import checkpoint_signature, load_checkpoint, load_model_weights


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_baseline")


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _validate_train_options(config: dict[str, Any]) -> None:
    train = config.get("train", {})
    if not isinstance(train, dict):
        raise ValueError("train must be a mapping")
    for key in (
        "learning_rate",
        "weight_decay",
        "lr_factor",
        "lr_min",
        "lr_threshold",
        "ema_decay",
        "grad_clip_norm",
    ):
        if key in train:
            value = train[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"train.{key} must be a finite number")
    if float(train.get("learning_rate", 1.0e-4)) <= 0:
        raise ValueError("train.learning_rate must be positive")
    if float(train.get("weight_decay", 0.0)) < 0:
        raise ValueError("train.weight_decay must be non-negative")
    if not 0 < float(train.get("lr_factor", 0.5)) < 1:
        raise ValueError("train.lr_factor must be in (0, 1)")
    if float(train.get("lr_min", 1.0e-7)) < 0:
        raise ValueError("train.lr_min must be non-negative")
    if not 0 <= float(train.get("ema_decay", 0.999)) < 1:
        raise ValueError("train.ema_decay must be in [0, 1)")
    _positive_int(train.get("lr_patience", 3), "train.lr_patience", minimum=0)
    _positive_int(train.get("accumulate_grad_batches", 1), "train.accumulate_grad_batches")
    _positive_int(train.get("log_every_n_steps", 20), "train.log_every_n_steps")
    _positive_int(train.get("save_top_k", 3), "train.save_top_k", minimum=-1)
    interval = train.get("val_check_interval", 1.0)
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) or float(interval) <= 0:
        raise ValueError("train.val_check_interval must be positive")
    check_every = _positive_int(
        train.get("check_val_every_n_epoch", 1), "train.check_val_every_n_epoch"
    )
    if check_every > 1 and not (
        isinstance(interval, float) and float(interval) == 1.0
    ):
        raise ValueError(
            "When validation is spaced by epoch, set val_check_interval: 1.0; "
            "an integer val_check_interval means training batches."
        )
    max_epochs = train.get("max_epochs", -1)
    max_steps = train.get("max_steps", -1)
    if not isinstance(max_epochs, int) or max_epochs < -1:
        raise ValueError("train.max_epochs must be >= -1")
    if not isinstance(max_steps, int) or max_steps < -1:
        raise ValueError("train.max_steps must be >= -1")
    if max_epochs == -1 and max_steps == -1:
        raise ValueError("train.max_epochs and train.max_steps cannot both be -1")
    scheduler = str(train.get("lr_scheduler", "plateau")).lower()
    if scheduler not in {"plateau", "cosine"}:
        raise ValueError("train.lr_scheduler must be plateau or cosine")
    train["lr_scheduler"] = scheduler


def _same_signature(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _same_signature(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _same_signature(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-12)
    return left == right


def _resume_compatible(path: Path, current_signature: dict[str, Any]) -> tuple[bool, str]:
    try:
        saved = checkpoint_signature(load_checkpoint(path))
    except Exception as exc:
        return False, f"cannot read checkpoint: {exc}"
    if not saved:
        return False, "checkpoint has no baseline experiment signature; use --load_weights"
    if not _same_signature(saved, current_signature):
        return False, "checkpoint architecture/data contract differs; use --load_weights"
    return True, "matching baseline architecture and data contract"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="baseline run YAML")
    parser.add_argument(
        "--overrides", nargs="*", default=None, help="dotted key=value overrides"
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume", default=None, help="resume full Lightning checkpoint"
    )
    checkpoint_group.add_argument(
        "--load_weights", default=None, help="load model/EMA weights and reset optimizer"
    )
    parser.add_argument(
        "--raw-weights", action="store_true", help="with --load_weights, ignore EMA shadows"
    )
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    validate_config(config)
    _validate_train_options(config)
    os.chdir(PROJECT_ROOT)

    train = config["train"]
    accelerator = str(train.get("accelerator", "gpu")).lower()
    if accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("train.accelerator=gpu but CUDA is unavailable")
    if accelerator not in {"cpu", "gpu", "mps", "tpu"}:
        raise ValueError(f"unsupported train.accelerator: {accelerator}")
    if accelerator == "gpu":
        torch.set_float32_matmul_precision("high")
    pl.seed_everything(int(train.get("seed", 42)), workers=True)

    train_loader, val_loader = build_dataloaders(config)
    adapter = get_adapter(config["model"]["name"])
    model = build_model(config["model"], scale=config["data"]["scale"])
    lit_model = LitSISRBaseline(model, config)
    signature = experiment_signature(config)
    signature["adapter"] = adapter.describe(config["model"], scale=config["data"]["scale"])
    lit_model._signature = signature

    output = config["output"]
    checkpoint_dir = resolve_path(output["checkpoint_dir"])
    log_root = resolve_path(output["log_dir"])
    run_name = str(output["experiment_name"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / "train_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

    callbacks: list[Any] = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="epoch={epoch:04d}-val_loss={val_loss:.6f}",
            monitor="val_loss",
            mode="min",
            save_top_k=int(train.get("save_top_k", 3)),
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    early_stop = train.get("early_stopping_patience")
    if early_stop is not None:
        callbacks.insert(
            1,
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=_positive_int(
                    early_stop, "train.early_stopping_patience", minimum=0
                ),
            ),
        )

    trainer_kwargs: dict[str, Any] = {
        "accelerator": accelerator,
        "devices": train.get("devices", 1),
        "precision": train.get("precision", "bf16-mixed"),
        "max_epochs": train.get("max_epochs", -1),
        "max_steps": train.get("max_steps", -1),
        "val_check_interval": train.get("val_check_interval", 1.0),
        "check_val_every_n_epoch": train.get("check_val_every_n_epoch", 1),
        "accumulate_grad_batches": train.get("accumulate_grad_batches", 1),
        "gradient_clip_val": train.get("grad_clip_norm", 1.0),
        "gradient_clip_algorithm": "norm",
        "log_every_n_steps": train.get("log_every_n_steps", 20),
        "callbacks": callbacks,
        "logger": TensorBoardLogger(str(log_root), name=run_name),
        "enable_progress_bar": bool(train.get("enable_progress_bar", True)),
        "num_sanity_val_steps": train.get("num_sanity_val_steps", 0),
    }
    trainer = pl.Trainer(**trainer_kwargs)

    resume_path: str | None = None
    if args.load_weights:
        report = load_model_weights(
            lit_model.model,
            load_checkpoint(args.load_weights),
            prefer_ema=not args.raw_weights,
        )
        logger.info("hot-start %s: %s", args.load_weights, report)
    else:
        candidate = Path(args.resume).expanduser() if args.resume else None
        source = "--resume"
        if candidate is None and bool(train.get("auto_resume", True)):
            last = checkpoint_dir / "last.ckpt"
            if last.is_file():
                candidate, source = last, "automatic last.ckpt"
        if candidate is not None:
            candidate = resolve_path(candidate, prefer_cwd=True)
            compatible, reason = _resume_compatible(candidate, signature)
            if compatible:
                resume_path = str(candidate)
                logger.info("resuming %s (%s): %s", source, candidate, reason)
            elif args.resume:
                raise RuntimeError(f"--resume checkpoint is incompatible: {reason}")
            else:
                logger.warning("ignoring incompatible %s (%s): %s", source, candidate, reason)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info("=" * 68)
    logger.info("SISR baseline training: %s / %s", config["dataset"].get("id", "dataset"), run_name)
    logger.info(
        "model=%s | params=%.2fM | scale=x%d | train LR/HR=%d/%d",
        config["model"]["name"],
        parameter_count / 1.0e6,
        config["data"]["scale"],
        config["data"]["train_lr_patch"],
        config["data"]["train_hr_patch"],
    )
    logger.info(
        "train=%d | val=%d | batch=%d x accumulate=%d | validation every %d epoch(s)",
        len(train_loader.dataset),
        len(val_loader.dataset),
        config["data"].get("batch_size", 1),
        train.get("accumulate_grad_batches", 1),
        train.get("check_val_every_n_epoch", 1),
    )
    logger.info(
        "optimizer=Adam lr=%.2e | scheduler=%s | loss=%s | EMA=%s",
        train["learning_rate"],
        train["lr_scheduler"],
        config["loss"]["name"],
        "on" if train.get("use_ema", True) else "off",
    )
    logger.info("=" * 68)
    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=resume_path)
    logger.info("training complete; best checkpoint: %s", trainer.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
