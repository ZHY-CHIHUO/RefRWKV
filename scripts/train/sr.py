#!/usr/bin/env python3
"""Train a single-image SR model from the new task-oriented config tree."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loaders import build_sr_loaders
from engines.sr import SRTrainer
from runtime.checkpoint import load_checkpoint, load_model_weights
from runtime.config import load_config, validate_config
from runtime.experiments import layout_from_config, save_config_snapshot

logger = logging.getLogger("train.sr")


def _accelerator(train: dict[str, Any]) -> str:
    value = str(train.get("accelerator", "auto")).lower()
    if value == "auto":
        return "gpu" if torch.cuda.is_available() else "cpu"
    if value == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("train.accelerator=gpu 但当前环境没有 CUDA")
    return value


def _callbacks(config: dict[str, Any], checkpoint_dir: Path) -> list[Any]:
    train = config["train"]
    callbacks: list[Any] = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="epoch={epoch:04d}-step={step:06d}",
            monitor="val/loss",
            mode="min",
            save_top_k=int(train.get("save_top_k", 3)),
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    patience = train.get("early_stopping_patience")
    if patience is not None:
        callbacks.insert(
            1,
            EarlyStopping(monitor="val/loss", mode="min", patience=int(patience)),
        )
    return callbacks


def run(config: dict[str, Any], *, resume: str | None = None, load_weights: str | None = None, raw_weights: bool = False) -> None:
    validate_config(config)
    train_cfg = config["train"]
    pl.seed_everything(int(train_cfg.get("seed", 42)), workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    layout = layout_from_config(config).create_train()
    save_config_snapshot(config, layout.train_dir / "config.json")
    with (layout.train_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    train_loader, val_loader = build_sr_loaders(config)
    module = SRTrainer.from_config(config)

    if load_weights:
        report = load_model_weights(module.model, load_checkpoint(load_weights), prefer_ema=not raw_weights)
        logger.info("已加载模型权重 %s: %s", load_weights, report)

    accelerator = _accelerator(train_cfg)
    logger_obj = TensorBoardLogger(str(layout.train_dir), name="logs", version="")
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=train_cfg.get("devices", 1),
        precision=train_cfg.get("precision", "32"),
        max_epochs=train_cfg.get("max_epochs", -1),
        max_steps=train_cfg.get("max_steps", -1),
        val_check_interval=train_cfg.get("val_check_interval", 1.0),
        check_val_every_n_epoch=train_cfg.get("check_val_every_n_epoch", 1),
        accumulate_grad_batches=train_cfg.get("accumulate_grad_batches", 1),
        log_every_n_steps=train_cfg.get("log_every_n_steps", 20),
        num_sanity_val_steps=train_cfg.get("num_sanity_val_steps", 0),
        callbacks=_callbacks(config, layout.checkpoints),
        logger=logger_obj,
        enable_progress_bar=bool(train_cfg.get("enable_progress_bar", True)),
    )

    checkpoint = Path(resume).expanduser() if resume else layout.checkpoints / "last.ckpt"
    checkpoint = checkpoint if checkpoint.is_absolute() else PROJECT_ROOT / checkpoint
    ckpt_path = str(checkpoint) if checkpoint.is_file() else None
    logger.info("SR 训练: model=%s dataset=%s scale=x%s", config["model"]["name"], config["dataset"]["id"], config["data"]["scale"])
    trainer.fit(module, train_loader, val_loader, ckpt_path=ckpt_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--load-weights", default=None)
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument("--overrides", nargs="*", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config = load_config(args.config, args.overrides)
    run(config, resume=args.resume, load_weights=args.load_weights, raw_weights=args.raw_weights)


if __name__ == "__main__":
    main()
