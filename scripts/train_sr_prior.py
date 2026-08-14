#!/usr/bin/env python
"""
RefSRWKV SR Prior 训练脚本

用法:
    # 从头训练
    python scripts/train_sr_prior.py --config configs/sr_prior.yaml

    # 断点续训
    python scripts/train_sr_prior.py --config configs/sr_prior.yaml \
        --resume checkpoints/refrwkv_sr/last.ckpt

    # 后台运行
    nohup python scripts/train_sr_prior.py --config configs/sr_prior.yaml \
        > train.log 2>&1 &
"""

import argparse
import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml

from RefRWKV.models.RefSRWKV import RefSRWKV, LitRefSRWKV
from RefRWKV.RefSR_data.RefDataset import RefPNGDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_sr_prior")


# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════
# 数据
# ═══════════════════════════════════════════════════════════════
def build_dataloaders(cfg):
    dc = cfg["data"]
    common = dict(
        data_dir=dc["root"],
        patch_size=dc.get("patch_size", 480),
        scale=dc.get("scale", 10),
        ref_aug_strengths=dc.get("ref_aug_strengths", [0.12, 0.12, 0.12, 0.03]),
        ref_aug_probs=dc.get("ref_aug_probs", [0.5, 0.5, 0.5, 0.5]),
        ref_gray_prob=dc.get("ref_gray_prob", 0.2),
        max_samples=(
            dc.get("max_samples_train"),
            dc.get("max_samples_val"),
            dc.get("max_samples_test"),
        ),
        sample_seed=42,
    )

    train_ds = RefPNGDataset(
        mode="train",
        augment=dc.get("augment", True),
        augment_ref=dc.get("augment_ref", True),
        **common,
    )
    val_ds = RefPNGDataset(mode="val", augment=False, augment_ref=False, **common)

    pin = dc.get("pin_memory", False)
    nw = dc.get("num_workers", 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=dc.get("batch_size", 4),
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=nw > 0,
        prefetch_factor=dc.get("prefetch_factor", 4) if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=dc.get("val_batch_size", 1),
        shuffle=False,
        num_workers=dc.get("val_num_workers", 2),
        pin_memory=pin,
    )
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════
# 模型
# ═══════════════════════════════════════════════════════════════
def build_model(cfg):
    mc = cfg["model"]
    model = RefSRWKV(
        inp_channels=mc.get("inp_channels", 3),
        out_channels=mc.get("out_channels", 3),
        dim=mc.get("dim", 48),
        num_blocks=tuple(mc.get("num_blocks", [4, 6, 6, 8])),
        num_refinement_blocks=mc.get("num_refinement_blocks", 4),
        scale=mc.get("scale", 10),
        drop_path_rate=mc.get("drop_path_rate", 0.1),
        hidden_rate=mc.get("hidden_rate", 4),
    )

    total = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("RefSRWKV 参数量: %.2fM", total)

    lit_model = LitRefSRWKV(
        model_sr=model,
        learning_rate=mc.get("learning_rate", 1e-4),
        warmup_steps=mc.get("warmup_steps", 500),
        grad_clip_norm=mc.get("grad_clip_norm", 1.0),
        ema_decay=mc.get("ema_decay", 0.999),
        use_ema=mc.get("use_ema", True),
    )
    return lit_model


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="RefSRWKV SR Prior 训练")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--load_weights", type=str, default=None,
                        help="仅加载模型权重（忽略不匹配的键），不恢复优化器状态")
    parser.add_argument("--resume", type=str, default=None,
                        help="从指定 checkpoint 恢复训练")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tc = cfg.get("train", {})
    oc = cfg.get("output", {})

    # 种子
    pl.seed_everything(tc.get("seed", 42), workers=True)

    # 数据
    train_loader, val_loader = build_dataloaders(cfg)
    logger.info("训练样本: %d, 验证样本: %d",
                len(train_loader.dataset), len(val_loader.dataset))

    # 模型
    lit_model = build_model(cfg)

    # 输出目录
    ckpt_dir = oc.get("checkpoint_dir", "checkpoints/refrwkv_sr")
    log_dir = oc.get("log_dir", "logs/refrwkv_sr")
    exp_name = oc.get("experiment_name", "refrwkv_sr")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 回调
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
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

    # Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=tc.get("devices", 1),
        precision=str(tc.get("precision", "bf16-mixed")),
        max_epochs=tc.get("max_epochs", 200),
        log_every_n_steps=tc.get("log_every_n_steps", 20),
        val_check_interval=tc.get("val_check_interval", 0.1),
        gradient_clip_val=tc.get("grad_clip_val", 1.0),
        gradient_clip_algorithm=tc.get("grad_clip_algorithm", "norm"),
        callbacks=callbacks,
        logger=TensorBoardLogger(log_dir, name=exp_name),
        enable_progress_bar=True,
        accumulate_grad_batches=tc.get("accumulate_grad_batches", 1),
    )

    # 恢复
    resume_ckpt = args.resume
    if resume_ckpt is None:
        last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            resume_ckpt = last_ckpt
            logger.info("自动检测到 last.ckpt，断点续训: %s", last_ckpt)

    mc = cfg["model"]
    logger.info("=" * 60)
    logger.info("  RefSRWKV SR Prior 训练")
    logger.info("  数据: %s", cfg["data"]["root"])
    logger.info("  Batch size: %d × accumulate %d = 等效 %d",
                cfg["data"].get("batch_size", 4),
                tc.get("accumulate_grad_batches", 1),
                cfg["data"].get("batch_size", 4) * tc.get("accumulate_grad_batches", 1))
    logger.info("  LR: %.1e | warmup: %d 步",
                mc.get("learning_rate", 1e-4), mc.get("warmup_steps", 500))
    logger.info("  梯度裁剪: %.2f (%s)",
                tc.get("grad_clip_val", 1.0), tc.get("grad_clip_algorithm", "norm"))
    logger.info("  EMA: %s",
                "on (decay=%.4f)" % mc.get("ema_decay", 0.999)
                if mc.get("use_ema", True) else "off")
    logger.info("  恢复: %s", resume_ckpt or "无（从头训练）")
    logger.info("=" * 60)

    # 训练
    # ── 加载权重 ──
    if args.resume:
        # 完整恢复（模型 + 优化器 + epoch），要求架构完全一致
        trainer.fit(lit_model, train_loader, val_loader, ckpt_path=args.resume)
    elif args.load_weights:
        # 仅加载模型权重（strict=False），优化器从头开始
        logger.info("加载模型权重（忽略不匹配的键）: %s", args.load_weights)
        ckpt = torch.load(args.load_weights, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        missing, unexpected = lit_model.load_state_dict(sd, strict=False)
        logger.info("  匹配成功，跳过 %d 个缺失键, %d 个多余键",
                     len(missing), len(unexpected))
        trainer.fit(lit_model, train_loader, val_loader)
    else:
        trainer.fit(lit_model, train_loader, val_loader)

    logger.info("训练完成！最佳模型: %s", trainer.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()