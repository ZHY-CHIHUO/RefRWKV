#!/usr/bin/env python
# train.py
import sys
import argparse
import yaml
import os
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import TensorBoardLogger

# ==================== 导入模型和数据集 ====================
from RefRWKV.models import RefSRWKV, RefDiffRWKV, EnRWKV
from RefRWKV import RefRWKV_PL
from RefRWKV.RefSR_data.RefSR_dataset import RefPNGDataset


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    # ========== 1. 加载配置文件 ==========
    cfg = load_config(args.config)

    # ========== 2. 自动创建输出目录 ==========
    output_cfg = cfg.get("output", {})
    checkpoint_dir = output_cfg.get("checkpoint_dir", "checkpoints")
    log_dir = output_cfg.get("log_dir", "logs")
    result_dir = output_cfg.get("result_dir", "results")   # 供 test/inference 使用

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    print(f"✅ 输出目录已创建/确认:")
    print(f"   Checkpoints: {checkpoint_dir}")
    print(f"   Logs       : {log_dir}")
    print(f"   Results    : {result_dir}")

    # ========== 3. 数据集 ==========
    data_cfg = cfg["data"]
    train_ds = RefPNGDataset(
        data_dir=data_cfg["root"],
        mode="train",
        patch_size=data_cfg["crop_size"],
        augment=True,
        augment_ref=True,
        max_samples=(data_cfg["max_samples_train"],
                     data_cfg["max_samples_val"],
                     data_cfg["max_samples_test"]),
        sample_seed=42,
    )

    val_ds = RefPNGDataset(
        data_dir=data_cfg["root"],
        mode="val",
        patch_size=160,
        augment=False,
        max_samples=(data_cfg["max_samples_train"],
                     data_cfg["max_samples_val"],
                     data_cfg["max_samples_test"]),
        sample_seed=42,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ========== 4. 模型实例化 ==========
    model_cfg = cfg["model"]
    sr_cfg = cfg["sr"]
    enhance_cfg = cfg["enhance"]

    model_sr = RefSRWKV(
        inp_channels=model_cfg["channels"],
        out_channels=model_cfg["channels"],
        dim=sr_cfg["dim"],
        num_blocks=sr_cfg["num_blocks"],
        num_refinement_blocks=sr_cfg["num_refinement_blocks"],
        scale=data_cfg["scale"],
    )

    model_diff = RefDiffRWKV(
        img_size=model_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        embed_dim=model_cfg["embed_dim"],
        channels=model_cfg["channels"],
        enc_blocks=model_cfg["enc_blocks"],
        dec_blocks=model_cfg["dec_blocks"],
        latent_blocks=model_cfg["latent_blocks"],
        drop_path_rate=model_cfg["drop_path_rate"],
        upsample_mode=model_cfg["upsample_mode"],
    )

    model_enhance = EnRWKV(
        inp_channels=model_cfg["channels"],
        out_channels=model_cfg["channels"],
        dim=enhance_cfg["dim"],
        num_blocks=enhance_cfg["num_blocks"],
        num_refinement_blocks=enhance_cfg["num_refinement_blocks"],
    )

    # ========== 5. Lightning 模块 ==========
    train_cfg = cfg["train"]
    pl_model = RefRWKV_PL(
        model_sr=model_sr,
        model_diff=model_diff,
        model_enhance=model_enhance,
        train_sr=train_cfg["train_sr"],
        train_diff=train_cfg["train_diff"],
        train_enhance=train_cfg["train_enhance"],
        t_enhance_threshold=train_cfg["t_enhance_threshold"],
        num_timesteps=1000,
        lr_sr=train_cfg["lr_sr"],
        lr_diff=train_cfg["lr_diff"],
        lr_enhance=train_cfg["lr_enhance"],
        weight_decay=1e-2,
        beta1=0.9,
        beta2=0.999,
        warmup_epochs=train_cfg["warmup_epochs"],
        scheduler="cosine",
        eta_min=None,
        loss_sr_weight=train_cfg["loss_sr_weight"],
        loss_enhance_weight=train_cfg["loss_enhance_weight"],
    )

    # ========== 6. Trainer 配置 ==========
    logger = TensorBoardLogger(log_dir, name="RefRWKV")

    callbacks = [
        EarlyStopping(
            monitor="val-loss_total",
            patience=15,
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,          # 从配置文件读取
            filename="refrwkv-{epoch:04d}-{val-loss_total:.5f}",
            monitor="val-loss_total",
            save_top_k=3,
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=train_cfg["precision"],
        max_epochs=train_cfg["max_epochs"],
        log_every_n_steps=20,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        accumulate_grad_batches=train_cfg["accumulate_grad_batches"],
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
    )

    print("\n🚀 开始训练 RefRWKV (CRefDiff) 全流程")
    print(f"   配置文件: {args.config}")
    print(f"   Max Epochs: {train_cfg['max_epochs']}")
    print(f"   Batch Size: {data_cfg['batch_size']}")

    # ========== 7. 检查是否有上次的检查点 ==========
    last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
    ckpt_path = last_ckpt if os.path.exists(last_ckpt) else None

    if ckpt_path:
        print(f"发现上次训练存档，将从 {ckpt_path} 恢复训练")
    else:
        print("未找到存档，开始全新训练")

    trainer.fit(
        pl_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()