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
from RefRWKV.models import RefSRWKV, RefDiffRWKV, EnRWKV, GlobalSemanticModule
from RefRWKV import RefRWKV_PL
from RefRWKV.RefSR_data.RefSR_dataset import RefPNGDataset


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    # ========== 1. 加载配置文件 ==========
    cfg = load_config(args.config)

    # ========== 2. 自动创建输出目录 ==========
    output_cfg = cfg.get("output", {})
    checkpoint_dir = output_cfg.get("checkpoint_dir", "checkpoints")
    log_dir = output_cfg.get("log_dir", "logs")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ========== 3. 数据集参数 ==========
    data_cfg = cfg["data"]

    # 构建 max_samples 元组
    max_samples_tuple = (
        data_cfg.get("max_samples_train"),
        data_cfg.get("max_samples_val"),
        data_cfg.get("max_samples_test"),
    )

    # 公共参数
    dataset_kwargs = {
        "data_dir": data_cfg["root"],
        "patch_size": data_cfg.get("patch_size"),  # 可以是 None
        "scale": data_cfg.get("scale", 10),
        "ref_aug_strengths": data_cfg.get(
            "ref_aug_strengths", [0.12, 0.12, 0.12, 0.03]
        ),
        "ref_aug_probs": data_cfg.get("ref_aug_probs", [0.5, 0.5, 0.5, 0.5]),
        "ref_gray_prob": data_cfg.get("ref_gray_prob", 0.2),
        "max_samples": max_samples_tuple,
        "sample_seed": 42,
    }

    # ========== 4. 数据集 ==========
    # Train dataset
    train_ds = RefPNGDataset(
        mode="train",
        augment=data_cfg.get("augment", True),
        augment_ref=data_cfg.get("augment_ref"),
        **dataset_kwargs,
    )

    # Val dataset
    val_ds = RefPNGDataset(
        mode="val", augment=False, augment_ref=False, **dataset_kwargs
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

    # ========== 5. 实例化三个模型 ==========
    model_cfg = cfg.get("model", {})
    sr_cfg = cfg.get("sr", {})
    enhance_cfg = cfg.get("enhance", {})

    # 创建全局语义模块
    global_semantic = GlobalSemanticModule(
        target_dim=cfg["model"].get("embed_dim", 64),
        num_tokens=32,
        use_rwkv=True,
    )

    # 5.1 RefDiffRWKV
    model_diff = RefDiffRWKV(
        img_size=model_cfg.get("img_size", 256),
        patch_size=model_cfg.get("patch_size", 4),
        embed_dim=model_cfg.get("embed_dim", 64),
        channels=model_cfg.get("channels", 3),
        enc_blocks=model_cfg.get("enc_blocks", [4, 6, 6]),
        dec_blocks=model_cfg.get("dec_blocks", [6, 6, 4]),
        latent_blocks=model_cfg.get("latent_blocks", 8),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
        hidden_rate=model_cfg.get("hidden_rate", 4),
        learn_sigma=model_cfg.get("learn_sigma", False),
        upsample_mode=model_cfg.get("upsample_mode", "cnn"),
        global_semantic=global_semantic,
    )

    # 5.2 RefSRWKV
    model_sr = RefSRWKV(
        inp_channels=sr_cfg.get("inp_channels", 3),
        out_channels=sr_cfg.get("out_channels", 3),
        dim=sr_cfg.get("dim", 48),
        num_blocks=sr_cfg.get("num_blocks", [4, 6, 6, 8]),
        num_refinement_blocks=sr_cfg.get("num_refinement_blocks", 8),
        scale=sr_cfg.get("scale", 10),
    )

    # 5.3 EnRWKV
    model_enhance = EnRWKV(
        inp_channels=enhance_cfg.get("inp_channels", 3),
        out_channels=enhance_cfg.get("out_channels", 3),
        dim=enhance_cfg.get("dim", 48),
        num_blocks=enhance_cfg.get("num_blocks", [4, 6, 6, 8]),
        num_refinement_blocks=enhance_cfg.get("num_refinement_blocks", 4),
    )

    # ========== 6. Lightning 模块 ==========
    train_cfg = cfg["train"]
    pl_model = RefRWKV_PL(
        model_sr=model_sr,
        model_diff=model_diff,
        model_enhance=model_enhance,
        global_semantic=global_semantic,
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

    # ========== 7. Trainer 配置 ==========
    logger = TensorBoardLogger(log_dir, name="RefRWKV")

    callbacks = [
        EarlyStopping(
            monitor="val-loss_total",
            patience=15,
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
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

    # ========== 8. 检查点恢复 ==========
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
