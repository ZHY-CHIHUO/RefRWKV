#!/usr/bin/env python
# train.py
import sys
from pathlib import Path
# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# ==================== 导入模型和数据集 ====================
from model import RefDiffRWKV,RefDiffRWKV_PL
from RWKV.RefSR_data.RefSR_dataset import RefPNGDataset


def main():
    # ====================== 模型参数 ======================
    model_config = {
        "img_size": 256,
        "patch_size": 4,              # ← 模型 PatchEmbed 使用（推荐 4 或 8）
        "embed_dim": 64,
        "enc_blocks": [4, 6, 6],
        "dec_blocks": [6, 6, 4],
        "latent_blocks": 8,
        "drop_path_rate": 0.1,
        "upsample_mode": "cnn",
        "channels": 3,
    }

    # ====================== 数据集参数 ======================
    data_config = {
        "data_root": r"/home/zhy/PROJECT/RWKV/RefSR_data/ALL_2",
        "crop_size": 480,
        "scale": 10,
        "max_samples": (1000, None, None),   # train, val, test
        "batch_size": 4,
        "num_workers": 2,
    }

    # ====================== 训练超参 ======================
    train_config = {
        "lr": 4e-4,
        "warmup_steps": 100,
        "max_epochs": 200,
        "accumulate_grad_batches": 4,
        "precision": "bf16",
    }

    # ====================== 数据集 ======================
    train_ds = RefPNGDataset(
        data_dir=data_config["data_root"],
        mode="train",
        patch_size=data_config["crop_size"],   # 数据集裁剪大小
        augment=True,
        max_samples=data_config["max_samples"],
        sample_seed=42,
    )

    val_ds = RefPNGDataset(
        data_dir=data_config["data_root"],
        mode="val",
        patch_size=None,                       # 验证使用全图
        augment=False,
        max_samples=data_config["max_samples"],
        sample_seed=42,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
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

    # ====================== 模型 ======================
    base_model = RefDiffRWKV(
        img_size=model_config["img_size"],
        patch_size=model_config["patch_size"],   # 模型的 patch_size
        embed_dim=model_config["embed_dim"],
        channels=model_config["channels"],
        enc_blocks=model_config["enc_blocks"],
        dec_blocks=model_config["dec_blocks"],
        latent_blocks=model_config["latent_blocks"],
        drop_path_rate=model_config["drop_path_rate"],
        upsample_mode=model_config["upsample_mode"],
    )

    pl_model = RefDiffRWKV_PL(
        model=base_model,
        lr=train_config["lr"],
        warmup_steps=train_config["warmup_steps"],
        max_epochs=train_config["max_epochs"],
        scheduler="cosine",
    )

    # ====================== Trainer 配置 ======================
    logger = TensorBoardLogger("logs", name="RefDiffRWKV")

    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="refdiff-{epoch:04d}-{val/loss:.5f}",
            monitor="val/loss",
            save_top_k=3,
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=train_config["precision"],
        max_epochs=train_config["max_epochs"],
        log_every_n_steps=20,
        val_check_interval=2000,
        gradient_clip_val=1.0,
        accumulate_grad_batches=train_config["accumulate_grad_batches"],
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
    )

    print("🚀 开始训练 RefDiffRWKV")
    print(f"   Model Patch Size : {model_config['patch_size']}")
    print(f"   Dataset Crop Size: {data_config['crop_size']}")
    print(f"   Embed Dim        : {model_config['embed_dim']}")
    print(f"   Batch Size       : {data_config['batch_size']}")

    trainer.fit(
        pl_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    main()
