#!/usr/bin/env python
# train.py
import sys
import argparse
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import os
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train RefRWKV (CRefDiff) full pipeline")

    # ====== 模型参数 ======
    parser.add_argument("--img_size", type=int, default=256, help="Image size for training")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size for PatchEmbed")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--enc_blocks", type=int, nargs="+", default=[4, 6, 6], help="Encoder blocks per level")
    parser.add_argument("--dec_blocks", type=int, nargs="+", default=[6, 6, 4], help="Decoder blocks per level")
    parser.add_argument("--latent_blocks", type=int, default=8, help="Number of latent blocks")
    parser.add_argument("--drop_path_rate", type=float, default=0.1, help="Drop path rate")
    parser.add_argument("--upsample_mode", type=str, default="cnn", choices=["bilinear", "cnn", "pixelshuffle"], help="Upsample mode for LR")
    parser.add_argument("--channels", type=int, default=3, help="Number of input/output channels")

    # 超分模型参数 (RefSRWKV)
    parser.add_argument("--sr_dim", type=int, default=48, help="Base dimension for RefSRWKV")
    parser.add_argument("--sr_num_blocks", type=int, nargs="+", default=[4, 6, 6, 8], help="Blocks per level for RefSRWKV")
    parser.add_argument("--sr_num_refinement", type=int, default=8, help="Refinement blocks for RefSRWKV")

    # 增强模型参数 (EnRWKV)
    parser.add_argument("--enhance_dim", type=int, default=48, help="Base dimension for EnRWKV")
    parser.add_argument("--enhance_num_blocks", type=int, nargs="+", default=[4, 6, 6, 8], help="Blocks per level for EnRWKV")
    parser.add_argument("--enhance_num_refinement", type=int, default=4, help="Refinement blocks for EnRWKV")

    # ====== 训练开关 ======
    parser.add_argument("--train_sr", action="store_true", help="Train RefSRWKV")
    parser.add_argument("--train_diff", action="store_true", default=True, help="Train RefDiffRWKV")
    parser.add_argument("--train_enhance", action="store_true", help="Train EnRWKV")
    parser.add_argument("--no_train_sr", dest="train_sr", action="store_false")
    parser.add_argument("--no_train_diff", dest="train_diff", action="store_false")
    parser.add_argument("--no_train_enhance", dest="train_enhance", action="store_false")
    parser.set_defaults(train_sr=True, train_diff=True, train_enhance=True)

    # ====== 数据集参数 ======
    parser.add_argument("--data_root", type=str, default="/home/zhy/PROJECT/RWKV/RefSR_data/ALL_2", help="Root directory of dataset")
    parser.add_argument("--crop_size", type=int, default=160, help="Crop size for training patches")
    parser.add_argument("--scale", type=int, default=10, help="Super-resolution scale factor")
    parser.add_argument("--max_samples_train", type=int, default=10000, help="Max train samples")
    parser.add_argument("--max_samples_val", type=int, default=None, help="Max val samples")
    parser.add_argument("--max_samples_test", type=int, default=None, help="Max test samples")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loading workers")

    # ====== 训练超参 ======
    parser.add_argument("--lr_sr", type=float, default=1e-4, help="Learning rate for RefSRWKV")
    parser.add_argument("--lr_diff", type=float, default=4e-4, help="Learning rate for RefDiffRWKV")
    parser.add_argument("--lr_enhance", type=float, default=1e-4, help="Learning rate for EnRWKV")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
    parser.add_argument("--max_epochs", type=int, default=200, help="Maximum number of epochs")
    parser.add_argument("--accumulate_grad_batches", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--precision", type=str, default="bf16", choices=["32", "16", "bf16"], help="Precision for training")

    # ====== 损失权重 ======
    parser.add_argument("--loss_sr_weight", type=float, default=0.1, help="Weight for SR loss")
    parser.add_argument("--loss_enhance_weight", type=float, default=0.1, help="Weight for enhancement loss")

    # ====== 增强阈值 ======
    parser.add_argument("--t_enhance_threshold", type=int, default=250, help="Only train enhance when t < threshold")

    return parser.parse_args()


def main():
    args = parse_args()

    # ====================== 数据集 ======================
    train_ds = RefPNGDataset(
        data_dir=args.data_root,
        mode="train",
        patch_size=args.crop_size,
        augment=True,
        augment_ref=True,
        max_samples=(args.max_samples_train, args.max_samples_val, args.max_samples_test),
        sample_seed=42,
    )

    val_ds = RefPNGDataset(
        data_dir=args.data_root,
        mode="val",
        patch_size=160,  # 验证使用全图
        augment=False,
        max_samples=(args.max_samples_train, args.max_samples_val, args.max_samples_test),
        sample_seed=42,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
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

    # ====================== 实例化三个模型 ======================
    # 1. RefSRWKV (Better Start)
    model_sr = RefSRWKV(
        inp_channels=args.channels,
        out_channels=args.channels,
        dim=args.sr_dim,
        num_blocks=args.sr_num_blocks,
        num_refinement_blocks=args.sr_num_refinement,
        scale=args.scale,
    )

    # 2. RefDiffRWKV (主扩散)
    model_diff = RefDiffRWKV(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        channels=args.channels,
        enc_blocks=args.enc_blocks,
        dec_blocks=args.dec_blocks,
        latent_blocks=args.latent_blocks,
        drop_path_rate=args.drop_path_rate,
        upsample_mode=args.upsample_mode,
    )

    # 3. EnRWKV (增强)
    model_enhance = EnRWKV(
        inp_channels=args.channels,
        out_channels=args.channels,
        dim=args.enhance_dim,
        num_blocks=args.enhance_num_blocks,
        num_refinement_blocks=args.enhance_num_refinement,
    )

    # ====================== Lightning 模块 ======================
    pl_model = RefRWKV_PL(
        model_sr=model_sr,
        model_diff=model_diff,
        model_enhance=model_enhance,
        train_sr=args.train_sr,
        train_diff=args.train_diff,
        train_enhance=args.train_enhance,
        t_enhance_threshold=args.t_enhance_threshold,
        num_timesteps=1000,  # 固定，可根据需要调整
        lr_sr=args.lr_sr,
        lr_diff=args.lr_diff,
        lr_enhance=args.lr_enhance,
        weight_decay=1e-2,
        beta1=0.9,
        beta2=0.999,
        warmup_epochs=args.warmup_epochs,
        scheduler="cosine",
        eta_min=None,
        loss_sr_weight=args.loss_sr_weight,
        loss_enhance_weight=args.loss_enhance_weight,
    )

    # ====================== Trainer 配置 ======================
    logger = TensorBoardLogger("logs", name="RefRWKV")

    callbacks = [
        EarlyStopping(
            monitor="val-loss_total",  # 注意指标名称带 _total 后缀
            patience=15,
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath="checkpoints",
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
        precision=args.precision,
        max_epochs=args.max_epochs,
        log_every_n_steps=20,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
    )

    print("🚀 开始训练 RefRWKV (CRefDiff) 全流程")
    print(f"   Train SR       : {args.train_sr}")
    print(f"   Train Diff     : {args.train_diff}")
    print(f"   Train Enhance  : {args.train_enhance}")
    print(f"   Patch Size     : {args.patch_size}")
    print(f"   Embed Dim      : {args.embed_dim}")
    print(f"   Batch Size     : {args.batch_size}")
    print(f"   t_enhance_thr  : {args.t_enhance_threshold}")

    # 自动查找 last.ckpt
    ckpt_path = None
    last_ckpt = os.path.join("checkpoints", "last.ckpt")
    if os.path.exists(last_ckpt):
        ckpt_path = last_ckpt
        print(f"发现上次训练存档，将从 {last_ckpt} 恢复训练")
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