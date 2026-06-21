#!/usr/bin/env python
"""
分阶段训练脚本：sr → diff → enhance
用法:
    python scripts/train.py --config configs/base_train_config.yaml
    python scripts/train.py --config configs/base_train_config.yaml --stage diff
"""

import sys
import argparse
import yaml
import os
import shutil
from pathlib import Path

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

torch.set_float32_matmul_precision("high")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==================== 导入模型和数据集 ====================
from RefRWKV.models.RefSRWKV import RefSRWKV
from RefRWKV.models.RefDiffRWKV import RefDiffRWKV
from RefRWKV.models.EnRWKV import EnRWKV
from RefRWKV.models.GlobalSemanticModule import GlobalSemanticModule
from RefRWKV.RefSR_data.RefSR_dataset import RefLMDBDataset
from RefRWKV.RefRWKV_PL import LitRefSRWKV, LitRefDiffRWKV, LitEnRWKV


# ============================================================
# 工具函数
# ============================================================
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_weights(model: torch.nn.Module, ckpt_path: str, prefix: str = ""):
    """从 Lightning checkpoint 加载模型权重，自动剥离前缀。"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    new_state_dict = {}
    for k, v in state_dict.items():
        # 剥离可能的 "model_sr." / "model_diff." / "model." 前缀
        for pre in ["model_sr.", "model_diff.", "model_enhance.", "model."]:
            if k.startswith(pre):
                k = k[len(pre):]
                break
        new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"⚠️  缺失键 ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"⚠️  多余键 ({len(unexpected)}): {unexpected[:5]}...")
    print(f"✅ 从 {ckpt_path} 加载权重完成")


def build_dataloaders(cfg: dict):
    """返回 (train_loader, val_loader, test_loader)。所有阶段共用。"""
    data_cfg = cfg["data"]

    max_samples_tuple = (
        data_cfg.get("max_samples_train"),
        data_cfg.get("max_samples_val"),
        data_cfg.get("max_samples_test"),
    )

    dataset_kwargs = {
        "data_dir": data_cfg["root"],
        "patch_size": data_cfg.get("patch_size"),
        "scale": data_cfg.get("scale", 10),
        "ref_aug_strengths": data_cfg.get("ref_aug_strengths", [0.12, 0.12, 0.12, 0.03]),
        "ref_aug_probs": data_cfg.get("ref_aug_probs", [0.5, 0.5, 0.5, 0.5]),
        "ref_gray_prob": data_cfg.get("ref_gray_prob", 0.2),
        "max_samples": max_samples_tuple,
        "sample_seed": 42,
    }

    # --- 训练集 ---
    train_ds = RefLMDBDataset(
        mode="train",
        augment=data_cfg.get("augment", True),
        augment_ref=data_cfg.get("augment_ref", False),
        **dataset_kwargs,
    )

    # --- 验证集（不做增强）---
    val_ds = RefLMDBDataset(
        mode="val", augment=False, augment_ref=False, **dataset_kwargs
    )

    # --- 测试集 ---
    test_ds = RefLMDBDataset(
        mode="test", augment=False, augment_ref=False, **dataset_kwargs
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=data_cfg.get("prefetch_factor", 4),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def build_model_diff(cfg: dict, global_semantic: GlobalSemanticModule = None):
    """根据配置构建 RefDiffRWKV。"""
    mc = cfg.get("model", {})
    return RefDiffRWKV(
        patch_size=mc.get("patch_size", 4),
        embed_dim=mc.get("embed_dim", 64),
        channels=mc.get("channels", 3),
        enc_blocks=mc.get("enc_blocks", [4, 6, 6]),
        dec_blocks=mc.get("dec_blocks", [6, 6, 4]),
        latent_blocks=mc.get("latent_blocks", 8),
        drop_path_rate=mc.get("drop_path_rate", 0.1),
        hidden_rate=mc.get("hidden_rate", 4),
        learn_sigma=mc.get("learn_sigma", False),
        upsample_mode=mc.get("upsample_mode", "cnn"),
        global_semantic=global_semantic,
        use_checkpoint=mc.get("use_checkpoint", True),
    )


# ============================================================
# 各阶段训练函数
# ============================================================
def train_stage_sr(cfg: dict, train_loader, val_loader):
    """阶段 1：训练 RefSRWKV 超分模型。"""
    stage_cfg = cfg["train"]["sr"]
    checkpoint_dir = cfg["output"]["checkpoint_dir"]
    log_dir = cfg["output"]["log_dir"]
    exp_name = cfg["output"].get("experiment_name", "refrwkv") + "_sr"

    # 构建模型
    sr_cfg = cfg.get("sr", cfg.get("model", {}).get("sr", {}))
    model_sr = RefSRWKV(
        inp_channels=sr_cfg.get("inp_channels", 3),
        out_channels=sr_cfg.get("out_channels", 3),
        dim=sr_cfg.get("dim", 48),
        num_blocks=sr_cfg.get("num_blocks", [4, 6, 6, 8]),
        num_refinement_blocks=sr_cfg.get("num_refinement_blocks", 4),
        scale=sr_cfg.get("scale", 10),
    )

    # 可选加载预训练权重
    pretrained = cfg.get("pretrained", {})
    if pretrained.get("sr_ckpt"):
        load_weights(model_sr, pretrained["sr_ckpt"])

    # Lightning 模块
    loss_fn = nn.L1Loss() if stage_cfg.get("loss_fn", "l1") == "l1" else nn.MSELoss()
    lit_module = LitRefSRWKV(
        model_sr=model_sr,
        learning_rate=stage_cfg["learning_rate"],
        warmup_steps=stage_cfg["warmup_steps"],
        loss_fn=loss_fn,
    )

    trainer, full_ckpt_dir = _make_trainer(cfg, exp_name, checkpoint_dir, log_dir, stage_cfg["max_epochs"])

    resume_ckpt = None
    last_ckpt = os.path.join(full_ckpt_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        resume_ckpt = last_ckpt
        print(f"🔁 检测到上次训练记录，从 {last_ckpt} 恢复")

    trainer.fit(lit_module, train_loader, val_loader, ckpt_path=resume_ckpt)
    return trainer.checkpoint_callback.best_model_path


def train_stage_diff(cfg: dict, train_loader, val_loader):
    """阶段 2：训练 RefDiffRWKV 扩散模型。"""
    stage_cfg = cfg["train"]["diff"]
    checkpoint_dir = cfg["output"]["checkpoint_dir"]
    log_dir = cfg["output"]["log_dir"]
    exp_name = cfg["output"].get("experiment_name", "refrwkv") + "_diff"

    # 构建模型（含 GlobalSemanticModule）
    global_semantic = GlobalSemanticModule(
        base_dim=64,
        unet_dim=cfg.get("model", {}).get("embed_dim", 128),
        freeze_dinov2=True,
    )
    model_diff = build_model_diff(cfg, global_semantic)

    pretrained = cfg.get("pretrained", {})
    if pretrained.get("diff_ckpt"):
        load_weights(model_diff, pretrained["diff_ckpt"])

    lit_module = LitRefDiffRWKV(
        model_diff=model_diff,
        num_timesteps=stage_cfg.get("num_timesteps", 1000),
        cfg_drop_prob=stage_cfg.get("cfg_drop_prob", 0.1),
        learning_rate=stage_cfg["learning_rate"],
        weight_decay=stage_cfg.get("weight_decay", 0.01),
        beta1=stage_cfg.get("beta1", 0.9),
        beta2=stage_cfg.get("beta2", 0.999),
        warmup_epochs=stage_cfg["warmup_epochs"],
        scheduler=stage_cfg.get("scheduler", "cosine"),
        eta_min=stage_cfg["learning_rate"] * stage_cfg.get("eta_min_ratio", 0.01),
    )

    trainer, full_ckpt_dir = _make_trainer(cfg, exp_name, checkpoint_dir, log_dir, stage_cfg["max_epochs"])

    resume_ckpt = None
    last_ckpt = os.path.join(full_ckpt_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        resume_ckpt = last_ckpt
        print(f"🔁 检测到上次训练记录，从 {last_ckpt} 恢复")

    trainer.fit(lit_module, train_loader, val_loader, ckpt_path=resume_ckpt)

    return trainer.checkpoint_callback.best_model_path


def train_stage_enhance(cfg: dict, train_loader, val_loader):
    """阶段 3：训练 EnRWKV 增强模型（依赖冻结的扩散模型）。"""
    stage_cfg = cfg["train"]["enhance"]
    checkpoint_dir = cfg["output"]["checkpoint_dir"]
    log_dir = cfg["output"]["log_dir"]
    exp_name = cfg["output"].get("experiment_name", "refrwkv") + "_enhance"

    # 构建扩散模型（冻结用）
    global_semantic = GlobalSemanticModule(
        base_dim=64,
        unet_dim=cfg.get("model", {}).get("embed_dim", 128),
        freeze_dinov2=True,
    )
    model_diff = build_model_diff(cfg, global_semantic)

    # 必须加载已训练好的扩散权重
    diff_ckpt = cfg.get("pretrained", {}).get("diff_ckpt")
    if diff_ckpt is None:
        raise ValueError("Enhance 阶段必须提供 pretrained.diff_ckpt！")
    load_weights(model_diff, diff_ckpt)

    # 增强模型
    enh_cfg = cfg.get("enhance", cfg.get("model", {}).get("enhance", {}))
    model_enhance = EnRWKV(
        inp_channels=enh_cfg.get("inp_channels", 3),
        out_channels=enh_cfg.get("out_channels", 3),
        dim=enh_cfg.get("dim", 48),
        num_blocks=enh_cfg.get("num_blocks", [4, 6, 6, 8]),
        num_refinement_blocks=enh_cfg.get("num_refinement_blocks", 4),
    )

    pretrained = cfg.get("pretrained", {})
    if pretrained.get("enhance_ckpt"):
        load_weights(model_enhance, pretrained["enhance_ckpt"])

    lit_module = LitEnRWKV(
        model_enhance=model_enhance,
        model_diff=model_diff,
        num_timesteps=stage_cfg.get("num_timesteps", 1000),
        t_threshold=stage_cfg.get("t_threshold", 250),
        learning_rate=stage_cfg["learning_rate"],
        weight_decay=stage_cfg.get("weight_decay", 0.01),
        warmup_epochs=stage_cfg["warmup_epochs"],
    )

    trainer, full_ckpt_dir = _make_trainer(cfg, exp_name, checkpoint_dir, log_dir, stage_cfg["max_epochs"])

    resume_ckpt = None    
    last_ckpt = os.path.join(full_ckpt_dir, "last.ckpt")    
    if os.path.exists(last_ckpt):        
        resume_ckpt = last_ckpt        
        print(f"🔁 检测到上次训练记录，从 {last_ckpt} 恢复")

    trainer.fit(lit_module, train_loader, val_loader, ckpt_path=resume_ckpt)    
    
    return trainer.checkpoint_callback.best_model_path


def _make_trainer(cfg: dict, exp_name: str, checkpoint_dir: str, log_dir: str, max_epochs: int):
    train_cfg = cfg["train"]
    full_checkpoint_dir = os.path.join(checkpoint_dir, exp_name)
    os.makedirs(full_checkpoint_dir, exist_ok=True)

    logger = TensorBoardLogger(log_dir, name=exp_name)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=full_checkpoint_dir,
            filename="{epoch:04d}-{val_loss:.5f}",
            monitor="val_loss",
            save_top_k=3,
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=train_cfg.get("precision", "bf16-mixed"),
        max_epochs=max_epochs,
        log_every_n_steps=train_cfg.get("log_every_n_steps", 20),
        check_val_every_n_epoch=train_cfg.get("check_val_every_n_epoch", 1),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=train_cfg.get("accumulate_grad_batches", 1),
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
    )

    return trainer, full_checkpoint_dir


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage", type=str, default=None,
                        help="覆盖 YAML 中的 stage: sr | diff | enhance")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stage = args.stage or cfg.get("stage", "sr")

    # 复制配置文件到 checkpoint 目录
    output_cfg = cfg.get("output", {})
    checkpoint_dir = output_cfg.get("checkpoint_dir", "checkpoints")
    log_dir = output_cfg.get("log_dir", "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy2(args.config, os.path.join(checkpoint_dir, "train_config.yaml"))

    # 构建 DataLoader
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    print(f"\n{'='*50}")
    print(f"  Stage: {stage}")
    print(f"  Data root: {cfg['data']['root']}")
    print(f"  Batch size: {cfg['data']['batch_size']}")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"{'='*50}\n")

    # 分派到对应阶段
    if stage == "sr":
        best_ckpt = train_stage_sr(cfg, train_loader, val_loader)
    elif stage == "diff":
        best_ckpt = train_stage_diff(cfg, train_loader, val_loader)
    elif stage == "enhance":
        best_ckpt = train_stage_enhance(cfg, train_loader, val_loader)
    else:
        raise ValueError(f"Unknown stage: {stage}. 可选: sr | diff | enhance")

    print(f"\n✅ 训练完成！最佳模型: {best_ckpt}")


if __name__ == "__main__":
    import torch.nn as nn

    main()
