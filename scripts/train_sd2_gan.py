#!/usr/bin/env python
"""
SD2RefGANSystem 训练脚本
(G/D 分离：SD2RefGenerator + SD2RefDiscriminator)

用法:
    python scripts/train_sd2_gan.py --config configs/sd2_ref_gan_config.yaml
    python scripts/train_sd2_gan.py --config configs/sd2_ref_gan_config.yaml --resume checkpoints/.../last.ckpt
"""

import sys
import argparse
import yaml
import os
import shutil
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    Callback,
)
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision("high")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==================== 导入模型和数据集 ====================
from RefRWKV.models.RefDiffRWKV.sd2_ref_generator import SD2RefGenerator
from RefRWKV.models.RefDiffRWKV.sd2_ref_discriminator import SD2RefDiscriminator
from RefRWKV.models.RefDiffRWKV.sd2_ref_gan_system import SD2RefGANSystem
from RefRWKV.models.RefSRWKV import RefSRWKV
from RefRWKV.RefSR_data.RefSR_dataset import RefLMDBDataset


# ============================================================
# SR prior 构建
# ============================================================
def build_sr_model(cfg: dict):
    mc = cfg.get("model", {})
    if not mc.get("sr_enabled", False):
        return None

    sr_cfg = mc.get("sr", {})
    model = RefSRWKV(
        inp_channels=sr_cfg.get("inp_channels", 3),
        out_channels=sr_cfg.get("out_channels", 3),
        dim=sr_cfg.get("dim", 48),
        num_blocks=tuple(sr_cfg.get("num_blocks", [4, 6, 6, 8])),
        num_refinement_blocks=sr_cfg.get("num_refinement_blocks", 4),
        scale=sr_cfg.get("scale", 10),
        drop_path_rate=sr_cfg.get("drop_path_rate", 0.1),
        hidden_rate=sr_cfg.get("hidden_rate", 4),
    )

    ckpt_path = sr_cfg.get("ckpt_path")
    if ckpt_path is not None and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if isinstance(ckpt, dict):
            state_dict = {}
            for k, v in ckpt.items():
                k = k.replace("module.", "")
                if k.startswith("model."):
                    k = k[len("model."):]
                state_dict[k] = v
            ckpt = state_dict
        model.load_state_dict(ckpt, strict=False)
        print(f"✅ Loaded SR prior weights from {ckpt_path}")
    elif mc.get("sr_enabled", False):
        print("⚠️ sr_enabled=True but sr.ckpt_path is null; using randomly initialized SR prior")

    if mc.get("sr_fixed", True):
        model.eval()
        model.requires_grad_(False)

    return model


# ============================================================
# NaN 监测回调
# ============================================================
class NaNMonitorCallback(Callback):
    """在每个 backward 后检查所有参数梯度是否含 NaN / Inf。"""

    def on_after_backward(self, trainer, model):
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if torch.isnan(param.grad).any():
                print(f"[NaN grad] {name} | step={trainer.global_step}")
            if torch.isinf(param.grad).any():
                print(f"[Inf grad] {name} | step={trainer.global_step}")


# ============================================================
# 工具函数
# ============================================================
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_weights(model: torch.nn.Module, ckpt_path: str):
    """从 Lightning checkpoint 加载模型权重，自动剥离可能的前缀。"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    new_state_dict = {}
    for k, v in state_dict.items():
        for pre in ["model.", "model_sr.", "model_diff.", "model_enhance.", "generator.", "discriminator."]:
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


# ============================================================
# 数据加载
# ============================================================
def build_dataloaders(cfg: dict):
    """返回 (train_loader, val_loader, test_loader)。"""
    data_cfg = cfg["data"]
    mc = cfg.get("model", {})

    max_samples_tuple = (
        data_cfg.get("max_samples_train"),
        data_cfg.get("max_samples_val"),
        data_cfg.get("max_samples_test"),
    )

    dataset_kwargs = {
        "data_dir": data_cfg["root"],
        "patch_size": data_cfg.get("patch_size", 480),
        "scale": data_cfg.get("scale", 10),
        "ref_aug_strengths": data_cfg.get(
            "ref_aug_strengths", [0.12, 0.12, 0.12, 0.03]
        ),
        "ref_aug_probs": data_cfg.get("ref_aug_probs", [0.5, 0.5, 0.5, 0.5]),
        "ref_gray_prob": data_cfg.get("ref_gray_prob", 0.2),
        "max_samples": max_samples_tuple,
        "sample_seed": 42,
        "lr_key": mc.get("lr_key", "lr"),
        "hr_key": mc.get("hr_key", "hr"),
        "ref_key": mc.get("ref_key", "ref"),
    }

    train_ds = RefLMDBDataset(
        mode="train",
        augment=data_cfg.get("augment", True),
        augment_ref=data_cfg.get("augment_ref", False),
        **dataset_kwargs,
    )

    val_ds = RefLMDBDataset(
        mode="val", augment=False, augment_ref=False, **dataset_kwargs
    )

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
        persistent_workers=(data_cfg["num_workers"] > 0),
        prefetch_factor=data_cfg.get("prefetch_factor", 4) if data_cfg["num_workers"] > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.get("val_batch_size", 1),
        shuffle=False,
        num_workers=data_cfg.get("val_num_workers", 2),
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=data_cfg.get("val_batch_size", 1),
        shuffle=False,
        num_workers=data_cfg.get("val_num_workers", 2),
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ============================================================
# 模型构建
# ============================================================
def build_model(cfg: dict):
    """构建 Generator + Discriminator + GAN System。"""
    mc = cfg.get("model", {})

    # SR prior（推理 Better Start 用，训练时固定）
    sr_model = build_sr_model(cfg)

    # Generator
    generator = SD2RefGenerator(
        # Data keys
        lr_key=mc.get("lr_key", "lr"),
        ref_key=mc.get("ref_key", "ref"),
        hr_key=mc.get("hr_key", "hr"),
        # Adapter
        strategy=mc.get("strategy", "rwkv"),
        rwkv_cfg=mc.get("rwkv_cfg", {"patch_size": 4, "embed_dim": 192}),
        # SD2
        sd_model_path=mc["sd_model_path"],
        use_lora=mc.get("use_lora", True),
        lora_rank=mc.get("lora_rank", 64),
        lora_target_modules=mc.get("lora_target_modules", None),
        sd_locked=mc.get("sd_locked", True),
        # DINOv2
        use_semantic=mc.get("use_semantic", True),
        dinov2_model_name=mc.get("dinov2_model_name", "facebook/dinov2-base"),
        # Diffusion
        num_train_timesteps=mc.get("num_train_timesteps", 1000),
        beta_start=mc.get("beta_start", 0.00085),
        beta_end=mc.get("beta_end", 0.012),
        beta_schedule=mc.get("beta_schedule", "scaled_linear"),
        prediction_type=mc.get("prediction_type", "epsilon"),
        t_min=mc.get("t_min", 300),
        t_max=mc.get("t_max", 700),
        cfg_drop_prob=mc.get("cfg_drop_prob", 0.1),
        control_scale=mc.get("control_scale", 1.0),
        # Optimizer (for standalone usage)
        learning_rate=mc.get("learning_rate", 1e-4),
        weight_decay=mc.get("weight_decay", 1e-3),
    )

    # Discriminator（可选，设为 None 则退化为纯 diffusion baseline）
    discriminator = None
    if mc.get("use_discriminator", True):
        discriminator = SD2RefDiscriminator(
            use_semantic_d=mc.get("use_semantic_d", True),
            use_texture_d=mc.get("use_texture_d", True),
            semantic_alpha=mc.get("semantic_alpha", 0.8),
            semantic_use_freq=mc.get("semantic_use_freq", True),
            semantic_trainable_stages=mc.get("semantic_trainable_stages", 1),
            semantic_precision=mc.get("semantic_precision", "fp32"),
            texture_base_ch=mc.get("texture_base_ch", 48),
            texture_num_scales=mc.get("texture_num_scales", 4),
            texture_use_spectral=mc.get("texture_use_spectral", True),
            lr_semantic=mc.get("lr_D", 5e-6),
            lr_texture=mc.get("lr_D_texture", 1e-6),
            weight_decay=mc.get("d_weight_decay", 1e-3),
            betas=mc.get("d_betas", [0.5, 0.999]),
        )

    # GAN System
    system = SD2RefGANSystem(
        generator=generator,
        discriminator=discriminator,
        # Loss weights
        lambda_gan_semantic=mc.get("lambda_gan", 0.0),
        lambda_gan_texture=mc.get("lambda_gan_texture", 0.0),
        lambda_lpips=mc.get("lambda_lpips", 0.0),
        # Training control
        accumulate_grad_batches=mc.get("accumulate_grad_batches", 8),
        use_amp=mc.get("use_amp", True),
        # Optimizers
        g_lr=mc.get("learning_rate", 1e-4),
        g_weight_decay=mc.get("weight_decay", 1e-3),
        d_lr_sem=mc.get("lr_D", 5e-6),
        d_lr_tex=mc.get("lr_D_texture", 1e-6),
        d_weight_decay=mc.get("d_weight_decay", 1e-3),
        betas=mc.get("d_betas", [0.5, 0.999]),
        # Validation
        sample_steps=mc.get("sample_steps", 50),
        fr_metrics=mc.get("fr_metrics", ["psnr", "ssim", "lpips", "dists"]),
        # Better Start SR model
        sr_model=sr_model,
    )

    return system


# ============================================================
# Trainer 构建
# ============================================================
def build_trainer(
    cfg: dict, exp_name: str, checkpoint_dir: str, log_dir: str, max_epochs: int
):
    """构建 PyTorch Lightning Trainer。"""
    train_cfg = cfg.get("train", {})
    full_checkpoint_dir = os.path.join(checkpoint_dir, exp_name)
    os.makedirs(full_checkpoint_dir, exist_ok=True)

    logger = TensorBoardLogger(log_dir, name=exp_name)

    callbacks = [
        NaNMonitorCallback(),
        EarlyStopping(
            monitor="val/loss_diff",
            patience=train_cfg.get("early_stopping_patience", 30),
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=full_checkpoint_dir,
            filename="{epoch:04d}-val_loss={val/loss_diff:.6f}",
            monitor="val/loss_diff",
            save_top_k=train_cfg.get("save_top_k", 3),
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=train_cfg.get("devices", 1),
        precision=train_cfg.get("precision", "32"),
        max_epochs=max_epochs,
        log_every_n_steps=train_cfg.get("log_every_n_steps", 20),
        val_check_interval=train_cfg.get("val_check_interval", 0.5),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
    )

    return trainer, full_checkpoint_dir


# ============================================================
# 训练函数
# ============================================================
def train(cfg: dict, resume_ckpt: str = None):
    """训练 SD2RefGANSystem。"""
    train_cfg = cfg["train"]
    output_cfg = cfg.get("output", {})
    mc = cfg.get("model", {})

    if train_cfg.get("detect_anomaly", False):
        torch.autograd.set_detect_anomaly(True)
        print("⚠️  autograd 异常检测已开启，训练速度会降低")

    checkpoint_dir = output_cfg.get("checkpoint_dir", "checkpoints/sd2_ref_gan")
    log_dir = output_cfg.get("log_dir", "logs/sd2_ref_gan")
    exp_name = output_cfg.get("experiment_name", "sd2_ref_gan")

    # 构建 system
    system = build_model(cfg)

    # 可选加载预训练权重
    pretrained = cfg.get("pretrained", {})
    if pretrained.get("sd2_control_ckpt") and not resume_ckpt:
        print(f"🔁 从 {pretrained['sd2_control_ckpt']} 加载预训练权重")
        load_weights(system, pretrained["sd2_control_ckpt"])

    # 数据
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # Trainer
    max_epochs = train_cfg.get("max_epochs", 100)
    trainer, full_ckpt_dir = build_trainer(
        cfg, exp_name, checkpoint_dir, log_dir, max_epochs
    )

    # 断点续训
    if resume_ckpt is None:
        last_ckpt = os.path.join(full_ckpt_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            resume_ckpt = last_ckpt
            print(f"🔁 检测到上次训练记录，从 {last_ckpt} 恢复")

    # 打印训练摘要
    print(f"{'='*60}")
    print(f"  SD2RefGANSystem 训练")
    print(f"  数据根目录    : {cfg['data']['root']}")
    print(f"  Batch size    : {cfg['data']['batch_size']} x accumulate {mc.get('accumulate_grad_batches', 8)}")
    print(f"  训练样本数    : {len(train_loader.dataset)}")
    print(f"  验证样本数    : {len(val_loader.dataset)}")
    print(f"  测试样本数    : {len(test_loader.dataset)}")
    print(f"  Strategy      : {mc.get('strategy', 'rwkv')}")
    print(f"  RWKN embed_dim: {mc.get('rwkv_cfg', {}).get('embed_dim', 192)}")
    print(f"  使用 Discriminator : {mc.get('use_discriminator', True)}")
    print(f"  GAN λ (sem/tex) : {mc.get('lambda_gan', 0.0)} / {mc.get('lambda_gan_texture', 0.0)}")
    print(f"  LPIPS λ       : {mc.get('lambda_lpips', 0.0)}")
    print(f"  LR (G/D_sem/D_tex): {mc.get('learning_rate', 1e-4)} / {mc.get('lr_D', 5e-6)} / {mc.get('lr_D_texture', 1e-6)}")
    print(f"  最大 epoch    : {max_epochs}")
    print(f"  CFG dropout   : {mc.get('cfg_drop_prob', 0.1)}")
    print(f"  Semantic      : {mc.get('use_semantic', True)}")
    print(f"  AMP           : {mc.get('use_amp', True)}")
    print(f"{'='*60}")

    trainer.fit(system, train_loader, val_loader, ckpt_path=resume_ckpt)

    return (
        trainer.checkpoint_callback.best_model_path,
        trainer.checkpoint_callback.best_model_score,
    )


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SD2RefGANSystem 训练")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument(
        "--resume", type=str, default=None, help="恢复训练的 checkpoint 路径"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 复制配置文件
    output_cfg = cfg.get("output", {})
    checkpoint_dir = output_cfg.get("checkpoint_dir", "checkpoints")
    log_dir = output_cfg.get("log_dir", "logs")
    experiment_name = output_cfg.get("experiment_name", "")

    if experiment_name:
        checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy2(args.config, os.path.join(checkpoint_dir, "train_config.yaml"))

    best_ckpt, best_score = train(cfg, resume_ckpt=args.resume)

    print(f"{'='*60}")
    print(f"  ✅ 训练完成")
    print(f"  最佳模型 : {best_ckpt}")
    print(f"  最佳 val/loss_diff : {best_score:.6f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
