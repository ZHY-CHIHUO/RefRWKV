#!/usr/bin/env python
"""
SD2ControlLDM 训练脚本（RefDiffRWKV + SD2 UNet 端到端扩散超分）

用法:
    python scripts/train_sd2_control.py --config configs/sd2_control_config.yaml
    python scripts/train_sd2_control.py --config configs/sd2_control_config.yaml --resume checkpoints/.../last.ckpt

注意：
    - 手动优化模式下 G/D 交替训练使用模型内置 AMP scaler (use_amp)。
    - Trainer precision 设为 "32"，避免与内置 scaler 双重混合精度冲突。
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
    Callback,
)
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision("high")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==================== 导入模型和数据集 ====================
from RefRWKV.models.RefDiffRWKV.sd2_control_ldm import SD2ControlLDM
from RefRWKV.models.RefSRWKV import RefSRWKV
from RefRWKV.RefSR_data.RefSR_dataset import RefLMDBDataset


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
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if isinstance(ckpt, dict):
            state_dict = {}
            for k, v in ckpt.items():
                k = k.replace("module.", "")
                if k.startswith("model."):
                    k = k[len("model.") :]
                state_dict[k] = v
            ckpt = state_dict
        model.load_state_dict(ckpt, strict=False)
        print(f"✅ Loaded SR prior weights from {ckpt_path}")
    elif mc.get("sr_enabled", False):
        print(
            "⚠️ sr_enabled=True but sr.ckpt_path is null; using randomly initialized SR prior"
        )

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
        for pre in ["model.", "model_sr.", "model_diff.", "model_enhance."]:
            if k.startswith(pre):
                k = k[len(pre) :]
                break
        new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"⚠️  缺失键 ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"⚠️  多余键 ({len(unexpected)}): {unexpected[:5]}...")
    print(f"✅ 从 {ckpt_path} 加载权重完成")


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
        persistent_workers=(data_cfg["num_workers"] > 0),
        prefetch_factor=data_cfg.get("prefetch_factor", 4),
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


def build_model(cfg: dict) -> SD2ControlLDM:
    """根据配置构建 SD2ControlLDM。

    所有可调超参均从配置文件读取，方便实验管理。
    """
    mc = cfg.get("model", {})

    sr_model = build_sr_model(cfg)

    return SD2ControlLDM(
        # Data keys
        lr_key=mc.get("lr_key", "lr"),
        ref_key=mc.get("ref_key", "ref"),
        hr_key=mc.get("hr_key", "hr"),
        # SD2
        sd_model_path=mc["sd_model_path"],
        use_lora=mc.get("use_lora", True),
        lora_rank=mc.get("lora_rank", 64),
        lora_target_modules=mc.get("lora_target_modules", None),
        sd_locked=mc.get("sd_locked", True),
        # SR prior
        sr_model=sr_model,
        sr_fixed=mc.get("sr_fixed", True),
        # RefDiffRWKV
        patch_size=mc.get("patch_size", 4),
        embed_dim=mc.get("embed_dim", 384),
        upsample_mode=mc.get("upsample_mode", "bilinear"),
        # GlobalSemantic
        use_semantic=mc.get("use_semantic", False),
        dinov2_model_name=mc.get("dinov2_model_name", "facebook/dinov2-base"),
        # Diffusion
        cfg_drop_prob=mc.get("cfg_drop_prob", 0.1),
        num_train_timesteps=mc.get("num_train_timesteps", 1000),
        beta_start=mc.get("beta_start", 0.00085),
        beta_end=mc.get("beta_end", 0.012),
        beta_schedule=mc.get("beta_schedule", "scaled_linear"),
        prediction_type=mc.get("prediction_type", "epsilon"),
        model_t=mc.get("model_t", 200),
        # Loss weights & GAN
        learning_rate=mc.get("learning_rate", 1e-4),
        lr_D=mc.get("lr_D", 1e-4),
        lr_D_texture=mc.get("lr_D_texture", 1e-4),  # ← 新增
        lambda_gan_texture=mc.get("lambda_gan_texture", 0.5),  # ← 新增
        disc_trainable_stages=mc.get("disc_trainable_stages", 1),  # ← 新增
        l_simple_weight=mc.get("l_simple_weight", 1.0),
        lambda_lpips=mc.get("lambda_lpips", 0.1),
        lambda_gan=mc.get("lambda_gan", 0.005),
        use_freq=mc.get("use_freq", True),
        weight_decay=mc.get("weight_decay", 1e-3),
        # AMP（手动优化模式下独立于 Trainer precision）
        use_amp=mc.get("use_amp", True),
        # Validation
        sample_steps=mc.get("sample_steps", 50),
        fr_metrics=mc.get("fr_metrics", None),
        iqa_device=mc.get("iqa_device", "cuda"),
        # Debug
        debug_nan=mc.get("debug_nan", True),
        accumulate_grad_batches=mc.get("accumulate_grad_batches", 8),
    )


def build_trainer(
    cfg: dict, exp_name: str, checkpoint_dir: str, log_dir: str, max_epochs: int
):
    """构建 PyTorch Lightning Trainer。

    注意：手动优化 + 内置 AMP scaler 时 trainer precision 应设为 "32"，
    避免 Lightning 注入额外的 autocast 导致双重混合精度。
    """
    train_cfg = cfg.get("train", {})
    full_checkpoint_dir = os.path.join(checkpoint_dir, exp_name)
    os.makedirs(full_checkpoint_dir, exist_ok=True)

    logger = TensorBoardLogger(log_dir, name=exp_name)

    callbacks = [
        NaNMonitorCallback(),
        EarlyStopping(
            monitor="val/loss",
            patience=train_cfg.get("early_stopping_patience", 30),
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=full_checkpoint_dir,
            filename="{epoch:04d}-val_loss={val_loss:.6f}",
            monitor="val/loss",
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
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
    )

    return trainer, full_checkpoint_dir


# ============================================================
# 训练函数
# ============================================================
def train(cfg: dict, resume_ckpt: str = None):
    """训练 SD2ControlLDM。"""
    train_cfg = cfg["train"]
    output_cfg = cfg.get("output", {})
    mc = cfg.get("model", {})

    # ── 开启自动异常检测（调试用，训练慢 10-20%）──
    if train_cfg.get("detect_anomaly", False):
        torch.autograd.set_detect_anomaly(True)
        print("⚠️  autograd 异常检测已开启，训练速度会降低")

    checkpoint_dir = output_cfg.get("checkpoint_dir", "checkpoints/sd2_control")
    log_dir = output_cfg.get("log_dir", "logs/sd2_control")
    exp_name = output_cfg.get("experiment_name", "sd2_control")

    # 构建模型
    model = build_model(cfg)

    # 可选加载预训练权重
    pretrained = cfg.get("pretrained", {})
    if pretrained.get("sd2_control_ckpt") and not resume_ckpt:
        load_weights(model, pretrained["sd2_control_ckpt"])

    # 构建数据
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # 构建 Trainer
    max_epochs = train_cfg.get("max_epochs", 100)
    trainer, full_ckpt_dir = build_trainer(
        cfg, exp_name, checkpoint_dir, log_dir, max_epochs
    )

    # 自动检测断点续训
    if resume_ckpt is None:
        last_ckpt = os.path.join(full_ckpt_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            resume_ckpt = last_ckpt
            print(f"🔁 检测到上次训练记录，从 {last_ckpt} 恢复")

    # ── 打印训练摘要 ──
    print(f"{'='*60}")
    print(f"  SD2ControlLDM 训练")
    print(f"  数据根目录    : {cfg['data']['root']}")
    print(
        f"  Batch size    : {cfg['data']['batch_size']} × "
        f"accumulate {mc.get('accumulate_grad_batches', 8)}"
    )
    print(f"  训练样本数    : {len(train_loader.dataset)}")
    print(f"  验证样本数    : {len(val_loader.dataset)}")
    print(f"  测试样本数    : {len(test_loader.dataset)}")
    print(f"  Trainer prec  : {train_cfg.get('precision', '32')}")
    print(f"  手动 AMP      : {mc.get('use_amp', True)}")
    print(f"  最大 epoch    : {max_epochs}")
    print(f"  LoRA rank     : {mc.get('lora_rank', 64)}")
    print(
        f"  LR (G / D_sem / D_tex) : {mc.get('learning_rate', 1e-4)} / {mc.get('lr_D', 1e-4)} / {mc.get('lr_D_texture', 1e-4)}"
    )
    print(f"  LPIPS λ       : {mc.get('lambda_lpips', 0.1)}")
    print(
        f"  GAN λ (sem / tex) : {mc.get('lambda_gan', 0.005)} / {mc.get('lambda_gan_texture', 0.5)}"
    )
    print(f"  Freq D        : {mc.get('use_freq', True)}")
    print(f"  D stages      : {mc.get('disc_trainable_stages', 1)}")
    print(f"  model_t       : {mc.get('model_t', 200)}")
    print(f"  CFG dropout   : {mc.get('cfg_drop_prob', 0.1)}")
    print(f"  Semantic      : {mc.get('use_semantic', False)}")
    print(f"  NaN debug     : {mc.get('debug_nan', True)}")
    print(f"{'='*60}")

    # 开始训练
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)

    return (
        trainer.checkpoint_callback.best_model_path,
        trainer.checkpoint_callback.best_model_score,
    )


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SD2ControlLDM 训练")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument(
        "--resume", type=str, default=None, help="恢复训练的 checkpoint 路径"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 复制配置文件到 checkpoint 目录
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
    print(f"  最佳 val/loss : {best_score:.6f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
