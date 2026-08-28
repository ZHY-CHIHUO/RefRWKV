#!/usr/bin/env python
"""
RefSRWKV SR Prior 训练脚本

用法:
    # 从头训练
    python scripts/train_sr_prior.py --config configs/sr_prior.yaml

    # 断点续训
    python scripts/train_sr_prior.py --config configs/sr_prior.yaml \
        --resume checkpoints/refrwkv_sr/last.ckpt

    # 热启动（仅加载形状匹配的权重，优化器从头开始）
    python scripts/train_sr_prior.py --config configs/sr_prior.yaml \
        --load_weights checkpoints/refrwkv_sr/best.ckpt

    # 后台运行
    nohup python scripts/train_sr_prior.py --config configs/sr_prior.yaml \
        > train.log 2>&1 &

last.ckpt 自动续训逻辑（入口只有 yaml 的 output.checkpoint_dir）:
    启动时自动检查 checkpoint_dir/last.ckpt:
      - 与当前模型结构完全一致 → 自动断点续训
      - 结构不匹配（如 ×10 权重用于 ×4 训练）→ 备份为
        last.ckpt.incompat_backup，从头训练并警告（不崩溃、不误加载）
    显式 --resume 时: 结构不匹配直接报错退出（明确意图，不静默降级）

用法示例（换 scale/数据集时只需在 yaml 中更换 output 三项即可隔离实验）:
    output:
      checkpoint_dir: "checkpoints/refrwkv_sr_x4_hrms"
      log_dir: "logs/refrwkv_sr_x4_hrms"
      experiment_name: "refrwkv_sr_x4_hrms"
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
    Callback,
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
# Checkpoint 兼容性校验
#   解决: 换 scale/数据集后，output.checkpoint_dir 下残留的旧
#   last.ckpt 被自动当作续训入口，形状不匹配直接崩溃。
#   现在: 自动检测到不兼容 → 备份 + 从头训练；显式指定 → 报错退出。
# ═══════════════════════════════════════════════════════════════
_CKPT_PREFIXES = (
    "generator.sr_model.", "sr_model.", "model_sr.", "model.", "module."
)


def _strip_prefix(key: str) -> str:
    for pre in _CKPT_PREFIXES:
        if key.startswith(pre):
            return key[len(pre):]
    return key


def _load_state_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    return {k: v for k, v in sd.items() if torch.is_tensor(v)}


def check_resume_compatible(ckpt_path: str, lit_model):
    """结构级校验：同名键全部存在且形状一致，才允许断点续训。

    返回 (ok, reason)。
    """
    try:
        sd = {_strip_prefix(k): v for k, v in _load_state_dict(ckpt_path).items()}
    except Exception as e:
        return False, f"checkpoint 读取失败: {e}"

    ref = {k: v for k, v in lit_model.state_dict().items() if torch.is_tensor(v)}

    overlap = [k for k in sd if k in ref]
    if not overlap:
        return False, "checkpoint 与当前模型没有任何同名参数（可能来自完全不同的模型）"

    mismatched = [
        (k, tuple(sd[k].shape), tuple(ref[k].shape))
        for k in overlap
        if sd[k].shape != ref[k].shape
    ]
    if mismatched:
        detail = "; ".join(
            f"{k}: ckpt{c} vs 现模型{r}" for k, c, r in mismatched[:3]
        )
        return False, (
            f"{len(mismatched)} 个参数形状不匹配"
            f"（典型原因: scale/架构不同的旧权重，如 ×10 用于 ×4）: {detail}"
        )

    missing = [k for k in ref if k not in sd]
    if missing:
        return False, (
            f"当前模型有 {len(missing)} 个参数在 checkpoint 中缺失: {missing[:3]}"
        )

    extra = [k for k in sd if k not in ref]
    if extra:
        return False, f"checkpoint 有 {len(extra)} 个多余参数: {extra[:3]}"

    return True, f"结构一致（{len(overlap)} 个张量全部匹配）"


def load_weights_filtered(lit_model, ckpt_path: str):
    """热启动：仅加载形状匹配的参数，形状不匹配/多余的键跳过（保持随机初始化）。

    strict=False 容忍键缺失/多余但不容忍形状不同，因此必须先按形状过滤。
    """
    sd = _load_state_dict(ckpt_path)
    ref = {k: v for k, v in lit_model.state_dict().items() if torch.is_tensor(v)}

    matched, skipped = {}, []
    for k, v in sd.items():
        kk = _strip_prefix(k)
        if kk not in ref:
            continue
        if tuple(v.shape) == tuple(ref[kk].shape):
            matched[kk] = v
        else:
            skipped.append(k)

    missing, _unexpected = lit_model.load_state_dict(matched, strict=False)
    logger.info(
        "热启动 %s: 匹配 %d/%d | 形状不匹配跳过 %d | 缺失 %d",
        ckpt_path, len(matched), len(ref), len(skipped), len(missing),
    )
    if skipped:
        logger.warning("  形状不匹配（保持随机初始化）示例: %s", skipped[:5])
    if len(missing) / max(len(ref), 1) > 0.5:
        logger.warning("  缺失参数超过 50%%，请确认 checkpoint 与当前任务是否匹配")


class ForceSaveLastCallback(Callback):
    """每个 epoch 结束强制保存 last.ckpt（移植自 train_sd2_gan.py）。

    save_last=True 只在 top-k checkpoint 更新时顺带写 last.ckpt；
    val_loss 长期不改进时会长时间零落盘。本回调保证每 epoch 至少落盘一次，
    中断最多丢失一个 epoch。
    """

    def on_train_epoch_end(self, trainer, pl_module):
        cb = trainer.checkpoint_callback
        if cb is None or cb.dirpath is None:
            return
        target = os.path.join(cb.dirpath, "last.ckpt")
        try:
            trainer.save_checkpoint(target)
            logger.info(
                "ForceSaveLastCallback: epoch %d 强制保存 -> %s",
                trainer.current_epoch, target,
            )
        except Exception as e:
            logger.warning("ForceSaveLastCallback 保存失败: %s", e)


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="RefSRWKV SR Prior 训练")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--load_weights", type=str, default=None,
                        help="仅加载形状匹配的模型权重（热启动），不恢复优化器状态")
    parser.add_argument("--resume", type=str, default=None,
                        help="断点续训（要求结构一致，不匹配时报错退出）")
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

    # 输出目录（全部由 yaml 的 output 段控制）
    ckpt_dir = oc.get("checkpoint_dir", "checkpoints/refrwkv_sr")
    log_dir = oc.get("log_dir", "logs/refrwkv_sr")
    exp_name = oc.get("experiment_name", "refrwkv_sr")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 保存完整配置副本（自包含，便于复现与审计）
    with open(os.path.join(ckpt_dir, "train_config.yaml"), "w",
              encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

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
        ForceSaveLastCallback(),
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

    # ═══════════════════════════════════════════════════════════
    # 恢复决策（入口只有 yaml 的 output.checkpoint_dir）
    #   优先级: --resume > --load_weights > 自动 last.ckpt > 从头
    # ═══════════════════════════════════════════════════════════
    resume_ckpt = None

    if args.load_weights:
        # 热启动：优先于自动续训，不自动检测 last.ckpt
        load_weights_filtered(lit_model, args.load_weights)
    else:
        candidate, source = args.resume, "显式 --resume"
        if candidate is None:
            last_ckpt = os.path.join(ckpt_dir, "last.ckpt")  # ← 只看 yaml 指定的目录
            if os.path.exists(last_ckpt):
                candidate, source = last_ckpt, "自动检测 last.ckpt"

        if candidate is not None:
            ok, reason = check_resume_compatible(candidate, lit_model)
            if ok:
                resume_ckpt = candidate
                logger.info("断点续训 (%s): %s（%s）", source, candidate, reason)
            elif args.resume:
                # 用户明确指定 → 不兼容直接退出，绝不静默降级
                logger.error(
                    "--resume 的 checkpoint 与当前模型不兼容，拒绝加载: %s\n"
                    "  原因: %s\n"
                    "  处理: ① 热启动改用 --load_weights（自动跳过形状不匹配层）\n"
                    "        ② 从头训练去掉 --resume\n"
                    "        ③ 检查是否误用了其他 scale/数据集实验的 checkpoint",
                    candidate, reason,
                )
                raise SystemExit(1)
            else:
                # 自动检测到但不兼容（典型: 换 scale/数据集后没换 output 目录）
                logger.warning(
                    "%s 与当前模型不兼容，已忽略，将从头训练: %s\n  原因: %s",
                    source, candidate, reason,
                )
                backup = candidate + ".incompat_backup"
                try:
                    if not os.path.exists(backup):
                        os.rename(candidate, backup)
                        logger.warning("  已备份为: %s（避免被新训练覆盖）", backup)
                    else:
                        logger.warning("  备份已存在: %s，原文件将被新训练覆盖", backup)
                except OSError as e:
                    logger.warning("  备份失败（%s），原文件将被新训练覆盖", e)
                logger.warning(
                    "  建议: 在 yaml 中更换 output.checkpoint_dir 以隔离实验；"
                    "如需利用旧权重请用 --load_weights"
                )
                resume_ckpt = None

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
    if args.load_weights:
        logger.info("  权重来源: 热启动 %s（形状过滤）", args.load_weights)
    else:
        logger.info("  恢复: %s", resume_ckpt or "无（从头训练）")
    logger.info("=" * 60)

    # 训练（resume_ckpt 为 None 即从头训练）
    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=resume_ckpt)

    logger.info("训练完成！最佳模型: %s", trainer.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
