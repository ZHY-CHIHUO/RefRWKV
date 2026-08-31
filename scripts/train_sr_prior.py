#!/usr/bin/env python
"""
RefSRWKV SR Prior 训练脚本
"""
import argparse
import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import yaml

from RefRWKV.models.RefSRWKV import RefSRWKV, LitRefSRWKV
from RefRWKV.RefSR_data.RefDataset import RefPNGDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("train_sr_prior")

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_dataloaders(cfg):
    dc = cfg["data"]
    common = dict(
        data_dir=dc["root"], patch_size=dc.get("patch_size", 480), scale=dc.get("scale", 10),
        ref_aug_strengths=dc.get("ref_aug_strengths", [0.12, 0.12, 0.12, 0.03]),
        ref_aug_probs=dc.get("ref_aug_probs", [0.5, 0.5, 0.5, 0.5]),
        ref_gray_prob=dc.get("ref_gray_prob", 0.2),
        max_samples=(dc.get("max_samples_train"), dc.get("max_samples_val"), dc.get("max_samples_test")),
        sample_seed=42,
    )
    train_ds = RefPNGDataset(mode="train", augment=dc.get("augment", True), augment_ref=dc.get("augment_ref", True), **common)
    val_ds = RefPNGDataset(mode="val", augment=False, augment_ref=False, **common)
    pin, nw = dc.get("pin_memory", False), dc.get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=dc.get("batch_size", 4), shuffle=True, num_workers=nw, pin_memory=pin, drop_last=True, persistent_workers=nw > 0, prefetch_factor=dc.get("prefetch_factor", 4) if nw > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=dc.get("val_batch_size", 1), shuffle=False, num_workers=dc.get("val_num_workers", 2), pin_memory=pin)
    return train_loader, val_loader

def build_model(cfg):
    mc = cfg["model"]
    dc = cfg["data"]
    
    # ★ 自动获取数据裁剪尺寸作为 hr_size，网络内部会自动推导 internal_size
    hr_size = dc.get("patch_size", 480)
    
    model = RefSRWKV(
        inp_channels=mc.get("inp_channels", 3), out_channels=mc.get("out_channels", 3),
        dim=mc.get("dim", 48), num_blocks=tuple(mc.get("num_blocks", [4, 6, 6, 8])),
        num_refinement_blocks=mc.get("num_refinement_blocks", 4), scale=mc.get("scale", 4),
        hr_size=hr_size, drop_path_rate=mc.get("drop_path_rate", 0.1), hidden_rate=mc.get("hidden_rate", 4),
    )
    logger.info("RefSRWKV 参数量: %.2fM (HR=%d, Internal=%d)", sum(p.numel() for p in model.parameters()) / 1e6, hr_size, model.internal_size)
    
    return LitRefSRWKV(
        model_sr=model, learning_rate=mc.get("learning_rate", 1e-4), warmup_steps=mc.get("warmup_steps", 500),
        grad_clip_norm=mc.get("grad_clip_norm", 1.0), ema_decay=mc.get("ema_decay", 0.999), use_ema=mc.get("use_ema", True),
        ssim_weight=mc.get("ssim_weight", 0.0), fft_weight=mc.get("fft_weight", 0.0), ref_drop_prob=mc.get("ref_drop_prob", 0.0),
    )

_CKPT_PREFIXES = ("generator.sr_model.", "sr_model.", "model_sr.", "model.", "module.")
def _strip_prefix(key: str) -> str:
    for pre in _CKPT_PREFIXES:
        if key.startswith(pre): return key[len(pre):]
    return key

def _load_state_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    return {k: v for k, v in sd.items() if torch.is_tensor(v)}

def check_resume_compatible(ckpt_path: str, lit_model):
    try: sd = {_strip_prefix(k): v for k, v in _load_state_dict(ckpt_path).items()}
    except Exception as e: return False, f"checkpoint 读取失败: {e}"
    ref = {_strip_prefix(k): v for k, v in lit_model.state_dict().items() if torch.is_tensor(v)}
    overlap = [k for k in sd if k in ref]
    if not overlap: return False, "checkpoint 与当前模型没有任何同名参数"
    mismatched = [(k, tuple(sd[k].shape), tuple(ref[k].shape)) for k in overlap if sd[k].shape != ref[k].shape]
    if mismatched: return False, f"{len(mismatched)} 个参数形状不匹配: {mismatched[0][0]}: ckpt{mismatched[0][1]} vs 现模型{mismatched[0][2]}"
    missing = [k for k in ref if k not in sd]
    if missing: return False, f"当前模型有 {len(missing)} 个参数在 checkpoint 中缺失: {missing[:3]}"
    extra = [k for k in sd if k not in ref]
    if extra: return False, f"checkpoint 有 {len(extra)} 个多余参数: {extra[:3]}"
    return True, f"结构一致（{len(overlap)} 个张量全部匹配）"

def load_weights_filtered(lit_model, ckpt_path: str):
    sd = _load_state_dict(ckpt_path)
    ref = {k: v for k, v in lit_model.state_dict().items() if torch.is_tensor(v)}
    ref_by_norm = {_strip_prefix(k): k for k in ref}
    matched, skipped = {}, []
    for k, v in sd.items():
        full_key = ref_by_norm.get(_strip_prefix(k))
        if full_key is None: continue
        if tuple(v.shape) == tuple(ref[full_key].shape): matched[full_key] = v
        else: skipped.append(k)
    missing, _unexpected = lit_model.load_state_dict(matched, strict=False)
    logger.info("热启动 %s: 匹配 %d/%d | 形状不匹配跳过 %d | 缺失 %d", ckpt_path, len(matched), len(ref), len(skipped), len(missing))
    if skipped: logger.warning("  形状不匹配（保持随机初始化）示例: %s", skipped[:5])
    if len(matched) == 0: raise SystemExit(f"热启动失败: checkpoint 没有匹配到任何参数（{ckpt_path}）")

class ForceSaveLastCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        cb = trainer.checkpoint_callback
        if cb is None or cb.dirpath is None: return
        target = os.path.join(cb.dirpath, "last.ckpt")
        try:
            trainer.save_checkpoint(target)
            logger.info("ForceSaveLastCallback: epoch %d 强制保存 -> %s", trainer.current_epoch, target)
        except Exception as e: logger.warning("ForceSaveLastCallback 保存失败: %s", e)

def main():
    parser = argparse.ArgumentParser(description="RefSRWKV SR Prior 训练")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--load_weights", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tc, oc = cfg.get("train", {}), cfg.get("output", {})
    pl.seed_everything(tc.get("seed", 42), workers=True)

    train_loader, val_loader = build_dataloaders(cfg)
    logger.info("训练样本: %d, 验证样本: %d", len(train_loader.dataset), len(val_loader.dataset))
    lit_model = build_model(cfg)

    ckpt_dir = oc.get("checkpoint_dir", "checkpoints/refrwkv_sr")
    log_dir = oc.get("log_dir", "logs/refrwkv_sr")
    exp_name = oc.get("experiment_name", "refrwkv_sr")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "train_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    callbacks = [
        ModelCheckpoint(dirpath=ckpt_dir, filename="{epoch:04d}-{val_loss:.6f}", monitor="val_loss", save_top_k=tc.get("save_top_k", 3), mode="min", save_last=True),
        EarlyStopping(monitor="val_loss", patience=tc.get("early_stopping_patience", 30), mode="min", verbose=True),
        LearningRateMonitor(logging_interval="step"),
        ForceSaveLastCallback(),
    ]

    trainer = pl.Trainer(
        accelerator="gpu", devices=tc.get("devices", 1), precision=str(tc.get("precision", "bf16-mixed")),
        max_epochs=tc.get("max_epochs", 200), log_every_n_steps=tc.get("log_every_n_steps", 20),
        val_check_interval=tc.get("val_check_interval", 0.1), gradient_clip_val=tc.get("grad_clip_val", 1.0),
        gradient_clip_algorithm=tc.get("gradient_clip_algorithm", "norm"), callbacks=callbacks,
        logger=TensorBoardLogger(log_dir, name=exp_name), enable_progress_bar=True, accumulate_grad_batches=tc.get("accumulate_grad_batches", 1),
    )

    resume_ckpt = None
    if args.load_weights:
        load_weights_filtered(lit_model, args.load_weights)
    else:
        candidate, source = args.resume, "显式 --resume"
        if candidate is None:
            last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
            if os.path.exists(last_ckpt): candidate, source = last_ckpt, "自动检测 last.ckpt"
        if candidate is not None:
            ok, reason = check_resume_compatible(candidate, lit_model)
            if ok:
                resume_ckpt = candidate
                logger.info("断点续训 (%s): %s（%s）", source, candidate, reason)
            elif args.resume:
                logger.error("--resume 不兼容: %s\n  原因: %s", candidate, reason)
                raise SystemExit(1)
            else:
                logger.warning("%s 不兼容，已忽略，从头训练: %s\n  原因: %s", source, candidate, reason)
                backup = candidate + ".incompat_backup"
                try:
                    if not os.path.exists(backup): os.rename(candidate, backup)
                except OSError: pass
                resume_ckpt = None

    mc = cfg["model"]
    logger.info("=" * 60)
    logger.info("  RefSRWKV SR Prior 训练 (Auto Internal Resolution)")
    logger.info("  数据: %s (patch_size=%d, scale=%d)", cfg["data"]["root"], cfg["data"].get("patch_size", 480), mc.get("scale", 4))
    logger.info("  Batch size: %d × accumulate %d = 等效 %d", cfg["data"].get("batch_size", 4), tc.get("accumulate_grad_batches", 1), cfg["data"].get("batch_size", 4) * tc.get("accumulate_grad_batches", 1))
    logger.info("  LR: %.1e | warmup: %d 步", mc.get("learning_rate", 1e-4), mc.get("warmup_steps", 500))
    logger.info("  SSIM weight: %.2f | FFT weight: %.2f | Ref dropout: %.2f", mc.get("ssim_weight", 0.0), mc.get("fft_weight", 0.0), mc.get("ref_drop_prob", 0.0))
    logger.info("  EMA: %s", "on (decay=%.4f)" % mc.get("ema_decay", 0.999) if mc.get("use_ema", True) else "off")
    if args.load_weights: logger.info("  权重来源: 热启动 %s", args.load_weights)
    else: logger.info("  恢复: %s", resume_ckpt or "无（从头训练）")
    logger.info("=" * 60)

    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=resume_ckpt)
    logger.info("训练完成！最佳模型: %s", trainer.checkpoint_callback.best_model_path)

if __name__ == "__main__":
    main()