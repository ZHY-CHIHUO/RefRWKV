#!/usr/bin/env python
"""
SD2RefGANSystem 训练脚本

用法:
    python scripts/train_sd2_gan.py --config configs/sd2_ref_gan_config.yaml
    python scripts/train_sd2_gan.py --config configs/sd2_ref_gan_config.yaml \
        --resume checkpoints/sd2_ref_gan/last.ckpt

本版变更：
1. 所有路径改为相对路径（相对于项目根目录），本地/服务器共用同一份配置
2. 脚本启动时自动 cd 到项目根目录
3. SR 模型加载逻辑：
   - sr_fixed=True（冻结）：
     ① sr.ckpt_path 存在 → 从独立 SR checkpoint 加载
        （第一次训练 resume_ckpt=null 时走这条路）
     ② 否则 resume_ckpt 存在 → 从训练 checkpoint 提取 sr_model.*
     ③ 都没有 → ValueError
   - sr_fixed=False（微调）：只能从 resume_ckpt 加载（SR 状态随训练更新，
     必须恢复最新权重），不存在 → ValueError
4. 跨阶段结构变化自动回退：当 checkpoint 的 optimizer 参数组与当前模型
   不匹配时（典型：Stage1→2 开启 semantic 新增 299 参数），自动回退为
   仅加载模型权重，optimizer 重新初始化。
"""

import sys
import argparse
import yaml
import os
import shutil
import random
import logging
from pathlib import Path
from typing import Optional

# ═══════════════════════════════════════════════════════════
# ★ 项目根目录 = scripts/ 的上级目录
#   本地: ~/PROJECT/RefRWKV/
#   服务器: /mnt/sda/home/zhangheyi/projects/RefRWKV/
# ═══════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))  # 让 import RefRWKV.xxx 生效
os.chdir(PROJECT_ROOT)  # ★ 所有相对路径基于此目录

import torch

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_sd2_gan")

from RefRWKV.models.RefDiffRWKV.sd2_ref_generator import SD2RefGenerator
from RefRWKV.models.RefDiffRWKV.sd2_ref_discriminator import SD2RefDiscriminator
from RefRWKV.models.RefDiffRWKV.sd2_ref_gan_system import SD2RefGANSystem
from RefRWKV.models.RefSRWKV import RefSRWKV
from RefRWKV.RefSR_data.RefDataset import RefPNGDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("随机种子已设置为 %d", seed)


def build_sr_model(cfg: dict, resume_ckpt_path: Optional[str] = None):
    """构建 SR prior 模型（RefSRWKV）。

    SR 权重来源（按存在性判断，不读 checkpoint 内容）：
      sr_fixed=True（冻结）：
        ① sr.ckpt_path 存在 → 从独立 SR checkpoint 加载
           （第一次训练 resume_ckpt=null 时走这条路）
        ② 否则 resume_ckpt 存在 → 从训练 checkpoint 提取 sr_model.*
        ③ 都没有 → ValueError
      sr_fixed=False（微调）：
        只能从 resume_ckpt 加载（SR 状态在训练中更新，
        必须恢复训练 checkpoint 的最新权重），不存在 → ValueError
    """
    mc = cfg.get("model", {})
    if not mc.get("sr_enabled", False):
        return None

    sr_cfg = mc.get("sr", {})
    sr_fixed = mc.get("sr_fixed", True)

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

    def _load_into(model, path, desc):
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if isinstance(ckpt, dict):
            state_dict = {}
            for k, v in ckpt.items():
                for pre in (
                    "generator.sr_model.",
                    "sr_model.",
                    "model_sr.",
                    "model.",
                    "module.",
                ):
                    if k.startswith(pre):
                        k = k[len(pre) :]
                        break
                state_dict[k] = v
            ckpt = state_dict
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing:
            logger.warning("SR prior 缺失键 (%d): %s", len(missing), missing[:3])
        if unexpected:
            logger.warning("SR prior 多余键 (%d): %s", len(unexpected), unexpected[:3])
        logger.info("SR 权重加载完成（%s）: %s", desc, path)

    sr_ckpt = sr_cfg.get("ckpt_path")

    if sr_fixed:
        # 冻结：SR 权重训练中不变，sr.ckpt_path 与 resume_ckpt 内容一致，
        # 优先独立 checkpoint（第一次训练 resume_ckpt=null 时唯一来源）
        if sr_ckpt and os.path.exists(sr_ckpt):
            _load_into(model, sr_ckpt, "sr.ckpt_path")
        elif resume_ckpt_path and os.path.exists(resume_ckpt_path):
            _load_into(model, resume_ckpt_path, "resume_ckpt 兜底")
        else:
            raise ValueError(
                "sr_fixed=True 但 SR 权重无处加载："
                f"sr.ckpt_path={sr_ckpt!r} 不存在，且无 resume_ckpt。"
                "请提供独立 SR checkpoint，或指定含 SR 权重的 resume_ckpt。"
            )
        model.eval()
        model.requires_grad_(False)
        logger.info("SR prior 已冻结")
    else:
        # 微调：SR 权重训练中更新，必须恢复训练 checkpoint 的最新状态
        if resume_ckpt_path and os.path.exists(resume_ckpt_path):
            _load_into(model, resume_ckpt_path, "resume_ckpt（SR 微调）")
        else:
            raise ValueError(
                "sr_fixed=False（SR 微调）必须有 resume_ckpt 以恢复 SR 训练状态，"
                "当前 resume_ckpt 不存在。"
            )
        model.train()
        model.requires_grad_(True)
        logger.info("SR prior 可训练")

    return model


class BestAllMetricsCallback(Callback):
    """当指定指标中至少 min_improved 项同时改进时，保存验证图像。"""

    def __init__(self, metrics=None, min_improved=2):
        self.metrics = metrics or ["psnr", "ssim", "lpips"]
        self.min_improved = min_improved
        self.best_psnr = -float("inf")
        self.best_ssim = -float("inf")
        self.best_lpips = float("inf")
        self.best_dists = float("inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        psnr, ssim = metrics.get("val/psnr"), metrics.get("val/ssim")
        if psnr is None or ssim is None:
            return

        psnr_v, ssim_v = psnr.item(), ssim.item()
        lpips_v = metrics.get("val/lpips")
        dists_v = metrics.get("val/dists")

        candidates = {}
        if "psnr" in self.metrics and psnr_v > self.best_psnr:
            candidates["psnr"] = psnr_v
        if "ssim" in self.metrics and ssim_v > self.best_ssim:
            candidates["ssim"] = ssim_v
        if (
            "lpips" in self.metrics
            and lpips_v is not None
            and lpips_v.item() < self.best_lpips
        ):
            candidates["lpips"] = lpips_v.item()
        if (
            "dists" in self.metrics
            and dists_v is not None
            and dists_v.item() < self.best_dists
        ):
            candidates["dists"] = dists_v.item()

        improved = list(candidates.keys())

        log_dir = trainer.logger.log_dir if trainer.logger else "."
        tmp_dir = os.path.join(log_dir, "validation_tmp")

        if len(improved) >= self.min_improved:
            for k, v in candidates.items():
                setattr(self, f"best_{k}", v)
            tag = "+".join(improved)
            target = os.path.join(
                log_dir, f"validation_best_step_{trainer.global_step}_{tag}"
            )
            if os.path.exists(target):
                shutil.rmtree(target)
            if os.path.exists(tmp_dir):
                shutil.move(tmp_dir, target)
            pl_module.log("val/best_saved_step", float(trainer.global_step))
        elif os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

    def state_dict(self):
        return {
            "best_psnr": self.best_psnr,
            "best_ssim": self.best_ssim,
            "best_lpips": self.best_lpips,
            "best_dists": self.best_dists,
        }

    def load_state_dict(self, state_dict):
        self.best_psnr = state_dict.get("best_psnr", -float("inf"))
        self.best_ssim = state_dict.get("best_ssim", -float("inf"))
        self.best_lpips = state_dict.get("best_lpips", float("inf"))
        self.best_dists = state_dict.get("best_dists", float("inf"))


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config(cfg):
    mc = cfg.get("model", {})
    t_min, t_max = mc.get("t_min", 300), mc.get("t_max", 700)
    if t_min > t_max:
        raise ValueError(f"t_min({t_min}) 必须 <= t_max({t_max})，请检查配置文件")

    train_t_min = mc.get("train_t_min", 0)
    train_t_max = mc.get("train_t_max", 999)
    if train_t_min > train_t_max:
        raise ValueError(
            f"train_t_min({train_t_min}) 必须 <= train_t_max({train_t_max})"
        )
    aux_t_min = mc.get("aux_t_min", 100)
    aux_t_max = mc.get("aux_t_max", 400)
    if aux_t_min > aux_t_max:
        raise ValueError(f"aux_t_min({aux_t_min}) 必须 <= aux_t_max({aux_t_max})")

    lambda_sr_noise = mc.get("lambda_sr_noise", 1.0)
    sr_noise_warmdown_start = mc.get("sr_noise_warmdown_start", 1.0)
    sr_noise_warmdown_steps = mc.get("sr_noise_warmdown_steps", 0)
    if lambda_sr_noise < 0:
        raise ValueError("lambda_sr_noise must be >= 0")
    if sr_noise_warmdown_start < 0:
        raise ValueError("sr_noise_warmdown_start must be >= 0")
    if sr_noise_warmdown_steps < 0:
        raise ValueError("sr_noise_warmdown_steps must be >= 0")

    bs = cfg.get("data", {}).get("batch_size", 1)
    accum = mc.get("accumulate_grad_batches", 8)
    logger.info("有效 batch size: %d × %d = %d", bs, accum, bs * accum)
    logger.info(
        "课程 timestep: train=[%d, %d], aux=[%d, %d], "
        "lambda_sr_noise=%s, sr_noise_warmdown=(%s -> %s, %s steps), "
        "gan_crop=%s, gan_warmup=%s",
        train_t_min,
        train_t_max,
        aux_t_min,
        aux_t_max,
        lambda_sr_noise,
        sr_noise_warmdown_start,
        lambda_sr_noise,
        sr_noise_warmdown_steps,
        mc.get("gan_crop_size", 256),
        mc.get("gan_warmup_steps", 3000),
    )


def load_weights(model, ckpt_path, max_missing_ratio=0.1, strip_prefix=True):
    """从 checkpoint 加载权重，含缺失键比例校验。"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    if strip_prefix:
        new_sd = {}
        for k, v in state_dict.items():
            while True:
                old_k = k
                for pre in (
                    "model.",
                    "model_sr.",
                    "model_diff.",
                    "model_enhance.",
                    "generator.",
                    "discriminator.",
                ):
                    if k.startswith(pre):
                        k = k[len(pre) :]
                        break
                if k == old_k:
                    break
            new_sd[k] = v
        state_dict = new_sd

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    total_params = len(dict(model.named_parameters()))

    if missing:
        ratio = len(missing) / max(total_params, 1)
        logger.warning("缺失键 (%d): %s", len(missing), missing[:5])
        if ratio > max_missing_ratio:
            logger.error(
                "缺失键比例 %.1f%% 超过阈值 %.1f%%，checkpoint 可能不兼容！",
                ratio * 100,
                max_missing_ratio * 100,
            )
    if unexpected:
        logger.warning("多余键 (%d): %s", len(unexpected), unexpected[:5])

    logger.info("从 %s 加载权重完成", ckpt_path)


def build_dataloaders(cfg):
    dc, mc = cfg["data"], cfg.get("model", {})
    dk = {
        "data_dir": dc["root"],
        "patch_size": dc.get("patch_size", 480),
        "scale": dc.get("scale", 10),
        "ref_aug_strengths": dc.get("ref_aug_strengths", [0.12, 0.12, 0.12, 0.03]),
        "ref_aug_probs": dc.get("ref_aug_probs", [0.5, 0.5, 0.5, 0.5]),
        "ref_gray_prob": dc.get("ref_gray_prob", 0.2),
        "max_samples": (
            dc.get("max_samples_train"),
            dc.get("max_samples_val"),
            dc.get("max_samples_test"),
        ),
        "sample_seed": 42,
        "lr_key": mc.get("lr_key", "lr"),
        "hr_key": mc.get("hr_key", "hr"),
        "ref_key": mc.get("ref_key", "ref"),
    }

    train_ds = RefPNGDataset(
        mode="train",
        augment=dc.get("augment", True),
        augment_ref=dc.get("augment_ref", False),
        **dk,
    )
    val_ds = RefPNGDataset(mode="val", augment=False, augment_ref=False, **dk)

    tl = DataLoader(
        train_ds,
        batch_size=dc["batch_size"],
        shuffle=True,
        num_workers=dc["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=dc["num_workers"] > 0,
        prefetch_factor=dc.get("prefetch_factor", 4) if dc["num_workers"] > 0 else None,
    )
    vl = DataLoader(
        val_ds,
        batch_size=dc.get("val_batch_size", 1),
        shuffle=False,
        num_workers=dc.get("val_num_workers", 2),
        pin_memory=True,
    )
    return tl, vl


def build_model(cfg, resume_ckpt_path=None):
    mc = cfg.get("model", {})
    sr_model = build_sr_model(cfg, resume_ckpt_path=resume_ckpt_path)

    t_min, t_max = mc.get("t_min", 300), mc.get("t_max", 700)
    logger.info("Diffusion 时间步范围: [%d, %d]", t_min, t_max)

    rwkv_cfg = mc.get("rwkv_cfg", {"patch_size": 4, "embed_dim": 192})
    logger.info(
        "v2 开关: use_sr_condition=%s, use_confidence_gate=%s, "
        "self_sim_transfer=%s (topk=%s), dtex_conf_weight=%s",
        mc.get("use_sr_condition", False),
        mc.get("use_confidence_gate", False),
        rwkv_cfg.get("use_self_sim_transfer", False),
        rwkv_cfg.get("self_sim_topk", 8),
        mc.get("dtex_conf_weight", False),
    )
    logger.info(
        "System 课程: train_t=[%d, %d], aux_t=[%d, %d], "
        "lambda_sr_noise=%s, sr_noise_warmdown=(%s -> %s, %s steps), "
        "gan_crop_size=%s, gan_warmup_steps=%s",
        mc.get("train_t_min", 0),
        mc.get("train_t_max", 999),
        mc.get("aux_t_min", 100),
        mc.get("aux_t_max", 400),
        mc.get("lambda_sr_noise", 1.0),
        mc.get("sr_noise_warmdown_start", 1.0),
        mc.get("lambda_sr_noise", 1.0),
        mc.get("sr_noise_warmdown_steps", 0),
        mc.get("gan_crop_size", 256),
        mc.get("gan_warmup_steps", 3000),
    )

    generator = SD2RefGenerator(
        lr_key=mc.get("lr_key", "lr"),
        ref_key=mc.get("ref_key", "ref"),
        hr_key=mc.get("hr_key", "hr"),
        strategy=mc.get("strategy", "rwkv"),
        rwkv_cfg=rwkv_cfg,
        sd_model_path=mc["sd_model_path"],
        use_lora=mc.get("use_lora", True),
        lora_rank=mc.get("lora_rank", 64),
        lora_target_modules=mc.get("lora_target_modules"),
        sd_locked=mc.get("sd_locked", True),
        use_semantic=mc.get("use_semantic", True),
        dinov2_model_name=mc.get("dinov2_model_name", "facebook/dinov2-base"),
        num_train_timesteps=mc.get("num_train_timesteps", 1000),
        beta_start=mc.get("beta_start", 0.00085),
        beta_end=mc.get("beta_end", 0.012),
        beta_schedule=mc.get("beta_schedule", "scaled_linear"),
        prediction_type=mc.get("prediction_type", "epsilon"),
        t_min=t_min,
        t_max=t_max,
        cfg_drop_prob=mc.get("cfg_drop_prob", 0.1),
        control_scale=mc.get("control_scale", 1.0),
        learning_rate=mc.get("learning_rate", 1e-4),
        weight_decay=mc.get("weight_decay", 1e-3),
        sr_model=sr_model,
        use_sr_latent_cond=mc.get("use_sr_latent_cond", False),
        use_sr_condition=mc.get("use_sr_condition", False),
        use_confidence_gate=mc.get("use_confidence_gate", False),
        confidence_alpha=mc.get("confidence_alpha", 0.4),
        use_temporal_gate=mc.get("use_temporal_gate", False),
        control_scale_min=mc.get("control_scale_min", 0.3),
        control_scale_max=mc.get("control_scale_max", 1.5),
    )

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

    system = SD2RefGANSystem(
        generator=generator,
        discriminator=discriminator,
        lambda_gan_semantic=mc.get("lambda_gan", 0.0),
        lambda_gan_texture=mc.get("lambda_gan_texture", 0.0),
        lambda_lpips=mc.get("lambda_lpips", 0.0),
        lambda_diff_sr=mc.get("lambda_diff_sr", 0.5),
        accumulate_grad_batches=mc.get("accumulate_grad_batches", 8),
        use_amp=mc.get("use_amp", True),
        g_d_ratio=mc.get("g_d_ratio", 1),
        g_lr=mc.get("learning_rate", 1e-4),
        g_weight_decay=mc.get("weight_decay", 1e-3),
        d_lr_sem=mc.get("lr_D", 5e-6),
        d_lr_tex=mc.get("lr_D_texture", 1e-6),
        d_weight_decay=mc.get("d_weight_decay", 1e-3),
        betas=mc.get("d_betas", [0.5, 0.999]),
        sample_steps=mc.get("sample_steps", 50),
        fr_metrics=mc.get("fr_metrics", ["psnr", "ssim", "lpips", "dists"]),
        sr_model=sr_model,
        sr_fixed=mc.get("sr_fixed", True),
        sr_lr=mc.get("sr_lr", 1e-5),
        gan_enabled=mc.get("gan_enabled", False),
        t_start=mc.get("t_start"),
        guidance_scale=mc.get("guidance_scale", 0.0),
        t_stop=mc.get("t_stop", 200),
        grad_clip_val=mc.get("grad_clip_val", 1.0),
        grad_warn_threshold=mc.get("grad_warn_threshold", 100.0),
        max_consecutive_nan=mc.get("max_consecutive_nan", 10),
        use_swap_test=mc.get("use_swap_test", False),
        swap_ratio=mc.get("swap_ratio", 0.5),
        dtex_conf_weight=mc.get("dtex_conf_weight", False),
        lambda_sr_noise=mc.get("lambda_sr_noise", 1.0),
        sr_noise_warmdown_start=mc.get("sr_noise_warmdown_start", 1.0),
        sr_noise_warmdown_steps=mc.get("sr_noise_warmdown_steps", 0),
        gan_crop_size=mc.get("gan_crop_size", 256),
        train_t_min=mc.get("train_t_min", 0),
        train_t_max=mc.get("train_t_max", 999),
        aux_t_min=mc.get("aux_t_min", 100),
        aux_t_max=mc.get("aux_t_max", 400),
        gan_warmup_steps=mc.get("gan_warmup_steps", 3000),
    )

    return system


def build_trainer(cfg, exp_name, checkpoint_dir, log_dir, max_epochs):
    tc = cfg.get("train", {})
    full_ckpt_dir = os.path.join(checkpoint_dir, exp_name)
    os.makedirs(full_ckpt_dir, exist_ok=True)

    mc = cfg.get("model", {})
    lambda_lpips = mc.get("lambda_lpips", 0.0)
    has_lpips_metric = "lpips" in mc.get("fr_metrics", [])

    if lambda_lpips > 0 and has_lpips_metric:
        es_monitor = "val/lpips"
        es_mode = "min"
    else:
        es_monitor = "val_psnr"
        es_mode = "max"

    best_cfg = tc.get("best_save", {})
    best_metrics = best_cfg.get("metrics", ["psnr", "ssim", "lpips"])
    best_min = best_cfg.get("min_improved", 2)

    callbacks = [
        BestAllMetricsCallback(metrics=best_metrics, min_improved=best_min),
        EarlyStopping(
            monitor=es_monitor,
            patience=tc.get("early_stopping_patience", 20),
            mode=es_mode,
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=full_ckpt_dir,
            filename="{epoch:04d}-{step:06d}-{val_psnr:.4f}",
            monitor="val_psnr",
            save_top_k=tc.get("save_top_k", 3),
            mode="max",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if lambda_lpips > 0 and has_lpips_metric:
        callbacks.insert(
            2,
            ModelCheckpoint(
                dirpath=full_ckpt_dir,
                filename="{epoch:04d}-{step:06d}-{val_lpips:.4f}",
                monitor="val/lpips",
                save_top_k=1,
                mode="min",
            ),
        )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=tc.get("devices", 1),
        precision=str(tc.get("precision", 32)),
        max_epochs=max_epochs,
        log_every_n_steps=tc.get("log_every_n_steps", 20),
        val_check_interval=tc.get("val_check_interval", 0.5),
        gradient_clip_val=None,
        callbacks=callbacks,
        logger=TensorBoardLogger(log_dir, name=exp_name),
        enable_progress_bar=True,
    )
    return trainer, full_ckpt_dir


def train(cfg, resume_ckpt=None):
    tc, oc, mc = cfg["train"], cfg.get("output", {}), cfg.get("model", {})

    seed = tc.get("seed", 42)
    set_seed(seed)
    pl.seed_everything(seed, workers=True)

    if tc.get("detect_anomaly", False):
        torch.autograd.set_detect_anomaly(True)
        logger.warning("autograd 异常检测已开启")

    validate_config(cfg)

    ckpt_dir = oc.get("checkpoint_dir", "checkpoints/sd2_ref_gan")
    log_dir = oc.get("log_dir", "logs/sd2_ref_gan")
    exp_name = oc.get("experiment_name", "sd2_ref_gan")
    full_ckpt_dir = os.path.join(ckpt_dir, exp_name)

    # ── ★ 确定 resume_ckpt（在 build_model 之前，SR 加载需要它）──
    if resume_ckpt is None:
        # ① 优先：当前实验目录的 last.ckpt（同阶段断点续训）
        last_ckpt = os.path.join(full_ckpt_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            resume_ckpt = last_ckpt
            logger.info("自动检测到 last.ckpt，断点续训: %s", last_ckpt)
        else:
            # ② 其次：配置文件中指定的 resume_ckpt（跨阶段首次启动）
            cfg_resume = tc.get("resume_ckpt")
            if cfg_resume and os.path.exists(cfg_resume):
                resume_ckpt = cfg_resume
                logger.info("从配置指定的 checkpoint 恢复: %s", cfg_resume)
            elif cfg_resume:
                logger.warning("配置指定的 checkpoint 不存在: %s，从头训练", cfg_resume)
            else:
                logger.info("未指定 resume_ckpt，无 last.ckpt，从头训练")

    # ── 直接构建模型，SR 来源校验在 build_sr_model 内部完成 ──
    system = build_model(cfg, resume_ckpt_path=resume_ckpt)

    # 加载预训练权重（非 resume 时，比如从 Stage1 → Stage2 的过渡）
    pretrained = cfg.get("pretrained", {})
    if pretrained.get("sd2_control_ckpt") and not resume_ckpt:
        logger.info("从 %s 加载预训练权重", pretrained["sd2_control_ckpt"])
        load_weights(
            system,
            pretrained["sd2_control_ckpt"],
            strip_prefix=pretrained.get("strip_prefix", False),
        )

    train_loader, val_loader = build_dataloaders(cfg)
    max_epochs = tc.get("max_epochs", 100)

    trainer, full_ckpt_dir = build_trainer(cfg, exp_name, ckpt_dir, log_dir, max_epochs)

    logger.info("=" * 60)
    logger.info("  SD2RefGANSystem 训练")
    logger.info("  项目根目录: %s", PROJECT_ROOT)
    logger.info("  数据根目录: %s", cfg["data"]["root"])
    logger.info(
        "  Batch size: %d x accumulate %d",
        cfg["data"]["batch_size"],
        mc.get("accumulate_grad_batches", 8),
    )
    logger.info("  训练样本数: %d", len(train_loader.dataset))
    logger.info("  验证样本数: %d", len(val_loader.dataset))
    logger.info("  恢复 checkpoint: %s", resume_ckpt or "无（从头训练）")
    logger.info("=" * 60)

    # ── ★ 跨阶段预检测与加载（避免双重加载导致 CPU OOM） ──
    fit_ckpt_path = resume_ckpt

    if resume_ckpt and os.path.exists(resume_ckpt):
        logger.info("正在读取 checkpoint 以检查 optimizer 兼容性...")
        # 这里会加载 checkpoint 到内存（可能很大）
        ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)

        opt_states = ckpt.get("optimizer_states", [])
        load_optimizer = True

        if opt_states:
            # 统计 checkpoint 里 optimizer 的参数组数 + 管理的参数数量
            saved_param_groups = len(opt_states[0].get("param_groups", []))
            saved_params = sum(
                len(pg.get("params", []))
                for pg in opt_states[0].get("param_groups", [])
            )

            # 当前模型 G 优化器的参数组数
            current_param_groups = 1
            current_params = sum(1 for p in system.parameters() if p.requires_grad)

            # 判断当前 G 优化器是否有参数分组（与 configure_optimizers 的划分一致）
            gen = system.generator
            current_param_groups = 2 if any(
                p.requires_grad
                and any(k in n for k in ("semantic_pyramid", "sem_proj", "sr_conditioner"))
                for n, p in gen.named_parameters()
            ) else 1

            if saved_params != current_params or saved_param_groups != current_param_groups:
                logger.warning(
                    "预检测：参数数(%d vs %d) 或参数组数(%d vs %d) 不匹配，"
                    "判定为跨阶段结构变化，仅加载模型权重，不恢复 optimizer",
                    saved_params, current_params, saved_param_groups, current_param_groups,
                )
                load_optimizer = False
            else:
                logger.info(
                    "预检测通过：参数数=%d，参数组数=%d，完整恢复 optimizer",
                    saved_params, current_param_groups
                )

        if not load_optimizer:
            # 【跨阶段】：手动加载权重，然后立刻丢弃 checkpoint 以释放内存
            logger.info("跨阶段恢复：仅加载模型权重，optimizer 重新初始化")
            sd = ckpt.get("state_dict", ckpt)
            system.load_state_dict(sd, strict=False)

            # ★ 彻底释放 checkpoint 内存，防止后续 OOM
            del ckpt, sd, opt_states
            import gc
            gc.collect()

            # 告诉 trainer 不要再去读 checkpoint 了
            fit_ckpt_path = None
        else:
            # 【同阶段】：兼容，释放我们手动加载的 ckpt，让 trainer.fit 自己去读
            del ckpt, opt_states
            import gc
            gc.collect()
            fit_ckpt_path = resume_ckpt

    # ── 正常训练循环（仅保留 CUDA OOM 重试） ──
    max_retries = 3
    for attempt in range(max_retries):
        try:
            trainer.fit(system, train_loader, val_loader, ckpt_path=fit_ckpt_path)
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and attempt < max_retries - 1:
                logger.warning(
                    "CUDA OOM (attempt %d/%d)，清理缓存后重试...",
                    attempt + 1,
                    max_retries,
                )
                torch.cuda.empty_cache()
                import time

                time.sleep(5)
                del system
                torch.cuda.empty_cache()

                system = build_model(cfg, resume_ckpt_path=resume_ckpt)
                last_ckpt = os.path.join(full_ckpt_dir, "last.ckpt")
                if os.path.exists(last_ckpt):
                    logger.info("OOM 重试：从 %s 恢复", last_ckpt)
                    resume_ckpt = last_ckpt
                    fit_ckpt_path = last_ckpt
                elif pretrained.get("sd2_control_ckpt"):
                    logger.info("OOM 重试：无 checkpoint，重新加载预训练权重")
                    load_weights(
                        system,
                        pretrained["sd2_control_ckpt"],
                        strip_prefix=pretrained.get("strip_prefix", False),
                    )
                    fit_ckpt_path = None

                trainer, full_ckpt_dir = build_trainer(
                    cfg, exp_name, ckpt_dir, log_dir, max_epochs
                )
            else:
                raise

    best_path, best_score = None, None
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint) and cb.monitor == "val_psnr":
            best_path = cb.best_model_path
            best_score = cb.best_model_score
            break

    return best_path, best_score


def main():
    parser = argparse.ArgumentParser(description="SD2RefGANSystem 训练")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    oc = cfg.get("output", {})
    ckpt_dir = oc.get("checkpoint_dir", "checkpoints")
    exp_name = oc.get("experiment_name", "")
    if exp_name:
        ckpt_dir = os.path.join(ckpt_dir, exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(oc.get("log_dir", "logs"), exist_ok=True)

    # 保存训练配置副本
    shutil.copy2(args.config, os.path.join(ckpt_dir, "train_config.yaml"))

    best_ckpt, best_score = train(cfg, resume_ckpt=args.resume)

    logger.info("=" * 60)
    logger.info("  训练完成")
    logger.info("  最佳模型: %s", best_ckpt or "N/A")
    if best_score is not None:
        logger.info("  最佳 val_psnr: %.6f", best_score)
    else:
        logger.info("  最佳 val_psnr: N/A（无验证记录）")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
