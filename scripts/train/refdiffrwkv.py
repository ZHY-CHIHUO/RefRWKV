#!/usr/bin/env python3
"""Train the ``RefDiffRWKV`` RefSR model.

The diffusion implementation owns its manual G/D loop. This entry point only
owns experiment setup: configuration materialization, direct-RefSR prior
loading, data loaders, checkpoint compatibility, and the Lightning callbacks
shared by the project. The model remains under ``models/refsr/RefDiffRWKV`` so
adding a compatible RefSR condition model does not create another framework.
"""

from __future__ import annotations

import argparse
import gc
import logging
import random
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.loaders import build_refsr_loaders
from models.refsr import build_model as build_refsr_model
from runtime.checkpoint import load_checkpoint, load_model_weights
from runtime.common import resolve_path
from runtime.config import load_config, validate_config
from runtime.experiments import ExperimentLayout, layout_from_config, save_config_snapshot

logger = logging.getLogger("train.refdiffrwkv")


def _model_cfg(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("model", {})
    if not isinstance(value, Mapping):
        raise ValueError("model must be a mapping")
    return value


def _load_refsr_prior(model: torch.nn.Module, path: str | Path, description: str) -> None:
    """Load a direct RefSR prior from raw, Lightning, or diffusion checkpoints."""
    report = load_model_weights(model, load_checkpoint(path), prefer_ema=True)
    logger.info("RefSR prior loaded (%s): %s; %s", description, path, report)


def build_sr_model(config: Mapping[str, Any], *, resume_ckpt_path: str | Path | None = None) -> torch.nn.Module | None:
    """Build the SR branch consumed by RefDiffRWKV.

    A diffusion checkpoint may contain the prior under ``generator.sr_model``
    or ``sr_model``; :func:`load_model_weights` normalizes both forms.  A
    missing prior is allowed with a warning so a user can deliberately start a
    fully fresh experiment, but a production run should set
    ``model.sr.name`` and ``model.sr.ckpt_path`` to a registered direct RefSR
    architecture and compatible checkpoint.
    """
    mc = _model_cfg(config)
    if not bool(mc.get("sr_enabled", True)):
        return None
    raw_sr = mc.get("sr", {})
    if not isinstance(raw_sr, Mapping):
        raise ValueError("model.sr must be a mapping")
    sr_cfg = dict(raw_sr)
    data = config.get("data", {})
    scale = int(data.get("scale", sr_cfg.get("scale", 1)))
    prior_name = str(sr_cfg.get("name", "refsrwkv"))
    sr_cfg.setdefault("name", prior_name)
    model = build_refsr_model(sr_cfg, scale=scale)

    sr_fixed = bool(mc.get("sr_fixed", True))
    configured = sr_cfg.get("ckpt_path")
    configured_path = resolve_path(configured) if configured else None
    resume_path = Path(resume_ckpt_path).expanduser() if resume_ckpt_path else None
    if resume_path is not None and not resume_path.is_absolute():
        resume_path = resolve_path(resume_path)

    source: Path | None = None
    description = "random initialization"
    if configured_path is not None and configured_path.is_file():
        source, description = configured_path, "model.sr.ckpt_path"
    elif resume_path is not None and resume_path.is_file():
        source, description = resume_path, "resume checkpoint fallback"

    if source is not None:
        _load_refsr_prior(model, source, description)
    else:
        logger.warning(
            "RefSR prior %s checkpoint not found (ckpt_path=%r, resume=%r); "
            "using random initialization",
            prior_name,
            configured,
            str(resume_path) if resume_path else None,
        )

    model.requires_grad_(not sr_fixed)
    model.train(not sr_fixed)
    if sr_fixed:
        model.eval()
    return model


def build_model(config: Mapping[str, Any], *, resume_ckpt_path: str | Path | None = None):
    """Construct the full RefDiffRWKV system from a materialized config."""
    # These imports are intentionally lazy: ``--help`` and SR-only tooling do
    # not need diffusers, LPIPS, or OpenCLIP installed.
    from engines.refsr.refdiff_trainer import RefDiffRWKVTrainer
    from models.refsr.RefDiffRWKV.sd2_ref_discriminator import SD2RefDiscriminator
    from models.refsr.RefDiffRWKV.sd2_ref_generator import SD2RefGenerator

    mc = dict(_model_cfg(config))
    train = config.get("train", {})
    if not isinstance(train, Mapping):
        raise ValueError("train must be a mapping")
    loss = config.get("loss", {})
    if not isinstance(loss, Mapping):
        raise ValueError("loss must be a mapping")
    # Optimizer settings belong to ``train`` and objective weights belong to
    # ``loss``.  The model-level fallbacks keep old in-memory configs usable
    # while the canonical YAML remains cleanly separated.
    train_value = lambda key, default: train.get(key, mc.get(key, default))
    loss_value = lambda key, default: loss.get(key, mc.get(key, default))
    sr_model = build_sr_model(config, resume_ckpt_path=resume_ckpt_path)
    sr_cfg = mc.get("sr", {}) if isinstance(mc.get("sr", {}), Mapping) else {}

    generator = SD2RefGenerator(
        strategy=mc.get("strategy", "rwkv"),
        sd_model_path=mc.get("sd_model_path", "sd2-community/stable-diffusion-2-1-base"),
        use_lora=bool(mc.get("use_lora", True)),
        lora_rank=int(mc.get("lora_rank", 64)),
        lora_target_modules=mc.get("lora_target_modules"),
        sd_locked=bool(mc.get("sd_locked", True)),
        rwkv_cfg=mc.get("rwkv_cfg"),
        use_semantic=bool(mc.get("use_semantic", False)),
        dinov2_model_name=mc.get("dinov2_model_name", "facebook/dinov2-base"),
        num_train_timesteps=int(mc.get("num_train_timesteps", 1000)),
        beta_start=float(mc.get("beta_start", 0.00085)),
        beta_end=float(mc.get("beta_end", 0.012)),
        beta_schedule=mc.get("beta_schedule", "scaled_linear"),
        prediction_type=mc.get("prediction_type", "epsilon"),
        t_min=int(mc.get("t_min", 300)),
        t_max=int(mc.get("t_max", 700)),
        cfg_drop_prob=float(mc.get("cfg_drop_prob", 0.2)),
        control_scale=float(mc.get("control_scale", 1.0)),
        learning_rate=float(train_value("learning_rate", 1.0e-5)),
        weight_decay=float(train_value("weight_decay", 1.0e-3)),
        lr_key=mc.get("lr_key", config.get("data", {}).get("lr_key", "lr")),
        ref_key=mc.get("ref_key", config.get("data", {}).get("ref_key", "ref")),
        hr_key=mc.get("hr_key", config.get("data", {}).get("hr_key", "hr")),
        local_files_only=bool(mc.get("local_files_only", True)),
        sr_model=sr_model,
        use_sr_latent_cond=bool(mc.get("use_sr_latent_cond", True)),
        use_sr_condition=bool(mc.get("use_sr_condition", False)),
        use_confidence_gate=bool(mc.get("use_confidence_gate", False)),
        confidence_alpha=float(mc.get("confidence_alpha", 0.4)),
        use_temporal_gate=bool(mc.get("use_temporal_gate", False)),
        control_scale_min=float(mc.get("control_scale_min", 0.3)),
        control_scale_max=float(mc.get("control_scale_max", 1.5)),
        wkv_backend=mc.get("wkv_backend", "torch"),
        use_reference=bool(mc.get("use_reference", True)),
    )

    discriminator = None
    if bool(mc.get("use_discriminator", False)):
        discriminator = SD2RefDiscriminator(
            use_semantic_d=bool(mc.get("use_semantic_d", True)),
            use_texture_d=bool(mc.get("use_texture_d", True)),
            semantic_alpha=float(mc.get("semantic_alpha", 0.8)),
            semantic_use_freq=bool(mc.get("semantic_use_freq", True)),
            semantic_trainable_stages=int(mc.get("semantic_trainable_stages", 1)),
            semantic_precision=mc.get("semantic_precision", "fp32"),
            texture_base_ch=int(mc.get("texture_base_ch", 48)),
            texture_num_scales=int(mc.get("texture_num_scales", 4)),
            texture_use_spectral=bool(mc.get("texture_use_spectral", True)),
            lr_semantic=float(train_value("lr_D", 5.0e-6)),
            lr_texture=float(train_value("lr_D_texture", 1.0e-6)),
            weight_decay=float(train_value("d_weight_decay", 1.0e-3)),
            betas=train_value("d_betas", [0.5, 0.999]),
        )

    return RefDiffRWKVTrainer(
        generator=generator,
        discriminator=discriminator,
        lambda_gan_semantic=float(loss_value("lambda_gan", loss_value("lambda_gan_semantic", 0.0))),
        lambda_gan_texture=float(loss_value("lambda_gan_texture", 0.0)),
        lambda_lpips=float(loss_value("lambda_lpips", 0.0)),
        lambda_diff_sr=float(loss_value("lambda_diff_sr", 0.0)),
        accumulate_grad_batches=int(train.get("accumulate_grad_batches", mc.get("accumulate_grad_batches", 8))),
        use_amp=bool(train_value("use_amp", False)),
        g_d_ratio=int(mc.get("g_d_ratio", 1)),
        g_lr=float(train_value("learning_rate", 1.0e-5)),
        g_weight_decay=float(train_value("weight_decay", 1.0e-3)),
        d_lr_sem=float(train_value("lr_D", 5.0e-6)),
        d_lr_tex=float(train_value("lr_D_texture", 1.0e-6)),
        d_weight_decay=float(train_value("d_weight_decay", 1.0e-3)),
        betas=train_value("d_betas", [0.5, 0.999]),
        sample_steps=int(mc.get("sample_steps", 20)),
        fr_metrics=mc.get("fr_metrics", ["psnr", "ssim"]),
        sr_model=sr_model,
        sr_fixed=bool(mc.get("sr_fixed", True)),
        sr_lr=float(mc.get("sr_lr", 1.0e-5)),
        gan_enabled=bool(mc.get("gan_enabled", False)),
        t_start=mc.get("t_start"),
        guidance_scale=float(mc.get("guidance_scale", 0.0)),
        t_stop=int(mc.get("t_stop", 200)),
        grad_clip_val=float(train_value("grad_clip_val", train_value("grad_clip_norm", 1.0))),
        grad_warn_threshold=float(mc.get("grad_warn_threshold", 100.0)),
        max_consecutive_nan=int(mc.get("max_consecutive_nan", 20)),
        use_swap_test=bool(mc.get("use_swap_test", False)),
        swap_ratio=float(mc.get("swap_ratio", 0.5)),
        dtex_conf_weight=bool(mc.get("dtex_conf_weight", False)),
        lambda_sr_noise=float(loss_value("lambda_sr_noise", 1.0)),
        sr_noise_warmdown_start=float(loss_value("sr_noise_warmdown_start", 1.0)),
        sr_noise_warmdown_steps=int(loss_value("sr_noise_warmdown_steps", 0)),
        gan_crop_size=int(mc.get("gan_crop_size", 256)),
        train_t_min=int(mc.get("train_t_min", 0)),
        train_t_max=int(mc.get("train_t_max", 999)),
        aux_t_min=int(mc.get("aux_t_min", 100)),
        aux_t_max=int(mc.get("aux_t_max", 400)),
        gan_warmup_steps=int(mc.get("gan_warmup_steps", 3000)),
    )


class ForceSaveLast(Callback):
    """Keep a resumable checkpoint even when no monitored metric improves."""

    def on_train_epoch_end(self, trainer, pl_module) -> None:  # pragma: no cover - Lightning callback
        callback = trainer.checkpoint_callback
        if callback is None or not getattr(callback, "dirpath", None):
            return
        target = Path(callback.dirpath) / "last.ckpt"
        try:
            trainer.save_checkpoint(str(target))
        except Exception as exc:
            logger.warning("could not force-save %s: %s", target, exc)


def _optimizer_signature(system: Any) -> tuple[int, int]:
    """Return optimizer count and G parameter-group count without allocation."""
    count = 1  # generator
    sr = getattr(system, "sr_model", None)
    if sr is not None and not bool(getattr(system, "sr_fixed", True)) and any(p.requires_grad for p in sr.parameters()):
        count += 1
    discriminator = getattr(system, "discriminator", None)
    if discriminator is not None:
        if bool(getattr(discriminator, "use_semantic_d", False)) and any(p.requires_grad for p in discriminator.D_sem.parameters()):
            count += 1
        if bool(getattr(discriminator, "use_texture_d", False)) and any(p.requires_grad for p in discriminator.D_tex.parameters()):
            count += 1
    generator = getattr(system, "generator", None)
    groups = 1
    if generator is not None and any(
        p.requires_grad and any(token in name for token in ("semantic_pyramid", "sem_proj", "sr_conditioner"))
        for name, p in generator.named_parameters()
    ):
        groups = 2
    return count, groups


def _prepare_resume(system: Any, resume: Path | None) -> Path | None:
    """Load weights manually when stage changes make optimizer state unsafe."""
    if resume is None or not resume.is_file():
        return None
    checkpoint = load_checkpoint(resume)
    optimizer_states = checkpoint.get("optimizer_states", []) if isinstance(checkpoint, Mapping) else []
    expected_count, expected_groups = _optimizer_signature(system)
    saved_count = len(optimizer_states) if isinstance(optimizer_states, list) else 0
    saved_groups = len(optimizer_states[0].get("param_groups", [])) if optimizer_states else expected_groups
    compatible = saved_count in {0, expected_count} and saved_groups == expected_groups
    if compatible:
        del checkpoint
        gc.collect()
        return resume

    logger.warning(
        "stage checkpoint optimizer layout differs (saved count/groups=%s/%s, current=%s/%s); "
        "loading weights only",
        saved_count,
        saved_groups,
        expected_count,
        expected_groups,
    )
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
    system.load_state_dict(state, strict=False)
    del checkpoint, state
    gc.collect()
    return None


def _accelerator(train: Mapping[str, Any]) -> str:
    value = str(train.get("accelerator", "auto")).lower()
    if value == "auto":
        return "gpu" if torch.cuda.is_available() else "cpu"
    if value == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("train.accelerator=gpu 但当前环境没有 CUDA")
    return value


def run(config: dict[str, Any], *, resume: str | None = None) -> Path:
    validate_config(config)
    train = config.get("train", {})
    model_cfg = _model_cfg(config)
    pl.seed_everything(int(train.get("seed", 42)), workers=True)
    random.seed(int(train.get("seed", 42)))
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    layout = layout_from_config(config).create_train()
    save_config_snapshot(config, layout.train_dir / "config.json")
    with (layout.train_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

    requested = Path(resume).expanduser() if resume else None
    if requested is not None and not requested.is_absolute():
        requested = resolve_path(requested)
    if requested is None and bool(train.get("auto_resume", True)):
        candidate = layout.checkpoints / "last.ckpt"
        if candidate.is_file():
            requested = candidate
    if requested is None and train.get("resume_ckpt"):
        candidate = resolve_path(train["resume_ckpt"])
        if candidate.is_file():
            requested = candidate

    system = build_model(config, resume_ckpt_path=requested)
    fit_ckpt = _prepare_resume(system, requested)
    train_loader, val_loader = build_refsr_loaders(config)

    monitor = str(train.get("ckpt_monitor", "val/loss_diff"))
    mode = str(train.get("ckpt_mode", "min"))
    if mode not in {"min", "max"}:
        raise ValueError("train.ckpt_mode must be min or max")
    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=str(layout.checkpoints),
            filename="epoch={epoch:04d}-step={step:06d}",
            monitor=monitor,
            mode=mode,
            save_top_k=int(train.get("save_top_k", 3)),
            save_last=True,
        ),
        ForceSaveLast(),
        LearningRateMonitor(logging_interval="step"),
    ]
    patience = train.get("early_stopping_patience")
    if patience is not None:
        callbacks.append(EarlyStopping(monitor=str(train.get("es_monitor") or monitor), mode=str(train.get("es_mode") or mode), patience=int(patience)))

    trainer = pl.Trainer(
        accelerator=_accelerator(train),
        devices=train.get("devices", 1),
        precision=train.get("precision", "32"),
        max_epochs=int(train.get("max_epochs", -1)),
        max_steps=int(train.get("max_steps", -1)),
        val_check_interval=train.get("val_check_interval", 1.0),
        check_val_every_n_epoch=int(train.get("check_val_every_n_epoch", 1)),
        # RefDiffRWKV performs its own G/D accumulation.
        accumulate_grad_batches=1,
        log_every_n_steps=int(train.get("log_every_n_steps", 20)),
        num_sanity_val_steps=int(train.get("num_sanity_val_steps", 0)),
        callbacks=callbacks,
        logger=TensorBoardLogger(str(layout.train_dir), name="logs", version=""),
        enable_progress_bar=bool(train.get("enable_progress_bar", True)),
        gradient_clip_val=None,
    )
    logger.info(
        "RefDiffRWKV training: dataset=%s model=%s scale=x%s resume=%s",
        config.get("dataset", {}).get("id"),
        model_cfg.get("name"),
        config.get("data", {}).get("scale"),
        fit_ckpt or "weights only / fresh optimizer",
    )
    trainer.fit(system, train_loader, val_loader, ckpt_path=str(fit_ckpt) if fit_ckpt else None)
    return layout.train_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--overrides", nargs="*", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run(load_config(args.config, args.overrides), resume=args.resume)


if __name__ == "__main__":
    main()
