"""Model-agnostic inference and metric writing for native test splits.

The runner keeps the filesystem contract in one place.  Task-specific model
construction remains in ``models/sr`` and ``models/refsr``; this module only
normalizes batches, invokes the model, writes PNG files, and aggregates metrics.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from data.loaders import build_refsr_test_loader, build_sr_test_loader
from models.refsr import build_model as build_refsr_model
from models.sr import build_model as build_sr_model
from runtime.checkpoint import load_checkpoint, load_model_weights
from runtime.common import gaussian_ssim, per_image_psnr, resolve_path
from runtime.config import normalize_reference_mode, validate_config
from runtime.experiments import layout_from_config

LOGGER = logging.getLogger(__name__)
VALID_SPLITS = {"test", "test_easy", "test_hard"}


def select_device(config: Mapping[str, Any], requested: str | None = None) -> torch.device:
    """Resolve ``--device`` while respecting the training accelerator default."""
    value = str(requested or config.get("test", {}).get("device", "auto")).lower()
    if value == "auto":
        value = str(config.get("train", {}).get("accelerator", "auto")).lower()
        if value in {"auto", "gpu", "cuda"}:
            value = "cuda" if torch.cuda.is_available() else "cpu"
    if value in {"gpu", "cuda"}:
        if not torch.cuda.is_available():
            raise RuntimeError("请求 CUDA 设备，但当前环境没有可用 GPU")
        value = "cuda"
    if value == "cpu":
        return torch.device("cpu")
    try:
        device = torch.device(value)
    except RuntimeError as exc:
        raise ValueError(f"无效的推理设备: {value!r}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"请求设备 {value!r}，但当前环境没有可用 CUDA")
    return device


def _move_batch(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, Mapping):
        return {key: _move_batch(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_batch(value, device) for value in batch)
    return batch


def _reference_from_lr(lr: torch.Tensor, hr: torch.Tensor, scale: int) -> torch.Tensor:
    expected = (int(lr.shape[-2]) * int(scale), int(lr.shape[-1]) * int(scale))
    if tuple(hr.shape[-2:]) != expected:
        raise ValueError(
            f"LR/HR 尺寸不匹配: LR={tuple(lr.shape[-2:])}, HR={tuple(hr.shape[-2:])}, x{scale}"
        )
    return F.interpolate(lr, size=expected, mode="bicubic", align_corners=False)


def _reference_for_refsr_batch(
    batch: Mapping[str, Any],
    *,
    lr: torch.Tensor,
    hr: torch.Tensor,
    scale: int,
    reference_mode: str,
    ref_key: str,
) -> torch.Tensor:
    """Resolve the only valid reference source for one RefSR batch."""
    if reference_mode == "lr_up":
        return _reference_from_lr(lr, hr, scale)
    ref = batch.get(ref_key)
    if ref is None:
        raise ValueError(
            f"data.reference_mode=paired requires batch[{ref_key!r}]; "
            "use a LR/HR/Ref dataset or select reference_mode=lr_up for RefSRWKV."
        )
    if not torch.is_tensor(ref):
        raise TypeError(f"batch[{ref_key!r}] must be a tensor, got {type(ref).__name__}")
    return ref


def _image_tensor(value: torch.Tensor, *, value_range: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return metric-space ``[-1, 1]`` and PNG-space ``[0, 1]`` tensors."""
    value = torch.nan_to_num(value.float(), nan=0.0, posinf=1.0, neginf=-1.0)
    if value_range == "zero_one":
        png = value.clamp(0.0, 1.0)
        return png * 2.0 - 1.0, png
    metric = value.clamp(-1.0, 1.0)
    return metric, ((metric + 1.0) * 0.5).clamp(0.0, 1.0)


def _save_png(batch: torch.Tensor, output_dir: Path, start: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for offset, image in enumerate(batch.detach().cpu()):
        array = (image.permute(1, 2, 0).numpy() * 255.0 + 0.5).clip(0, 255).astype("uint8")
        Image.fromarray(array).save(output_dir / f"{start + offset:06d}.png")


def _build_refsr_model(
    config: Mapping[str, Any],
    checkpoint: Any,
    device: torch.device,
    *,
    raw_weights: bool = False,
):
    model_name = str(config.get("model", {}).get("name", "RefSRWKV")).lower()
    if model_name != "refdiffrwkv":
        model = build_refsr_model(
            config["model"], scale=int(config["data"]["scale"])
        )
        report = load_model_weights(model, checkpoint, prefer_ema=not raw_weights)
        LOGGER.info("loaded direct RefSR checkpoint (%s): %s", model_name, report)
        return model.to(device).eval(), "minus_one_one", None

    # The builder is shared with the training entry point so its prior-loading
    # and Stable-Diffusion compatibility rules cannot drift between commands.
    from scripts.train.refdiffrwkv import build_model

    system = build_model(config)
    report = load_model_weights(system, checkpoint, prefer_ema=not raw_weights)
    LOGGER.info("loaded RefDiffRWKV checkpoint: %s", report)
    system = system.to(device).eval()
    if getattr(system, "sr_model", None) is not None:
        system.sr_model.eval()
    return system, "zero_one", getattr(system, "generator", system)


def run_inference(
    config: Mapping[str, Any],
    *,
    checkpoint: str | Path,
    split: str = "test",
    output: str | Path | None = None,
    device: str | None = None,
    steps: int | None = None,
    batch_size: int | None = None,
    raw_weights: bool = False,
) -> dict[str, Any]:
    """Run one split and write ``images/*.png`` plus ``metrics.json``.

    ``output`` is the test-run root; the selected split is always nested below
    it.  Omitting it uses the canonical ``experiments/test/...`` layout.
    """
    split = str(split).strip()
    if split not in VALID_SPLITS:
        raise ValueError("split 必须是 test、test_easy 或 test_hard")
    config = dict(config)
    validate_config(config)
    task = str(config.get("task", "")).lower()
    model_name = str(config.get("model", {}).get("name", "")).lower()
    if task not in {"sr", "refsr"}:
        task = "refsr" if model_name in {"refsrwkv", "refdiffrwkv"} else "sr"
    reference_mode = (
        normalize_reference_mode(config["data"].get("reference_mode", "paired"))
        if task == "refsr"
        else None
    )
    selected_device = select_device(config, device)
    checkpoint_obj = load_checkpoint(resolve_path(checkpoint, prefer_cwd=True))

    if task == "sr":
        loader = build_sr_test_loader(config, split=split, batch_size=batch_size)
        model = build_sr_model(config["model"], scale=int(config["data"]["scale"]))
        report = load_model_weights(model, checkpoint_obj, prefer_ema=not raw_weights)
        LOGGER.info("loaded SR checkpoint: %s", report)
        model = model.to(selected_device).eval()
        value_range, generator = "minus_one_one", None
    else:
        loader = build_refsr_test_loader(config, split=split, batch_size=batch_size)
        model, value_range, generator = _build_refsr_model(
            config, checkpoint_obj, selected_device, raw_weights=raw_weights
        )

    layout = layout_from_config(dict(config))
    test_root = resolve_path(output, prefer_cwd=True) if output else layout.test_dir
    split_root = test_root / split
    image_root = split_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    scale = int(config["data"]["scale"])
    sample_count = 0
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    inference_steps = int(steps if steps is not None else config.get("model", {}).get("sample_steps", 20))
    if inference_steps < 1:
        raise ValueError("steps 必须为正整数")

    with torch.inference_mode():
        for batch in loader:
            batch = _move_batch(batch, selected_device)
            lr, hr = batch[config["data"].get("lr_key", "lr")], batch[config["data"].get("hr_key", "hr")]
            if task == "sr":
                prediction = model(lr)
                prediction_metric, prediction_png = _image_tensor(prediction, value_range=value_range)
            elif model_name != "refdiffrwkv":
                ref = _reference_for_refsr_batch(
                    batch,
                    lr=lr,
                    hr=hr,
                    scale=scale,
                    reference_mode=reference_mode,
                    ref_key=config["data"].get("ref_key", "ref"),
                )
                prediction = model(lr, ref)
                prediction_metric, prediction_png = _image_tensor(prediction, value_range=value_range)
            else:
                ref = _reference_for_refsr_batch(
                    batch,
                    lr=lr,
                    hr=hr,
                    scale=scale,
                    reference_mode=reference_mode,
                    ref_key=config["data"].get("ref_key", "ref"),
                )
                prediction_png = generator.generate_sr(
                    lr,
                    ref,
                    steps=inference_steps,
                    sr_model=getattr(model, "sr_model", None),
                    t_start=getattr(model, "t_start", None),
                    guidance_scale=float(getattr(model, "guidance_scale", 0.0)),
                    t_stop=int(getattr(model, "t_stop", 200)),
                )
                prediction_metric, prediction_png = _image_tensor(prediction_png, value_range="zero_one")

            hr_metric, _ = _image_tensor(hr, value_range="minus_one_one")
            if prediction_metric.shape != hr_metric.shape:
                raise ValueError(
                    f"模型输出与 HR 尺寸不一致: {tuple(prediction_metric.shape)} vs {tuple(hr_metric.shape)}"
                )
            psnr_values.extend(per_image_psnr(prediction_metric, hr_metric).detach().cpu().tolist())
            ssim_values.extend(gaussian_ssim(prediction_metric, hr_metric).detach().cpu().tolist())
            _save_png(prediction_png, image_root, sample_count)
            sample_count += int(prediction_png.shape[0])

    if not psnr_values:
        raise RuntimeError(f"split {split!r} 没有产生任何样本")
    metrics = {
        "task": task,
        "model": model_name,
        "dataset": str(config.get("dataset", {}).get("id", "dataset")),
        "scale": scale,
        "split": split,
        "samples": sample_count,
        "psnr": {"mean": sum(psnr_values) / len(psnr_values), "per_image": psnr_values},
        "ssim": {"mean": sum(ssim_values) / len(ssim_values), "per_image": ssim_values},
        "checkpoint": str(resolve_path(checkpoint, prefer_cwd=True)),
    }
    metrics_path = split_root / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("saved %d predictions and metrics to %s", sample_count, split_root)
    return metrics


__all__ = ["VALID_SPLITS", "run_inference", "select_device"]
