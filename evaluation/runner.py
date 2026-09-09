"""Model-agnostic inference and metric writing for native test splits.

The runner keeps the filesystem contract in one place. Task-specific model
construction remains in ``models/sr`` and ``models/refsr``; this module only
normalizes batches, invokes the model, writes prediction rasters, and aggregates metrics.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data.loaders import build_refsr_test_loader, build_sr_test_loader
from models.refsr import build_model as build_refsr_model
from models.sr import build_model as build_sr_model
from runtime.checkpoint import load_checkpoint, load_model_weights
from runtime.common import gaussian_ssim, per_image_psnr, resolve_path
from runtime.config import is_refsr_model, normalize_reference_mode, validate_config
from runtime.experiments import layout_from_config
from runtime.tiling import tiled_forward
from metrics.wuhan import wuhan_metric_tensors

LOGGER = logging.getLogger(__name__)
VALID_SPLITS = {"test", "test_easy", "test_hard"}
VALID_METRICS = {
    "psnr",
    "ssim",
    "rmse",
    "uiqi",
    "psnr_wuhan",
    "sam_rad",
    "sam_deg",
    "ergas",
}
WUHAN_METRICS = {"rmse", "uiqi", "psnr_wuhan", "sam_rad", "sam_deg", "ergas"}
_METRIC_ALIASES = {
    "psnr": ("psnr",),
    "ssim": ("ssim",),
    "rmse": ("rmse",),
    "uiqi": ("uiqi",),
    "psnr_wuhan": ("psnr_wuhan",),
    "sam": ("sam_rad", "sam_deg"),
    "sam_rad": ("sam_rad",),
    "sam_deg": ("sam_deg",),
    "ergas": ("ergas",),
    "all": tuple(sorted(VALID_METRICS)),
}


def normalize_test_metrics(value: Any = None) -> list[str]:
    """Normalize ``test.metrics`` while preserving the requested order."""
    if value is None:
        raw: list[Any] = ["psnr", "ssim"]
    elif isinstance(value, str):
        raw = value.replace(",", " ").split()
    elif isinstance(value, Sequence):
        raw = list(value)
    else:
        raise ValueError("test.metrics must be a metric name or a sequence of names")
    result: list[str] = []
    for item in raw:
        name = str(item).strip().lower()
        if not name:
            continue
        try:
            expanded = _METRIC_ALIASES[name]
        except KeyError as exc:
            options = ", ".join(sorted((*VALID_METRICS, "sam", "all")))
            raise ValueError(f"unknown test metric {item!r}; choose from {options}") from exc
        for metric in expanded:
            if metric not in result:
                result.append(metric)
    return result


def _test_options(
    config: Mapping[str, Any],
    *,
    metrics: Sequence[str] | str | None,
    save_images: bool | None,
    device: str | None,
    steps: int | None,
    batch_size: int | None,
    output: str | Path | None,
) -> tuple[list[str], bool, str | None, int | None, int | None, str | Path | None]:
    test = config.get("test", {})
    if not isinstance(test, Mapping):
        raise ValueError("test 必须是 mapping")
    selected_metrics = normalize_test_metrics(test.get("metrics") if metrics is None else metrics)
    configured_save = test.get("save_images", test.get("infer_images", True))
    if save_images is None:
        save_predictions = configured_save
    else:
        save_predictions = save_images
    if not isinstance(save_predictions, bool):
        raise ValueError("test.save_images 必须是布尔值")
    configured_device = test.get("device")
    effective_device = device if device is not None else (
        str(configured_device) if configured_device is not None else None
    )
    configured_steps = test.get("steps")
    effective_steps = steps if steps is not None else configured_steps
    if effective_steps is not None:
        if isinstance(effective_steps, bool) or not isinstance(effective_steps, int) or effective_steps < 1:
            raise ValueError("test.steps 必须是正整数或 null")
    configured_batch = test.get("batch_size")
    effective_batch = batch_size if batch_size is not None else configured_batch
    if effective_batch is not None:
        if isinstance(effective_batch, bool) or not isinstance(effective_batch, int) or effective_batch < 1:
            raise ValueError("test.batch_size 必须是正整数或 null")
    configured_output = test.get("output")
    effective_output = output if output is not None else configured_output
    if effective_output is not None and not isinstance(effective_output, (str, Path)):
        raise ValueError("test.output 必须是路径或 null")
    return selected_metrics, save_predictions, effective_device, effective_steps, effective_batch, effective_output


def select_device(config: Mapping[str, Any], requested: str | None = None) -> torch.device:
    """Resolve ``--device`` while respecting the training accelerator default."""
    configured = config.get("test", {}).get("device")
    value = str(requested if requested is not None else (configured or "auto")).lower()
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
            "use a LR/HR/Ref dataset or a model/configuration that supports lr_up."
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


def _normalize_sample_ids(value: Any, batch_size: int) -> list[str]:
    """Normalize the DataLoader-collated sample IDs for one batch."""
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise TypeError(
            "test loader must return a string/list sample_id field; "
            f"got {type(value).__name__}"
        )
    if len(values) != batch_size or any(not isinstance(item, str) for item in values):
        raise ValueError(
            "sample_id count/type does not match the prediction batch: "
            f"ids={values!r}, batch_size={batch_size}"
        )
    result: list[str] = []
    for item in values:
        sample_id = item.strip()
        if not sample_id or sample_id in {".", ".."} or "/" in sample_id or "\\" in sample_id:
            raise ValueError(f"invalid sample_id for output filename: {item!r}")
        result.append(sample_id)
    return result


def _save_predictions(batch: torch.Tensor, output_dir: Path, sample_ids: list[str]) -> None:
    """Save predictions without collapsing non-RGB bands into a PNG mode.

    PNG remains convenient for grayscale/RGB previews. Four-band and wider
    predictions are scientific rasters, so keep their floating-point values
    in TIFF (or NPY when the optional TIFF dependency is unavailable).
    """
    if batch.ndim != 4:
        raise ValueError(f"prediction batch must be NCHW, got {tuple(batch.shape)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(sample_ids) != int(batch.shape[0]):
        raise ValueError(
            f"sample_id count {len(sample_ids)} does not match image batch {int(batch.shape[0])}"
        )
    channels = int(batch.shape[1])
    if channels < 1:
        raise ValueError("prediction batch must contain at least one channel")
    raster_writer = None
    if channels not in {1, 3}:
        try:
            import tifffile  # type: ignore

            raster_writer = tifffile.imwrite
        except ImportError:  # pragma: no cover - requirements include tifffile
            raster_writer = None
    for offset, image in enumerate(batch.detach().cpu()):
        array = image.permute(1, 2, 0).numpy()
        sample_id = sample_ids[offset]
        if channels in {1, 3}:
            png_array = (array * 255.0 + 0.5).clip(0, 255).astype("uint8")
            if channels == 1:
                png_array = png_array[..., 0]
            Image.fromarray(png_array).save(output_dir / f"{sample_id}.png")
            continue
        raster = array.astype(np.float32, copy=False)
        if raster_writer is not None:
            raster_writer(
                output_dir / f"{sample_id}.tif",
                raster,
                photometric="minisblack",
            )
        else:
            np.save(output_dir / f"{sample_id}.npy", raster)


def _build_refsr_model(
    config: Mapping[str, Any],
    checkpoint: Any,
    device: torch.device,
    *,
    raw_weights: bool = False,
):
    model_name = str(config.get("model", {}).get("name", "RefSRWKV")).lower()
    train_cfg = config.get("train", {})
    if not isinstance(train_cfg, Mapping):
        train_cfg = {}
    channel_adaptation = str(train_cfg.get("channel_adaptation", "none"))
    if model_name != "refdiffrwkv":
        model = build_refsr_model(
            config["model"], scale=int(config["data"]["scale"])
        )
        report = load_model_weights(
            model,
            checkpoint,
            prefer_ema=not raw_weights,
            channel_adaptation=channel_adaptation,
        )
        LOGGER.info("loaded direct RefSR checkpoint (%s): %s", model_name, report)
        return model.to(device).eval(), "minus_one_one", None

    # Share the builder with the training entry point so prior loading and
    # Stable-Diffusion construction use the same rules.
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
    checkpoint: str | Path | None = None,
    split: str = "test",
    output: str | Path | None = None,
    device: str | None = None,
    steps: int | None = None,
    batch_size: int | None = None,
    raw_weights: bool = False,
    metrics: Sequence[str] | str | None = None,
    save_images: bool | None = None,
) -> dict[str, Any]:
    """Run one split and write the configured metrics and optional PNGs.

    ``output`` is the test-run root; the selected split is always nested below
    it.  Omitting it uses ``test.output`` or the canonical
    ``experiments/test/...`` layout.  ``metrics`` and ``save_images`` override
    the corresponding ``test`` YAML settings for this call.
    """
    split = str(split).strip()
    if split not in VALID_SPLITS:
        raise ValueError("split 必须是 test、test_easy 或 test_hard")
    config = dict(config)
    validate_config(config)
    task = str(config.get("task", "")).lower()
    model_name = str(config.get("model", {}).get("name", "")).lower()
    if task not in {"sr", "refsr"}:
        task = "refsr" if is_refsr_model(model_name) else "sr"
    reference_mode = (
        normalize_reference_mode(config["data"].get("reference_mode", "paired"))
        if task == "refsr"
        else None
    )
    (
        selected_metrics,
        save_predictions,
        requested_device,
        configured_steps,
        configured_batch_size,
        configured_output,
    ) = _test_options(
        config,
        metrics=metrics,
        save_images=save_images,
        device=device,
        steps=steps,
        batch_size=batch_size,
        output=output,
    )
    selected_device = select_device(config, requested_device)
    if not selected_metrics and not save_predictions:
        raise ValueError("test.metrics 为空且 test.save_images=false；至少启用一项输出")
    is_bicubic = task == "sr" and model_name == "bicubic"
    if checkpoint is None and not is_bicubic:
        raise ValueError("--checkpoint is required for every trainable SR/RefSR model; Bicubic is the only checkpoint-free baseline")
    checkpoint_path = resolve_path(checkpoint, prefer_cwd=True) if checkpoint is not None else None
    # A Bicubic run may still receive a checkpoint solely to supply the
    # embedded dataset/model metadata to the standalone test config.  It has
    # no learned weights and must never try to load that checkpoint into the
    # parameter-free adapter.
    checkpoint_obj = (
        load_checkpoint(checkpoint_path)
        if checkpoint_path is not None and not is_bicubic
        else None
    )

    if task == "sr":
        loader = build_sr_test_loader(config, split=split, batch_size=configured_batch_size)
        model = build_sr_model(config["model"], scale=int(config["data"]["scale"]))
        if checkpoint_obj is not None:
            report = load_model_weights(model, checkpoint_obj, prefer_ema=not raw_weights)
            LOGGER.info("loaded SR checkpoint: %s", report)
        else:
            LOGGER.info("running parameter-free Bicubic baseline without a checkpoint")
        model = model.to(selected_device).eval()
        value_range, generator = "minus_one_one", None
    else:
        assert checkpoint_obj is not None
        loader = build_refsr_test_loader(config, split=split, batch_size=configured_batch_size)
        model, value_range, generator = _build_refsr_model(
            config, checkpoint_obj, selected_device, raw_weights=raw_weights
        )

    layout = layout_from_config(dict(config))
    test_root = resolve_path(configured_output, prefer_cwd=True) if configured_output else layout.test_dir
    split_root = test_root / split
    image_root = split_root / "images"
    split_root.mkdir(parents=True, exist_ok=True)
    if save_predictions:
        image_root.mkdir(parents=True, exist_ok=True)
    scale = int(config["data"]["scale"])
    sample_count = 0
    sample_ids: list[str] = []
    metric_values: dict[str, list[float]] = {name: [] for name in selected_metrics}
    wuhan_values: dict[str, list[float]] = {
        key: [] for key in ("rmse", "uiqi", "psnr", "sam_rad", "sam_deg", "ergas")
    }
    wuhan_band_rmse: list[list[float]] = []
    dataset_meta = config.get("dataset", {})
    if not isinstance(dataset_meta, Mapping):
        dataset_meta = {}
    data_meta = config.get("data", {})
    if not isinstance(data_meta, Mapping):
        data_meta = {}
    wuhan_run = any(
        "wuhan" in str(value).strip().lower()
        for value in (
            dataset_meta.get("id"),
            dataset_meta.get("kind"),
            data_meta.get("dataset_format"),
            data_meta.get("dataset_kind"),
            data_meta.get("format"),
            str(data_meta.get("root", "")).split("/")[-1],
        )
        if value is not None
    )
    unsupported_wuhan = sorted(WUHAN_METRICS.intersection(selected_metrics) - set(WUHAN_METRICS if wuhan_run else ()))
    if unsupported_wuhan:
        raise ValueError(
            f"metrics {unsupported_wuhan} 只适用于 Wuhan 数据集；当前数据集不是 Wuhan"
        )
    wuhan_metric_keys = {
        "rmse": "rmse",
        "uiqi": "uiqi",
        "psnr_wuhan": "psnr",
        "sam_rad": "sam_rad",
        "sam_deg": "sam_deg",
        "ergas": "ergas",
    }
    requested_wuhan_keys = {
        wuhan_metric_keys[name] for name in selected_metrics if name in wuhan_metric_keys
    }
    wuhan_ratio = float(data_meta.get("physical_resolution_ratio", 30.0 / 8.0))
    eval_tile_size = data_meta.get("eval_tile_size") if wuhan_run else None
    eval_tile_overlap = data_meta.get("eval_tile_overlap", 0) if wuhan_run else 0
    if eval_tile_size is not None and (
        isinstance(eval_tile_size, bool) or not isinstance(eval_tile_size, int) or eval_tile_size < 1
    ):
        raise ValueError("data.eval_tile_size must be a positive integer or null")
    if (
        isinstance(eval_tile_overlap, bool)
        or not isinstance(eval_tile_overlap, int)
        or eval_tile_overlap < 0
        or (eval_tile_size is not None and eval_tile_overlap >= eval_tile_size)
    ):
        raise ValueError("data.eval_tile_overlap must be in [0, data.eval_tile_size)")
    inference_steps = int(
        configured_steps
        if configured_steps is not None
        else config.get("model", {}).get("sample_steps", 20)
    )
    if inference_steps < 1:
        raise ValueError("steps 必须为正整数")

    with torch.inference_mode():
        for batch in loader:
            batch = _move_batch(batch, selected_device)
            lr, hr = batch[config["data"].get("lr_key", "lr")], batch[config["data"].get("hr_key", "hr")]
            batch_sample_ids = _normalize_sample_ids(batch.get("sample_id"), int(hr.shape[0]))
            if task == "sr":
                prediction = tiled_forward(
                    model,
                    lr,
                    scale=scale,
                    tile_size=eval_tile_size,
                    overlap=eval_tile_overlap,
                )
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
                prediction = tiled_forward(
                    model,
                    lr,
                    ref,
                    scale=scale,
                    tile_size=eval_tile_size,
                    overlap=eval_tile_overlap,
                )
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
            if "psnr" in metric_values:
                metric_values["psnr"].extend(
                    per_image_psnr(prediction_metric, hr_metric).detach().cpu().tolist()
                )
            if "ssim" in metric_values:
                metric_values["ssim"].extend(
                    gaussian_ssim(prediction_metric, hr_metric).detach().cpu().tolist()
                )
            if wuhan_run and requested_wuhan_keys:
                values = wuhan_metric_tensors(
                    prediction_metric,
                    hr_metric,
                    resolution_ratio=wuhan_ratio,
                    value_range="minus_one_one",
                )
                for key in wuhan_values:
                    if key in requested_wuhan_keys:
                        wuhan_values[key].extend(values[key].detach().cpu().tolist())
                if "rmse" in requested_wuhan_keys:
                    wuhan_band_rmse.extend(values["rmse_per_band"].detach().cpu().tolist())
            if set(batch_sample_ids) & set(sample_ids):
                duplicate = sorted(set(batch_sample_ids) & set(sample_ids))
                raise ValueError(f"duplicate sample_id encountered during evaluation: {duplicate}")
            if save_predictions:
                _save_predictions(prediction_png, image_root, batch_sample_ids)
            sample_ids.extend(batch_sample_ids)
            sample_count += int(prediction_png.shape[0])

    if not sample_ids:
        raise RuntimeError(f"split {split!r} 没有产生任何样本")
    implementation = (
        "reference_only"
        if is_bicubic
        else str(config.get("model", {}).get("implementation", "native"))
    )
    variant = "bicubic" if is_bicubic else str(
        config.get("model", {}).get("variant", model_name)
    )
    metrics = {
        "task": task,
        "model": model_name,
        "implementation": implementation,
        "variant": variant,
        "dataset": str(config.get("dataset", {}).get("id", "dataset")),
        "scale": scale,
        "split": split,
        "reference_mode": reference_mode,
        "samples": sample_count,
        "sample_ids": sample_ids,
        "image_naming": "sample_id",
        "metrics": selected_metrics,
        "save_images": save_predictions,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "test_config": str(config.get("_test_config_path"))
        if config.get("_test_config_path")
        else None,
    }
    for name, values in metric_values.items():
        if name in WUHAN_METRICS:
            continue
        if not values:
            raise RuntimeError(f"metric {name!r} did not produce any values")
        metrics[name] = {"mean": sum(values) / len(values), "per_image": values}
    if wuhan_run and requested_wuhan_keys:
        wuhan_payload: dict[str, Any] = {
            "resolution_ratio": wuhan_ratio,
            "value_range": "reflectance [0,1]",
        }
        for key in ("rmse", "uiqi", "psnr", "sam_rad", "sam_deg", "ergas"):
            if key not in requested_wuhan_keys:
                continue
            values = wuhan_values[key]
            if not values:
                raise RuntimeError(f"Wuhan metric {key!r} did not produce any values")
            wuhan_payload[key] = {"mean": sum(values) / len(values), "per_image": values}
        if "rmse" in requested_wuhan_keys:
            wuhan_payload["rmse_per_band"] = (
                torch.as_tensor(wuhan_band_rmse, dtype=torch.float64).mean(dim=0).tolist()
                if wuhan_band_rmse
                else []
            )
        metrics["wuhan"] = wuhan_payload
        for name, key in wuhan_metric_keys.items():
            if name in selected_metrics and key in wuhan_payload:
                metrics[name] = wuhan_payload[key]
    metrics_path = split_root / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("saved %d predictions and metrics to %s", sample_count, split_root)
    return metrics


__all__ = ["VALID_SPLITS", "VALID_METRICS", "normalize_test_metrics", "run_inference", "select_device"]
