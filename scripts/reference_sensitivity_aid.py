#!/usr/bin/env python3
"""Run a zero-shot AID reference-sensitivity scan with an HRMS checkpoint.

The experiment keeps the HRMS-trained RefSRWKV model and AID LR/HR pairs
fixed.  For every image it constructs one sensor-mismatched reference from
HR, then mixes that reference with bicubic(LR) using ``p``.  The synthetic
reference includes a fixed RGB spectral response, seeded photometric changes,
and an independent seeded translation of 2--4 pixels in each axis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loaders import build_sr_test_loader
from evaluation.runner import (
    _build_refsr_model,
    _image_tensor,
    _move_batch,
    select_device,
)
from runtime.checkpoint import load_checkpoint
from runtime.common import gaussian_ssim, per_image_psnr, resolve_path
from runtime.config import load_test_config, validate_config
from runtime.tiling import tiled_forward


LOGGER = logging.getLogger(__name__)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "experiments/train/refsr/refsrwkv/hrms_scd/x4/hrms_scd_trefsr_x4/checkpoints/last.ckpt"
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs/test/aid_reference_sensitivity.yaml"
DEFAULT_SR_METRICS = (
    PROJECT_ROOT
    / "experiments/test/refsr/refsrwkv/aid/x4/hrms_scd_sr_x4_zero_shot/test/metrics.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments/test/comparison/aid_reference_sensitivity"
DEFAULT_P = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SHIFT_MAGNITUDES = (-4, -3, -2, 2, 3, 4)
SPECTRAL_RESPONSE = (
    (0.86, 0.10, 0.04),
    (0.05, 0.90, 0.05),
    (0.03, 0.11, 0.86),
)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _pearson_per_image(reference: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    ref_flat = reference.float().flatten(1)
    hr_flat = hr.float().flatten(1)
    ref_centered = ref_flat - ref_flat.mean(dim=1, keepdim=True)
    hr_centered = hr_flat - hr_flat.mean(dim=1, keepdim=True)
    numerator = (ref_centered * hr_centered).sum(dim=1)
    denominator = torch.sqrt(
        ref_centered.square().sum(dim=1) * hr_centered.square().sum(dim=1)
    ).clamp_min(1.0e-12)
    return (numerator / denominator).clamp(-1.0, 1.0)


def _paired_t_pvalue(values: Sequence[float]) -> float:
    """Return the two-sided paired t p-value, if scipy is available."""
    if len(values) < 2:
        return float("nan")
    try:
        from scipy.stats import ttest_1samp
    except ImportError:  # pragma: no cover - optional reporting dependency
        return float("nan")
    return float(ttest_1samp(values, 0.0).pvalue)


def _load_metrics(path: Path) -> tuple[dict[str, float], list[str], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("sample_ids")
    values = payload.get("psnr", {}).get("per_image")
    if not isinstance(ids, list) or not isinstance(values, list) or len(ids) != len(values):
        raise ValueError(f"baseline metrics must contain aligned sample_ids and psnr.per_image: {path}")
    if any(not isinstance(item, str) for item in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"baseline sample_ids are invalid or duplicated: {path}")
    result = {sample_id: float(value) for sample_id, value in zip(ids, values)}
    return result, list(ids), float(payload["psnr"]["mean"])


def _seed_for_sample(sample_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _translate_replicate(image: torch.Tensor, *, dy: int, dx: int) -> torch.Tensor:
    """Translate NCHW images with edge replication, never circular wrapping."""
    if image.ndim != 3:
        raise ValueError(f"expected CHW image, got {tuple(image.shape)}")
    height, width = int(image.shape[-2]), int(image.shape[-1])
    left, right = max(dx, 0), max(-dx, 0)
    top, bottom = max(dy, 0), max(-dy, 0)
    padded = F.pad(image.unsqueeze(0), (left, right, top, bottom), mode="replicate")
    return padded[..., :height, :width].squeeze(0)


def _synthetic_reference(
    hr: torch.Tensor,
    degraded: torch.Tensor,
    sample_ids: Sequence[str],
    *,
    seed: int,
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    """Build the strongest synthetic sensor reference and its provenance."""
    matrix = torch.tensor(SPECTRAL_RESPONSE, dtype=hr.dtype, device=hr.device)
    hr01 = ((hr.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    degraded01 = ((degraded.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    spectral = torch.einsum("ij,bjhw->bihw", matrix, hr01)
    transformed: list[torch.Tensor] = []
    provenance: list[dict[str, float | int]] = []
    for offset, sample_id in enumerate(sample_ids):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_seed_for_sample(sample_id, seed))
        brightness = 0.88 + 0.24 * float(torch.rand((), generator=generator))
        contrast = 0.90 + 0.20 * float(torch.rand((), generator=generator))
        gamma = 0.95 + 0.10 * float(torch.rand((), generator=generator))
        dy = int(SHIFT_MAGNITUDES[int(torch.randint(len(SHIFT_MAGNITUDES), (), generator=generator))])
        dx = int(SHIFT_MAGNITUDES[int(torch.randint(len(SHIFT_MAGNITUDES), (), generator=generator))])
        image = spectral[offset]
        image = ((image - 0.5) * contrast + 0.5) * brightness
        image = image.clamp(0.0, 1.0).pow(gamma)
        image = _translate_replicate(image, dy=dy, dx=dx)
        transformed.append(image)
        provenance.append(
            {
                "sample_id": sample_id,
                "brightness": brightness,
                "contrast": contrast,
                "gamma": gamma,
                "shift_dy": dy,
                "shift_dx": dx,
            }
        )
    sensor = torch.stack(transformed, dim=0).to(hr.dtype)
    return sensor * 2.0 - 1.0, provenance


def _parse_p(values: list[str] | None) -> list[float]:
    raw = DEFAULT_P if values is None else tuple(float(value) for value in values)
    result = sorted(set(raw))
    if not result or any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in result):
        raise ValueError("p values must be finite numbers in [0, 1]")
    return result


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "p",
        "control",
        "samples",
        "psnr_mean",
        "psnr_std",
        "ssim_mean",
        "ssim_std",
        "delta_psnr_mean",
        "delta_psnr_std",
        "improved_fraction",
        "paired_t_pvalue",
        "reference_correlation_mean",
        "reference_correlation_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_per_image_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "p",
        "psnr",
        "ssim",
        "baseline_psnr",
        "delta_psnr",
        "reference_correlation",
        "brightness",
        "contrast",
        "gamma",
        "shift_dy",
        "shift_dx",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional reporting dependency
        LOGGER.warning("matplotlib is unavailable; skipping the sensitivity plot")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plotted = [row for row in rows if float(row["p"]) > 0.0]
    plt.figure(figsize=(7.0, 4.8))
    plt.plot(
        [float(row["p"]) for row in plotted],
        [float(row["delta_psnr_mean"]) for row in plotted],
        marker="o",
        linewidth=2.0,
    )
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xlabel("Reference intensity p")
    plt.ylabel("TRefSR - SR PSNR (dB)")
    plt.title("AID reference sensitivity (HRMS zero-shot)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _prepare_config(source: Mapping[str, Any], *, samples: int, seed: int) -> dict[str, Any]:
    config = json.loads(json.dumps(source))
    config["task"] = "refsr"
    dataset = config.setdefault("dataset", {})
    dataset.update(
        {
            "id": "aid",
            "name": "AID (Aerial Image Dataset)",
            "kind": "synthetic_sisr",
            "root": "data/sr/AID",
        }
    )
    data = config.setdefault("data", {})
    data.update(
        {
            "root": "data/sr/AID",
            "reference_mode": "lr_up",
            "max_samples_test": int(samples),
            "sample_seed": int(seed),
            "lr_source": "auto",
            "lr_native_scale": 4,
            "lr_provenance": "bicubic",
        }
    )
    for key in ("augment_ref", "ref_aug_strengths", "ref_aug_probs", "ref_gray_prob"):
        data.pop(key, None)
    loss = config.get("loss")
    if isinstance(loss, dict):
        loss.pop("ref_drop_prob", None)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--sr-metrics", default=str(DEFAULT_SR_METRICS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--p", action="append", dest="p_values", help="reference intensity in [0,1]")
    parser.add_argument("--device", default=None)
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument("--overrides", nargs="*", default=None)
    args = parser.parse_args()
    if args.samples < 1 or args.batch_size < 1:
        parser.error("--samples and --batch-size must be positive")
    p_values = _parse_p(args.p_values)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    checkpoint_path = resolve_path(args.checkpoint, prefer_cwd=True)
    checkpoint = load_checkpoint(checkpoint_path)
    source = load_test_config(args.config, checkpoint, args.overrides)
    config = _prepare_config(source, samples=args.samples, seed=args.seed)
    config.setdefault("test", {})["device"] = args.device or config.get("test", {}).get("device") or "auto"
    config["test"]["batch_size"] = args.batch_size
    validate_config(config)
    device = select_device(config, args.device)
    model, value_range, generator = _build_refsr_model(
        config, checkpoint, device, raw_weights=args.raw_weights
    )
    if value_range != "minus_one_one" or generator is not None:
        raise ValueError("reference_sensitivity_aid.py supports direct RefSR models only")

    baseline_path = resolve_path(args.sr_metrics, prefer_cwd=True)
    baseline, baseline_ids, baseline_mean = _load_metrics(baseline_path)
    loader = build_sr_test_loader(config, split="test", batch_size=args.batch_size)
    output = resolve_path(args.output, prefer_cwd=True)
    output.mkdir(parents=True, exist_ok=True)
    per_image_rows: list[dict[str, Any]] = []
    grouped: dict[float, dict[str, list[float]]] = {
        p: {key: [] for key in ("psnr", "ssim", "delta", "correlation")} for p in p_values
    }
    seen: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            batch = _move_batch(batch, device)
            lr = batch[config["data"].get("lr_key", "lr")]
            hr = batch[config["data"].get("hr_key", "hr")]
            sample_ids = batch.get("sample_id")
            if isinstance(sample_ids, str):
                sample_ids = [sample_ids]
            elif isinstance(sample_ids, (tuple, list)):
                sample_ids = list(sample_ids)
            else:
                raise TypeError("AID test loader must return sample_id")
            if len(sample_ids) != int(hr.shape[0]):
                raise ValueError("sample_id count does not match batch size")
            missing = [sample_id for sample_id in sample_ids if sample_id not in baseline]
            if missing:
                raise ValueError(f"baseline metrics are missing AID sample IDs: {missing[:5]}")
            seen.extend(sample_ids)
            expected = (int(lr.shape[-2]) * 4, int(lr.shape[-1]) * 4)
            degraded = F.interpolate(lr, size=expected, mode="bicubic", align_corners=False)
            sensor, provenance = _synthetic_reference(
                hr,
                degraded,
                sample_ids,
                seed=args.seed,
            )
            hr_metric, _ = _image_tensor(hr, value_range="minus_one_one")
            for p in p_values:
                reference = ((1.0 - p) * ((degraded + 1.0) * 0.5) + p * ((sensor + 1.0) * 0.5))
                reference = reference.clamp(0.0, 1.0) * 2.0 - 1.0
                prediction = tiled_forward(model, lr, reference, scale=4)
                prediction_metric, _ = _image_tensor(prediction, value_range="minus_one_one")
                psnr_values = per_image_psnr(prediction_metric, hr_metric).detach().cpu().tolist()
                ssim_values = gaussian_ssim(prediction_metric, hr_metric).detach().cpu().tolist()
                correlations = _pearson_per_image(reference, hr).detach().cpu().tolist()
                for index, sample_id in enumerate(sample_ids):
                    delta = float(psnr_values[index]) - baseline[sample_id]
                    grouped[p]["psnr"].append(float(psnr_values[index]))
                    grouped[p]["ssim"].append(float(ssim_values[index]))
                    grouped[p]["delta"].append(delta)
                    grouped[p]["correlation"].append(float(correlations[index]))
                    per_image_rows.append(
                        {
                            "sample_id": sample_id,
                            "p": p,
                            "psnr": float(psnr_values[index]),
                            "ssim": float(ssim_values[index]),
                            "baseline_psnr": baseline[sample_id],
                            "delta_psnr": delta,
                            "reference_correlation": float(correlations[index]),
                            **provenance[index],
                        }
                    )
    if len(seen) != len(set(seen)):
        raise ValueError("duplicate sample IDs in AID loader")
    extra = sorted(set(seen) - set(baseline_ids))
    if extra:
        raise ValueError(f"AID sample set mismatch: unexpected={extra[:5]}")

    summary_rows: list[dict[str, Any]] = []
    for p in p_values:
        values = grouped[p]
        deltas = values["delta"]
        summary_rows.append(
            {
                "p": p,
                "control": p == 0.0,
                "samples": len(deltas),
                "psnr_mean": _mean(values["psnr"]),
                "psnr_std": _std(values["psnr"]),
                "ssim_mean": _mean(values["ssim"]),
                "ssim_std": _std(values["ssim"]),
                "delta_psnr_mean": _mean(deltas),
                "delta_psnr_std": _std(deltas),
                "improved_fraction": sum(value > 0.0 for value in deltas) / len(deltas),
                "paired_t_pvalue": _paired_t_pvalue(deltas),
                "reference_correlation_mean": _mean(values["correlation"]),
                "reference_correlation_std": _std(values["correlation"]),
            }
        )

    protocol = {
        "experiment": "HRMS-trained RefSRWKV zero-shot reference sensitivity on AID",
        "checkpoint": str(checkpoint_path),
        "baseline_sr_metrics": str(baseline_path),
        "samples": args.samples,
        "sample_seed": args.seed,
        "split": "AID/test",
        "p_values": p_values,
        "p_zero_is_control": True,
        "reference_definition": "Ref(p) = (1-p) * bicubic(LR) + p * SensorMismatch(HR)",
        "spectral_response": SPECTRAL_RESPONSE,
        "photometric_ranges": {
            "brightness": [0.88, 1.12],
            "contrast": [0.90, 1.10],
            "gamma": [0.95, 1.05],
        },
        "translation": "independent dy/dx chosen from {-4,-3,-2,2,3,4}; edge replication",
        "baseline_psnr_mean": baseline_mean,
        "device": str(device),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "protocol": protocol,
                "rows": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_summary_csv(output / "summary.csv", summary_rows)
    _write_per_image_csv(output / "per_image.csv", per_image_rows)
    _write_plot(output / "delta_psnr_vs_p.png", summary_rows)
    LOGGER.info("saved AID reference sensitivity results to %s", output)
    print(json.dumps({"output": str(output), "rows": summary_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
