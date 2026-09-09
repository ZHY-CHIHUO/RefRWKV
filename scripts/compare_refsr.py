#!/usr/bin/env python3
"""Compare paired RefSR/SR results by the original sample ID.

The report is deliberately based on per-image metrics from ``metrics.json``;
it never aligns images by experiment-directory names or by an accidental
filesystem traversal order.  Old metrics written before sample IDs were
added can be read with ``--dataset-root`` as an explicitly labelled legacy
fallback because the test loader's lexical ordering is deterministic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_RUNS = {
    "Bicubic": PROJECT_ROOT / "experiments/test/sr/bicubic/hrms_scd/x4/hrms_scd_bicubic_x4",
    "SwinIR": PROJECT_ROOT / "experiments/test/sr/swinir/hrms_scd/x4/hrms_scd_swinir_x4",
    "RefSRWKV-SR": PROJECT_ROOT / "experiments/test/refsr/refsrwkv/hrms_scd/x4/hrms_scd_sr_x4",
    "TRefSR": PROJECT_ROOT / "experiments/test/refsr/refsrwkv/hrms_scd/x4/hrms_scd_trefsr_x4",
}
SPLITS = ("test_easy", "test_hard")


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must use LABEL=TEST_RUN_ROOT")
    label, raw_path = value.split("=", 1)
    label, raw_path = label.strip(), raw_path.strip()
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("--run must use a non-empty LABEL=TEST_RUN_ROOT")
    return label, Path(raw_path).expanduser()


def _legacy_ids(dataset_root: Path | None, split: str, count: int) -> list[str] | None:
    if dataset_root is None:
        return None
    hr_dir = dataset_root / split / "HR"
    ids = sorted(path.stem for path in hr_dir.glob("*.png"))
    if len(ids) != count:
        raise ValueError(
            f"legacy fallback for {split} expected {count} HR files, found {len(ids)} in {hr_dir}"
        )
    return ids


def _load_run(
    label: str,
    root: Path,
    split: str,
    dataset_root: Path | None,
) -> tuple[dict[str, dict[str, float]], str]:
    path = root / split / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"{label}: metrics.json not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("psnr", {}).get("per_image")
    ssim_values = payload.get("ssim", {}).get("per_image")
    if not isinstance(values, list) or not isinstance(ssim_values, list):
        raise ValueError(f"{label}: {path} has no per-image PSNR/SSIM arrays")
    if len(values) != len(ssim_values):
        raise ValueError(f"{label}: PSNR/SSIM per-image lengths differ in {path}")
    ids = payload.get("sample_ids")
    source = "metrics.sample_ids"
    if not isinstance(ids, list) or len(ids) != len(values) or any(not isinstance(item, str) for item in ids):
        ids = _legacy_ids(dataset_root, split, len(values))
        if ids is None:
            raise ValueError(
                f"{label}: {path} has no valid sample_ids. Re-run evaluation with the current code "
                "or pass --dataset-root for the deterministic legacy-order fallback."
            )
        source = "legacy lexical HR order (compatibility fallback)"
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label}: duplicate sample_ids in {path}")
    result = {
        str(sample_id): {"psnr": float(psnr), "ssim": float(ssim)}
        for sample_id, psnr, ssim in zip(ids, values, ssim_values)
    }
    return result, source


def _paired_runs(
    runs: dict[str, Path], split: str, dataset_root: Path | None
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, str]]:
    loaded: dict[str, dict[str, dict[str, float]]] = {}
    sources: dict[str, str] = {}
    for label, root in runs.items():
        loaded[label], sources[label] = _load_run(label, root, split, dataset_root)
    expected = set(next(iter(loaded.values())))
    for label, values in loaded.items():
        actual = set(values)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"{split}: sample IDs do not match for {label}; "
                f"missing={missing[:8]} extra={extra[:8]}"
            )
    return loaded, sources


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _paired_t(delta: list[float]) -> tuple[float, float]:
    if len(delta) < 2:
        return float("nan"), float("nan")
    mean = _mean(delta)
    variance = sum((value - mean) ** 2 for value in delta) / (len(delta) - 1)
    std = math.sqrt(variance)
    if std == 0.0:
        return (float("inf") if mean > 0 else float("-inf") if mean < 0 else 0.0), 0.0 if mean else 1.0
    try:
        from scipy.stats import ttest_1samp
    except ImportError as exc:  # pragma: no cover - depends on optional eval env
        raise RuntimeError(
            "paired t-tests require scipy; install the evaluation dependencies or run in rwkv7"
        ) from exc
    result = ttest_1samp(delta, popmean=0.0)
    return float(result.statistic), float(result.pvalue)


def _summary(delta: list[float]) -> dict[str, float | int]:
    ordered = sorted(delta)
    t_stat, p_value = _paired_t(delta)
    def percentile(q: float) -> float:
        position = (len(ordered) - 1) * q
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "n": len(delta),
        "mean": _mean(delta),
        "median": percentile(0.5),
        "std": math.sqrt(sum((value - _mean(delta)) ** 2 for value in delta) / max(1, len(delta) - 1)),
        "min": ordered[0],
        "max": ordered[-1],
        "p05": percentile(0.05),
        "p25": percentile(0.25),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "improved": sum(value > 0.0 for value in delta) / len(delta),
        "t_stat": t_stat,
        "p_value": p_value,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _write_histogram(path: Path, deltas: dict[str, list[float]], split: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional eval env
        raise RuntimeError(
            "histogram output requires matplotlib; install the evaluation dependencies"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 5.0))
    for label, values in deltas.items():
        plt.hist(values, bins=30, alpha=0.42, density=True, label=label)
    plt.axvline(0.0, color="black", linewidth=1.0)
    plt.xlabel("Per-image PSNR difference (dB)")
    plt.ylabel("Density")
    plt.title(f"Paired PSNR differences: {split}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _write_split_report(
    path: Path,
    split: str,
    loaded: dict[str, dict[str, dict[str, float]]],
    sources: dict[str, str],
) -> dict[str, dict[str, float | int]]:
    labels = list(loaded)
    sample_ids = sorted(next(iter(loaded.values())))
    tref_label = "TRefSR"
    if tref_label not in loaded:
        raise ValueError("comparison requires a run labelled TRefSR")
    lines = [
        f"# {split} paired comparison",
        "",
        "All rows use the same sample IDs. PSNR/SSIM are means over per-image values.",
        "",
        "| Method | N | PSNR (dB) | SSIM | ΔPSNR vs Bicubic (dB) | ΔPSNR vs SwinIR (dB) | ΔPSNR vs RefSRWKV-SR (dB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    means = {
        label: {
            "psnr": _mean([loaded[label][sample_id]["psnr"] for sample_id in sample_ids]),
            "ssim": _mean([loaded[label][sample_id]["ssim"] for sample_id in sample_ids]),
        }
        for label in labels
    }
    for label in labels:
        def delta_vs(baseline: str) -> float:
            baseline_mean = means.get(baseline)
            return means[label]["psnr"] - baseline_mean["psnr"] if baseline_mean is not None else float("nan")
        lines.append(
            f"| {label} | {len(sample_ids)} | {_fmt(means[label]['psnr'])} | {_fmt(means[label]['ssim'])} | "
            f"{_fmt(delta_vs('Bicubic'), 4)} | "
            f"{_fmt(delta_vs('SwinIR'), 4)} | "
            f"{_fmt(delta_vs('RefSRWKV-SR'), 4)} |"
        )
    lines.extend([
        "",
        "## Per-image significance",
        "",
        "`improved` is the fraction of images with ΔPSNR > 0. The p-value is a two-sided paired t-test over the same image IDs.",
        "",
        "| Comparison | Mean Δ (dB) | Median Δ (dB) | Improved | t | p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    summaries: dict[str, dict[str, float | int]] = {}
    comparisons = [
        ("TRefSR - Bicubic", "Bicubic"),
        ("TRefSR - SwinIR", "SwinIR"),
        ("TRefSR - RefSRWKV-SR", "RefSRWKV-SR"),
    ]
    histogram_deltas: dict[str, list[float]] = {}
    for name, baseline in comparisons:
        if baseline not in loaded:
            continue
        delta = [loaded[tref_label][sample_id]["psnr"] - loaded[baseline][sample_id]["psnr"] for sample_id in sample_ids]
        summary = _summary(delta)
        summaries[name] = summary
        histogram_deltas[name] = delta
        lines.append(
            f"| {name} | {_fmt(summary['mean'])} | {_fmt(summary['median'])} | "
            f"{_fmt(summary['improved'] * 100, 2)}% | {_fmt(summary['t_stat'], 3)} | {_fmt(summary['p_value'], 6)} |"
        )
    lines.extend([
        "",
        "## Pairing provenance",
        "",
    ])
    for label in labels:
        lines.append(f"- {label}: `{sources[label]}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summaries, histogram_deltas, sample_ids


def _write_per_image(
    path: Path,
    sample_ids: list[str],
    loaded: dict[str, dict[str, dict[str, float]]],
) -> None:
    labels = list(loaded)
    fields = ["sample_id"] + [f"{label}_psnr" for label in labels] + [f"{label}_ssim" for label in labels]
    fields.extend([
        "TRefSR_minus_Bicubic",
        "TRefSR_minus_SwinIR",
        "TRefSR_minus_RefSRWKV-SR",
    ])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_id in sample_ids:
            row: dict[str, Any] = {"sample_id": sample_id}
            for label in labels:
                row[f"{label}_psnr"] = loaded[label][sample_id]["psnr"]
                row[f"{label}_ssim"] = loaded[label][sample_id]["ssim"]
            for baseline in ("Bicubic", "SwinIR", "RefSRWKV-SR"):
                key = f"TRefSR_minus_{baseline}"
                row[key] = (
                    loaded["TRefSR"][sample_id]["psnr"] - loaded[baseline][sample_id]["psnr"]
                    if baseline in loaded
                    else ""
                )
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_run, help="LABEL=TEST_RUN_ROOT; repeatable")
    parser.add_argument("--dataset-root", default="data/refsr/HRMS_SCD", help="used only for legacy metrics without sample_ids")
    parser.add_argument("--output", default="experiments/test/comparison/hrms_scd_x4")
    parser.add_argument("--split", action="append", choices=SPLITS, dest="splits")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    runs = dict(args.run) if args.run else dict(DEFAULT_RUNS)
    output = Path(args.output).expanduser()
    dataset_root = Path(args.dataset_root).expanduser() if args.dataset_root else None
    for split in args.splits or list(SPLITS):
        loaded, sources = _paired_runs(runs, split, dataset_root)
        report_path = output / f"{split}.md"
        summaries, histogram_deltas, sample_ids = _write_split_report(report_path, split, loaded, sources)
        _write_per_image(output / f"{split}_per_image.csv", sample_ids, loaded)
        if not args.no_plots:
            _write_histogram(output / f"{split}_delta_histogram.png", histogram_deltas, split)
        (output / f"{split}_summary.json").write_text(
            json.dumps({"split": split, "samples": len(sample_ids), "comparisons": summaries, "sources": sources}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(report_path)


if __name__ == "__main__":
    main()
