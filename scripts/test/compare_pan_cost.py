#!/usr/bin/env python3
"""Compare FusionMamba (official full/cut) vs RDM-PAN (tiled) FLOPs and latency."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loaders import build_refsr_test_loader
from evaluation.runner import _build_refsr_model
from runtime.checkpoint import load_checkpoint
from runtime.config import load_test_config
from runtime.tiling import _starts, tiled_forward


def n_tiles(length: int, tile_size: int | None, overlap: int) -> int:
    if tile_size is None or length <= tile_size:
        return 1
    starts = _starts(length, tile_size, tile_size - overlap)
    return len(starts) ** 2


def count_params(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def load_ready_model(checkpoint: str | Path, device: torch.device):
    payload = load_checkpoint(checkpoint)
    config = load_test_config("configs/test/test.yaml", payload)
    config.setdefault("data", {})
    config["data"]["num_workers"] = 0
    config["data"]["val_num_workers"] = 0
    config["data"]["persistent_workers"] = False
    model, _, _ = _build_refsr_model(config, payload, device)
    model.eval()
    return model, config


def infer_fn(model, tile_size: int | None, overlap: int, scale: int):
    def _run(lr: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if tile_size is None:
            return model(lr, ref)
        return tiled_forward(
            model,
            lr,
            ref,
            scale=scale,
            tile_size=tile_size,
            overlap=overlap,
            input_scales=(1, scale),
        )

    return _run


@torch.inference_mode()
def measure_flops(run, lr: torch.Tensor, ref: torch.Tensor) -> float:
    from torch.utils.flop_counter import FlopCounterMode

    # One warmup so CUDA kernels are loaded before the counter starts.
    run(lr, ref)
    torch.cuda.synchronize()
    with FlopCounterMode(display=False) as counter:
        run(lr, ref)
        torch.cuda.synchronize()
        return float(counter.get_total_flops())


@torch.inference_mode()
def measure_time(
    run,
    loader,
    *,
    warmup: int,
    device: torch.device,
    max_samples: int | None,
) -> dict[str, float | int]:
    times: list[float] = []
    peak = 0
    seen = 0
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for batch in loader:
        lr = batch["lr"].to(device, non_blocking=True)
        ref = batch["ref"].to(device, non_blocking=True)
        if seen < warmup:
            run(lr, ref)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            seen += 1
            continue
        torch.cuda.reset_peak_memory_stats(device)
        starter.record()
        run(lr, ref)
        ender.record()
        torch.cuda.synchronize()
        times.append(float(starter.elapsed_time(ender)))
        peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
        seen += 1
        if max_samples is not None and (seen - warmup) >= max_samples:
            break
    if not times:
        raise RuntimeError("no timed samples; increase split size or lower --warmup")
    mean = sum(times) / len(times)
    return {
        "n": len(times),
        "ms_mean": mean,
        "ms_min": min(times),
        "ms_max": max(times),
        "images_per_s": 1000.0 / mean,
        "peak_mem_mb": peak / (1024 ** 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fusion-ckpt",
        default="experiments/train/refsr/fusion_mamba/pancollection_wv3/x4/fusion_mamba_pancollection_wv3_official/checkpoints/420.ckpt",
    )
    parser.add_argument(
        "--rdm-ckpt",
        default="experiments/train/refsr/rdm_pan/pancollection_wv3/x4/pancollection_wv3_rdm_pan_x4/checkpoints/last.ckpt",
    )
    parser.add_argument("--splits", nargs="+", default=["test", "test_hard"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all remaining after warmup")
    parser.add_argument("--output", default="experiments/test/refsr/compare_pan_cost.json")
    return parser.parse_args()


def protocols(name: str, split: str) -> list[tuple[str, int | None, int, str]]:
    """Return rows of (tag, lr_tile, overlap, note)."""
    if name == "fusion_mamba":
        if split == "test_hard":
            official = (64, 0, "official cut_size=256 PAN (LR tile=64, overlap=0)")
        else:
            official = (None, 0, "official full image")
        return [
            ("official", *official),
            ("tile16_ov8", 16, 8, "same tile as RDM-PAN: LR 16 / overlap 8 (PAN 64)"),
        ]
    return [
        ("tile16_ov8", 16, 8, "RDM-PAN tiled LR 16 / overlap 8 (PAN 64)"),
    ]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this comparison needs CUDA")
    device = torch.device("cuda")
    jobs = [
        ("fusion_mamba", args.fusion_ckpt),
        ("rdm_pan", args.rdm_ckpt),
    ]
    results = []
    print(
        f"{'model':<14} {'infer':<12} {'split':<10} {'MS':<12} {'tiles':>5} "
        f"{'params':>8} {'FLOPs':>10} {'ms/img':>8} {'img/s':>7} {'peakMB':>8}  protocol"
    )
    for name, ckpt in jobs:
        model, config = load_ready_model(ckpt, device)
        params = count_params(model)
        scale = int(config["data"]["scale"])
        for split in args.splits:
          for tag, tile, overlap, note in protocols(name, split):
            run = infer_fn(model, tile, overlap, scale)
            loader = build_refsr_test_loader(config, split=split, batch_size=1)
            first = next(iter(loader))
            lr = first["lr"].to(device)
            ref = first["ref"].to(device)
            tiles = n_tiles(int(lr.shape[-2]), tile, overlap)
            flops = measure_flops(run, lr, ref)
            timed = measure_time(
                run,
                build_refsr_test_loader(config, split=split, batch_size=1),
                warmup=args.warmup,
                device=device,
                max_samples=None if args.max_samples <= 0 else args.max_samples,
            )
            row = {
                "model": name,
                "infer": tag,
                "checkpoint": str(Path(ckpt).resolve()),
                "split": split,
                "lr_shape": list(lr.shape),
                "ref_shape": list(ref.shape),
                "tile_size": tile,
                "overlap": overlap,
                "tiles": tiles,
                "params": params,
                "flops": flops,
                "protocol": note,
                **timed,
            }
            results.append(row)
            ms = f"{lr.shape[-2]}x{lr.shape[-1]}"
            print(
                f"{name:<14} {tag:<12} {split:<10} {ms:<12} {tiles:>5} "
                f"{params/1e6:7.2f}M {flops/1e9:8.2f}G {timed['ms_mean']:7.1f} "
                f"{timed['images_per_s']:7.2f} {timed['peak_mem_mb']:7.0f}  {note}"
            )
        del model
        torch.cuda.empty_cache()

    out = Path(args.output)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
