#!/usr/bin/env python3
"""Evaluate the official FusionMamba WV3 checkpoint on PanCollection H5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.runner import run_inference
from runtime.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", required=True, help="official FusionMamba checkpoint, e.g. weights/420.pth")
    parser.add_argument("--config", default="configs/runs/refsr/fusion_mamba_pancollection_wv3_x4.yaml")
    parser.add_argument("--split", default="test", choices=("test", "test_easy", "test_hard"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-images", dest="save_images", action="store_true")
    parser.add_argument("--no-save-images", dest="save_images", action="store_false")
    parser.set_defaults(save_images=False)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config).resolve())
    if args.batch_size is not None:
        config["test"]["batch_size"] = args.batch_size
    config["test"]["device"] = args.device
    config["test"]["save_images"] = bool(args.save_images)
    if args.output is not None:
        config["test"]["output"] = args.output
    if args.max_samples is not None:
        config["data"]["max_samples_test"] = args.max_samples

    result = run_inference(
        config,
        checkpoint=Path(args.weight).resolve(),
        split=args.split,
        batch_size=config["test"].get("batch_size"),
        device=config["test"].get("device"),
        save_images=config["test"].get("save_images"),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
