#!/usr/bin/env python3
"""Run RefSRWKV or RefDiffRWKV on one native RefSR test split."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.runner import run_inference
from runtime.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("test", "test_easy", "test_hard"))
    parser.add_argument("--output", default=None, help="test run root; split is written below it")
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--raw-weights", action="store_true", help="ignore EMA shadows")
    parser.add_argument("--overrides", nargs="*", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config = load_config(args.config, args.overrides)
    if str(config.get("task", "refsr")).lower() != "refsr":
        raise ValueError("scripts/test/refsr.py 只能用于 task: refsr 配置")
    result = run_inference(
        config,
        checkpoint=args.checkpoint,
        split=args.split,
        output=args.output,
        device=args.device,
        steps=args.steps,
        batch_size=args.batch_size,
        raw_weights=args.raw_weights,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
