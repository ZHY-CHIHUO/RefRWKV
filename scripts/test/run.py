#!/usr/bin/env python3
"""Config-driven test/inference entry point for SR and RefSR models.

The YAML ``test`` section controls splits, metrics, image writing, device,
batch size, sampling steps, and output root. CLI flags are one-run overrides.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.runner import run_inference
from runtime.checkpoint import load_checkpoint
from runtime.config import load_config, load_test_config, load_test_yaml, merge_test_policy
from runtime.common import resolve_path
from runtime.experiments import layout_from_config


def _configured_splits(config: dict) -> list[str]:
    test_cfg = config.get("test", {})
    raw = test_cfg.get("split", test_cfg.get("splits", ["test"]))
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("test.split(s) 必须是非空 split 列表")
    allowed = {"test", "test_easy", "test_hard"}
    splits = [str(item).strip() for item in raw]
    if any(item not in allowed for item in splits):
        raise ValueError("test.split(s) 只能是 test、test_easy 或 test_hard")
    if len(set(splits)) != len(splits):
        raise ValueError("test.split(s) 不能重复")
    return splits


def _write_snapshot(
    config: dict,
    *,
    config_path: Path,
    checkpoint_path: Path | None,
    training_config_path: Path | None,
    split: str,
    output_root: Path,
    overrides: list[str] | None = None,
) -> Path:
    """Write an editable split policy once, without overwriting user edits."""
    test_policy = copy.deepcopy(config["test"])
    test_policy["splits"] = [split]
    payload = {
        "test": test_policy,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "source_config": str(config_path),
        "split": split,
    }
    if training_config_path is not None:
        payload["training_config"] = str(training_config_path)
    if overrides:
        payload["overrides"] = list(overrides)
    target = output_root / f"{split}.yaml"
    output_root.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="训练模型 checkpoint；Bicubic 可省略并改用 --training-config",
    )
    parser.add_argument(
        "--training-config",
        default=None,
        help="仅用于无 checkpoint 的参数免费基线（当前为 Bicubic）",
    )
    parser.add_argument("--split", action="append", dest="splits", choices=("test", "test_easy", "test_hard"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--metrics", nargs="+", default=None, help="metric names; overrides test.metrics")
    parser.add_argument("--save-images", dest="save_images", action="store_true")
    parser.add_argument("--no-save-images", dest="save_images", action="store_false")
    parser.set_defaults(save_images=None)
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument("--overrides", nargs="*", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    checkpoint_path = (
        resolve_path(args.checkpoint, prefer_cwd=True)
        if args.checkpoint is not None
        else None
    )
    if checkpoint_path is not None and args.training_config is not None:
        parser.error("--checkpoint 和 --training-config 只能指定一个")
    checkpoint = load_checkpoint(checkpoint_path) if checkpoint_path is not None else None
    training_config_path: Path | None = None
    if checkpoint is not None:
        # The command-line config is the bootstrap policy.  Once a split has
        # been evaluated, its editable policy under the output root becomes
        # the source of truth for later runs (similar to train/config.yaml).
        bootstrap_config = load_test_config(args.config, checkpoint, args.overrides)
    elif args.training_config is not None:
        training_config_path = resolve_path(args.training_config, prefer_cwd=True)
        training_config = load_config(training_config_path, args.overrides)
        test_source, test_config_path = load_test_yaml(args.config, args.overrides)
        bootstrap_config = merge_test_policy(
            training_config,
            test_source,
            test_config_path=test_config_path,
        )
    else:
        parser.error("训练模型需要 --checkpoint；无 checkpoint 的测试仅支持通过 --training-config 运行 Bicubic")
    bootstrap_test = bootstrap_config["test"]
    if args.splits:
        bootstrap_test["splits"] = list(args.splits)
    if args.metrics is not None:
        bootstrap_test["metrics"] = list(args.metrics)
    if args.save_images is not None:
        bootstrap_test["save_images"] = args.save_images
    if args.device is not None:
        bootstrap_test["device"] = args.device
    if args.steps is not None:
        bootstrap_test["steps"] = args.steps
    if args.batch_size is not None:
        bootstrap_test["batch_size"] = args.batch_size
    if args.output is not None:
        bootstrap_test["output"] = args.output
    splits = _configured_splits(bootstrap_config)
    bootstrap_layout = layout_from_config(bootstrap_config)
    configured_output = bootstrap_test.get("output")
    test_root = (
        resolve_path(configured_output, prefer_cwd=True)
        if configured_output
        else bootstrap_layout.test_dir
    )
    results = []
    for split in splits:
        snapshot_path = test_root / f"{split}.yaml"
        snapshot_exists = snapshot_path.is_file()
        source_config_path = snapshot_path if snapshot_exists else Path(args.config)
        if snapshot_exists:
            snapshot_source, snapshot_config_path = load_test_yaml(
                snapshot_path, args.overrides
            )
            config = merge_test_policy(
                bootstrap_config,
                snapshot_source,
                test_config_path=snapshot_config_path,
            )
        else:
            config = copy.deepcopy(bootstrap_config)
            config["_test_config_path"] = str(snapshot_path.resolve())
        test_cfg = config["test"]
        # A split invocation must never inherit another split from a snapshot.
        test_cfg["splits"] = [split]
        if args.metrics is not None:
            test_cfg["metrics"] = list(args.metrics)
        if args.save_images is not None:
            test_cfg["save_images"] = args.save_images
        if args.device is not None:
            test_cfg["device"] = args.device
        if args.steps is not None:
            test_cfg["steps"] = args.steps
        if args.batch_size is not None:
            test_cfg["batch_size"] = args.batch_size
        if args.output is not None:
            test_cfg["output"] = args.output
        effective_output = test_cfg.get("output")
        effective_test_root = (
            resolve_path(effective_output, prefer_cwd=True)
            if effective_output
            else test_root
        )
        snapshot_path = effective_test_root / f"{split}.yaml"
        config["_test_config_path"] = str(snapshot_path.resolve())
        _write_snapshot(
            config,
            config_path=source_config_path,
            checkpoint_path=checkpoint_path,
            training_config_path=training_config_path,
            split=split,
            output_root=effective_test_root,
            overrides=args.overrides,
        )
        result = run_inference(
            config,
            checkpoint=checkpoint_path,
            split=split,
            output=effective_output,
            device=None,
            steps=None,
            batch_size=None,
            raw_weights=args.raw_weights,
            metrics=None,
            save_images=None,
        )
        results.append(result)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
