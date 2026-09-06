#!/usr/bin/env python3
"""Materialize a layered experiment config into an editable YAML file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.config import load_config
from runtime.experiments import layout_from_config


def _write_yaml(config: dict, target: Path, *, force: bool) -> Path:
    if target.exists() and not force:
        raise FileExistsError(
            f"完整配置已存在：{target}；如需覆盖请添加 --force，避免误覆盖手工修改。"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    return target


def _train_entry(config: dict) -> str:
    model_name = str(config.get("model", {}).get("name", "")).strip().lower()
    if model_name == "refdiffrwkv":
        return "refdiffrwkv.py"
    if str(config.get("task", "sr")).strip().lower() == "sr":
        return "sr.py"
    return "refsrwkv.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="分层 run YAML 或已有完整 YAML")
    parser.add_argument(
        "--output",
        default=None,
        help="输出路径；省略时写入该实验目录的 experiments/.../config.yaml",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许覆盖已有完整配置",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="用 section.key=value 覆盖配置",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.overrides, prefer_existing=False)
    target = (
        layout_from_config(config).train_dir / "config.yaml"
        if args.output is None
        else Path(args.output).expanduser()
    )
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    target = _write_yaml(config, target, force=args.force)
    print(f"完整配置已生成：{target}")
    print(f"可直接运行：python scripts/train/{_train_entry(config)} --config {target}")


if __name__ == "__main__":
    main()
