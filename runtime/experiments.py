"""Stable filesystem layout for training runs, tests, and logs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import PROJECT_ROOT, json_safe, resolve_path


def _slug(value: Any, fallback: str) -> str:
    text = str(value or fallback).strip().replace("\\", "_").replace("/", "_")
    return text or fallback


@dataclass(frozen=True)
class ExperimentLayout:
    """Paths for one task/model/dataset/scale/run tuple."""

    task: str
    model: str
    dataset: str
    scale: int | str
    run: str
    train_root: Path = PROJECT_ROOT / "experiments" / "train"
    test_root: Path = PROJECT_ROOT / "experiments" / "test"

    @property
    def train_dir(self) -> Path:
        return self.train_root / self.task / self.model / self.dataset / f"x{self.scale}" / self.run

    @property
    def checkpoints(self) -> Path:
        return self.train_dir / "checkpoints"

    @property
    def logs(self) -> Path:
        return self.train_dir / "logs"

    @property
    def test_dir(self) -> Path:
        return self.test_root / self.task / self.model / self.dataset / f"x{self.scale}" / self.run

    def test_split(self, split: str) -> Path:
        return self.test_dir / _slug(split, "test")

    def create_train(self) -> "ExperimentLayout":
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        return self

    def create_test(self, split: str) -> Path:
        path = self.test_split(split)
        path.mkdir(parents=True, exist_ok=True)
        return path


def layout_from_config(config: dict[str, Any]) -> ExperimentLayout:
    data = config.get("data", {})
    model = config.get("model", {})
    dataset = config.get("dataset", {})
    output = config.get("output", {})
    run = config.get("run", {})
    scale = data.get("scale", run.get("scale", "native"))
    train_root = resolve_path(output.get("train_root", "experiments/train"))
    test_root = resolve_path(output.get("test_root", "experiments/test"))
    return ExperimentLayout(
        task=_slug(config.get("task"), "sr"),
        model=_slug(model.get("name"), "model").lower(),
        dataset=_slug(dataset.get("id"), "dataset").lower(),
        scale=scale,
        run=_slug(run.get("name") or output.get("experiment_name"), "run"),
        train_root=train_root,
        test_root=test_root,
    )


def save_config_snapshot(config: dict[str, Any], path: str | Path) -> Path:
    """Write a JSON snapshot; YAML remains the human-edited source of truth."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(config), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return target


__all__ = ["ExperimentLayout", "layout_from_config", "save_config_snapshot"]
