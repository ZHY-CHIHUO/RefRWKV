from pathlib import Path

import pytest
import yaml

from runtime.config import load_test_config


def test_load_test_config_uses_checkpoint_trainer_config(tmp_path: Path) -> None:
    path = tmp_path / "test.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "test": {
                    "splits": ["test_easy", "test_hard"],
                    "metrics": ["psnr", "rmse"],
                    "save_images": False,
                }
            }
        ),
        encoding="utf-8",
    )
    checkpoint = {
        "trainer_config": {
            "task": "sr",
            "model": {"name": "swinir", "embed_dim": 180},
            "dataset": {"id": "hrms_scd"},
            "data": {"root": "data/refsr/HRMS_SCD", "scale": 4},
            "train": {},
            "loss": {},
            "output": {"test_dir": "experiments/test/example"},
        }
    }
    config = load_test_config(path, checkpoint)
    assert config["model"]["embed_dim"] == 180
    assert config["dataset"]["id"] == "hrms_scd"
    assert config["test"]["metrics"] == ["psnr", "rmse"]
    assert config["test"]["save_images"] is False
    assert config["_test_config_path"] == str(path.resolve())


def test_standalone_test_policy_replaces_checkpoint_test_section(tmp_path: Path) -> None:
    path = tmp_path / "test.yaml"
    path.write_text(
        yaml.safe_dump({"test": {"splits": ["test_easy"], "metrics": ["psnr"]}}),
        encoding="utf-8",
    )
    checkpoint = {
        "trainer_config": {
            "task": "sr",
            "model": {"name": "swinir"},
            "dataset": {"id": "hrms_scd"},
            "data": {"root": "data/refsr/HRMS_SCD", "scale": 4},
            "train": {},
            "loss": {},
            "output": {},
            "test": {"split": "test", "save_images": False},
        }
    }
    config = load_test_config(path, checkpoint)
    assert config["test"] == {
        "splits": ["test_easy"],
        "metrics": ["psnr"],
        "save_images": True,
        "device": None,
        "batch_size": None,
        "steps": None,
        "output": None,
    }


def test_load_test_config_rejects_checkpoint_without_training_config(tmp_path: Path) -> None:
    path = tmp_path / "test.yaml"
    path.write_text("test:\n  splits: [test]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trainer_config"):
        load_test_config(path, {})
