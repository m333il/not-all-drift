from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from interpretability_gepa.artifacts import RunDirectory
from interpretability_gepa.config import ExperimentConfig, load_config
from interpretability_gepa.errors import ArtifactError


def test_config_hash_redacts_secrets(config_dict: dict) -> None:
    config_dict["credentials"] = {"token": "super-secret"}
    cfg = ExperimentConfig.model_validate(config_dict)
    assert cfg.public_dict()["credentials"] == {"token": "***"}
    assert "super-secret" not in str(cfg.public_dict())
    assert len(cfg.content_hash()) == 64


def test_config_rejects_seed_mismatch(config_dict: dict) -> None:
    config_dict["prefix"] = {"seeds": [1]}
    with pytest.raises(ValidationError, match="seeds must match"):
        ExperimentConfig.model_validate(config_dict)


def test_config_rejects_non_greedy_task_provider(config_dict: dict) -> None:
    config_dict["task_provider"]["temperature"] = 0.7

    with pytest.raises(ValidationError, match="task provider temperature"):
        ExperimentConfig.model_validate(config_dict)


def test_yaml_override(tmp_path: Path, config_dict: dict) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    cfg = load_config(path, ["dataset.train_size=300"])
    assert cfg.dataset.train_size == 300


def test_run_directory_is_timestamped_and_atomic(config: ExperimentConfig) -> None:
    now = datetime(2026, 7, 22, 12, 34, 56, tzinfo=UTC)
    with RunDirectory(config, "test", now=now) as run:
        assert run.path.name.startswith("20260722T123456000000Z_test_test-run_")
        assert yaml.safe_load((run.path / "config.resolved.yaml").read_text())
        run.write_json("nested/value.json", {"x": 1})
    assert json.loads((run.path / "status.json").read_text())["state"] == "completed"
    with pytest.raises(ArtifactError), RunDirectory(config, "test", now=now):
        pass
