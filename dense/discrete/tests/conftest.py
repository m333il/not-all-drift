from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from interpretability_gepa.config import ExperimentConfig


@pytest.fixture
def config_dict(tmp_path: Path) -> dict[str, Any]:
    return {
        "name": "test-run",
        "dataset": {
            "id": "synthetic",
            "revision": "fixed",
            "train_size": 10,
            "optimizer_val_size": 5,
            "probe_train_size": 20,
            "probe_val_size": 10,
            "intervention_val_size": 10,
            "mechanistic_eval_size": 10,
            "natural_eval_size": 20,
        },
        "models": [{"id": "tiny", "revision": "fixed", "role": "primary"}],
        "task_provider": {"kind": "fake", "model": "fake"},
        "gepa": {
            "seeds": [42, 43, 44],
            "max_metric_calls": 10,
            "pilot_metric_calls": 5,
            "reflection_minibatch_size": 2,
            "local_reflector": {"kind": "fake", "model": "reflect"},
        },
        "prefix": {"seeds": [42, 43, 44]},
        "output": {"root": str(tmp_path)},
    }


@pytest.fixture
def config(config_dict: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig.model_validate(config_dict)
