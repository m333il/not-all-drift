from __future__ import annotations

import json
import platform
from pathlib import Path

import pytest

from interpretability_gepa.artifacts import ArtifactStore
from interpretability_gepa.config import load_config
from interpretability_gepa.datasets import dataset_factory
from interpretability_gepa.errors import ConfigurationError
from interpretability_gepa.modeling import model_family_factory
from interpretability_gepa.orchestration import Stage, expand_jobs
from interpretability_gepa.preregistration import load_preregistration


def test_core_config_resolves_updated_four_model_matrix() -> None:
    config = load_config(Path("configs/experiments/core.yaml"))

    assert {model.id for model in config.models} == {
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-8B",
    }
    assert {model.endpoint for model in config.models} == {
        "http://127.0.0.1:8010/v1",
        "http://127.0.0.1:8011/v1",
        "http://127.0.0.1:8012/v1",
        "http://127.0.0.1:8013/v1",
    }
    assert all(model.revision != "main" for model in config.models)
    assert all(model.non_thinking for model in config.models if model.family == "qwen3")
    assert config.provider_for_model("qwen3_8b").chat_template_kwargs == {
        "enable_thinking": False
    }
    assert config.task_provider.temperature == 0.0
    assert config.provider_for_model("gemma2_2b").temperature == 0.0
    assert config.gepa.local_reflector.temperature == 0.7


def test_matrix_jobs_are_deterministic_and_resource_scoped() -> None:
    config = load_config(Path("configs/experiments/core.yaml"))

    first = expand_jobs(config, Stage.GEPA, dataset_ids=("civil_comments", "goemotions"))
    second = expand_jobs(config, Stage.GEPA, dataset_ids=("civil_comments", "goemotions"))

    assert first == second
    assert len(first) == 4 * 2 * 3
    assert len({job.id for job in first}) == len(first)
    assert all(job.resource_pool == "hf_single" for job in first)
    assert all(job.payload["model_revision"] != "main" for job in first)


def test_artifact_store_resumes_only_hash_matching_success(tmp_path: Path) -> None:
    config = load_config(Path("configs/experiments/core.yaml"))
    job = expand_jobs(
        config,
        Stage.PHASE0,
        dataset_ids=("civil_comments",),
        model_keys=("gemma2_9b",),
    )[0]
    store = ArtifactStore(tmp_path, study_id="test-study")

    assert store.try_claim(job)
    assert not store.try_claim(job)
    path = store.job_path(job)
    assert not store.is_complete(job)
    store.commit(job, {"parity_passed": True})

    assert store.is_complete(job)
    assert (path / "_SUCCESS").exists()
    assert store.outputs(job)["parity_passed"] is True

    stale_job = expand_jobs(
        config,
        Stage.PREFIX,
        dataset_ids=("civil_comments",),
        model_keys=("gemma2_9b",),
    )[1]
    stale_path = store.job_path(stale_job)
    stale_path.mkdir(parents=True)
    (stale_path / ".claim").write_text(
        json.dumps({"hostname": platform.node(), "pid": 2_000_000_000}),
        encoding="utf-8",
    )
    assert store.try_claim(stale_job)


def test_model_family_adapters_expose_architecture_metadata() -> None:
    gemma = model_family_factory("gemma2")
    qwen = model_family_factory("qwen3")

    gemma_layers = gemma.layer_metadata(4)
    assert [layer.attention for layer in gemma_layers] == ["local", "global", "local", "global"]
    assert all(layer.attention == "global" for layer in qwen.layer_metadata(4))
    assert qwen.output_is_non_thinking("<think>\n\n</think>\n\n[\"joy\"]")
    assert not qwen.output_is_non_thinking("<think>secret reasoning</think>[\"joy\"]")
    assert dataset_factory("hallmarks_of_cancer").__name__ == "HallmarksSource"


def test_preregistration_forbids_final_eval_model_selection() -> None:
    preregistration = load_preregistration(Path("configs/preregistration.yaml"))

    preregistration.assert_selection_split("probe_val")
    with pytest.raises(ConfigurationError, match="forbidden"):
        preregistration.assert_selection_split("eval_mechanistic")
    assert len(preregistration.content_hash()) == 64
