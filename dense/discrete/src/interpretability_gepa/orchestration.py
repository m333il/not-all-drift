from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .config import ExperimentConfig
from .errors import ConfigurationError


class Stage(StrEnum):
    DATA = "data"
    GEPA = "gepa"
    PREFIX = "prefix"
    PHASE0 = "phase0"
    ACTIVATIONS = "activations"
    PROBES = "probes"
    LOGIT_LENS = "logit_lens"
    PATCH = "patch"
    STEER = "steer"
    GEOMETRY = "geometry"
    LEACE = "leace"
    SAE = "sae"
    REPORT = "report"


@dataclass(frozen=True, slots=True)
class JobSpec:
    id: str
    stage: Stage
    model_key: str
    dataset_id: str
    seed: int
    resource_pool: str
    payload: dict[str, Any]
    dependencies: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        stage: Stage,
        model_key: str,
        dataset_id: str,
        seed: int,
        resource_pool: str,
        payload: dict[str, Any],
        dependencies: tuple[str, ...] = (),
    ) -> JobSpec:
        identity = {
            "stage": stage.value,
            "model_key": model_key,
            "dataset_id": dataset_id,
            "seed": seed,
            "resource_pool": resource_pool,
            "payload": payload,
            "dependencies": list(dependencies),
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return cls(
            hashlib.sha256(encoded.encode()).hexdigest()[:24],
            stage,
            model_key,
            dataset_id,
            seed,
            resource_pool,
            payload,
            dependencies,
        )


def expand_jobs(
    config: ExperimentConfig,
    stage: Stage,
    *,
    dataset_ids: tuple[str, ...] | None = None,
    model_keys: tuple[str, ...] | None = None,
) -> tuple[JobSpec, ...]:
    selected_datasets = dataset_ids or config.matrix.dataset_ids or (config.dataset.id,)
    selected_models = model_keys or config.matrix.model_keys or tuple(
        model.key or model.id for model in config.models
    )
    unknown_datasets = set(selected_datasets) - set(config.matrix.dataset_ids or selected_datasets)
    if unknown_datasets:
        raise ConfigurationError(f"unknown matrix datasets: {sorted(unknown_datasets)}")

    cpu_stages = {Stage.DATA, Stage.PROBES, Stage.GEOMETRY, Stage.REPORT}
    unseeded_stages = {Stage.DATA, Stage.PHASE0, Stage.REPORT}
    if stage == Stage.DATA:
        selected_models = (config.model().key or config.model().id,)
    seeds = (0,) if stage in unseeded_stages else config.gepa.seeds
    jobs: list[JobSpec] = []
    for model_key in selected_models:
        model = config.model(model_key)
        for dataset_id in selected_datasets:
            for seed in seeds:
                payload = {
                    "config_hash": config.content_hash(),
                    "model_id": model.id,
                    "model_revision": model.revision,
                    "endpoint": model.endpoint,
                    "family": model.family,
                    "dataset_revision": (
                        config.dataset_config(dataset_id).revision
                    ),
                    "non_thinking": model.non_thinking,
                }
                jobs.append(
                    JobSpec.create(
                        stage=stage,
                        model_key=model_key,
                        dataset_id=dataset_id,
                        seed=seed,
                        resource_pool="cpu_probe" if stage in cpu_stages else model.resource_pool,
                        payload=payload,
                    )
                )
    return tuple(jobs)


__all__ = ["JobSpec", "Stage", "expand_jobs"]
