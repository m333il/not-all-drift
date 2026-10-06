from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from .errors import ConfigurationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(StrictModel):
    key: str | None = None
    id: str
    revision: str
    tokenizer_revision: str | None = None
    role: Literal["primary", "replication"]
    family: Literal["gemma2", "qwen3", "synthetic"] = "synthetic"
    scale: Literal["small", "large", "synthetic"] = "synthetic"
    endpoint: str | None = None
    non_thinking: bool = False
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    resource_pool: str = "hf_single"
    layers: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_family_contract(self) -> ModelConfig:
        if self.family == "qwen3" and not self.non_thinking:
            raise ValueError("Qwen3 models must set non_thinking=true")
        return self


class ProviderConfig(StrictModel):
    kind: Literal["vllm", "openai", "anthropic", "fake"]
    model: str
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    revision: str | None = None
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"
    quantization: Literal["fp8"] | None = None
    chat_template_kwargs: dict[str, bool] = Field(default_factory=dict)
    # Passed verbatim to OpenAI-compatible request bodies.
    extra_body: dict[str, Any] = Field(default_factory=dict)
    base_url: str | None = None
    api_key_env: str | None = None
    thinking: Literal["adaptive", "disabled"] = "disabled"
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=3, ge=0)


class DatasetConfig(StrictModel):
    id: Literal[
        "goemotions",
        "civil_comments",
        "hallmarks_of_cancer",
        "synthetic",
    ]
    revision: str
    train_size: int = Field(default=100, gt=0)
    optimizer_val_size: int = Field(default=50, gt=0)
    probe_train_size: int = Field(default=4000, gt=0)
    probe_val_size: int = Field(default=500, gt=0)
    intervention_val_size: int = Field(default=500, gt=0)
    mechanistic_eval_size: int = Field(default=1000, gt=0)
    natural_eval_size: int = Field(default=5000, gt=0)
    group_field: str | None = None
    binarization_threshold: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def validate_grouping(self) -> DatasetConfig:
        if self.id == "hallmarks_of_cancer" and self.group_field != "pmid":
            raise ValueError("Hallmarks of Cancer must use group_field=pmid")
        return self


class GenerationConfig(StrictModel):
    temperature: float = Field(default=0.0, ge=0)
    max_new_tokens: int = Field(default=128, gt=0)
    do_sample: bool = False

    @model_validator(mode="after")
    def deterministic_confirmatory_runs(self) -> GenerationConfig:
        if self.temperature != 0 or self.do_sample:
            raise ValueError("confirmatory generation must be greedy")
        return self


class GepaConfig(StrictModel):
    seeds: tuple[int, ...] = (42, 43, 44)
    max_metric_calls: int = Field(default=10_000, gt=0)
    pilot_metric_calls: int = Field(default=1_000, gt=0)
    reflection_minibatch_size: int = Field(default=5, gt=0)
    reflection_max_tokens: int = Field(default=16_384, gt=0)
    reflection_instruction_budget_tokens: int = Field(default=8_192, gt=0)
    reflection_failure_limit: int = Field(default=20, gt=0)
    local_reflector: ProviderConfig
    api_reflector: ProviderConfig | None = None


class PrefixConfig(StrictModel):
    seeds: tuple[int, ...] = (42, 43, 44)
    virtual_tokens: tuple[int | Literal["length_match"], ...] = (10, "length_match")
    learning_rates: tuple[float, ...] = (1e-3, 3e-4, 1e-4)
    max_steps: int = Field(default=500, gt=0)
    pilot_steps: int = Field(default=300, gt=0)
    patience: int = Field(default=50, gt=0)
    batch_size: int = Field(default=4, gt=0)
    validation_interval: int = Field(default=10, gt=0)
    checkpoint_fractions: tuple[float, ...] = (0.1, 0.25, 0.5)


class ProbeConfig(StrictModel):
    c_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    threshold: float = Field(default=0.5, gt=0, lt=1)


class InterventionConfig(StrictModel):
    alphas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    relative_norms: tuple[float, ...] = (0.05, 0.1, 0.2)
    layer_fractions: tuple[float, ...] = (0.25, 0.4, 0.55, 0.7)
    bootstrap_samples: int = Field(default=1000, gt=0)
    parse_failure_limit: float = Field(default=0.1, ge=0, le=1)


class StatisticsConfig(StrictModel):
    bootstrap_samples: int = Field(default=1000, gt=0)
    confidence: float = Field(default=0.95, gt=0, lt=1)
    correction: Literal["holm"] = "holm"


class SaeConfig(StrictModel):
    enabled: bool = True
    reconstruction_cosine_min: float = Field(default=0.8, ge=-1, le=1)
    fve_min: float = Field(default=0.5, ge=0, le=1)
    max_dead_feature_fraction: float = Field(default=0.95, ge=0, le=1)
    max_lm_loss_delta: float = Field(default=0.1, ge=0)
    max_next_token_kl: float = Field(default=0.1, ge=0)
    matched_random_draws: int = Field(default=5, gt=0)


class DeviceConfig(StrictModel):
    id: str
    visible_index: int = Field(ge=0)
    kind: Literal["H200", "other"] = "H200"


class ResourcePoolConfig(StrictModel):
    gpus_per_job: int = Field(default=0, ge=0)
    tensor_parallel_size: int = Field(default=1, gt=0)
    exclusive: bool = True
    max_workers: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def validate_tensor_parallelism(self) -> ResourcePoolConfig:
        if self.gpus_per_job == 0 and self.tensor_parallel_size != 1:
            raise ValueError("CPU pools cannot use tensor parallelism")
        if self.gpus_per_job and self.tensor_parallel_size > self.gpus_per_job:
            raise ValueError("tensor_parallel_size cannot exceed gpus_per_job")
        return self


class ResourceConfig(StrictModel):
    launcher: Literal["local", "slurm"] = "local"
    devices: tuple[DeviceConfig, ...] = ()
    pools: dict[str, ResourcePoolConfig] = Field(default_factory=dict)
    scratch_root: Path = Path("results")


class MatrixConfig(StrictModel):
    model_keys: tuple[str, ...] = ()
    dataset_ids: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ("C_seed", "C_adapt", "C_prefix")


class DecisionConfig(StrictModel):
    practical_f1: float = 0.02
    latent_gap: float = 0.05
    elicitation_fraction: float = 0.5
    equivalence_margin: float = 0.02
    strong_cross_recovery: float = 0.5
    cross_recovery_ci_floor: float = 0.3


class OutputConfig(StrictModel):
    root: Path = Path("results")
    timezone: Literal["UTC"] = "UTC"
    activation_compressor: Literal["zstd", "none"] = "zstd"


class ExperimentConfig(StrictModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    dataset: DatasetConfig
    dataset_catalog: dict[str, DatasetConfig] = Field(default_factory=dict)
    models: tuple[ModelConfig, ...]
    task_provider: ProviderConfig
    generation: GenerationConfig = GenerationConfig()
    gepa: GepaConfig
    prefix: PrefixConfig = PrefixConfig()
    probes: ProbeConfig = ProbeConfig()
    interventions: InterventionConfig = InterventionConfig()
    statistics: StatisticsConfig = StatisticsConfig()
    sae: SaeConfig = SaeConfig()
    resources: ResourceConfig = ResourceConfig()
    matrix: MatrixConfig = MatrixConfig()
    decisions: DecisionConfig = DecisionConfig()
    output: OutputConfig = OutputConfig()
    credentials: dict[str, SecretStr] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_design(self) -> ExperimentConfig:
        roles = [model.role for model in self.models]
        if roles.count("primary") != 1:
            raise ValueError("exactly one primary model is required")
        if self.dataset.id == "civil_comments" and self.dataset.mechanistic_eval_size < 1000:
            raise ValueError("Civil Comments mechanistic eval must contain at least 1000 examples")
        if tuple(self.gepa.seeds) != tuple(self.prefix.seeds):
            raise ValueError("GEPA and prefix seeds must match")
        if self.task_provider.temperature != self.generation.temperature:
            raise ValueError("task provider temperature must match generation temperature")
        keys = [model.key or model.id for model in self.models]
        if len(keys) != len(set(keys)):
            raise ValueError("model keys must be unique")
        endpoints = [model.endpoint for model in self.models if model.endpoint]
        if len(endpoints) != len(set(endpoints)):
            raise ValueError("model endpoints must be unique")
        unknown_models = set(self.matrix.model_keys) - set(keys)
        if unknown_models:
            raise ValueError(f"matrix references unknown model keys: {sorted(unknown_models)}")
        if self.matrix.dataset_ids and self.dataset.id not in self.matrix.dataset_ids:
            raise ValueError("selected dataset must be included in matrix.dataset_ids")
        for key, dataset in self.dataset_catalog.items():
            if key != dataset.id:
                raise ValueError(f"dataset catalog key {key!r} does not match id {dataset.id!r}")
        unknown_datasets = set(self.matrix.dataset_ids) - (
            set(self.dataset_catalog) or {self.dataset.id}
        )
        if unknown_datasets:
            raise ValueError(f"matrix references unknown dataset IDs: {sorted(unknown_datasets)}")
        unknown_pools = {
            model.resource_pool
            for model in self.models
            if model.resource_pool not in self.resources.pools
        }
        if self.resources.pools and unknown_pools:
            raise ValueError(f"models reference unknown resource pools: {sorted(unknown_pools)}")
        return self

    def model(self, key: str | None = None) -> ModelConfig:
        if key is None:
            return next(model for model in self.models if model.role == "primary")
        for model in self.models:
            if (model.key or model.id) == key:
                return model
        raise ConfigurationError(f"unknown model key: {key}")

    def provider_for_model(self, key: str | None = None) -> ProviderConfig:
        model = self.model(key)
        if model.endpoint is None:
            raise ConfigurationError(f"model has no endpoint: {model.id}")
        return self.task_provider.model_copy(
            update={
                "model": model.id,
                "base_url": model.endpoint,
                "chat_template_kwargs": ({"enable_thinking": False} if model.non_thinking else {}),
            }
        )

    def dataset_config(self, dataset_id: str | None = None) -> DatasetConfig:
        if dataset_id is None or dataset_id == self.dataset.id:
            return self.dataset
        try:
            return self.dataset_catalog[dataset_id]
        except KeyError as exc:
            raise ConfigurationError(f"unknown dataset ID: {dataset_id}") from exc

    def public_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        if data.get("credentials"):
            data["credentials"] = {key: "***" for key in data["credentials"]}
        return data

    def content_hash(self) -> str:
        payload = json.dumps(self.public_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _set_override(raw: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigurationError(f"override must be key=value: {override}")
    dotted, encoded = override.split("=", 1)
    keys = dotted.split(".")
    cursor = raw
    for key in keys[:-1]:
        child = cursor.setdefault(key, {})
        if not isinstance(child, dict):
            raise ConfigurationError(f"cannot descend into override key {key}")
        cursor = child
    cursor[keys[-1]] = yaml.safe_load(encoded)


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigurationError(f"expected a YAML mapping in {source}")
    for override in overrides or []:
        _set_override(raw, override)
    return ExperimentConfig.model_validate(raw)
