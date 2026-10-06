from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from .artifacts import sha256_path
from .errors import ArtifactError
from .prompts import render_messages

PROMPT_REGISTRY_SCHEMA_VERSION = 1


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid prompt registry source file: {path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError(f"prompt registry source must be a mapping: {path}")
    return payload


def _declared_hash(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ArtifactError(f"missing prompt hash file: {path}") from exc
    if len(value) != 64:
        raise ArtifactError(f"invalid prompt hash file: {path}")
    return value


def _require_sha256(value: object, *, source: str) -> str:
    result = str(value)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ArtifactError(f"invalid SHA-256 in {source}")
    return result


@dataclass(frozen=True, slots=True)
class PromptRecord:
    prompt_id: str
    kind: Literal["C_seed", "C_adapt"]
    instruction: str
    instruction_sha256: str
    gepa_seed: int | None
    gepa_train_size: int | None
    source_run: str | None
    optimized_prompt_sha256: str | None
    optimizer_train_sha256: str | None
    optimizer_val_sha256: str | None
    gepa_config_sha256: str | None
    gepa_git_commit: str | None
    setup_assertion_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "kind": self.kind,
            "instruction": self.instruction,
            "instruction_sha256": self.instruction_sha256,
            "gepa_seed": self.gepa_seed,
            "gepa_train_size": self.gepa_train_size,
            "source_run": self.source_run,
            "optimized_prompt_sha256": self.optimized_prompt_sha256,
            "optimizer_train_sha256": self.optimizer_train_sha256,
            "optimizer_val_sha256": self.optimizer_val_sha256,
            "gepa_config_sha256": self.gepa_config_sha256,
            "gepa_git_commit": self.gepa_git_commit,
            "setup_assertion_sha256": self.setup_assertion_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PromptRecord:
        kind = str(payload["kind"])
        if kind not in {"C_seed", "C_adapt"}:
            raise ArtifactError(f"unknown prompt kind: {kind}")
        record = cls(
            prompt_id=str(payload["prompt_id"]),
            kind=cast(Literal["C_seed", "C_adapt"], kind),
            instruction=str(payload["instruction"]),
            instruction_sha256=str(payload["instruction_sha256"]),
            gepa_seed=None if payload.get("gepa_seed") is None else int(payload["gepa_seed"]),
            gepa_train_size=(
                None
                if payload.get("gepa_train_size") is None
                else int(payload["gepa_train_size"])
            ),
            source_run=None if payload.get("source_run") is None else str(payload["source_run"]),
            optimized_prompt_sha256=(
                None
                if payload.get("optimized_prompt_sha256") is None
                else str(payload["optimized_prompt_sha256"])
            ),
            optimizer_train_sha256=(
                None
                if payload.get("optimizer_train_sha256") is None
                else str(payload["optimizer_train_sha256"])
            ),
            optimizer_val_sha256=(
                None
                if payload.get("optimizer_val_sha256") is None
                else str(payload["optimizer_val_sha256"])
            ),
            gepa_config_sha256=(
                None
                if payload.get("gepa_config_sha256") is None
                else str(payload["gepa_config_sha256"])
            ),
            gepa_git_commit=(
                None if payload.get("gepa_git_commit") is None else str(payload["gepa_git_commit"])
            ),
            setup_assertion_sha256=(
                None
                if payload.get("setup_assertion_sha256") is None
                else str(payload["setup_assertion_sha256"])
            ),
        )
        if _sha256_text(record.instruction) != record.instruction_sha256:
            raise ArtifactError(f"instruction hash mismatch for {record.prompt_id}")
        return record


@dataclass(frozen=True, slots=True)
class PromptRegistry:
    schema_version: int
    dataset_id: str
    model_key: str | None
    model_id: str
    model_revision: str
    tokenizer_revision: str
    non_thinking: bool
    labels: tuple[str, ...]
    prompt_contract_id: str
    seed_prompt_sha256: str
    split_manifest_sha256: str
    prompts: tuple[PromptRecord, ...]
    registry_sha256: str

    def prompt(self, prompt_id: str) -> PromptRecord:
        for item in self.prompts:
            if item.prompt_id == prompt_id:
                return item
        raise ArtifactError(f"unknown prompt ID: {prompt_id}")

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "model_key": self.model_key,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "non_thinking": self.non_thinking,
            "labels": list(self.labels),
            "prompt_contract_id": self.prompt_contract_id,
            "seed_prompt_sha256": self.seed_prompt_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "prompts": [item.to_dict() for item in self.prompts],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_hash(), "registry_sha256": self.registry_sha256}

    def validate(self) -> None:
        if self.schema_version != PROMPT_REGISTRY_SCHEMA_VERSION:
            raise ArtifactError(f"unsupported prompt registry schema: {self.schema_version}")
        prompt_ids = [item.prompt_id for item in self.prompts]
        if len(prompt_ids) != len(set(prompt_ids)):
            raise ArtifactError("duplicate prompt IDs in prompt registry")
        if sum(item.kind == "C_seed" for item in self.prompts) != 1:
            raise ArtifactError("prompt registry must contain exactly one C_seed")
        encoded = json.dumps(
            self._payload_without_hash(), sort_keys=True, separators=(",", ":")
        ).encode()
        expected = hashlib.sha256(encoded).hexdigest()
        if self.registry_sha256 != expected:
            raise ArtifactError("prompt registry hash mismatch")
        for item in self.prompts:
            if _sha256_text(item.instruction) != item.instruction_sha256:
                raise ArtifactError(f"instruction hash mismatch for {item.prompt_id}")
            rendered = render_messages(item.instruction, self.labels, "{text}").messages[0][
                "content"
            ]
            rendered_hash = _sha256_text(rendered)
            if item.kind == "C_seed":
                if any(
                    value is not None
                    for value in (
                        item.gepa_seed,
                        item.gepa_train_size,
                        item.source_run,
                        item.optimized_prompt_sha256,
                        item.optimizer_train_sha256,
                        item.optimizer_val_sha256,
                    )
                ):
                    raise ArtifactError("C_seed must not contain adapted-prompt provenance")
                if rendered_hash != self.seed_prompt_sha256:
                    raise ArtifactError("seed rendered prompt hash mismatch")
            else:
                required = (
                    item.gepa_seed,
                    item.gepa_train_size,
                    item.source_run,
                    item.optimized_prompt_sha256,
                    item.optimizer_train_sha256,
                    item.optimizer_val_sha256,
                    item.gepa_config_sha256,
                    item.gepa_git_commit,
                )
                if any(value is None for value in required):
                    raise ArtifactError(f"incomplete adapted prompt provenance: {item.prompt_id}")
                if rendered_hash != item.optimized_prompt_sha256:
                    raise ArtifactError(f"adapted rendered prompt hash mismatch: {item.prompt_id}")
            for name, value in (
                ("instruction", item.instruction_sha256),
                ("optimized prompt", item.optimized_prompt_sha256),
                ("optimizer train", item.optimizer_train_sha256),
                ("optimizer validation", item.optimizer_val_sha256),
                ("GEPA config", item.gepa_config_sha256),
                ("setup assertion", item.setup_assertion_sha256),
            ):
                if value is not None:
                    _require_sha256(value, source=f"{item.prompt_id} {name}")
        _require_sha256(self.seed_prompt_sha256, source="seed prompt")
        _require_sha256(self.split_manifest_sha256, source="split manifest")


def _registry_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _asserted_setup(run_dir: Path, assertion: dict[str, Any]) -> tuple[dict[str, Any], str]:
    setup = assertion.get("setup")
    evidence = assertion.get("evidence")
    if not isinstance(setup, dict) or not isinstance(evidence, list) or not evidence:
        raise ArtifactError(f"legacy setup assertion is incomplete: {run_dir}")
    for item in evidence:
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            raise ArtifactError(f"legacy setup assertion has invalid evidence: {run_dir}")
        evidence_path = Path(str(item["path"]))
        if sha256_path(evidence_path) != str(item["sha256"]):
            raise ArtifactError(f"legacy setup evidence hash mismatch: {evidence_path}")
    encoded = json.dumps(assertion, sort_keys=True, separators=(",", ":")).encode()
    return setup, hashlib.sha256(encoded).hexdigest()


def _resolved_model_provenance(
    run_dir: Path, model_key: str | None
) -> tuple[str, str, str, bool]:
    try:
        payload = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
        models = payload["models"]
    except (FileNotFoundError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise ArtifactError(f"invalid resolved GEPA config: {run_dir}") from exc
    if not isinstance(models, list):
        raise ArtifactError(f"resolved GEPA models must be a list: {run_dir}")
    matches = [
        model
        for model in models
        if isinstance(model, dict) and model.get("key") == model_key
    ]
    if len(matches) != 1:
        raise ArtifactError(f"cannot resolve GEPA task model config: {run_dir}")
    model = matches[0]
    model_id = str(model["id"])
    revision = str(model["revision"])
    tokenizer_revision = str(model.get("tokenizer_revision") or revision)
    return model_id, revision, tokenizer_revision, bool(model.get("non_thinking", False))


def build_prompt_registry(
    run_dirs: Sequence[Path],
    *,
    seed_instruction: str,
    expected_cells: Sequence[tuple[int, int]],
    setup_assertions: dict[str, dict[str, Any]] | None = None,
) -> PromptRegistry:
    expected = tuple(expected_cells)
    if not expected or len(expected) != len(set(expected)):
        raise ArtifactError("expected prompt cells must be unique and non-empty")
    records: dict[tuple[int, int], PromptRecord] = {}
    common: dict[str, Any] | None = None
    for run_dir in run_dirs:
        status = _read_json(run_dir / "status.json")
        if status.get("state") != "completed":
            raise ArtifactError(f"GEPA run is not completed: {run_dir}")
        setup_assertion_sha256: str | None = None
        if (run_dir / "setup.json").exists():
            setup = _read_json(run_dir / "setup.json")
        else:
            assertion = (setup_assertions or {}).get(str(run_dir.resolve()))
            if assertion is None:
                raise ArtifactError(f"missing GEPA setup and legacy assertion: {run_dir}")
            setup, setup_assertion_sha256 = _asserted_setup(run_dir, assertion)
        meta = _read_json(run_dir / "meta.json")
        contract = _read_json(run_dir / "prompt_contract.json")
        inputs = _read_json(run_dir / "inputs.json")
        task = setup.get("task")
        if not isinstance(task, dict):
            raise ArtifactError(f"GEPA setup has no task mapping: {run_dir}")
        manifest = inputs.get("manifest")
        if not isinstance(manifest, dict) or not manifest.get("sha256"):
            raise ArtifactError(f"GEPA inputs have no split manifest hash: {run_dir}")
        labels = contract.get("labels")
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise ArtifactError(f"GEPA prompt contract has invalid labels: {run_dir}")
        model_key = None if task.get("key") is None else str(task["key"])
        resolved_id, resolved_revision, tokenizer_revision, non_thinking = (
            _resolved_model_provenance(run_dir, model_key)
        )
        if resolved_id != str(task["model"]) or resolved_revision != str(task["revision"]):
            raise ArtifactError(f"GEPA setup task differs from resolved config: {run_dir}")
        current = {
            "dataset_id": str(setup["dataset"]),
            "model_key": model_key,
            "model_id": str(task["model"]),
            "model_revision": str(task["revision"]),
            "tokenizer_revision": tokenizer_revision,
            "non_thinking": non_thinking,
            "labels": tuple(labels),
            "prompt_contract_id": str(contract["id"]),
            "seed_prompt_sha256": str(contract["seed_prompt_sha256"]),
            "split_manifest_sha256": str(manifest["sha256"]),
        }
        if common is None:
            common = current
        elif current != common:
            raise ArtifactError(f"GEPA prompt provenance differs across runs: {run_dir}")
        seed_hash = str(contract.get("seed_instruction_sha256", ""))
        if seed_hash != _sha256_text(seed_instruction):
            raise ArtifactError(f"seed instruction hash mismatch: {run_dir}")
        seed = int(setup["seed"])
        train_size = int(setup["train_size"])
        cell = (seed, train_size)
        if cell in records:
            raise ArtifactError(f"duplicate GEPA prompt cell: {cell}")
        instruction = (run_dir / "optimized_instructions.txt").read_text(encoding="utf-8")
        instruction_hash = _sha256_text(instruction)
        if _declared_hash(run_dir / "prompt.sha256") != instruction_hash:
            raise ArtifactError(f"instruction hash mismatch: {run_dir}")
        optimized_prompt = (run_dir / "optimized_prompt.txt").read_text(encoding="utf-8")
        optimized_prompt_hash = _sha256_text(optimized_prompt)
        if _declared_hash(run_dir / "optimized_prompt.sha256") != optimized_prompt_hash:
            raise ArtifactError(f"optimized prompt hash mismatch: {run_dir}")
        rendered_prompt = render_messages(instruction, labels, "{text}").messages[0]["content"]
        if optimized_prompt != rendered_prompt:
            raise ArtifactError(f"optimized prompt does not render from instruction: {run_dir}")
        train = inputs.get("train")
        validation = inputs.get("validation")
        if not isinstance(train, dict) or not isinstance(validation, dict):
            raise ArtifactError(f"GEPA inputs have invalid split records: {run_dir}")
        if "seed" not in inputs or int(inputs["seed"]) != seed:
            raise ArtifactError(f"GEPA input seed differs from setup: {run_dir}")
        if int(train.get("rows", -1)) != train_size:
            raise ArtifactError(f"GEPA train rows differ from setup: {run_dir}")
        if int(validation.get("rows", -1)) != int(setup["validation_size"]):
            raise ArtifactError(f"GEPA validation rows differ from setup: {run_dir}")
        train_sha256 = _require_sha256(train.get("sha256"), source=str(run_dir / "inputs.json"))
        validation_sha256 = _require_sha256(
            validation.get("sha256"), source=str(run_dir / "inputs.json")
        )
        git = meta.get("git")
        if not isinstance(git, dict) or not git.get("commit") or not meta.get("config_hash"):
            raise ArtifactError(f"GEPA meta lacks config/git provenance: {run_dir}")
        records[cell] = PromptRecord(
            prompt_id=f"C_adapt_s{seed}_n{train_size}",
            kind="C_adapt",
            instruction=instruction,
            instruction_sha256=instruction_hash,
            gepa_seed=seed,
            gepa_train_size=train_size,
            source_run=str(run_dir.resolve()),
            optimized_prompt_sha256=optimized_prompt_hash,
            optimizer_train_sha256=train_sha256,
            optimizer_val_sha256=validation_sha256,
            gepa_config_sha256=str(meta["config_hash"]),
            gepa_git_commit=str(git["commit"]),
            setup_assertion_sha256=setup_assertion_sha256,
        )
    if common is None:
        raise ArtifactError("no GEPA run directories supplied")
    missing = set(expected) - set(records)
    extra = set(records) - set(expected)
    if missing or extra:
        raise ArtifactError(
            f"GEPA prompt grid mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    validation_by_seed: dict[int, str] = {}
    for (seed, _), record in records.items():
        validation_hash = str(record.optimizer_val_sha256)
        previous = validation_by_seed.setdefault(seed, validation_hash)
        if previous != validation_hash:
            raise ArtifactError(f"optimizer validation split differs within seed {seed}")
    prompts = (
        PromptRecord(
            prompt_id="C_seed",
            kind="C_seed",
            instruction=seed_instruction,
            instruction_sha256=_sha256_text(seed_instruction),
            gepa_seed=None,
            gepa_train_size=None,
            source_run=None,
            optimized_prompt_sha256=None,
            optimizer_train_sha256=None,
            optimizer_val_sha256=None,
            gepa_config_sha256=None,
            gepa_git_commit=None,
            setup_assertion_sha256=None,
        ),
        *(records[cell] for cell in expected),
    )
    payload = {
        "schema_version": PROMPT_REGISTRY_SCHEMA_VERSION,
        **common,
        "labels": list(common["labels"]),
        "prompts": [item.to_dict() for item in prompts],
    }
    registry = PromptRegistry(
        schema_version=PROMPT_REGISTRY_SCHEMA_VERSION,
        dataset_id=common["dataset_id"],
        model_key=common["model_key"],
        model_id=common["model_id"],
        model_revision=common["model_revision"],
        tokenizer_revision=common["tokenizer_revision"],
        non_thinking=common["non_thinking"],
        labels=common["labels"],
        prompt_contract_id=common["prompt_contract_id"],
        seed_prompt_sha256=common["seed_prompt_sha256"],
        split_manifest_sha256=common["split_manifest_sha256"],
        prompts=prompts,
        registry_sha256=_registry_hash(payload),
    )
    registry.validate()
    return registry


def save_prompt_registry(path: Path, registry: PromptRegistry) -> None:
    registry.validate()
    encoded = json.dumps(registry.to_dict(), indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ArtifactError(f"refusing to overwrite a different prompt registry: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def load_prompt_registry(path: Path) -> PromptRegistry:
    payload = _read_json(path)
    prompts_payload = payload.get("prompts")
    if not isinstance(prompts_payload, list):
        raise ArtifactError("prompt registry prompts must be a list")
    try:
        registry = PromptRegistry(
            schema_version=int(payload["schema_version"]),
            dataset_id=str(payload["dataset_id"]),
            model_key=None if payload.get("model_key") is None else str(payload["model_key"]),
            model_id=str(payload["model_id"]),
            model_revision=str(payload["model_revision"]),
            tokenizer_revision=str(payload["tokenizer_revision"]),
            non_thinking=bool(payload["non_thinking"]),
            labels=tuple(str(label) for label in payload["labels"]),
            prompt_contract_id=str(payload["prompt_contract_id"]),
            seed_prompt_sha256=str(payload["seed_prompt_sha256"]),
            split_manifest_sha256=str(payload["split_manifest_sha256"]),
            prompts=tuple(PromptRecord.from_dict(item) for item in prompts_payload),
            registry_sha256=str(payload["registry_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"invalid prompt registry: {path}") from exc
    registry.validate()
    return registry


__all__ = [
    "PromptRecord",
    "PromptRegistry",
    "build_prompt_registry",
    "load_prompt_registry",
    "save_prompt_registry",
]
