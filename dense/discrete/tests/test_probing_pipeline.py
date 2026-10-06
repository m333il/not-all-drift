from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from interpretability_gepa.activations import ActivationBatch, save_activation_store
from interpretability_gepa.artifacts import sha256_path
from interpretability_gepa.cli import app
from interpretability_gepa.errors import ArtifactError
from interpretability_gepa.metrics import empty_aware_sample_f1_rows, sample_f1_rows
from interpretability_gepa.probe_data import (
    assemble_probe_dataset,
    build_frozen_probe_subsets,
    verify_split_manifest,
)
from interpretability_gepa.probe_workflow import evaluate_frozen_probes, fit_frozen_probes
from interpretability_gepa.probes import run_probe_protocol
from interpretability_gepa.prompt_registry import (
    build_prompt_registry,
    load_prompt_registry,
    save_prompt_registry,
)
from interpretability_gepa.prompts import CIVIL_COMMENTS_PROMPT_CONTRACT_ID, render_messages

LABELS = (
    "toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
)
SEED_INSTRUCTION = (
    "You are a text classifier. Classify the text using the provided labels. "
    "More than one label may apply."
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_gepa_run(
    path: Path, *, seed: int, train_size: int, manifest_sha256: str | None = None
) -> None:
    path.mkdir()
    instruction = f"Adapted instruction for seed {seed} and N {train_size}."
    instruction_hash = hashlib.sha256(instruction.encode()).hexdigest()
    optimized_prompt = render_messages(instruction, LABELS, "{text}").messages[0]["content"]
    _write_json(path / "status.json", {"state": "completed"})
    _write_json(
        path / "meta.json",
        {"config_hash": "b" * 64, "git": {"commit": "git-commit"}},
    )
    _write_json(
        path / "setup.json",
        {
            "dataset": "civil_comments",
            "seed": seed,
            "train_size": train_size,
            "validation_size": 200,
            "task": {
                "key": "gemma2_2b",
                "model": "google/gemma-2-2b-it",
                "revision": "model-revision",
            },
        },
    )
    (path / "config.resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "key": "gemma2_2b",
                        "id": "google/gemma-2-2b-it",
                        "revision": "model-revision",
                        "tokenizer_revision": None,
                        "non_thinking": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_json(
        path / "prompt_contract.json",
        {
            "id": CIVIL_COMMENTS_PROMPT_CONTRACT_ID,
            "labels": list(LABELS),
            "seed_instruction_sha256": hashlib.sha256(SEED_INSTRUCTION.encode()).hexdigest(),
            "seed_prompt_sha256": hashlib.sha256(
                render_messages(SEED_INSTRUCTION, LABELS, "{text}").messages[0]["content"].encode()
            ).hexdigest(),
        },
    )
    _write_json(
        path / "inputs.json",
        {
            "seed": seed,
            "train": {
                "rows": train_size,
                "sha256": hashlib.sha256(f"train-{seed}-{train_size}".encode()).hexdigest(),
            },
            "validation": {
                "rows": 200,
                "sha256": hashlib.sha256(f"val-{seed}".encode()).hexdigest(),
            },
            "manifest": {"sha256": manifest_sha256 or "a" * 64},
        },
    )
    (path / "optimized_instructions.txt").write_text(instruction, encoding="utf-8")
    (path / "optimized_prompt.txt").write_text(optimized_prompt, encoding="utf-8")
    (path / "prompt.sha256").write_text(instruction_hash + "\n", encoding="utf-8")
    (path / "optimized_prompt.sha256").write_text(
        hashlib.sha256(optimized_prompt.encode()).hexdigest() + "\n", encoding="utf-8"
    )


def _write_split(path: Path, ids: tuple[str, ...]) -> None:
    rows = []
    for index, example_id in enumerate(ids):
        rows.append(
            {
                "id": example_id,
                "dataset": "civil_comments",
                "text": f"comment {example_id}",
                "labels": [LABELS[index % len(LABELS)]] if index % 2 else [],
                "source_id": example_id,
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_split_manifest(splits: Path, *, train_size: int, val_size: int) -> str:
    records: dict[str, object] = {}
    for split_file in sorted(splits.glob("*.jsonl")):
        ids = tuple(json.loads(line)["id"] for line in split_file.read_text().splitlines())
        records[split_file.stem] = {
            "size": len(ids),
            "id_hash": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "example_ids": list(ids),
        }
    _write_json(
        splits / "manifest.json",
        {
            "dataset": "google/civil_comments",
            "dataset_revision": "fixed",
            "labels": list(LABELS),
            "contract": {
                "probe_train_size": train_size,
                "probe_val_size": val_size,
                "probe_fraction": 0.5,
            },
            "splits": records,
        },
    )
    return sha256_path(splits / "manifest.json")


def _activation_metadata(ids: tuple[str, ...], *, split_file: Path) -> dict[str, object]:
    return {
        "model": "google/gemma-2-2b-it",
        "model_revision": "model-revision",
        "tokenizer_revision": "model-revision",
        "split": split_file.name,
        "split_sha256": sha256_path(split_file),
        "condition": "C_seed",
        "prompt_id": "C_seed",
        "prompt_registry_sha256": "registry-hash",
        "example_id_hash": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
    }


def test_prompt_registry_builds_complete_hashed_grid(tmp_path: Path) -> None:
    cells = tuple((seed, size) for seed in (42, 43, 44) for size in (100, 200, 500))
    run_dirs = []
    for seed, train_size in cells:
        run_dir = tmp_path / f"run-s{seed}-n{train_size}"
        _write_gepa_run(run_dir, seed=seed, train_size=train_size)
        run_dirs.append(run_dir)

    registry = build_prompt_registry(
        tuple(run_dirs),
        seed_instruction=SEED_INSTRUCTION,
        expected_cells=cells,
    )
    output = tmp_path / "prompt_registry.json"
    save_prompt_registry(output, registry)
    restored = load_prompt_registry(output)

    assert len(restored.prompts) == 10
    assert restored.prompts[0].prompt_id == "C_seed"
    assert {item.prompt_id for item in restored.prompts[1:]} == {
        f"C_adapt_s{seed}_n{size}" for seed, size in cells
    }
    assert restored.registry_sha256 == registry.registry_sha256
    assert len(restored.registry_sha256) == 64


def test_prompt_registry_rejects_instruction_hash_mismatch(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_gepa_run(run, seed=42, train_size=100)
    (run / "prompt.sha256").write_text("0" * 64 + "\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="instruction hash"):
        build_prompt_registry(
            (run,),
            seed_instruction=SEED_INSTRUCTION,
            expected_cells=((42, 100),),
        )


def test_prompt_registry_accepts_hashed_legacy_setup_assertion(tmp_path: Path) -> None:
    run = tmp_path / "legacy-run"
    _write_gepa_run(run, seed=42, train_size=100)
    setup = json.loads((run / "setup.json").read_text())
    (run / "setup.json").unlink()
    evidence = run / "status.json"
    assertion = {
        "setup": setup,
        "evidence": [{"path": str(evidence), "sha256": sha256_path(evidence)}],
        "reason": "Legacy run predates setup.json; reconstructed from frozen launch record.",
    }

    registry = build_prompt_registry(
        (run,),
        seed_instruction=SEED_INSTRUCTION,
        expected_cells=((42, 100),),
        setup_assertions={str(run.resolve()): assertion},
    )

    assert registry.prompt("C_adapt_s42_n100").setup_assertion_sha256 is not None


def test_prompt_registry_requires_explicit_input_seed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_gepa_run(run, seed=42, train_size=100)
    inputs = json.loads((run / "inputs.json").read_text())
    del inputs["seed"]
    _write_json(run / "inputs.json", inputs)

    with pytest.raises(ArtifactError, match="input seed"):
        build_prompt_registry(
            (run,),
            seed_instruction=SEED_INSTRUCTION,
            expected_cells=((42, 100),),
        )


def test_probe_assembler_requires_exact_order_and_builds_targets(tmp_path: Path) -> None:
    ids = ("a", "b", "c", "d")
    split = tmp_path / "probe_train.jsonl"
    _write_split(split, ids)
    last = np.arange(4 * 3 * 2, dtype=np.float16).reshape(4, 3, 2)
    mean = last + 100
    store = tmp_path / "store"
    save_activation_store(
        store,
        ActivationBatch(ids, last, mean, _activation_metadata(ids, split_file=split)),
    )

    primary = assemble_probe_dataset(store, split, LABELS, position="last_prompt")
    robustness = assemble_probe_dataset(store, split, LABELS, position="mean_text")

    assert primary.example_ids == ids
    assert np.array_equal(primary.activations, last)
    assert np.array_equal(robustness.activations, mean)
    assert primary.targets.shape == (4, len(LABELS))
    assert primary.targets[0].sum() == 0
    assert primary.targets[1, 1] == 1
    assert np.array_equal(primary.label_counts, primary.targets.sum(axis=1))

    tampered_rows = [json.loads(line) for line in split.read_text().splitlines()]
    tampered_rows[1]["labels"] = []
    split.write_text("".join(json.dumps(row) + "\n" for row in tampered_rows), encoding="utf-8")
    with pytest.raises(ArtifactError, match="split SHA-256"):
        assemble_probe_dataset(store, split, LABELS, position="last_prompt")
    _write_split(split, ids)

    reordered_split = tmp_path / "probe_val.jsonl"
    _write_split(reordered_split, tuple(reversed(ids)))
    reordered_store = tmp_path / "reordered"
    save_activation_store(
        reordered_store,
        ActivationBatch(
            ids,
            last,
            mean,
            _activation_metadata(ids, split_file=reordered_split),
        ),
    )
    with pytest.raises(ArtifactError, match="ordered example IDs"):
        assemble_probe_dataset(reordered_store, reordered_split, LABELS, position="last_prompt")


def test_frozen_probe_subsets_map_ids_without_resampling(tmp_path: Path) -> None:
    master_ids = tuple(f"x{index}" for index in range(10))
    master_split = tmp_path / "probe_train.jsonl"
    _write_split(master_split, master_ids)
    activations = np.zeros((10, 2, 3), dtype=np.float16)
    store = tmp_path / "store"
    save_activation_store(
        store,
        ActivationBatch(
            master_ids,
            activations,
            activations,
            _activation_metadata(master_ids, split_file=master_split),
        ),
    )
    master = assemble_probe_dataset(store, master_split, LABELS, position="last_prompt")
    subset_files: dict[int, Path] = {}
    expected: dict[int, tuple[str, ...]] = {}
    for seed in range(5):
        ids = tuple(master_ids[(seed + offset) % len(master_ids)] for offset in range(8))
        path = tmp_path / f"probe_train_seed{seed}.jsonl"
        _write_split(path, ids)
        subset_files[seed] = path
        expected[seed] = ids

    subsets = build_frozen_probe_subsets(master, subset_files, expected_size=8)

    assert subsets.seeds == (0, 1, 2, 3, 4)
    for seed, indices in zip(subsets.seeds, subsets.indices, strict=True):
        assert tuple(master.example_ids[index] for index in indices) == expected[seed]
        assert len(set(indices.tolist())) == 8
        assert len(subsets.id_hashes[seed]) == 64

    invalid = tmp_path / "invalid.jsonl"
    _write_split(invalid, (*expected[0][:-1], "outside-master"))
    with pytest.raises(ArtifactError, match="outside master"):
        build_frozen_probe_subsets(master, {0: invalid}, expected_size=8)


def test_split_manifest_authenticates_ordered_ids(tmp_path: Path) -> None:
    splits = tmp_path / "splits"
    splits.mkdir()
    _write_split(splits / "probe_train.jsonl", ("a", "b"))
    _write_split(splits / "probe_val.jsonl", ("v",))
    manifest_hash = _write_split_manifest(splits, train_size=2, val_size=1)

    verify_split_manifest(
        splits,
        expected_sha256=manifest_hash,
        split_names=("probe_train", "probe_val"),
    )
    _write_split(splits / "probe_val.jsonl", ("changed",))
    with pytest.raises(ArtifactError, match="ordered IDs"):
        verify_split_manifest(
            splits,
            expected_sha256=manifest_hash,
            split_names=("probe_train", "probe_val"),
        )


def test_probe_protocol_uses_supplied_frozen_subsets() -> None:
    rng = np.random.default_rng(9)
    train_targets = rng.integers(0, 2, size=(12, 2))
    val_targets = rng.integers(0, 2, size=(6, 2))
    train = rng.normal(size=(12, 1, 4))
    val = rng.normal(size=(6, 1, 4))
    subsets = {
        0: np.asarray([0, 2, 4, 6, 8, 10]),
        1: np.asarray([1, 3, 5, 7, 9, 11]),
    }
    controls = {
        "length": (np.arange(12), np.arange(6)),
        "condition": (train_targets[:, :1], val_targets[:, :1]),
        "label_count": (train_targets.sum(axis=1), val_targets.sum(axis=1)),
    }

    rows = run_probe_protocol(
        train,
        train_targets,
        val,
        val_targets,
        controls=controls,
        subset_indices=subsets,
        subset_hashes={0: "hash-0", 1: "hash-1"},
        train_sizes=(3, 6),
        c_values=(1.0,),
        threshold=0.5,
        permutations=1,
    )

    assert {(int(row["seed"]), int(row["subset_size"])) for row in rows} == {
        (0, 6),
        (1, 6),
    }
    assert {str(row["subset_hash"]) for row in rows} == {"hash-0", "hash-1"}
    assert all([point["train_size"] for point in row["train_size_curve"]] == [3, 6] for row in rows)


def test_empty_aware_sample_f1_rewards_correct_none_prediction() -> None:
    targets = np.asarray([[0, 0], [1, 0], [1, 1]])
    predictions = np.asarray([[0, 0], [1, 0], [1, 0]])

    assert np.array_equal(sample_f1_rows(targets, predictions), [0.0, 1.0, 2 / 3])
    assert np.array_equal(empty_aware_sample_f1_rows(targets, predictions), [1.0, 1.0, 2 / 3])


def test_preregistered_probe_curve_ends_at_the_frozen_subset_size() -> None:
    # ``probes assemble`` refuses to run unless the top rung equals the frozen per-seed
    # subset size exactly, so the ladder is pinned to the split contract rather than to a
    # literal. The v1 rungs are kept below the new ceiling so the curve extends instead of
    # moving, and earlier points stay comparable with the numbers already reported.
    from interpretability_gepa.civil_splits_v2.contract import SplitContractV2

    contract = SplitContractV2()
    preregistration = yaml.safe_load(Path("configs/preregistration.yaml").read_text())
    train_sizes = preregistration["probe"]["train_sizes"]

    assert train_sizes == [250, 500, 1000, 2000, 3200, 6400]
    assert train_sizes[-1] == contract.probe_train_size * contract.probe_fraction


def test_probes_assemble_cli_writes_frozen_arrays_and_provenance(tmp_path: Path) -> None:
    config: dict[str, Any] = {
        "name": "probe-test",
        "dataset": {
            "id": "civil_comments",
            "revision": "fixed",
            "train_size": 100,
            "optimizer_val_size": 200,
            "probe_train_size": 4,
            "probe_val_size": 2,
            "intervention_val_size": 2,
            "mechanistic_eval_size": 1000,
            "natural_eval_size": 10,
            "binarization_threshold": 0.5,
        },
        "models": [
            {
                "key": "gemma2_2b",
                "id": "google/gemma-2-2b-it",
                "revision": "model-revision",
                "role": "primary",
            }
        ],
        "task_provider": {"kind": "fake", "model": "fake"},
        "gepa": {
            "seeds": [42, 43, 44],
            "max_metric_calls": 10,
            "pilot_metric_calls": 5,
            "reflection_minibatch_size": 2,
            "local_reflector": {"kind": "fake", "model": "reflect"},
        },
        "prefix": {"seeds": [42, 43, 44]},
        "probes": {"seeds": [0, 1, 2, 3, 4]},
        "output": {"root": str(tmp_path / "runs")},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    preregistration = tmp_path / "preregistration.yaml"
    preregistration.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "probe": {
                    "position": "last_prompt",
                    "c_values": [0.01, 0.1, 1.0, 10.0],
                    "train_sizes": [1, 2],
                    "seeds": [0, 1, 2, 3, 4],
                    "permutations": 1,
                },
                "selection_splits": ["probe_val"],
                "final_splits": ["test"],
            }
        ),
        encoding="utf-8",
    )
    splits = tmp_path / "splits"
    splits.mkdir()
    train_ids = ("a", "b", "c", "d")
    val_ids = ("v0", "v1")
    _write_split(splits / "probe_train.jsonl", train_ids)
    _write_split(splits / "probe_val.jsonl", val_ids)
    for seed in range(5):
        _write_split(splits / f"probe_train_seed{seed}.jsonl", train_ids[seed % 2 : seed % 2 + 2])
    manifest_sha256 = _write_split_manifest(splits, train_size=4, val_size=2)
    run = tmp_path / "gepa-run"
    _write_gepa_run(run, seed=42, train_size=100, manifest_sha256=manifest_sha256)
    registry = build_prompt_registry(
        (run,),
        seed_instruction=SEED_INSTRUCTION,
        expected_cells=((42, 100),),
    )
    registry_path = tmp_path / "prompt_registry.json"
    save_prompt_registry(registry_path, registry)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    adapter_result = CliRunner().invoke(
        app,
        [
            "activations",
            "extract",
            "--config",
            str(config_path),
            "--split-file",
            str(splits / "probe_val.jsonl"),
            "--prompt-registry",
            str(registry_path),
            "--prompt-id",
            "C_seed",
            "--adapter",
            str(adapter),
        ],
    )
    assert adapter_result.exit_code != 0
    assert "adapter cannot be combined" in adapter_result.output
    pilot_without_limit = CliRunner().invoke(
        app,
        [
            "activations",
            "extract",
            "--config",
            str(config_path),
            "--split-file",
            str(splits / "probe_val.jsonl"),
            "--prompt-registry",
            str(registry_path),
            "--prompt-id",
            "C_seed",
            "--pilot-verify",
        ],
    )
    assert pilot_without_limit.exit_code != 0
    assert "requires an explicit --limit" in pilot_without_limit.output
    prompt = registry.prompt("C_seed")
    common_metadata = {
        "model": registry.model_id,
        "model_revision": registry.model_revision,
        "tokenizer_revision": registry.model_revision,
        "condition": "C_seed",
        "prompt_id": "C_seed",
        "instruction_sha256": prompt.instruction_sha256,
        "prompt_registry_sha256": registry.registry_sha256,
        "template_hash": "template-hash",
        "adapter_hash": None,
        "non_thinking": False,
    }
    train_store = tmp_path / "train-store"
    val_store = tmp_path / "val-store"
    train_acts = np.zeros((4, 2, 3), dtype=np.float16)
    val_acts = np.zeros((2, 2, 3), dtype=np.float16)
    save_activation_store(
        train_store,
        ActivationBatch(
            train_ids,
            train_acts,
            train_acts,
            {
                **common_metadata,
                **_activation_metadata(train_ids, split_file=splits / "probe_train.jsonl"),
                "instruction_sha256": prompt.instruction_sha256,
                "prompt_registry_sha256": registry.registry_sha256,
            },
        ),
    )
    save_activation_store(
        val_store,
        ActivationBatch(
            val_ids,
            val_acts,
            val_acts,
            {
                **common_metadata,
                **_activation_metadata(val_ids, split_file=splits / "probe_val.jsonl"),
                "instruction_sha256": prompt.instruction_sha256,
                "prompt_registry_sha256": registry.registry_sha256,
            },
        ),
    )
    output = tmp_path / "assembled"

    result = CliRunner().invoke(
        app,
        [
            "probes",
            "assemble",
            "--config",
            str(config_path),
            "--preregistration",
            str(preregistration),
            "--prompt-registry",
            str(registry_path),
            "--prompt-id",
            "C_seed",
            "--train-store",
            str(train_store),
            "--val-store",
            str(val_store),
            "--splits-dir",
            str(splits),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    arrays = np.load(output / "arrays.npz")
    assert arrays["x_train"].shape == (4, 2, 3)
    assert arrays["subset_indices"].shape == (5, 2)
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["prompt_id"] == "C_seed"
    assert provenance["prompt_registry_sha256"] == registry.registry_sha256
    assert provenance["train_sizes"] == [1, 2]

    fitted = tmp_path / "fitted"
    fit_result = CliRunner().invoke(
        app,
        [
            "probes",
            "fit-frozen",
            "--config",
            str(config_path),
            "--assembled",
            str(output),
            "--output",
            str(fitted),
        ],
    )
    assert fit_result.exit_code == 0, fit_result.output
    assert len(list(fitted.glob("probe_seed*.npz"))) == 5

    ledger = tmp_path / "selection_ledger.json"
    selection_result = CliRunner().invoke(
        app,
        [
            "probes",
            "select-layer",
            "--fitted-seed",
            str(fitted),
            "--output",
            str(ledger),
        ],
    )
    assert selection_result.exit_code == 0, selection_result.output
    assert json.loads(ledger.read_text())["selected_layer"] in {0, 1}

    evaluation = tmp_path / "seed-to-seed.jsonl"
    evaluation_result = CliRunner().invoke(
        app,
        [
            "probes",
            "evaluate-frozen",
            "--fitted",
            str(fitted),
            "--assembled",
            str(output),
            "--output",
            str(evaluation),
        ],
    )
    assert evaluation_result.exit_code == 0, evaluation_result.output
    evaluation_rows = [json.loads(line) for line in evaluation.read_text().splitlines()]
    assert len(evaluation_rows) == 10
    assert {row["train_prompt_id"] for row in evaluation_rows} == {"C_seed"}
    assert {row["eval_prompt_id"] for row in evaluation_rows} == {"C_seed"}

    tampered_fit_input = tmp_path / "tampered-fit-input"
    shutil.copytree(output, tampered_fit_input)
    with np.load(tampered_fit_input / "arrays.npz") as archive:
        tampered_arrays = {name: archive[name] for name in archive.files}
    tampered_arrays["subset_indices"][0, 1] = tampered_arrays["subset_indices"][0, 0]
    np.savez(tampered_fit_input / "arrays.npz", **tampered_arrays)
    with pytest.raises(ArtifactError, match="subset indices are invalid"):
        fit_frozen_probes(
            tampered_fit_input,
            tmp_path / "tampered-fit",
            c_values=(0.01, 0.1, 1.0, 10.0),
            threshold=0.5,
        )

    tampered_transfer = tmp_path / "tampered-transfer"
    shutil.copytree(output, tampered_transfer)
    with np.load(tampered_transfer / "arrays.npz") as archive:
        transfer_arrays = {name: archive[name] for name in archive.files}
    transfer_arrays["example_ids_val"][0] = "different-id"
    np.savez(tampered_transfer / "arrays.npz", **transfer_arrays)
    with pytest.raises(ArtifactError, match="validation example IDs differ"):
        evaluate_frozen_probes(fitted, tampered_transfer, tmp_path / "tampered-eval.jsonl")
