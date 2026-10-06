from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from interpretability_gepa.artifacts import sha256_path
from interpretability_gepa.cli import _artifact_slug, app
from interpretability_gepa.gepa_runner import GepaRunResult
from interpretability_gepa.providers import FakeProvider
from interpretability_gepa.reporting import build_report, claim_gate_summary


def test_report_writes_machine_readable_files(tmp_path: Path) -> None:
    path = build_report([{"condition": "seed", "metric": "f1", "value": 0.5}], tmp_path)
    assert json.loads(path.read_text())["conditions"] == ["seed"]
    assert (tmp_path / "results.parquet").exists()
    assert (tmp_path / "results.csv").exists()


def test_cli_help_and_synthetic_smoke(tmp_path: Path, config_dict: dict) -> None:
    config_path = tmp_path / "config.yaml"
    config_dict["output"]["root"] = str(tmp_path / "runs")
    config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    runner = CliRunner()
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    for command in (
        "data",
        "gepa",
        "evaluate",
        "activations",
        "probes",
        "causal",
        "geometry",
        "report",
        "pipeline",
        "phase0",
        "logit-lens",
    ):
        assert command in help_result.output
    result = runner.invoke(app, ["synthetic-smoke", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    run = Path(result.output.strip())
    assert json.loads((run / "status.json").read_text())["state"] == "completed"
    assert (run / "report" / "results.parquet").exists()


def test_claim_gates_preserve_negative_results() -> None:
    gates = claim_gate_summary(
        {
            "optimizer_gain": 0.03,
            "optimizer_gain_ci_low": 0.01,
            "parse_gain_fraction": 0.1,
            "patch_recovery": 0.7,
            "patch_beats_mismatch": True,
        }
    )
    assert gates["optimizer_gain"]
    assert gates["paired_patching"]
    assert not gates["global_steering"]
    assert not gates["sae_feature_claim"]


def test_gepa_cli_persists_prompt_contract_and_graph(
    tmp_path: Path,
    config_dict: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dict["dataset"].update(
        {
            "id": "civil_comments",
            "train_size": 1,
            "optimizer_val_size": 1,
            "mechanistic_eval_size": 1000,
        }
    )
    config_dict["output"]["root"] = str(tmp_path / "runs")
    config_dict["models"][0]["endpoint"] = "http://127.0.0.1:9999/v1"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    example = {
        "id": "example-1",
        "dataset": "civil_comments",
        "text": "Example text.",
        "labels": [],
        "source_id": "1",
    }
    row = json.dumps(example) + "\n"
    train_path = splits_dir / "optimizer_train_seed42_n1.jsonl"
    validation_path = splits_dir / "optimizer_val_seed42.jsonl"
    manifest_path = splits_dir / "manifest.json"
    train_path.write_text(row, encoding="utf-8")
    validation_path.write_text(row, encoding="utf-8")
    manifest_path.write_text("{}\n", encoding="utf-8")

    def fake_optimize(**kwargs: Any) -> GepaRunResult:
        assert kwargs["max_metric_calls"] == 10
        assert "stop_callbacks" not in kwargs
        run_dir = Path(kwargs["run_dir"])
        (run_dir / "gepa_state.bin").write_bytes(b"state")
        # A real optimize reflects at least once, and the CLI now refuses to
        # freeze a prompt that no reflection ever touched.
        reflection_provider = kwargs["reflection_provider"]
        reflection_provider.provider = FakeProvider(["```Selected instruction.```"])
        reflection_provider.complete([{"role": "user", "content": "Reflect."}])
        raw = SimpleNamespace(
            to_dict=lambda: {"best_idx": 0},
            candidate_tree_html=lambda: "<html>tree</html>",
        )
        return GepaRunResult(
            selected_instruction="Selected instruction.",
            candidates=({"index": 0, "instruction": "Selected instruction."},),
            raw_result=raw,
        )

    monkeypatch.setattr("interpretability_gepa.gepa_runner.optimize", fake_optimize)
    result = CliRunner().invoke(
        app,
        [
            "gepa",
            "optimize",
            "--config",
            str(config_path),
            "--splits-dir",
            str(splits_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    run = Path(result.output.strip())
    assert (
        "_gepa-optimize-civil-comments-task-tiny-reflect-reflect-train-1-val-1-seed-42_"
    ) in run.name
    expected_seed = (
        "You are a text classifier. Classify the text using the provided labels. "
        "More than one label may apply.\n"
        "Labels (use exact names): toxicity, obscene, threat, insult, "
        "identity_attack\n"
        "Text:\n{text}\n"
        "Return every applicable label in the order listed, separated by commas. "
        "If no label applies, return exactly NONE. Return no other text.\n"
        "Answer:"
    )
    assert (run / "seed_prompt.txt").read_text() == expected_seed
    assert json.loads((run / "prompt_contract.json").read_text())["id"] == (
        "civil_comments_seed_v2"
    )
    assert (run / "gepa" / "gepa_state.bin").read_bytes() == b"state"
    assert (run / "gepa" / "candidate_tree.html").read_text() == "<html>tree</html>"
    assert json.loads((run / "gepa" / "result.json").read_text()) == {"best_idx": 0}
    inputs = json.loads((run / "inputs.json").read_text())
    assert inputs["train"]["rows"] == 1
    assert inputs["validation"]["rows"] == 1
    assert inputs["train"]["path"] == str(train_path.resolve())
    assert inputs["train"]["sha256"] == sha256_path(train_path)
    assert inputs["validation"]["path"] == str(validation_path.resolve())
    assert inputs["validation"]["sha256"] == sha256_path(validation_path)
    assert inputs["manifest"]["path"] == str(manifest_path.resolve())
    assert inputs["manifest"]["sha256"] == sha256_path(manifest_path)
    setup = json.loads((run / "setup.json").read_text())
    assert setup["task"] == {
        "endpoint": "http://127.0.0.1:9999/v1",
        "key": None,
        "model": "tiny",
        "revision": "fixed",
    }
    assert setup["reflector"]["model"] == "reflect"
    assert json.loads((run / "status.json").read_text())["state"] == "completed"


def test_artifact_slug_is_safe_bounded_and_collision_resistant() -> None:
    first = _artifact_slug("org/Model.Name_with spaces/" + "A" * 80)
    second = _artifact_slug("org/Model.Name_with spaces/" + "B" * 80)

    assert len(first) <= 32
    assert re.fullmatch(r"[a-z0-9-]+", first)
    assert first == _artifact_slug("org/Model.Name_with spaces/" + "A" * 80)
    assert first != second


def test_gepa_cli_rejects_split_size_mismatch_before_creating_run(
    tmp_path: Path, config_dict: dict[str, Any]
) -> None:
    config_dict["dataset"].update(
        {
            "id": "civil_comments",
            "train_size": 2,
            "optimizer_val_size": 1,
            "mechanistic_eval_size": 1000,
        }
    )
    output_root = tmp_path / "runs"
    config_dict["output"]["root"] = str(output_root)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    row = (
        json.dumps(
            {
                "id": "example-1",
                "dataset": "civil_comments",
                "text": "Example text.",
                "labels": [],
                "source_id": "1",
            }
        )
        + "\n"
    )
    (splits_dir / "optimizer_train_seed42_n2.jsonl").write_text(row, encoding="utf-8")
    (splits_dir / "optimizer_val_seed42.jsonl").write_text(row, encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "gepa",
            "optimize",
            "--config",
            str(config_path),
            "--splits-dir",
            str(splits_dir),
        ],
    )

    assert result.exit_code == 2
    assert "train split has 1 rows, expected 2" in result.output
    assert not output_root.exists()


