from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from interpretability_gepa.gepa_runner import optimize
from interpretability_gepa.providers import FakeProvider


def test_optimize_adapter_exposes_gepa_proposal_hook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, Any] = {}

    class StubEvaluationBatch:
        def __init__(
            self,
            *,
            outputs: list[dict[str, float]],
            scores: list[float],
            trajectories: list[dict[str, Any]] | None,
        ) -> None:
            self.outputs = outputs
            self.scores = scores
            self.trajectories = trajectories

    def fake_gepa_optimize(**kwargs: Any) -> SimpleNamespace:
        adapter = kwargs["adapter"]
        observed["propose_new_texts"] = adapter.propose_new_texts
        observed["run_dir"] = kwargs["run_dir"]
        observed["reflection_prompt_template"] = kwargs["reflection_prompt_template"]
        return SimpleNamespace(
            best_candidate={"instructions": "selected"},
            candidates=[{"instructions": "selected"}],
            to_dict=lambda: {"best_idx": 0},
            candidate_tree_html=lambda: "<html>tree</html>",
        )

    modules = {
        "gepa": ModuleType("gepa"),
        "gepa.api": ModuleType("gepa.api"),
        "gepa.core": ModuleType("gepa.core"),
        "gepa.core.adapter": ModuleType("gepa.core.adapter"),
        "gepa.core.result": ModuleType("gepa.core.result"),
        "gepa.strategies": ModuleType("gepa.strategies"),
        "gepa.strategies.instruction_proposal": ModuleType("gepa.strategies.instruction_proposal"),
    }
    modules["gepa.api"].__dict__["optimize"] = fake_gepa_optimize
    modules["gepa.core.adapter"].__dict__["EvaluationBatch"] = StubEvaluationBatch
    modules["gepa.core.result"].__dict__["GEPAResult"] = object
    modules["gepa.strategies.instruction_proposal"].__dict__["InstructionProposalSignature"] = (
        SimpleNamespace(
            default_prompt_template=(
                "<curr_param> <side_info>\n\nProvide the new instructions within ``` blocks."
            )
        )
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    result = optimize(
        train=(),
        validation=(),
        seed_instruction="seed",
        task_provider=FakeProvider([]),
        reflection_provider=FakeProvider([]),
        labels=("label",),
        seed=42,
        max_metric_calls=1,
        reflection_minibatch_size=1,
        reflection_max_tokens=16_384,
        reflection_instruction_budget_tokens=8_192,
        run_dir=tmp_path / "gepa",
    )

    template = observed["reflection_prompt_template"]
    assert "<curr_param>" in template and "<side_info>" in template
    assert "under 8192 tokens" in template
    assert template.rstrip().endswith("Provide the new instructions within ``` blocks.")
    assert observed["propose_new_texts"] is None
    assert observed["run_dir"] == str(tmp_path / "gepa")
    assert result.selected_instruction == "selected"
    assert result.result_snapshot() == {"best_idx": 0}
    assert result.candidate_tree_html() == "<html>tree</html>"


def test_feedback_uses_the_harness_wire_format() -> None:
    from interpretability_gepa.datasets import CIVIL_LABELS
    from interpretability_gepa.gepa_runner import MultilabelGepaAdapter
    from interpretability_gepa.prompts import parse_labels

    adapter = MultilabelGepaAdapter(FakeProvider([]), CIVIL_LABELS)
    text = adapter.feedback(
        {
            "text": "t",
            "gold": ["insult", "toxicity"],
            "predicted": [],
            "error": None,
        }
    )
    expected = next(line for line in text.splitlines() if line.startswith("Expected: "))
    rendered = expected.removeprefix("Expected: ")

    # A Python repr, or alphabetical order, would both fail the parser.
    assert rendered == "toxicity, insult"
    assert parse_labels(rendered, CIVIL_LABELS) == ("toxicity", "insult")
    assert "Predicted: NONE" in text


def test_output_contract_names_the_schema_order() -> None:
    from interpretability_gepa.datasets import CIVIL_LABELS
    from interpretability_gepa.prompts import describe_output_contract

    contract = describe_output_contract(CIVIL_LABELS)

    assert ", ".join(CIVIL_LABELS) in contract
    assert "not alphabetical" in contract
