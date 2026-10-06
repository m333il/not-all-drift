import importlib.util
import json
from pathlib import Path
from threading import Barrier, Lock

import pytest

from mrd.civil_v2_contract import (
    LABELS,
    SEED_INSTRUCTION,
    parse_labels,
    reflection_feedback,
    render_user_prompt,
    score_response,
)
from mrd.llm_client import LLMResponse


spec = importlib.util.spec_from_file_location(
    "optimize_gepa",
    Path(__file__).parents[1] / "scripts/optimize_gepa.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_redact_proxy_credentials():
    assert runner.redact_proxy("socks5h://alice:secret@example.test:49173") == (
        "socks5h://example.test:49173"
    )


def test_exact_user_only_prompt_contract():
    assert render_user_prompt(SEED_INSTRUCTION, "hello") == (
        f"{SEED_INSTRUCTION}\n"
        f"Labels (use exact names): {', '.join(LABELS)}\n"
        "Text:\nhello\n"
        "Return every applicable label in the order listed, separated by commas. "
        "If no label applies, return exactly NONE. Return no other text.\n"
        "Answer:"
    )


def test_v2_split_rows_receive_stable_keys_and_groups():
    row = {"id": "abc", "group_id": None, "text": "hello", "labels": []}
    normalized = runner.normalize_example(row)
    assert normalized["key"] == "abc"
    assert normalized["split_group"] == "abc"


def test_strict_parser_and_empty_aware_f1():
    assert parse_labels("toxicity, insult") == ("toxicity", "insult")
    with pytest.raises(ValueError, match="canonical order"):
        parse_labels("insult, toxicity")
    assert score_response({"labels": []}, "NONE")[0] == 1.0
    assert score_response({"labels": ["toxicity", "insult"]}, "insult")[0] == 2 / 3
    assert score_response({"labels": ["toxicity"]}, "Toxicity")[0] == 0.0


def test_feedback_matches_the_dense_gepa_wire_shape():
    feedback = reflection_feedback(
        {"text": "sample", "labels": ["toxicity", "insult"]},
        ["insult"],
        None,
    )
    assert feedback == (
        "Text: sample\nExpected: toxicity, insult\nPredicted: insult\n"
        "Missed: toxicity\nExtra: NONE\nParse error: None"
    )


def test_adapter_sends_candidate_in_only_user_message(tmp_path):
    observed = {}

    class Client:
        def generate(self, prompt, **kwargs):
            observed.update(prompt=prompt, kwargs=kwargs)
            return LLMResponse("NONE", "model", finish_reason="stop")

    with (tmp_path / "eval.jsonl").open("w") as stream:
        adapter = runner.CivilV2Adapter(Client(), stream)
        batch = [{"key": "k", "text": "hello", "labels": []}]
        result = adapter.evaluate(batch, {runner.COMPONENT: "candidate"}, capture_traces=True)
    assert observed == {
        "prompt": render_user_prompt("candidate", "hello"),
        "kwargs": {"max_tokens": runner.TASK_MAX_TOKENS},
    }
    assert result.scores == [1.0]
    reflective = adapter.make_reflective_dataset(
        {runner.COMPONENT: "candidate"}, result, [runner.COMPONENT]
    )
    assert reflective[runner.COMPONENT][0]["Generated Outputs"] == []


def test_adapter_rejects_truncated_task_response(tmp_path):
    class Client:
        def generate(self, prompt, **kwargs):
            return LLMResponse("", "model", finish_reason="length")

    with (tmp_path / "eval.jsonl").open("w") as stream:
        adapter = runner.CivilV2Adapter(Client(), stream)
        with pytest.raises(RuntimeError, match="token ceiling"):
            adapter.evaluate(
                [{"key": "k", "text": "hello", "labels": []}],
                {runner.COMPONENT: "candidate"},
            )


def test_adapter_runs_task_batch_concurrently_and_logs_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "TASK_CONCURRENCY", 4)
    barrier = Barrier(4)
    lock = Lock()
    active = 0
    peak = 0

    class Client:
        def generate(self, prompt, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            barrier.wait(timeout=2)
            with lock:
                active -= 1
            return LLMResponse("NONE", "model", finish_reason="stop")

    path = tmp_path / "eval.jsonl"
    with path.open("w") as stream:
        adapter = runner.CivilV2Adapter(Client(), stream)
        batch = [
            {"key": f"k{index}", "text": f"text {index}", "labels": []}
            for index in range(4)
        ]
        result = adapter.evaluate(batch, {runner.COMPONENT: "candidate"})
    assert peak == 4
    assert result.scores == [1.0] * 4
    assert [json.loads(line)["key"] for line in path.read_text().splitlines()] == [
        "k0", "k1", "k2", "k3"
    ]


def test_reflection_template_states_fixed_contract_and_budget():
    template = runner.reflection_prompt_template()
    assert "<curr_param>" in template and "<side_info>" in template
    assert "under 8192 tokens" in template
    assert "toxicity, obscene, threat, insult, identity_attack" in template
    assert template.rstrip().endswith("Provide the new instructions within ``` blocks.")
