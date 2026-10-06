"""Reasoning tokens must be counted apart from answer tokens.

A reasoning-mode routing map exists to show where the model routes while it
thinks. Labelling the whole response "answer" would fold that into the same
counter as the two-word verdict and hide it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from mrd_pruning.task import FINAL_CHANNEL

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "measure_routing_map", ROOT / "scripts" / "measure_routing_map.py"
)
measure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measure)


class CharTokenizer:
    """One token per character: token counts are string lengths."""

    chat_template = "<|channel|> template"

    def __call__(self, text, **kw):
        return {"input_ids": [ord(c) for c in text]}


def test_plain_answer_is_all_answer():
    ids, stages = measure.split_answer(CharTokenizer(), "toxicity, insult")
    assert len(ids) == len("toxicity, insult")
    assert set(stages) == {"answer"}


def test_reasoning_is_split_at_the_final_marker():
    head = "<|channel|>analysis<|message|>we weigh the labels<|end|>"
    tail = "toxicity"
    ids, stages = measure.split_answer(CharTokenizer(), head + FINAL_CHANNEL + tail)

    assert len(ids) == len(head + FINAL_CHANNEL + tail)
    assert stages.count("reasoning") == len(head + FINAL_CHANNEL)
    assert stages.count("answer") == len(tail)
    # The marker itself belongs to the reasoning side, the verdict to the answer.
    assert stages[len(head + FINAL_CHANNEL) - 1] == "reasoning"
    assert stages[len(head + FINAL_CHANNEL)] == "answer"


def test_empty_answer_after_the_marker_still_splits():
    ids, stages = measure.split_answer(CharTokenizer(), FINAL_CHANNEL)
    assert stages.count("answer") == 0
    assert stages.count("reasoning") == len(FINAL_CHANNEL)
    assert len(ids) == len(stages)


def test_stage_lists_always_match_token_counts():
    for text in ("", "NONE", "a<|channel|>final<|message|>b", FINAL_CHANNEL + "x"):
        ids, stages = measure.split_answer(CharTokenizer(), text)
        assert len(ids) == len(stages), text


def test_reasoning_stage_is_known_to_the_frequency_loader():
    from mrd_pruning.frequency import STAGES, validate_stage_spec

    assert "reasoning" in STAGES
    validate_stage_spec("reasoning")
    validate_stage_spec("reasoning+answer")
