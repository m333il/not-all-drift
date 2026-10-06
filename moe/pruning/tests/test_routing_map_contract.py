"""The routing map must be measured on the contract the arm was trained under.

Measuring a v25 arm with the v24 renderer counts routing on a prompt the arm
never saw, which silently makes the frequency map - and every pruning decision
taken from it - describe the wrong model.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mrd_pruning.task import (
    SEED_SYSTEM_PROMPT,
    V25_SEED_INSTRUCTION,
    render_user_prompt,
    render_user_prompt_v25,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "measure_routing_map", ROOT / "scripts" / "measure_routing_map.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


measure = _load_module()


def _args(contract: str, arm: str = "prompt_tuning", adapter: Path | None = None):
    return measure.parse_args(
        [
            "--arm", arm,
            "--data", "d.jsonl",
            "--answers", "a.jsonl",
            "--out", "o",
            "--prompt-contract", contract,
            *(["--adapter", str(adapter)] if adapter else []),
        ]
    )


def test_v24_is_the_default_contract():
    args = measure.parse_args(
        ["--arm", "base", "--data", "d.jsonl", "--answers", "a.jsonl", "--out", "o"]
    )
    assert args.prompt_contract == "v24"


def test_base_keeps_the_system_prompt_under_v24():
    spec = measure.build_arm(_args("v24", arm="base"))
    assert spec.system_prompt_text == SEED_SYSTEM_PROMPT
    assert spec.prompt_in_user_turn is False


def test_base_drops_the_system_prompt_under_v25():
    spec = measure.build_arm(_args("v25", arm="base"))
    assert spec.system_prompt_text is None
    assert spec.prompt_in_user_turn is True


def test_adapter_arm_records_the_user_turn_contract(tmp_path):
    spec = measure.build_arm(_args("v25", adapter=tmp_path))
    assert spec.prompt_in_user_turn is True
    assert spec.system_prompt_text is None


def test_the_two_contracts_render_different_prompts():
    comment = "some comment"
    v24 = render_user_prompt(comment)
    v25 = render_user_prompt_v25(comment, V25_SEED_INSTRUCTION)
    assert v24 != v25
    assert V25_SEED_INSTRUCTION in v25
    assert V25_SEED_INSTRUCTION not in v24


@pytest.mark.parametrize("contract", ["v24", "v25"])
def test_contract_is_accepted_by_the_cli(contract):
    assert _args(contract, arm="base").prompt_contract == contract
