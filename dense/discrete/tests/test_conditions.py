from __future__ import annotations

from interpretability_gepa.conditions import build_text_conditions
from interpretability_gepa.prompts import render_condition_messages


class WordTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text.split())))

    def decode(self, token_ids: list[int]) -> str:
        return " ".join("x" for _ in token_ids)


def test_control_conditions_are_deterministic_and_length_matched() -> None:
    tokenizer = WordTokenizer()
    first = build_text_conditions(
        "short seed", "One sentence. Two sentence. Three sentence.", tokenizer, 42
    )
    second = build_text_conditions(
        "short seed", "One sentence. Two sentence. Three sentence.", tokenizer, 42
    )
    assert first == second
    assert {condition.id for condition in first} >= {
        "C_null",
        "C_bland",
        "C_seed",
        "C_adapt",
        "C_seed_pad",
        "C_adapt_rand",
    }
    padded = next(x for x in first if x.id == "C_seed_pad")
    assert len(tokenizer.encode(padded.instruction)) == len(
        tokenizer.encode("One sentence. Two sentence. Three sentence.")
    )


def test_prefix_conditions_share_seed_instruction() -> None:
    conditions = build_text_conditions(
        "seed",
        "adapted instruction",
        WordTokenizer(),
        42,
        prefix_adapter="best-prefix",
        random_prefix_adapter="random-prefix",
    )

    prefix = next(condition for condition in conditions if condition.id == "C_prefix")
    placebo = next(condition for condition in conditions if condition.id == "C_prefix_rand")
    assert prefix.instruction == placebo.instruction == "seed"
    assert prefix.adapter == "best-prefix"
    assert placebo.control and placebo.adapter == "random-prefix"


def test_null_condition_contains_no_label_or_format_instruction() -> None:
    rendered = render_condition_messages("C_null", "classify", ("joy", "anger"), "hello")

    assert rendered.messages == ({"role": "user", "content": "hello"},)
