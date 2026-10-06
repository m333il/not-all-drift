from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from .prompts import (
    MULTILABEL_PADDING_FILLER,
    neutral_length_padding,
)


@dataclass(frozen=True)
class Condition:
    id: str
    instruction: str
    adapter: str | None = None
    control: bool = False

    def content_hash(self) -> str:
        payload = f"{self.id}\0{self.instruction}\0{self.adapter or ''}\0{self.control}"
        return hashlib.sha256(payload.encode()).hexdigest()


def build_text_conditions(
    seed: str,
    adapted: str,
    tokenizer: Any,
    seed_value: int,
    *,
    prefix_adapter: str | None = None,
    random_prefix_adapter: str | None = None,
    bland_instruction: str = "List every applicable label from the allowed category list.",
    padding_filler: str = MULTILABEL_PADDING_FILLER,
) -> tuple[Condition, ...]:
    """Build the text condition ladder (null, bland, seed, padded seed, adapted, ...)."""
    sentences = [part.strip() for part in adapted.replace("!", ".").split(".") if part.strip()]
    rng = random.Random(seed_value)
    shuffled = []
    for index in range(3):
        copy = sentences.copy()
        rng.shuffle(copy)
        shuffled.append(Condition(f"C_adapt_shuf{index}", ". ".join(copy) + ".", control=True))
    irrelevant = neutral_length_padding(
        tokenizer,
        "Summarize the typography and punctuation without changing the required output format.",
        adapted,
        filler=padding_filler,
    )
    conditions = [
        Condition("C_null", ""),
        Condition("C_bland", bland_instruction),
        Condition("C_seed", seed),
        Condition("C_adapt", adapted),
        Condition(
            "C_seed_pad",
            neutral_length_padding(tokenizer, seed, adapted, filler=padding_filler),
            control=True,
        ),
        *shuffled,
        Condition("C_adapt_rand", irrelevant, control=True),
    ]
    if prefix_adapter is not None:
        conditions.append(Condition("C_prefix", seed, adapter=prefix_adapter))
    if random_prefix_adapter is not None:
        conditions.append(
            Condition("C_prefix_rand", seed, adapter=random_prefix_adapter, control=True)
        )
    return tuple(conditions)
