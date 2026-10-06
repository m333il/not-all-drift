from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from interpretability_gepa.logit_lens import (  # noqa: E402
    layerwise_teacher_forced_logprobs,
)
from interpretability_gepa.modeling import register_model_family  # noqa: E402


class _Tokenizer:
    """Character-level tokenizer, so offsets and token indices coincide."""

    eos_token_id = 0

    def __call__(self, text: str, **kwargs: Any) -> Any:
        ids = [ord(character) % 16 + 1 for character in text]
        offsets = [(index, index + 1) for index in range(len(text))]
        return type("Encoded", (), {"input_ids": ids, "offset_mapping": offsets})()


class _Block(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class _Model(torch.nn.Module):
    """A model that optionally prepends virtual tokens, as prompt tuning does.

    Every scored position carries a one-hot state identifying the position it came
    from, so a shifted readout is visible directly in which token wins.
    """

    def __init__(self, vocab: int, virtual_tokens: int) -> None:
        super().__init__()
        self.virtual_tokens = virtual_tokens
        self.layers = torch.nn.ModuleList([_Block(), _Block()])
        self.lm_head = torch.nn.Linear(vocab, vocab, bias=False)
        self.lm_head.weight.data = torch.eye(vocab) * 10.0
        self.config = type("Config", (), {"final_logit_softcapping": None})()
        self.vocab = vocab

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        if self.virtual_tokens >= 0:
            padded = torch.cat(
                [torch.zeros(1, self.virtual_tokens, dtype=torch.long), input_ids], dim=1
            )
        else:
            padded = input_ids[:, -self.virtual_tokens :]
        hidden = torch.nn.functional.one_hot(padded, num_classes=self.vocab).float()
        for layer in self.layers:
            hidden = layer(hidden)
        return type("Output", (), {"logits": hidden})()


@register_model_family("logit_lens_test")
class _Adapter:
    family = "logit_lens_test"

    def layer_metadata(self, layers: int) -> tuple:
        return ()

    @staticmethod
    def final_norm(model: Any) -> None:
        return None


def _score(virtual_tokens: int) -> np.ndarray:
    model = _Model(vocab=32, virtual_tokens=virtual_tokens)
    scores, counts = layerwise_teacher_forced_logprobs(
        model=model,
        tokenizer=_Tokenizer(),
        family="logit_lens_test",
        prompts=["abcd"],
        candidate_strings=["ef", "gh"],
    )
    assert counts.tolist() == [2, 2]
    return scores


def test_prompt_tuning_offset_keeps_the_scored_positions_aligned() -> None:
    # The model copies each input token into its own residual slot, so a correct
    # readout scores the token that actually precedes each target. If the virtual
    # tokens were ignored the readout would land on the padding instead and the
    # scores would move; they must not.
    baseline = _score(virtual_tokens=0)
    shifted = _score(virtual_tokens=7)

    assert np.allclose(baseline, shifted, atol=1e-5)


def test_a_shorter_residual_stream_is_rejected() -> None:
    model = _Model(vocab=32, virtual_tokens=0)
    model.virtual_tokens = -2

    with pytest.raises(ValueError, match="shorter than the input sequence"):
        layerwise_teacher_forced_logprobs(
            model=model,
            tokenizer=_Tokenizer(),
            family="logit_lens_test",
            prompts=["abcd"],
            candidate_strings=["ef"],
        )
