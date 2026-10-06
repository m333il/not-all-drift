"""Calibration counter: counts what was selected, ignores padding."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from mrd_pruning.calibrate import ExpertCounter  # noqa: E402

N_EXPERTS = 8
TOP_K = 2


class _Block(nn.Module):
    def __init__(self, bias: torch.Tensor) -> None:
        super().__init__()
        self.gate = nn.Linear(4, N_EXPERTS, bias=True)
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.bias.copy_(bias)
        self.experts = nn.ModuleList(nn.Identity() for _ in range(N_EXPERTS))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.gate(hidden.reshape(-1, hidden.shape[-1]))
        return hidden


class _Layer(nn.Module):
    def __init__(self, bias: torch.Tensor) -> None:
        super().__init__()
        self.mlp = _Block(bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden)


class _Model(nn.Module):
    """Gate logits are constant, so the expected counts are exact, not noisy."""

    def __init__(self) -> None:
        super().__init__()
        first = torch.zeros(N_EXPERTS)
        first[3], first[5] = 10.0, 9.0
        second = torch.zeros(N_EXPERTS)
        second[0], second[1] = 10.0, 9.0
        self.layers = nn.ModuleList([_Layer(first), _Layer(second)])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        hidden = input_ids.unsqueeze(-1).float().expand(*input_ids.shape, 4)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def test_counts_every_selected_expert() -> None:
    model = _Model()
    ids = torch.ones(2, 3, dtype=torch.long)
    with ExpertCounter(model, top_k=TOP_K) as counter:
        counter.set_keep_mask(torch.ones_like(ids).bool())
        model(ids, torch.ones_like(ids))
        counts = counter.result(source="test", stage="prompt", n_examples=2)
    assert counts.counts.shape == (2, N_EXPERTS)
    np.testing.assert_array_equal(counts.counts[0], [0, 0, 0, 6, 0, 6, 0, 0])
    np.testing.assert_array_equal(counts.counts[1], [6, 6, 0, 0, 0, 0, 0, 0])


def test_padding_positions_are_excluded() -> None:
    """Left padding is real here: counting pad rows biases the frequencies that
    decide which experts get deleted."""
    model = _Model()
    ids = torch.ones(2, 3, dtype=torch.long)
    mask = torch.tensor([[0, 1, 1], [0, 0, 1]])
    with ExpertCounter(model, top_k=TOP_K) as counter:
        counter.set_keep_mask(mask.bool())
        model(ids, mask)
        counts = counter.result(source="test", stage="prompt", n_examples=2)
    assert counts.counts[0].sum() == 3 * TOP_K  # 3 real positions, top-2 each


def test_mismatched_keep_mask_raises_instead_of_miscounting() -> None:
    model = _Model()
    ids = torch.ones(2, 3, dtype=torch.long)
    with ExpertCounter(model, top_k=TOP_K) as counter:
        counter.set_keep_mask(torch.ones(2, 5).bool())
        with pytest.raises(RuntimeError, match="does not describe this batch"):
            model(ids, torch.ones_like(ids))


def test_result_refuses_when_nothing_was_collected() -> None:
    model = _Model()
    with ExpertCounter(model, top_k=TOP_K) as counter:
        with pytest.raises(RuntimeError, match="hooks never fired"):
            counter.result(source="test", stage="prompt", n_examples=0)
