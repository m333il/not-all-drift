from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from interpretability_gepa.activations import select_hidden_states  # noqa: E402


def _states(batch: int, tokens: int, hidden: int) -> list[torch.Tensor]:
    values = torch.arange(batch * tokens * hidden, dtype=torch.float32)
    return [values.reshape(batch, tokens, hidden)]


def test_virtual_token_prefix_is_dropped_before_selecting_positions() -> None:
    # Input-layer prompt tuning makes the residual stream longer than the attention
    # mask. If the prefix is not dropped, the "last prompt" position lands inside the
    # virtual tokens and the text mean averages over states that are not text at all.
    tokens, virtual = 4, 3
    mask = torch.tensor([[1, 1, 1, 0]])
    text = torch.tensor([[0, 1, 1, 0]])
    plain = _states(1, tokens, 2)
    padded = [torch.cat([torch.full((1, virtual, 2), -99.0), plain[0]], dim=1)]

    baseline_last, baseline_mean = select_hidden_states(plain, mask, text)
    shifted_last, shifted_mean = select_hidden_states(padded, mask, text)

    assert np.array_equal(baseline_last, shifted_last)
    assert np.array_equal(baseline_mean, shifted_mean)


def test_a_residual_stream_shorter_than_the_mask_is_rejected() -> None:
    mask = torch.tensor([[1, 1, 1, 1, 1]])
    text = torch.tensor([[0, 1, 1, 0, 0]])

    with pytest.raises(ValueError, match="shorter than the attention mask"):
        select_hidden_states(_states(1, 3, 2), mask, text)


class _Block(torch.nn.Module):
    def __init__(self, tag: float) -> None:
        super().__init__()
        self.tag = tag

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.tag


class _Stack(torch.nn.Module):
    def __init__(self, depth: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(_Block(float(i + 1)) for i in range(depth))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def test_store_index_maps_onto_the_block_one_below_it() -> None:
    # The store keeps one entry more than the model has blocks: index 0 is the embedding
    # output. An intervention derived from store index i must therefore hook block i-1.
    # Passing the store index straight to a block hook steers one layer too deep, which is
    # exactly what the first two E3 sweeps did.
    from interpretability_gepa.activations import forward_with_residuals
    from interpretability_gepa.causal import block_for_residual_index

    model = _Stack(4)
    hidden = torch.zeros(1, 1, 2)
    seen: dict[int, torch.Tensor] = {}
    handles = [
        block.register_forward_hook(
            lambda _m, _i, output, index=index: seen.__setitem__(index, output.detach())
        )
        for index, block in enumerate(model.layers)
    ]
    try:
        _, captured = forward_with_residuals(model, hidden=hidden)
    finally:
        for handle in handles:
            handle.remove()

    assert len(captured) == len(model.layers) + 1
    for store_index in range(1, len(captured)):
        block = block_for_residual_index(store_index)
        assert torch.equal(captured[store_index], seen[block])

    with pytest.raises(ValueError, match="embedding output"):
        block_for_residual_index(0)
