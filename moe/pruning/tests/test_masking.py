"""Expert masking against a miniature MoE-shaped model."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from mrd_pruning.masking import ExpertMask, discover_gates, iter_prune_levels  # noqa: E402

N_EXPERTS = 16
TOP_K = 4


class _Block(nn.Module):
    """Mimics the shape a Qwen3 MoE block presents: a gate plus experts."""

    def __init__(self, n_experts: int = N_EXPERTS) -> None:
        super().__init__()
        self.gate = nn.Linear(8, n_experts, bias=False)
        self.experts = nn.ModuleList(nn.Identity() for _ in range(n_experts))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.gate(hidden)          # [tokens, n_experts]
        chosen = logits.topk(TOP_K, -1).indices
        self.last_chosen = chosen
        return hidden


class _Layer(nn.Module):
    def __init__(self, dense: bool = False) -> None:
        super().__init__()
        self.mlp = nn.Identity() if dense else _Block()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden)


class _Model(nn.Module):
    def __init__(self, n_layers: int = 3, dense_first: bool = False) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _Layer(dense=(dense_first and i == 0)) for i in range(n_layers)
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def _hidden(tokens: int = 5) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(tokens, 8)


def test_discover_gates_skips_dense_layers() -> None:
    gates = discover_gates(_Model(n_layers=3, dense_first=True))
    assert [g.layer_idx for g in gates] == [1, 2]
    assert all(g.num_experts == N_EXPERTS for g in gates)


def test_discover_gates_through_wrapper() -> None:
    class _Wrapped(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_model = _Model()

    assert len(discover_gates(_Wrapped())) == 3


def test_pruned_experts_are_never_selected() -> None:
    model = _Model()
    pruned = {0: [0, 1, 2, 3, 4], 1: [10, 11], 2: []}
    with ExpertMask(model, pruned, top_k=TOP_K) as mask:
        model(_hidden())
        mask.assert_applied()
    for idx, layer in enumerate(model.layers):
        chosen = set(layer.mlp.last_chosen.reshape(-1).tolist())
        assert not chosen & set(pruned[idx]), f"layer {idx} selected a pruned expert"


def test_mask_is_active_during_every_forward_not_just_the_first() -> None:
    """Pruning is permanent: unlike the transplant override, it must hold on
    every decode step, not only prefill."""
    model = _Model()
    with ExpertMask(model, {0: [0, 1]}, top_k=TOP_K) as mask:
        for _ in range(3):
            model(_hidden())
        assert mask.audit.calls[0] == 3
    assert mask.audit.max_selected_pruned == 0


def test_zero_pruning_leaves_logits_untouched() -> None:
    model = _Model()
    hidden = _hidden()
    baseline = model(hidden)
    chosen_before = [layer.mlp.last_chosen.clone() for layer in model.layers]
    with ExpertMask(model, {}, top_k=TOP_K) as mask:
        after = model(hidden)
        mask.assert_applied()
    assert torch.equal(baseline, after)
    for before, layer in zip(chosen_before, model.layers):
        assert torch.equal(before, layer.mlp.last_chosen)


def test_assert_applied_raises_when_hooks_never_fired() -> None:
    model = _Model()
    with ExpertMask(model, {0: [0]}, top_k=TOP_K) as mask:
        pass  # no forward at all
    with pytest.raises(RuntimeError, match="never fired"):
        mask.assert_applied()


def test_refuses_to_leave_fewer_experts_than_top_k() -> None:
    model = _Model()
    with pytest.raises(ValueError, match="top_k"):
        ExpertMask(model, {0: list(range(N_EXPERTS - TOP_K + 1))}, top_k=TOP_K)


def test_rejects_unknown_layers_and_expert_ids() -> None:
    model = _Model()
    with pytest.raises(ValueError, match="non-MoE layers"):
        ExpertMask(model, {99: [0]}, top_k=TOP_K)
    with pytest.raises(ValueError, match="out of range"):
        ExpertMask(model, {0: [N_EXPERTS]}, top_k=TOP_K)


def test_mask_value_matches_logit_dtype() -> None:
    """A float32 constant written into a bf16 tensor killed a whole run once."""
    model = _Model().to(torch.bfloat16)
    with ExpertMask(model, {0: [0, 1]}, top_k=TOP_K) as mask:
        model(_hidden().to(torch.bfloat16))
        mask.assert_applied()


def test_iter_prune_levels_validates_before_loading() -> None:
    assert list(iter_prune_levels([0, 8], 128, 8)) == [0, 8]
    with pytest.raises(ValueError, match="below top_k"):
        list(iter_prune_levels([124], 128, 8))
    with pytest.raises(ValueError, match="negative"):
        list(iter_prune_levels([-1], 128, 8))


def test_discover_gates_accepts_the_router_spelling() -> None:
    """gpt-oss names the gate ``router`` and keeps the matrix itself rather than
    an nn.Linear, so a name- or type-based lookup silently finds nothing and the
    model reads as dense."""
    import torch
    from torch import nn

    class _Router(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_experts = 32
            self.weight = nn.Parameter(torch.zeros(32, 8))

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = _Router()
            self.experts = nn.ModuleList(nn.Identity() for _ in range(32))

    class _Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = _Block()

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(_Layer() for _ in range(3))

    gates = discover_gates(_Model())
    assert [g.layer_idx for g in gates] == [0, 1, 2]
    assert {g.num_experts for g in gates} == {32}
