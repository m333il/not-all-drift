"""The tally must count what the model actually routed, under the mask.

Counting is worthless if it counts the wrong thing, and every way of getting it
wrong here is quiet. A tally that includes pruned experts would say the mask
leaked. One that counts logits instead of winners would report every expert on
every token. One that misses the decode steps would describe the prompt only.

The gates come in two shapes and the hook takes a different branch for each, so
both are exercised: a plain linear that returns logits (Qwen's) and a router
that picks its own top-k and hands back winners (gpt-oss's).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from mrd_pruning.masking import ExpertMask, RoutingTally


N_EXPERTS = 8
N_LAYERS = 3
TOP_K = 2


class LogitGate(nn.Linear):
    """Qwen's shape: raw logits out, the block takes top-k downstream."""

    def __init__(self, seed: int) -> None:
        super().__init__(4, N_EXPERTS, bias=False)
        torch.manual_seed(seed)
        with torch.no_grad():
            self.weight.copy_(torch.randn(N_EXPERTS, 4))


class TopKGate(nn.Linear):
    """gpt-oss's shape: picks its own top-k, returns (scores, indices)."""

    def __init__(self, seed: int) -> None:
        super().__init__(4, N_EXPERTS, bias=True)
        self.top_k = TOP_K
        torch.manual_seed(seed)
        with torch.no_grad():
            self.weight.copy_(torch.randn(N_EXPERTS, 4))
            self.bias.zero_()

    def forward(self, x):  # noqa: D102
        logits = super().forward(x)
        scores, idx = logits.topk(self.top_k, dim=-1)
        return scores, idx


class Mlp(nn.Module):
    """A block is recognised as MoE by owning both a gate and experts."""

    def __init__(self, gate: nn.Module) -> None:
        super().__init__()
        self.gate = gate
        self.experts = nn.ModuleList(nn.Identity() for _ in range(N_EXPERTS))

    def forward(self, hidden):
        return self.gate(hidden)


class Layer(nn.Module):
    def __init__(self, gate: nn.Module) -> None:
        super().__init__()
        self.mlp = Mlp(gate)


class Tiny(nn.Module):
    """Enough structure for `discover_gates` to find the gates in layer order."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        make = LogitGate if kind == "logit" else TopKGate
        self.layers = nn.ModuleList([Layer(make(seed=i)) for i in range(N_LAYERS)])

    def forward(self, x):
        for layer in self.layers:
            layer.mlp(x)
        return x


@pytest.fixture(params=["logit", "topk"])
def kind(request):
    return request.param


def _run(model: Tiny, pruned: dict[int, list[int]], batches: int = 3):
    tally = RoutingTally(n_layers=N_LAYERS, n_experts=N_EXPERTS)
    torch.manual_seed(0)
    with ExpertMask(model, pruned, top_k=TOP_K, tally=tally):
        for _ in range(batches):
            model(torch.randn(5, 4))
    return tally


def test_every_token_is_counted_exactly_top_k_times(kind):
    """Each token picks top_k experts, so the total is rows × top_k × layers.

    A tally that misses the decode steps, or double-counts a batch, shows up
    here and nowhere else.
    """
    model = Tiny(kind)
    tally = _run(model, {0: [0, 1]}, batches=3)
    assert tally.total == 3 * 5 * TOP_K * N_LAYERS


def test_a_pruned_expert_never_appears_in_the_tally(kind):
    """The count is over winners, so a masked expert must be at zero.

    If it is not, either the mask did not apply or the tally is reading the
    logits instead of the selection - and both make every conclusion about
    where the traffic moved wrong.
    """
    model = Tiny(kind)
    removed = [0, 1, 2]
    tally = _run(model, {1: removed})
    counts = tally.as_array().numpy()
    assert counts[1][removed].sum() == 0, (
        f"knocked out experts got {counts[1][removed].sum()} assignments")
    assert counts[1].sum() > 0, "The masked layer received no assignments."


def test_layers_without_a_mask_are_counted_too(kind):
    """Only one layer is pruned here; the others must still be measured.

    The early-return branch for unmasked layers is a separate code path, and it
    is the one a sweep hits on most layers.
    """
    model = Tiny(kind)
    tally = _run(model, {1: [0, 1]})
    counts = tally.as_array().numpy()
    assert (counts.sum(axis=1) > 0).all(), (
        f"layers without designation: {np.flatnonzero(counts.sum(axis=1) == 0).tolist()}")


def test_the_tally_matches_a_hand_count(kind):
    """Against the selection recomputed outside the hook, expert by expert."""
    model = Tiny(kind)
    pruned = {2: [5, 6, 7]}
    torch.manual_seed(0)
    xs = [torch.randn(5, 4) for _ in range(2)]

    tally = RoutingTally(n_layers=N_LAYERS, n_experts=N_EXPERTS)
    with ExpertMask(model, pruned, top_k=TOP_K, tally=tally):
        for x in xs:
            model(x)
    got = tally.as_array().numpy()

    want = np.zeros((N_LAYERS, N_EXPERTS), dtype=int)
    for x in xs:
        for i, layer in enumerate(model.layers):
            gate = layer.mlp.gate
            logits = nn.functional.linear(x, gate.weight,
                                          getattr(gate, "bias", None))
            if i in pruned:
                logits = logits.clone()
                logits[:, pruned[i]] = torch.finfo(logits.dtype).min
            idx = logits.topk(TOP_K, dim=-1).indices.reshape(-1)
            for e in idx.tolist():
                want[i][e] += 1
    np.testing.assert_array_equal(got, want)


def test_counting_does_not_change_what_the_model_routes(kind):
    """The tally is an observer; with it off or on the selection is identical.

    A measurement that perturbs the thing measured would make the quality number
    and the distribution describe different runs.
    """
    pruned = {0: [3, 4]}
    torch.manual_seed(0)
    xs = [torch.randn(5, 4) for _ in range(2)]

    picks = []
    for tally in (None, RoutingTally(n_layers=N_LAYERS, n_experts=N_EXPERTS)):
        model = Tiny(kind)
        seen: list[list[int]] = []
        with ExpertMask(model, pruned, top_k=TOP_K, tally=tally):
            for x in xs:
                for i, layer in enumerate(model.layers):
                    out = layer.mlp.gate(x)
                    idx = (out[1] if isinstance(out, tuple)
                           else out.topk(TOP_K, dim=-1).indices)
                    seen.append(idx.reshape(-1).tolist())
        picks.append(seen)
    assert picks[0] == picks[1]


def test_the_tally_never_reads_a_value_back_to_the_host(kind, monkeypatch):
    """The counting path must queue kernels, not synchronise on them.

    `torch.bincount` sizes its output from the maximum of its input, and that
    read stalls the GPU queue - once per layer per decode step, behind every
    other tenant's work on a shared card. Measured before the fix, 300 steps on
    an H200 with neighbours: Qwen 182s → 350s, gpt-oss 0.5s → 75s.

    A CPU test cannot time that, so it pins the cause instead: the operations
    that force a device read must not appear on the hot path.
    """
    import torch as t

    forbidden = []
    for name in ("bincount", "unique", "max"):
        original = getattr(t, name)

        def trap(*args, _name=name, _orig=original, **kwargs):
            forbidden.append(_name)
            return _orig(*args, **kwargs)

        monkeypatch.setattr(t, name, trap)

    model = Tiny(kind)
    _run(model, {0: [0, 1]}, batches=2)
    assert not forbidden, f"on the hot path synchronizing calls: {sorted(set(forbidden))}"


def test_the_counts_table_is_allocated_once(kind):
    """The table and the ones-buffer are reused; growth only when a batch is wider."""
    model = Tiny(kind)
    tally = RoutingTally(n_layers=N_LAYERS, n_experts=N_EXPERTS)
    with ExpertMask(model, {1: [0]}, top_k=TOP_K, tally=tally):
        model(torch.randn(5, 4))
        first = tally._table.data_ptr()
        model(torch.randn(5, 4))
        assert tally._table.data_ptr() == first, "The table was moved between steps"
        model(torch.randn(40, 4))     # wider batch: the ones buffer must grow
        assert tally._table.data_ptr() == first, "The table should not depend on the size of the batch"
    assert tally.total == (5 + 5 + 40) * TOP_K * N_LAYERS
