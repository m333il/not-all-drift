"""Pruning must work on a gate that takes its own top-k (gpt-oss's router).

Qwen's gate returns raw logits and the block picks top-k downstream, so masking
the gate output is enough. gpt-oss picks top-k inside the router and returns
``(scores, indices)``; masking that output would only zero an already-selected
expert instead of removing it from the candidates, and the old code crashed on
it outright with ``'tuple' object has no attribute 'device'``.
"""

from __future__ import annotations

import torch
from torch import nn

from mrd_pruning.masking import ExpertMask, takes_own_topk

N_EXPERTS = 8
TOP_K = 2
HIDDEN = 4


class SelfTopKRouter(nn.Module):
    """Same shape of contract as transformers' GptOssTopKRouter."""

    def __init__(self) -> None:
        super().__init__()
        self.top_k = TOP_K
        self.num_experts = N_EXPERTS
        self.weight = nn.Parameter(torch.randn(N_EXPERTS, HIDDEN))
        self.bias = nn.Parameter(torch.zeros(N_EXPERTS))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, HIDDEN)
        logits = torch.nn.functional.linear(hidden_states, self.weight, self.bias)
        top_value, indices = torch.topk(logits, self.top_k, dim=-1)
        scores = torch.zeros_like(logits).scatter_(1, indices, top_value.softmax(dim=1))
        return scores, indices


class LogitGate(nn.Linear):
    """Qwen's shape of contract: raw logits, top-k taken downstream."""

    def __init__(self) -> None:
        super().__init__(HIDDEN, N_EXPERTS, bias=False)


class Block(nn.Module):
    def __init__(self, gate: nn.Module) -> None:
        super().__init__()
        self.gate = gate
        self.experts = nn.ModuleList([nn.Identity() for _ in range(N_EXPERTS)])


class Layer(nn.Module):
    def __init__(self, gate: nn.Module) -> None:
        super().__init__()
        self.mlp = Block(gate)


class Model(nn.Module):
    def __init__(self, gate_factory, n_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Layer(gate_factory()) for _ in range(n_layers)])

    def forward(self, hidden):
        out = None
        for layer in self.layers:
            out = layer.mlp.gate(hidden)
        return out


def test_router_with_own_topk_is_detected():
    assert takes_own_topk(SelfTopKRouter()) is True


def test_plain_logit_gate_is_not_detected():
    gate = LogitGate()
    assert takes_own_topk(gate) is False


def test_pruned_experts_are_never_selected():
    torch.manual_seed(0)
    model = Model(SelfTopKRouter)
    hidden = torch.randn(16, HIDDEN)
    pruned = {0: [1, 3, 5], 1: [0, 2]}

    with ExpertMask(model, pruned, top_k=TOP_K) as mask:
        for layer_idx, layer in enumerate(model.layers):
            _, indices = layer.mlp.gate(hidden)
            for expert in pruned[layer_idx]:
                assert not bool((indices == expert).any()), (
                    f"layer {layer_idx}: pruned expert {expert} was still selected"
                )
        assert mask.audit.max_selected_pruned == 0


def test_bias_is_restored_on_exit():
    torch.manual_seed(0)
    model = Model(SelfTopKRouter, n_layers=1)
    before = model.layers[0].mlp.gate.bias.detach().clone()

    with ExpertMask(model, {0: [2, 4]}, top_k=TOP_K):
        during = model.layers[0].mlp.gate.bias.detach().clone()
        assert during[2] < before[2]

    after = model.layers[0].mlp.gate.bias.detach()
    assert torch.equal(after, before)


def test_unpruned_run_leaves_the_router_untouched():
    torch.manual_seed(0)
    model = Model(SelfTopKRouter, n_layers=1)
    hidden = torch.randn(8, HIDDEN)
    _, baseline = model.layers[0].mlp.gate(hidden)

    with ExpertMask(model, {}, top_k=TOP_K):
        _, masked = model.layers[0].mlp.gate(hidden)
    assert torch.equal(baseline, masked)


def test_logit_gate_path_still_masks_the_output():
    torch.manual_seed(0)
    model = Model(LogitGate, n_layers=1)
    hidden = torch.randn(8, HIDDEN)

    with ExpertMask(model, {0: [1, 6]}, top_k=TOP_K):
        logits = model.layers[0].mlp.gate(hidden)
    assert torch.all(logits[:, 1] == torch.finfo(logits.dtype).min)
    assert torch.all(logits[:, 6] == torch.finfo(logits.dtype).min)


def test_audit_counts_the_masked_experts_on_both_gate_kinds():
    torch.manual_seed(0)
    for factory in (SelfTopKRouter, LogitGate):
        model = Model(factory, n_layers=1)
        hidden = torch.randn(8, HIDDEN)
        with ExpertMask(model, {0: [3, 7]}, top_k=TOP_K) as mask:
            model.layers[0].mlp.gate(hidden)
        assert mask.audit.masked_experts[0] == 2, factory.__name__
        assert mask.audit.calls[0] >= 1, factory.__name__
