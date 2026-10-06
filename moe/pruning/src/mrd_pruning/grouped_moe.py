"""Replace Qwen3-MoE's per-expert loop with one batched matmul at decode.

What the stock implementation does, in `Qwen3MoeSparseMoeBlock.forward`
(transformers 4.57.6):

    for expert_idx in expert_hit:
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[...]
        final_hidden_states.index_add_(0, top_x, current_hidden_states)

`expert_hit` is every expert that received at least one token. During decode
with a batch of 64 and top-k 8 that is essentially all 128 of them, so each
layer issues ~128 gathers, ~384 small GEMMs and ~128 scatters; across 48 layers
that is thousands of kernel launches to move one token forward. The arithmetic
is trivial - 64 rows against a 2048×768 weight - so the card spends its time
launching rather than multiplying.

The batched form computes every expert for every token in three `bmm` calls and
weights the results by the router, which is exactly what the loop produces:
experts the router did not pick get weight zero. That is the same trick
`GptOssExperts` already uses on GPU, and it is why gpt-oss does not show this
cost.

Two things keep it honest.

Memory. Stacking the expert weights into `(E, H, I)` tensors would double what
the experts occupy - 1.2 GB for Qwen3-30B - if the originals stayed. So the
conversion takes ownership: the stacked parameters replace the `ModuleList`, and
the per-expert Linears are dropped. Peak during conversion is one layer's worth.

Cost at prefill. The dense form materialises `E × tokens × hidden`, which is
fine for 64 decode tokens (128 × 64 × 2048) and ruinous for a 2000-token
prefill. So the batched path is taken only while the token count is small; above
the threshold the original loop runs, unchanged.

This module is opt-in and touches nothing by default: the pruning run that is
already half measured must keep the kernels it started with, because a different
reduction order moves the last bits of the logits and greedy decoding turns that
into a different token on a near-tie.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["group_qwen3_moe", "GROUPED_TOKEN_BUDGET"]

# `E × tokens` rows are materialised, each `hidden` wide. At 128 experts and
# 2048 hidden this is 64 tokens → 16.8M elements → 34 MB in bf16, which is
# nothing next to what the KV cache already holds. A 2000-token prefill would
# be 512 million elements, so it stays on the loop.
GROUPED_TOKEN_BUDGET = 256


class _ExpertCount:
    """Stands in for the `ModuleList` the conversion consumed.

    Holds no tensors: `len()` is all the rest of the code needs from it once the
    weights live in the stacked buffers.
    """

    __slots__ = ("n",)

    def __init__(self, n: int) -> None:
        self.n = int(n)

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<{self.n} Experts, weights collected in w_gate/w_up/w_down>"


def group_qwen3_moe(model, *, token_budget: int = GROUPED_TOKEN_BUDGET,
                    legacy_poison: bool = False) -> int:
    """Convert every Qwen3-MoE block in `model` to the batched form.

    Returns the number of blocks converted, so a caller can fail loudly when it
    patched nothing - a silent no-op would look exactly like "no speedup".

    `legacy_poison` puts the defect back: the combine multiplies instead of
    selecting, so a pruned expert that overflows turns its token into `NaN`.
    Nothing in the pipeline sets it - it exists so the damage can be reproduced
    on demand and shown beside the corrected answers, which is the only way to
    say what the broken levels actually looked like rather than assert it.
    """
    import torch

    converted = 0
    for module in model.modules():
        if type(module).__name__ != "Qwen3MoeSparseMoeBlock":
            continue
        if getattr(module, "_grouped", False):
            continue
        _convert_block(module, torch)
        module._grouped = True
        module._token_budget = int(token_budget)
        module._legacy_poison = bool(legacy_poison)
        module.forward = _grouped_forward.__get__(module, type(module))
        converted += 1
    logger.info("grouped MoE: rebuilt blocks %d (threshold %d tokens)",
                converted, token_budget)
    return converted


def _stack_transposed(experts, attr: str, torch, *, release: bool):
    """Build one (E, in, out) buffer, copying expert by expert.

    `torch.stack(...).transpose(1, 2).contiguous()` is the obvious way to write
    this and it peaks at three copies of the projection: the originals, the
    stack, and the contiguous transpose. Across the three projections that is
    nine copies of a block alive at once where three would do - on Qwen3-30B
    about 3.6 GB of transient per block.

    That transient is what killed the run sharing a card with a gpt-oss
    neighbour on 23-09: the weights fit, the conversion did not, and it died
    asking for 384 MiB with 356 MiB left on a 139.8 GB card.

    Writing straight into the destination and dropping each expert's weight as
    it is consumed holds one extra copy instead of six. `release` is the part
    that frees as it goes - the caller deletes the modules immediately after,
    so nothing reads those tensors again.
    """
    # Shape and dtype only - binding the tensor itself would keep the first
    # expert resident through the whole loop, which is the thing being avoided.
    probe = getattr(experts[0], attr).weight
    out_f, in_f = probe.shape
    dtype, device = probe.dtype, probe.device
    del probe
    # `no_grad` plus `detach` is not belt-and-braces here, it is the whole
    # saving. A plain `copy_` from a Parameter records an autograd edge on the
    # destination, and that edge holds the source alive: clearing the attribute
    # then frees nothing, every original stays resident, and the conversion
    # needs a second full copy of the model. A weakref test caught it.
    with torch.no_grad():
        out = torch.empty((len(experts), in_f, out_f), dtype=dtype, device=device)
        for i, expert in enumerate(experts):
            linear = getattr(expert, attr)
            out[i].copy_(linear.weight.detach().t())
            if release:
                linear.weight = None
    return out


def _convert_block(block, torch, *, release: bool = True) -> None:
    """Stack the experts' weights and drop the per-expert modules.

    Transposed once here so the hot path is a plain `bmm` with no permute:
    (E, tokens, H) @ (E, H, I).
    """
    experts = block.experts
    # The activation comes first: taking it after `release` would read a module
    # whose weights are already gone. It holds no weights itself.
    block.act_fn = experts[0].act_fn
    for name, attr in (("w_gate", "gate_proj"), ("w_up", "up_proj"),
                       ("w_down", "down_proj")):
        block.register_buffer(name, _stack_transposed(experts, attr, torch,
                                                      release=release),
                              persistent=False)
    # Drop the originals, or the experts occupy twice what they did. Measured on
    # Qwen3-30B: 1.2 GB of pure duplication across 48 layers.
    n_experts = len(experts)
    del block.experts
    del experts
    # …but leave something countable in its place. `masking.find_gates` accepts a
    # block only when it has both `.gate` and `.experts`, so a block with the
    # attribute gone is invisible to it and the whole model reads as dense:
    # measured 22-09-2026, the probe died at `no MoE gates found` after eighteen
    # minutes of loading. This placeholder holds no weights - the stacked
    # buffers do - and answers the two questions that code asks of it.
    block.experts = _ExpertCount(n_experts)


def _grouped_forward(self, hidden_states):
    import torch
    import torch.nn.functional as F

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(flat)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(flat.dtype)

    n_tokens = flat.shape[0]
    if n_tokens > self._token_budget:
        return _loop_forward(self, flat, routing_weights, selected,
                             router_logits, batch_size, sequence_length, hidden_dim)

    n_experts = self.w_gate.shape[0]
    # Every expert sees every token; the router's zeros do the selecting. One
    # `scatter` turns the top-k weights into the full (tokens, E) matrix.
    weights = flat.new_zeros((n_tokens, n_experts))
    weights.scatter_(1, selected, routing_weights)

    x = flat.unsqueeze(0).expand(n_experts, n_tokens, hidden_dim)
    inter = self.act_fn(torch.bmm(x, self.w_gate)) * torch.bmm(x, self.w_up)
    out = torch.bmm(inter, self.w_down)                      # (E, tokens, H)

    w = weights.t().unsqueeze(-1)                            # (E, tokens, 1)
    # `out * w` alone is where the two kernels stop being the same function.
    # The loop evaluates an expert only when the router picked it; this path
    # evaluates all of them and lets the zeros cancel the rest - and `0 * inf`
    # is `NaN`, not `0`. Under pruning that is not hypothetical: a masked expert
    # keeps its weights, keeps being multiplied here, and runs on tokens it was
    # never selected for. One overflow anywhere in a layer turns the whole
    # token's hidden state into `NaN`, the rest of the forward carries it, and
    # the model repeats a single token to the ceiling.
    #
    # Selecting instead of multiplying keeps a zero-weight expert at exactly
    # zero whatever it computed. An expert the router *did* pick still shows its
    # `NaN` - that is a fact about the checkpoint and both kernels report it.
    if getattr(self, "_legacy_poison", False):
        final = (out * w).sum(dim=0)                         # the defect, on purpose
    else:
        final = torch.where(w == 0, out.new_zeros(()), out * w).sum(dim=0)
    return final.view(batch_size, sequence_length, hidden_dim), router_logits


def _loop_forward(self, flat, routing_weights, selected, router_logits,
                  batch_size, sequence_length, hidden_dim):
    """The stock path, for token counts where the dense form would not fit."""
    import torch

    final = torch.zeros_like(flat)
    n_experts = self.w_gate.shape[0]
    mask = torch.nn.functional.one_hot(selected, num_classes=n_experts).permute(2, 1, 0)
    for expert_idx in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
        idx, top_x = torch.where(mask[expert_idx].squeeze(0))
        state = flat[None, top_x].reshape(-1, hidden_dim)
        i = expert_idx.item()
        inter = self.act_fn(state @ self.w_gate[i]) * (state @ self.w_up[i])
        out = (inter @ self.w_down[i]) * routing_weights[top_x, idx, None]
        final.index_add_(0, top_x, out.to(flat.dtype))
    return final.view(batch_size, sequence_length, hidden_dim), router_logits
