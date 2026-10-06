"""Where the massive activation and the attention sink actually sit.

The Super-Expert profile says what an expert's down projection emits. It cannot
say whether the *mechanism* that emission feeds - a massive activation carried in
the residual stream, and the attention sink that forms on it - is still present.
Under prefix tuning that distinction is the whole question: the adapter injects
learned keys and values at every layer, so the sink can be served by the prefix
without any expert producing anything large, and a quiet expert would then mean
"no longer needed" rather than "broken".

Two probes, deliberately separate from the expert profiler:

``ResidualNormProbe``
    The largest absolute value in the residual stream at each position after each
    decoder layer. A massive activation is visible here regardless of which
    expert, or whether any expert, produced it.
``attention_sink_summary``
    Where attention actually goes in the early layers, split between the injected
    prefix keys and the real positions. Requires the eager attention
    implementation, which is what this project measures under anyway.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LayerNorms:
    example: int
    layer: int
    per_position: list[float]
    """max |channel| of the residual stream at each position after this layer."""


class ResidualNormProbe:
    """Context manager recording per-position residual magnitudes for every layer."""

    def __init__(self, model, layers: list[int] | None = None) -> None:
        self.records: list[LayerNorms] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._example: int | None = None
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self._layers = list(base.model.layers)
        self._wanted = set(range(len(self._layers)) if layers is None else layers)

    def begin_example(self, index: int) -> None:
        self._example = index

    def __enter__(self) -> "ResidualNormProbe":
        for index, layer in enumerate(self._layers):
            if index in self._wanted:
                self._handles.append(layer.register_forward_hook(self._make_hook(index)))
        return self

    def __exit__(self, *_exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, layer: int):
        @torch.no_grad()
        def hook(_module, _inputs, output):
            if self._example is None:
                raise RuntimeError("Call begin_example(index) before each forward pass")
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[0] != 1:
                raise RuntimeError("Residual attribution requires batch size 1")
            self.records.append(LayerNorms(example=self._example, layer=layer,
                                           per_position=hidden[0].abs().float().amax(dim=-1).tolist()))
        return hook


@torch.no_grad()
def attention_sink_summary(model, input_ids, layers: list[int], prefix_length: int, offset: int = 0) -> list[dict]:
    """Split early-layer attention between injected prefix keys and real positions.

    ``prefix_length`` is how many keys the adapter prepended that are not tokens of
    the input - the prefix-tuning case. It is zero for every other arm, and then
    the prefix share is zero by construction rather than by measurement. GPT-OSS
    also places one learned sink logit in the softmax denominator and drops it from
    the returned key weights, so its mass is the returned weights' deficit from one.

    The ``from3_*`` fields are the split drawn in the paper's attention figure:
    queries from the third real position on, keys grouped into the virtual block
    (injected prefix keys plus the ``offset`` virtual positions of prompt tuning),
    real positions 0, 1 and 2, the rest, and the learned sink.
    """
    # Weights are read by hooks on the requested layers only and reduced at once:
    # output_attentions keeps every layer's quadratic tensor alive until the end of
    # the forward, which does not fit for a 4,000-token GEPA instruction.
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    captured = {}

    def hook_for(layer):
        def hook(_module, _args, output):
            weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if weights is None:
                raise RuntimeError(
                    "No attention weights returned; load the model with attn_implementation='eager'")
            captured[layer] = _split(weights, layer, prefix_length, offset)
        return hook

    handles = [base.model.layers[layer].self_attn.register_forward_hook(hook_for(layer)) for layer in layers]
    try:
        model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return [captured[layer] for layer in layers]


@torch.no_grad()
def _split(weights, layer: int, prefix_length: int, offset: int) -> dict:
    # [batch, heads, queries, keys] -> average over heads, keep the query axis.
    per_head = weights[0].float()
    learned_sink = (1.0 - per_head.sum(dim=-1)).clamp(0.0, 1.0)
    mass = per_head.mean(dim=0)
    prefix = mass[:, :prefix_length].sum(dim=-1) if prefix_length else torch.zeros(mass.shape[0])
    real = mass[:, prefix_length:]
    argmax = real.argmax(dim=-1)
    queries = mass[offset + 3:]
    keys = queries.shape[-1] - prefix_length
    stream = queries[:, prefix_length:]
    return {
        "layer": layer,
        "prefix_keys": prefix_length,
        "prefix_mass_mean": float(prefix.mean()),
        "learned_sink_mass_mean": float(learned_sink.mean()),
        "learned_sink_is_largest_share": float(
            (learned_sink > per_head.amax(dim=-1)).float().mean()),
        "real_mass_mean": float(real.sum(dim=-1).mean()),
        "real_key_0_mass_mean": float(real[:, 0].mean()) if real.shape[1] else None,
        "real_key_1_mass_mean": float(real[:, 1].mean()) if real.shape[1] > 1 else None,
        "argmax_real_key_mode": int(torch.mode(argmax).values),
        "argmax_real_key_share": float((argmax == torch.mode(argmax).values).float().mean()),
        "from3_queries": int(queries.shape[0]),
        "from3_virtual": float(queries[:, :prefix_length + offset].sum(-1).mean()),
        **{f"from3_key{k}": float(stream[:, offset + k].mean()) if offset + k < keys else 0.0
           for k in range(3)},
        "from3_rest": float(stream[:, offset + 3:].sum(-1).mean()),
        "from3_learned_sink": float(learned_sink.mean(0)[offset + 3:].mean()),
    }
