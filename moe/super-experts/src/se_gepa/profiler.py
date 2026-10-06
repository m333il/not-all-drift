"""Super-Expert profiling on fused-expert MoE runtimes (transformers 5.x).

Upstream (``ZunhaiSu/Super-Experts-Profilling``, pinned in ``../upstream``) hooks
every per-expert ``down_proj`` ``nn.Linear`` and keeps one number per
(layer, expert): the largest absolute value ever seen at that projection's
output, taken *before* the routing weight is applied. transformers 5.16.1 stores
Qwen3-MoE experts as three-dimensional parameters inside one fused
``Qwen3MoeExperts`` module, so ``"down" in name`` matches nothing and upstream's
hook records an empty profile. This module reproduces the same statistic by
instrumenting the fused expert forward itself.

Both experts implementations this project uses are instrumented:

``grouped_mm``
    The production backend, and the default here. One grouped GEMM per
    projection over expert-sorted rows, so the whole profile is a segmented
    reduction over ``(S, hidden)`` -- no per-expert Python loop and no extra
    matmuls. Profiling costs almost nothing on top of the forward pass.
``eager``
    The reference loop over hit experts. Correct but launch-bound at
    ``num_experts x num_layers`` small GEMMs per token batch; kept because it is
    the implementation upstream's numbers were produced under and because it is
    the second opinion the tests check ``grouped_mm`` against.

Each instrumented forward is a copy of the corresponding transformers 5.16.1
implementation with recording inserted, so the model's own output is unchanged.
``tests/test_profiler.py`` pins that: both backends must return logits identical
to the stock implementation, and the two must record the same maxima.

Two things are recorded that upstream does not keep, both needed downstream:

* which token produced each maximum. Upstream reduces to a corpus-level max, so
  it cannot say which token triggered an expert -- the paper's token-level claim
  (sink tokens) is argued from router scores, not from this profile.
* an optional per-token trace for a named set of experts, which is what a
  prompt-level objective has to read.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F

from transformers.integrations.moe import _grouped_linear

NEGATIVE_INFINITY = float("-inf")


@dataclass(frozen=True)
class OutlierRecord:
    """The corpus maximum of ``|down_proj output|`` for one (layer, expert).

    ``input_max`` is the largest absolute value at the *input* of the same
    projection, maximised independently of ``output_max`` exactly as upstream
    does. It is kept because a large output can come either from a large input
    or from the expert's own weights.

    ``example``/``position``/``token_id`` locate the token that produced
    ``output_max``; ``channel`` is the hidden-dimension index it landed in.
    ``position`` counts tokens of the sequence that was fed to the model, so
    with a prompt-tuning adapter installed position 0 is a virtual token, not a
    text token, and ``token_id`` is ``-1`` there.
    """

    output_max: float = 0.0
    input_max: float = 0.0
    channel: int = -1
    example: int = -1
    position: int = -1
    token_id: int = -1
    hits: int = 0
    """Number of (token, expert) assignments seen, i.e. this expert's traffic."""


@dataclass
class ExampleTrace:
    """Per-token ``max_channel |down_proj output|`` for the tracked experts."""

    example: int
    input_ids: list[int]
    values: dict[tuple[int, int], dict[int, float]] = field(default_factory=dict)


class _LayerState:
    """Running maxima for one MoE layer, kept on the model's device."""

    def __init__(self, num_experts: int, device) -> None:
        self.output_max = torch.full((num_experts,), NEGATIVE_INFINITY, device=device)
        self.input_max = torch.full((num_experts,), NEGATIVE_INFINITY, device=device)
        self.hits = torch.zeros(num_experts, dtype=torch.long, device=device)
        self.example = torch.full((num_experts,), -1, dtype=torch.long, device=device)
        self.position = torch.full((num_experts,), -1, dtype=torch.long, device=device)
        self.channel = torch.full((num_experts,), -1, dtype=torch.long, device=device)

    def update(self, experts, output_max, input_max, position, channel, hits, example) -> None:
        improved = output_max > self.output_max[experts]
        rows = experts[improved]
        self.output_max[rows] = output_max[improved]
        self.position[rows] = position[improved]
        self.channel[rows] = channel[improved]
        self.example[rows] = example
        self.input_max[experts] = torch.maximum(self.input_max[experts], input_max)
        self.hits[experts] += hits


class FusedExpertProfiler:
    """Context manager that instruments every fused MoE block of ``model``.

    Usage is one sequence at a time::

        with FusedExpertProfiler(model) as profiler:
            for index, ids in enumerate(segments):
                profiler.begin_example(index, ids)
                model(input_ids=tensor)
        records = profiler.records

    ``begin_example`` is what makes token attribution meaningful; calling the
    model without it raises rather than silently attributing to the previous
    example. Batched calls are refused for the same reason, and so are decode
    steps -- profile a prompt with one teacher-forced pass, not with
    ``generate``.
    """

    def __init__(self, model, track: set[tuple[int, int]] | None = None, backend: str | None = None) -> None:
        self.model = model
        self.track = set(track or ())
        self.traces: list[ExampleTrace] = []
        self._blocks: list[tuple[int, torch.nn.Module, torch.nn.Module]] = []
        self._saved: list[tuple[torch.nn.Module, object]] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._state: dict[int, _LayerState] = {}
        self._examples: dict[int, list[int]] = {}
        self._example: ExampleTrace | None = None
        self._seq_len: int | None = None
        for name, module in model.named_modules():
            experts = getattr(module, "experts", None)
            if experts is None or not isinstance(getattr(experts, "down_proj", None), torch.nn.Parameter):
                continue
            layer = int(name.split("layers.")[1].split(".")[0])
            self._blocks.append((layer, module, experts))
        if not self._blocks:
            raise RuntimeError(
                "No fused MoE blocks found: this profiler expects experts with a 3D "
                "down_proj parameter (transformers>=5 Qwen3-MoE and friends)."
            )
        self._blocks.sort(key=lambda row: row[0])
        configured = self._blocks[0][2].config._experts_implementation
        self.backend = backend or configured
        if self.backend not in {"eager", "grouped_mm"}:
            raise NotImplementedError(
                f"Only the eager and grouped_mm experts implementations are instrumented, not {self.backend!r}"
            )
        if backend is not None and backend != configured:
            raise ValueError(
                f"The model runs {configured!r} experts; profiling it as {backend!r} would report "
                "numbers the model did not produce. Load the model with the backend you want to profile."
            )
        self.num_experts = int(self._blocks[0][2].num_experts)
        unknown = [pair for pair in self.track
                   if pair[0] not in set(self.layer_ids) or not 0 <= pair[1] < self.num_experts]
        if unknown:
            raise ValueError(
                f"Tracked experts outside this model ({len(self._blocks)} MoE layers, "
                f"{self.num_experts} experts each): {sorted(unknown)}")
        device = self._blocks[0][2].down_proj.device
        for layer, _block, _experts in self._blocks:
            self._state[layer] = _LayerState(self.num_experts, device)

    @property
    def layer_ids(self) -> list[int]:
        return [layer for layer, _, _ in self._blocks]

    @property
    def records(self) -> dict[tuple[int, int], OutlierRecord]:
        """Materialise the running state; only experts that were routed to appear."""
        out = {}
        for layer, state in self._state.items():
            hits = state.hits.tolist()
            output_max = state.output_max.tolist()
            input_max = state.input_max.tolist()
            position = state.position.tolist()
            channel = state.channel.tolist()
            example = state.example.tolist()
            for expert in range(self.num_experts):
                if hits[expert] == 0:
                    continue
                ids = self._examples.get(example[expert], [])
                token_id = ids[position[expert]] if 0 <= position[expert] < len(ids) else -1
                out[(layer, expert)] = OutlierRecord(
                    output_max=output_max[expert],
                    input_max=input_max[expert],
                    channel=channel[expert],
                    example=example[expert],
                    position=position[expert],
                    token_id=token_id,
                    hits=hits[expert],
                )
        return out

    def begin_example(self, index: int, input_ids: list[int], virtual_tokens: int = 0) -> None:
        """Announce the sequence the next forward pass will run on.

        ``virtual_tokens`` is how many continuous positions a prompt-tuning
        adapter prepends. They occupy real positions in the expert input stream
        but have no token id, so they are recorded as ``-1`` -- which is the
        difference between "the Super Expert fired on the first text token" and
        "it fired on a learned vector".
        """
        ids = [-1] * virtual_tokens + list(input_ids)
        self._example = ExampleTrace(example=index, input_ids=ids)
        self._examples[index] = self._example.input_ids
        if self.track:
            self.traces.append(self._example)

    def __enter__(self) -> "FusedExpertProfiler":
        make = self._make_grouped_forward if self.backend == "grouped_mm" else self._make_eager_forward
        for layer, block, experts in self._blocks:
            self._handles.append(block.register_forward_pre_hook(self._shape_hook))
            self._saved.append((experts, experts.forward))
            experts.forward = make(layer, experts)
        return self

    def __exit__(self, *_exc) -> None:
        for handle in self._handles:
            handle.remove()
        for experts, forward in self._saved:
            experts.forward = forward
        self._handles.clear()
        self._saved.clear()

    def _shape_hook(self, _module, inputs) -> None:
        self._seq_len = int(inputs[0].shape[-2])

    def _check_positions(self, token_index: torch.Tensor) -> None:
        """Token attribution is only well defined for one un-cached sequence.

        ``Qwen3MoeSparseMoeBlock`` flattens the batch before calling the experts,
        so with batch > 1 a flat index is ambiguous, and during cached decoding
        every step would report position 0.
        """
        if self._seq_len is None:
            raise RuntimeError("The MoE block pre-hook did not run")
        if self._seq_len != len(self._example.input_ids):
            raise RuntimeError(
                f"Expected one full sequence of {len(self._example.input_ids)} tokens, got {self._seq_len}; "
                "profile a prompt with a single teacher-forced pass, not with generate()"
            )
        if int(token_index.max()) >= self._seq_len:
            raise RuntimeError("Profiling requires batch size 1 for token attribution")

    # ── instrumented forwards ────────────────────────────────────────────────

    def _make_eager_forward(self, layer: int, module):
        """``Qwen3MoeExperts.forward`` (transformers 5.16.1) with recording."""

        def forward(hidden_states, top_k_index, top_k_weights):
            self._require_example()
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = F.one_hot(top_k_index, num_classes=module.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            seen = []
            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                if expert_idx == module.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                gate, up = F.linear(current_state, module.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                current_hidden_states = module.act_fn(gate) * up
                projected = F.linear(current_hidden_states, module.down_proj[expert_idx])
                seen.append((expert_idx, current_hidden_states, projected, token_idx))
                current_hidden_states = projected * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
            self._record_eager(layer, seen)
            return final_hidden_states

        return forward

    def _make_grouped_forward(self, layer: int, module):
        """``grouped_mm_experts_forward`` (transformers 5.16.1) with recording."""

        def forward(hidden_states, top_k_index, top_k_weights):
            self._require_example()
            device = hidden_states.device
            num_top_k = top_k_index.size(-1)
            num_tokens = hidden_states.size(0)
            hidden_dim = hidden_states.size(-1)

            sample_weights = top_k_weights.reshape(-1)
            expert_ids = top_k_index.reshape(-1)

            expert_ids_g, perm = torch.sort(expert_ids)
            selected_hidden_states_g = hidden_states[perm // num_top_k]
            sample_weights_g = sample_weights[perm]

            histc_input = expert_ids_g.float() if device.type in ("cpu", "mps") else expert_ids_g.int()
            tokens_per_expert = torch.histc(histc_input, bins=module.num_experts, min=0, max=module.num_experts - 1)
            offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

            sentinel_mask = (expert_ids_g >= module.num_experts).unsqueeze(-1)
            expert_ids_g.clamp_(max=module.num_experts - 1)

            if module.has_gate:
                selected_weights = module.gate_up_proj
                selected_biases = module.gate_up_proj_bias[expert_ids_g] if module.has_bias else None
            else:
                selected_weights = module.up_proj
                selected_biases = module.up_proj_bias[expert_ids_g] if module.has_bias else None

            selected_hidden_states_g.masked_fill_(sentinel_mask, 0.0)

            proj_out = _grouped_linear(
                selected_hidden_states_g, selected_weights, offsets,
                bias=selected_biases, is_transposed=module.is_transposed,
            )
            proj_out = module._apply_gate(proj_out) if module.has_gate else module.act_fn(proj_out)

            entry = proj_out
            selected_weights = module.down_proj
            selected_biases = module.down_proj_bias[expert_ids_g] if module.has_bias else None
            proj_out = _grouped_linear(
                entry, selected_weights, offsets, bias=selected_biases, is_transposed=module.is_transposed,
            )
            self._record_grouped(layer, entry, proj_out, expert_ids_g, perm // num_top_k, sentinel_mask)

            weighted_out = proj_out * sample_weights_g.unsqueeze(-1)
            weighted_out.masked_fill_(sentinel_mask, 0.0)

            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(perm.size(0), device=device)
            weighted_out = weighted_out[inv_perm]
            final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
            return final_hidden_states.to(hidden_states.dtype)

        return forward

    # ── recording ────────────────────────────────────────────────────────────

    def _require_example(self) -> None:
        if self._example is None:
            raise RuntimeError("Call begin_example(index, input_ids) before each forward pass")

    @torch.no_grad()
    def _record_eager(self, layer: int, seen) -> None:
        if not seen:
            return
        experts, output_max, input_max, position, channel, hits = [], [], [], [], [], []
        for expert_idx, entry, projected, token_idx in seen:
            self._check_positions(token_idx)
            magnitudes = projected.abs().float()
            flat = int(magnitudes.argmax())
            row, column = divmod(flat, magnitudes.shape[-1])
            experts.append(int(expert_idx))
            output_max.append(float(magnitudes.view(-1)[flat]))
            input_max.append(float(entry.abs().float().max()))
            position.append(int(token_idx[row]))
            channel.append(column)
            hits.append(int(token_idx.numel()))
            if (layer, int(expert_idx)) in self.track:
                self._trace(layer, int(expert_idx), magnitudes.amax(dim=-1), token_idx)
        device = self._state[layer].output_max.device
        self._state[layer].update(
            torch.tensor(experts, dtype=torch.long, device=device),
            torch.tensor(output_max, device=device),
            torch.tensor(input_max, device=device),
            torch.tensor(position, dtype=torch.long, device=device),
            torch.tensor(channel, dtype=torch.long, device=device),
            torch.tensor(hits, dtype=torch.long, device=device),
            self._example.example,
        )

    @torch.no_grad()
    def _record_grouped(self, layer, entry, projected, expert_ids_g, token_index, sentinel_mask) -> None:
        valid = ~sentinel_mask.squeeze(-1)
        self._check_positions(token_index[valid])
        experts = expert_ids_g[valid]
        magnitudes = projected[valid].abs().float().amax(dim=-1)
        entry_magnitudes = entry[valid].abs().float().amax(dim=-1)
        rows = valid.nonzero().flatten()

        size = self.num_experts
        device = magnitudes.device
        output_max = torch.full((size,), NEGATIVE_INFINITY, device=device)
        output_max.scatter_reduce_(0, experts, magnitudes, reduce="amax")
        input_max = torch.full((size,), NEGATIVE_INFINITY, device=device)
        input_max.scatter_reduce_(0, experts, entry_magnitudes, reduce="amax")
        hits = torch.zeros(size, dtype=torch.long, device=device)
        hits.scatter_add_(0, experts, torch.ones_like(experts))

        # Lowest row index that attained each expert's maximum, so the attribution
        # is deterministic when several tokens tie.
        order = torch.arange(experts.numel(), device=device)
        best = torch.full((size,), experts.numel(), dtype=torch.long, device=device)
        missing = torch.full_like(order, experts.numel())
        best.scatter_reduce_(0, experts, torch.where(magnitudes == output_max[experts], order, missing), reduce="amin")

        present = hits > 0
        selected = best[present]
        channel = projected[rows[selected]].abs().float().argmax(dim=-1)
        self._state[layer].update(
            present.nonzero().flatten(),
            output_max[present],
            input_max[present],
            token_index[rows[selected]],
            channel,
            hits[present],
            self._example.example,
        )
        for key in self.track:
            if key[0] != layer:
                continue
            chosen = experts == key[1]
            if bool(chosen.any()):
                self._trace(layer, key[1], magnitudes[chosen], token_index[rows[chosen]])

    def _trace(self, layer: int, expert: int, values, positions) -> None:
        target = self._example.values.setdefault((layer, expert), {})
        for position, value in zip(positions.tolist(), values.tolist()):
            target[position] = max(target.get(position, 0.0), value)


def profile_to_json(profiler: FusedExpertProfiler) -> dict:
    """Serializable profile: one row per (layer, expert) that was routed to."""
    return {
        "backend": profiler.backend,
        "layer_ids": profiler.layer_ids,
        "num_experts": profiler.num_experts,
        "records": [
            {"layer": layer, "expert": expert, **asdict(record)}
            for (layer, expert), record in sorted(profiler.records.items())
        ],
    }


class SharedExpertRecorder:
    """Largest ``|output|`` of each MoE block's always-on shared expert, per layer.

    The Super-Expert criterion is defined over routed experts, but in some families
    (DeepSeek-R1) shared experts meet it as well. Models with a shared expert get it
    recorded beside the routed profile, never mixed into its percentile. Blocks
    without one are skipped.
    """

    def __init__(self, model) -> None:
        self.records: dict[int, dict] = {}
        self._modules: list[tuple[int, torch.nn.Module]] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for name, module in model.named_modules():
            shared = getattr(module, "shared_experts", None) or getattr(module, "shared_expert", None)
            if shared is None or getattr(module, "experts", None) is None:
                continue
            self._modules.append((int(name.split("layers.")[1].split(".")[0]), shared))

    def __enter__(self) -> "SharedExpertRecorder":
        for layer, shared in self._modules:
            self._handles.append(shared.register_forward_hook(self._make_hook(layer)))
        return self

    def __exit__(self, *_exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, layer: int):
        @torch.no_grad()
        def hook(_module, _inputs, output):
            flat = output.detach().float().abs().reshape(-1, output.shape[-1])
            values, channels = flat.max(dim=-1)
            position = int(values.argmax())
            peak = float(values[position])
            if layer not in self.records or peak > self.records[layer]["max"]:
                self.records[layer] = {"max": peak, "position": position, "channel": int(channels[position])}
        return hook
