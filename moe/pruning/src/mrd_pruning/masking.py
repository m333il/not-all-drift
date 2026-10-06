"""Frequency pruning of MoE experts, applied as a router-level mask.

An expert is "pruned" by driving its router logit to the dtype's minimum before
the block takes its top-k, so the expert can never be selected. Quality-wise
this is equivalent to deleting the expert's weights, and it needs no surgery on
the checkpoint, so the same loaded model serves every pruning level in a sweep.

Two facts about this hook point, both inherited from the routing-transplant code
that already ran on this architecture:

* the gate is an ``nn.Linear`` whose forward output is the raw router logits,
  shaped ``[B*seq, num_experts]`` - indices and combining weights are derived
  downstream from whatever logits it returns;
* a forward hook stays active for every forward, prefill and decode alike. That
  is what we want here (pruning is permanent) and is the opposite of the
  transplant, where the override was deliberately prefill-only.

The mask value is ``torch.finfo(dtype).min``, taken from the live tensor rather
than a constant: the donor-transplant run died silently for a whole day because
a float32 tensor was written into a bf16 one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import TracebackType
from typing import Iterator, Mapping, Sequence

import torch
from torch import nn

logger = logging.getLogger(__name__)


class RoutingTally:
    """Counts, per layer, which experts won a slot while the mask was on.

    The map measured before pruning says where the traffic used to go. It says
    nothing about where it goes once the quiet experts are gone - and that is
    the question the quality number raises: did the load spread over the
    survivors, or pile onto a few of them.

    The hook already sees the gate's decision on every forward, prefill and
    decode alike, so the tally is over exactly the tokens the evaluated run
    produced. Nothing is sampled and nothing is re-run.

    **It must not synchronise.** The obvious implementation - `torch.bincount`
    into a per-layer tensor - does: bincount reads the maximum of its input to
    size the output, and that read drags the GPU queue back to the host. Once
    per layer per decode step is twenty-four to forty-eight stalls per token,
    and on a card shared with other tenants each stall waits behind *their*
    queued kernels too. Measured on an H200 with neighbours, 300 decode steps:
    Qwen 182s → 350s, gpt-oss 0.5s → 75s. The counting itself is trivial; the
    synchronising was the whole cost.

    So the counts live in one preallocated table on the routing device and grow
    by `scatter_add_`, which queues like any other kernel and never looks at the
    values. Nothing leaves the GPU until `as_array`.
    """

    def __init__(self, n_layers: int, n_experts: int) -> None:
        self.n_experts = int(n_experts)
        self.n_layers = int(n_layers)
        self._table: "torch.Tensor | None" = None
        self._ones: "torch.Tensor | None" = None

    def add(self, layer_idx: int, picked: torch.Tensor) -> None:
        flat = picked.reshape(-1).to(torch.long)
        if self._table is None:
            self._table = torch.zeros(
                self.n_layers, self.n_experts, dtype=torch.long, device=flat.device)
        # One buffer of ones, grown when a batch needs more. Allocating it per
        # call would put an allocation on the hot path for no reason.
        if self._ones is None or self._ones.numel() < flat.numel():
            self._ones = torch.ones(flat.numel(), dtype=torch.long, device=flat.device)
        self._table[layer_idx].scatter_add_(0, flat, self._ones[:flat.numel()])

    def as_array(self) -> "torch.Tensor":
        if self._table is None:
            return torch.zeros(self.n_layers, self.n_experts, dtype=torch.long)
        return self._table.cpu()

    @property
    def total(self) -> int:
        return 0 if self._table is None else int(self._table.sum())


@dataclass(frozen=True)
class GateRef:
    """One MoE gate: its layer index in the model and the module itself."""

    layer_idx: int
    module: nn.Module
    num_experts: int


def takes_own_topk(module: nn.Module) -> bool:
    """True for a gate that selects its own top-k and returns the winners.

    Qwen's gate is an ``nn.Linear`` returning raw logits, so masking its output
    happens before the block picks top-k. gpt-oss's router picks top-k *inside*
    its forward and returns ``(scores, indices)``, so masking that output would
    only zero an already-chosen expert's contribution - a different intervention
    that never reassigns the token. Such a gate is pruned at its ``bias``
    instead, which lands before the ``topk`` and is what this module promises.
    """
    return hasattr(module, "top_k") and isinstance(
        getattr(module, "bias", None), torch.Tensor
    )


def discover_gates(model: nn.Module) -> list[GateRef]:
    """Find every MoE gate, in layer order.

    Detection is structural (a block that owns both a gate and ``experts``), not
    name-based, so a model whose early layers are dense - Ling's layer 0 -
    simply contributes fewer gates instead of raising. The gate itself is looked
    up under either spelling: Qwen calls it ``gate``, gpt-oss ``router``.
    """
    gates: list[GateRef] = []
    layers = _decoder_layers(model)
    for idx, layer in enumerate(layers):
        block = getattr(layer, "mlp", None)
        if block is None:
            continue
        gate = getattr(block, "gate", None) or getattr(block, "router", None)
        experts = getattr(block, "experts", None)
        if gate is None or experts is None:
            continue
        num_experts = getattr(gate, "out_features", None)
        if num_experts is None:
            # gpt-oss's router is not an nn.Linear; it keeps the matrix itself.
            num_experts = getattr(gate, "num_experts", None)
        if num_experts is None:
            weight = getattr(gate, "weight", None)
            num_experts = weight.shape[0] if weight is not None else len(experts)
        gates.append(GateRef(layer_idx=idx, module=gate, num_experts=int(num_experts)))
    if not gates:
        raise ValueError("no MoE gates found - is this a dense model?")
    return gates


def _decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    """Reach the decoder layer list through PEFT and HF wrappers alike."""
    node = model
    for _ in range(6):  # base_model -> model -> model -> layers is the deepest seen
        layers = getattr(node, "layers", None)
        if layers is not None:
            return layers
        for attr in ("base_model", "model", "transformer"):
            child = getattr(node, attr, None)
            if child is not None:
                node = child
                break
        else:
            break
    raise ValueError("could not locate decoder layers on this model")


@dataclass
class MaskAudit:
    """What the hooks actually did, so a run can assert instead of assume."""

    calls: dict[int, int] = field(default_factory=dict)
    masked_experts: dict[int, int] = field(default_factory=dict)
    max_selected_pruned: int = 0

    def layers_touched(self) -> int:
        return sum(1 for n in self.calls.values() if n > 0)


class ExpertMask:
    """Context manager masking a per-layer set of experts at the router.

    ``pruned`` maps layer index to the expert ids removed in that layer. Layers
    absent from the mapping keep every expert, so a sweep's zero-pruning cell
    goes through exactly the same code path as the others and still reproduces
    the unhooked forward pass bit for bit.

    ``verify_topk`` re-derives the block's top-k from the masked logits on the
    first ``verify_batches`` calls per layer and records any pruned expert that
    still won a slot. It is cheap, and it is the only direct evidence that the
    intervention did what the config says.
    """

    def __init__(
        self,
        model: nn.Module,
        pruned: Mapping[int, Sequence[int]],
        *,
        top_k: int,
        verify_topk: bool = True,
        verify_batches: int = 2,
        tally: "RoutingTally | None" = None,
    ) -> None:
        self.gates = discover_gates(model)
        self.top_k = int(top_k)
        self.verify_topk = verify_topk
        self.verify_batches = int(verify_batches)
        # None keeps the hook on exactly the path it had before counting existed.
        self.tally = tally
        self.audit = MaskAudit()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._bias_backup: dict[int, torch.Tensor] = {}

        known = {g.layer_idx: g for g in self.gates}
        unknown = sorted(set(pruned) - set(known))
        if unknown:
            raise ValueError(f"pruning requested for non-MoE layers: {unknown}")

        self._pruned: dict[int, torch.Tensor] = {}
        for layer_idx, experts in pruned.items():
            gate = known[layer_idx]
            ids = sorted({int(e) for e in experts})
            if not ids:
                continue
            if min(ids) < 0 or max(ids) >= gate.num_experts:
                raise ValueError(
                    f"layer {layer_idx}: expert id out of range for "
                    f"{gate.num_experts} experts: {ids[:5]}..."
                )
            n_keep = gate.num_experts - len(ids)
            if n_keep < self.top_k:
                raise ValueError(
                    f"layer {layer_idx}: pruning {len(ids)} of {gate.num_experts} "
                    f"experts leaves {n_keep} < top_k={self.top_k}; the block "
                    "cannot fill its top-k and the run would be meaningless"
                )
            self._pruned[layer_idx] = torch.tensor(ids, dtype=torch.long)

    @property
    def n_pruned_total(self) -> int:
        return sum(len(v) for v in self._pruned.values())

    def _make_hook(self, layer_idx: int):
        pruned_cpu = self._pruned.get(layer_idx)

        def hook(_module: nn.Module, _inputs, output):
            self.audit.calls[layer_idx] = self.audit.calls.get(layer_idx, 0) + 1
            # A gate that took its own top-k hands back (scores, indices); it was
            # pruned at the bias, so here we only audit what it actually picked.
            picked = output[1] if isinstance(output, tuple) else None
            if pruned_cpu is None or pruned_cpu.numel() == 0:
                self.audit.masked_experts[layer_idx] = 0
                if self.tally is not None:
                    chosen = (picked if picked is not None
                              else output.topk(self.top_k, dim=-1).indices)
                    self.tally.add(layer_idx, chosen)
                return None if picked is not None else output
            if picked is not None:
                self.audit.masked_experts[layer_idx] = int(pruned_cpu.numel())
                if self.tally is not None:
                    self.tally.add(layer_idx, picked)
                if self.verify_topk and self.audit.calls[layer_idx] <= self.verify_batches:
                    idx = pruned_cpu.to(picked.device)
                    leaked = int(torch.isin(picked, idx).sum())
                    self.audit.max_selected_pruned = max(
                        self.audit.max_selected_pruned, leaked
                    )
                return None  # the module's own output stands
            logits = output
            idx = pruned_cpu.to(logits.device)
            masked = logits.clone()
            masked[:, idx] = torch.finfo(masked.dtype).min
            self.audit.masked_experts[layer_idx] = int(idx.numel())
            chosen = None
            if self.tally is not None:
                chosen = masked.topk(self.top_k, dim=-1).indices
                self.tally.add(layer_idx, chosen)
            if self.verify_topk and self.audit.calls[layer_idx] <= self.verify_batches:
                if chosen is None:
                    chosen = masked.topk(self.top_k, dim=-1).indices
                leaked = int(torch.isin(chosen, idx).sum())
                self.audit.max_selected_pruned = max(self.audit.max_selected_pruned, leaked)
            return masked

        return hook

    def __enter__(self) -> "ExpertMask":
        self._mask_biases()
        self._handles = [
            g.module.register_forward_hook(self._make_hook(g.layer_idx)) for g in self.gates
        ]
        return self

    def __exit__(self, exc_type, exc, tb: TracebackType | None) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._restore_biases()

    def _mask_biases(self) -> None:
        """Sink the router bias of every pruned expert on self-top-k gates.

        ``router_logits = linear(h, weight, bias)`` runs before the gate's own
        ``topk``, so a floored bias removes the expert from the candidates -
        the same effect the logit mask has on Qwen. The minimum is read off the
        live tensor's dtype, never a constant: writing a float32 value into a
        bf16 parameter is how the transplant run died silently for a day.
        """
        for gate in self.gates:
            pruned = self._pruned.get(gate.layer_idx)
            if pruned is None or pruned.numel() == 0:
                continue
            if not takes_own_topk(gate.module):
                continue
            bias = gate.module.bias
            self._bias_backup[gate.layer_idx] = bias.detach().clone()
            with torch.no_grad():
                bias[pruned.to(bias.device)] = torch.finfo(bias.dtype).min

    def _restore_biases(self) -> None:
        for layer_idx, saved in self._bias_backup.items():
            gate = next(g for g in self.gates if g.layer_idx == layer_idx)
            with torch.no_grad():
                gate.module.bias.copy_(saved)
        self._bias_backup = {}

    def assert_applied(self, *, expect_layers: int | None = None) -> None:
        """Fail loudly if the mask never fired or a pruned expert was selected.

        A run whose hooks silently did nothing must not produce a summary file:
        that is precisely how a broken measurement gets reported as a result.
        """
        touched = self.audit.layers_touched()
        if touched == 0:
            raise RuntimeError("expert mask never fired - no gate forward was hooked")
        if expect_layers is not None and touched != expect_layers:
            raise RuntimeError(
                f"expert mask fired on {touched} layers, expected {expect_layers}"
            )
        if self.audit.max_selected_pruned > 0:
            raise RuntimeError(
                f"{self.audit.max_selected_pruned} pruned experts still won a top-k slot; "
                "the mask is not being applied where the block reads its logits"
            )
        applied = sum(self.audit.masked_experts.values())
        if self.n_pruned_total > 0 and applied == 0:
            raise RuntimeError("pruning was configured but no expert was masked")


def iter_prune_levels(levels: Sequence[int], num_experts: int, top_k: int) -> Iterator[int]:
    """Validate a sweep's pruning levels before any model is loaded."""
    for level in levels:
        if level < 0:
            raise ValueError(f"negative pruning level {level}")
        if num_experts - level < top_k:
            raise ValueError(
                f"level {level} leaves {num_experts - level} experts, below top_k={top_k}"
            )
        yield int(level)
