"""Prune named experts by zeroing their down projection.

Upstream prunes exactly this way - `eval_utils.py` copies a zero tensor over
`model.layers.<l>.mlp.experts.<e>.down_proj.weight` - so an expert still receives
tokens and still consumes its share of the routing weights, but contributes
nothing to the residual stream. Routing weights are **not** renormalised over the
survivors, which matters: a pruned expert removes its share of the mixture rather
than handing it to another expert.

On transformers 5.16.1 the same operation is a slice assignment into the fused
`down_proj` parameter, which needs no forward of our own and therefore works
unchanged under either experts backend, during cached generation, and with a PEFT
adapter installed.
"""
from __future__ import annotations

import torch


class ExpertAblation:
    """Zero the down projection of ``experts`` = {(layer, expert)} inside the block."""

    def __init__(self, model, experts: set[tuple[int, int]]) -> None:
        self.experts = set(experts)
        self._blocks: dict[int, torch.nn.Module] = {}
        self._saved: list[tuple[torch.nn.Parameter, int, torch.Tensor]] = []
        for name, module in model.named_modules():
            fused = getattr(module, "experts", None)
            if fused is None or not isinstance(getattr(fused, "down_proj", None), torch.nn.Parameter):
                continue
            self._blocks[int(name.split("layers.")[1].split(".")[0])] = fused
        if not self._blocks:
            raise RuntimeError("No fused MoE blocks found")
        count = int(next(iter(self._blocks.values())).num_experts)
        unknown = [pair for pair in self.experts
                   if pair[0] not in self._blocks or not 0 <= pair[1] < count]
        if unknown:
            raise ValueError(f"Experts outside this model ({count} per layer): {sorted(unknown)}")

    def __enter__(self) -> "ExpertAblation":
        with torch.no_grad():
            for layer, expert in sorted(self.experts):
                parameter = self._blocks[layer].down_proj
                self._saved.append((parameter, expert, parameter[expert].detach().clone()))
                parameter[expert].zero_()
        return self

    def __exit__(self, *_exc) -> None:
        with torch.no_grad():
            for parameter, expert, original in reversed(self._saved):
                parameter[expert].copy_(original)
        self._saved.clear()


class RouterMask:
    """Keep an expert out of the top-k, which is what deleting it would do.

    Not the same intervention as zeroing ``down_proj``. That leaves the router
    untouched, so the expert still wins its slot, still takes its share of the
    gate weights and returns nothing -- the block's output at that token is
    attenuated by exactly that share. Masking removes it from the competition:
    the freed slot goes to the next expert and the weights renormalise, so the
    token still receives a full-magnitude mixture. Deleting an expert from a
    shipped model drops its row from the gate, which is this one; the paper's
    "skipping the experts when selected by the router" is the other.

    The hook point is forced by the runtime. In transformers 5.16.1 the gate is a
    ``Qwen3MoeTopKRouter`` that takes its own top-k and returns
    ``(logits, scores, indices)``, of which the block uses only the last two --
    so writing into the logits it returns changes nothing at all. This replaces
    the whole tuple instead: it masks the router's own logits and redoes the
    selection the way the router does, which is why ``__enter__`` checks that
    recomputing an *unmasked* forward reproduces the router's own indices before
    trusting the reimplementation.
    """

    def __init__(self, model, experts: set[tuple[int, int]]) -> None:
        self.experts = set(experts)
        self._routers: dict[int, torch.nn.Module] = {}
        self._handles: list = []
        for name, module in model.named_modules():
            router = getattr(module, "gate", None) or getattr(module, "router", None)
            if router is None or getattr(module, "experts", None) is None or "layers." not in name:
                continue
            self._routers[int(name.split("layers.")[1].split(".")[0])] = router
        missing = sorted({layer for layer, _ in self.experts} - set(self._routers))
        if missing:
            raise ValueError(f"No MoE router found for layers {missing}")
        count = int(next(iter(self._routers.values())).num_experts)
        outside = sorted(pair for pair in self.experts if not 0 <= pair[1] < count)
        if outside:
            raise ValueError(
                f"Experts outside this model ({count} per layer): {outside}. A set identified on "
                "one model does not carry to another; run the replication gate on this one.")

    @staticmethod
    def _select(router, logits, selected_softmax):
        """The router's own selection, applied to logits we may have masked."""
        top_logits, indices = torch.topk(logits, router.top_k, dim=-1)
        if selected_softmax:
            values = torch.nn.functional.softmax(top_logits, dim=-1)
        else:
            probs = torch.nn.functional.softmax(logits, dtype=torch.float, dim=-1)
            values = probs.gather(-1, indices)
            if getattr(router, "norm_topk_prob", False):
                values = values / values.sum(dim=-1, keepdim=True)
        return values.to(logits.dtype), indices

    def __enter__(self) -> "RouterMask":
        for layer, router in self._routers.items():
            targets = sorted(expert for other, expert in self.experts if other == layer)
            if not targets:
                continue

            def hook(_module, _inputs, output, router=router, targets=targets):
                logits, scores, indices = output
                selected_values, check_indices = self._select(router, logits, True)
                full_values, _ = self._select(router, logits, False)
                if not torch.equal(check_indices, indices):
                    raise RuntimeError(
                        "Recomputing the router's selection does not reproduce it; "
                        "the runtime's routing differs from what this mask assumes")
                if torch.allclose(selected_values, scores, rtol=1e-3, atol=1e-3):
                    selected_softmax = True
                elif torch.allclose(full_values, scores, rtol=1e-3, atol=1e-3):
                    selected_softmax = False
                else:
                    raise RuntimeError(
                        "Recomputing the router's weights does not reproduce them; "
                        "the runtime's normalization differs from what this mask assumes")
                masked = logits.clone()
                masked[..., targets] = torch.finfo(masked.dtype).min
                values, chosen = self._select(router, masked, selected_softmax)
                return masked, values, chosen

            self._handles.append(router.register_forward_hook(hook))
        return self

    def __exit__(self, *_exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
