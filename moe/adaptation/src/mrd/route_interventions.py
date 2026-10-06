from contextlib import AbstractContextManager
from dataclasses import dataclass

import torch



@dataclass
class RouteState:
    logits: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor


@dataclass
class RoutePatch:
    positions: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor


class RouteHooks(AbstractContextManager):
    """Capture actual post-selection routes; optionally replace IDs and coefficients at prefill."""

    def __init__(self, adapter, patches=None):
        self.adapter = adapter
        self.patches = patches or {}
        unknown = set(self.patches) - set(adapter.layer_ids)
        if unknown:
            raise ValueError(f"unknown patch layers: {sorted(unknown)}")
        self.captured = {}
        self.coverage = {}
        self.handles = []

    def __enter__(self):
        for layer, gate in self.adapter.gates:
            self.handles.append(gate.register_forward_hook(self._hook(layer)))
        return self

    def _hook(self, layer):
        calls = 0

        def hook(_gate, _inputs, output):
            nonlocal calls
            calls += 1
            if calls > 1:
                return output
            logits, weights, indices = output
            patch = self.patches.get(layer)
            self.coverage[layer] = 0
            if patch is not None:
                pos = patch.positions.to(indices.device)
                ids = patch.indices.to(indices.device)
                coeff = patch.weights.to(weights.device, weights.dtype)
                if ids.shape != (len(pos), self.adapter.top_k) or coeff.shape != ids.shape:
                    raise ValueError("patch shape must be [number of positions, top_k]")
                if pos.ndim != 1 or (pos < 0).any() or (pos >= len(indices)).any() or pos.unique().numel() != pos.numel():
                    raise ValueError("patch positions must be unique valid prefill rows")
                if (ids < 0).any() or (ids >= self.adapter.num_experts).any() or (ids.sort(-1).values.diff(dim=-1) == 0).any():
                    raise ValueError("patch must select distinct valid expert IDs")
                if not torch.isfinite(coeff).all() or (coeff < 0).any():
                    raise ValueError("patch coefficients must be finite and nonnegative")
                if getattr(_gate, "norm_topk_prob", True) and not torch.allclose(
                    coeff.float().sum(-1), torch.ones(len(pos), device=coeff.device), atol=0.01, rtol=0,
                ):
                    raise ValueError("patch coefficients must sum to one")
                indices, weights = indices.clone(), weights.clone()
                indices[pos], weights[pos] = ids, coeff
                self.coverage[layer] = len(pos)
            self.captured[layer] = RouteState(
                logits.detach().cpu(), indices.detach().cpu(), weights.detach().cpu(),
            )
            return logits, weights, indices
        return hook

    def __exit__(self, *exc_info):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
