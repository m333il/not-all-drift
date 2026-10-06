"""Predictors and losses for real-layer-skipping early exits."""

from __future__ import annotations

import torch


class ResidualLinear(torch.nn.Linear):
    """Predict a residual correction; the reconstructed state is x + f(x)."""

    def __init__(self, mean_residual: torch.Tensor) -> None:
        if mean_residual.ndim != 1 or not torch.isfinite(mean_residual).all():
            raise ValueError("mean_residual must be one finite vector")
        super().__init__(mean_residual.numel(), mean_residual.numel(), bias=True)
        torch.nn.init.zeros_(self.weight)
        with torch.no_grad():
            self.bias.copy_(mean_residual)

    def state_prediction(self, states: torch.Tensor) -> torch.Tensor:
        return states + self(states)


def forward_kl(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> torch.Tensor:
    log_p = reference_logits.float().log_softmax(-1)
    log_q = candidate_logits.float().log_softmax(-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1).clamp_min(0)
