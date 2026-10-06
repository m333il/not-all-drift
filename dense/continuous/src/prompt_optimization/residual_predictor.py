"""Linear and nonlinear predictors of condition-induced residual shifts."""

from __future__ import annotations

import torch


class BiasOnlyResidualPredictor(torch.nn.Module):
    """Input-independent residual correction ``f(h) = b``.

    Initializing ``b`` with the train mean shift makes this the exact
    constant least-squares baseline for the dense-only objective.  Objectives
    that include output KL may subsequently move the shared vector, but the
    prediction never depends on the input state.
    """

    def __init__(self, mean_shift: torch.Tensor) -> None:
        if mean_shift.ndim != 1 or not torch.isfinite(mean_shift).all():
            raise ValueError("mean_shift must be one finite vector")
        super().__init__()
        self.bias = torch.nn.Parameter(mean_shift.detach().clone())

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim < 1 or states.shape[-1] != self.bias.numel():
            raise ValueError("Input hidden dimension does not match bias")
        return self.bias.expand(*states.shape[:-1], -1)

    def predict(self, states: torch.Tensor) -> torch.Tensor:
        return self(states)

    def state_prediction(self, states: torch.Tensor) -> torch.Tensor:
        values = states.to(device=self.bias.device, dtype=self.bias.dtype)
        return values + self(values)


class LinearResidualPredictor(torch.nn.Linear):
    def __init__(self, mean_shift: torch.Tensor) -> None:
        if mean_shift.ndim != 1:
            raise ValueError("mean_shift must be one vector")
        super().__init__(mean_shift.numel(), mean_shift.numel(), bias=True)
        torch.nn.init.zeros_(self.weight)
        with torch.no_grad():
            self.bias.copy_(mean_shift)

    def predict(self, states: torch.Tensor) -> torch.Tensor:
        return self(states.to(device=self.weight.device, dtype=self.weight.dtype))

    def state_prediction(self, states: torch.Tensor) -> torch.Tensor:
        values = states.to(device=self.weight.device, dtype=self.weight.dtype)
        return values + self(values)


class MLPResidualPredictor(torch.nn.Module):
    def __init__(
        self,
        dimension: int,
        width: int,
        input_mean: torch.Tensor,
        input_scale: torch.Tensor,
        mean_shift: torch.Tensor,
    ) -> None:
        super().__init__()
        if input_mean.shape != (dimension,) or mean_shift.shape != (dimension,):
            raise ValueError("input_mean and mean_shift must match the residual dimension")
        if input_scale.numel() != 1 or float(input_scale) <= 0:
            raise ValueError("input_scale must be one positive scalar")
        self.dimension = dimension
        self.width = width
        self.register_buffer("input_mean", input_mean.detach().clone())
        self.register_buffer("input_scale", input_scale.detach().clone().reshape(()))
        self.up = torch.nn.Linear(dimension, width)
        self.down = torch.nn.Linear(width, dimension)
        torch.nn.init.zeros_(self.down.weight)
        with torch.no_grad():
            self.down.bias.copy_(mean_shift)

    def predict(self, states: torch.Tensor) -> torch.Tensor:
        values = states.to(device=self.input_mean.device, dtype=self.input_mean.dtype)
        normalized = (values - self.input_mean) / self.input_scale
        return self.down(torch.nn.functional.gelu(self.up(normalized)))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.predict(states)

    def state_prediction(self, states: torch.Tensor) -> torch.Tensor:
        values = states.to(device=self.input_mean.device, dtype=self.input_mean.dtype)
        return values + self.predict(values)


class LowRankResidualPredictor(torch.nn.Module):
    """Affine residual predictor with a rank-constrained input-dependent term.

    The correction is ``U V h + b``. ``U`` starts at zero and ``b`` starts at
    the train mean shift, so the initial predictor is exactly the mean-shift
    baseline while gradients can update ``U`` on the first optimization step.
    """

    def __init__(self, mean_shift: torch.Tensor, rank: int) -> None:
        if mean_shift.ndim != 1 or not torch.isfinite(mean_shift).all():
            raise ValueError("mean_shift must be one finite vector")
        dimension = mean_shift.numel()
        if rank < 1 or rank > dimension:
            raise ValueError(f"rank must be in [1, {dimension}], got {rank}")
        super().__init__()
        self.dimension = dimension
        self.rank = int(rank)
        layer_options = {"device": mean_shift.device, "dtype": mean_shift.dtype}
        self.down = torch.nn.Linear(dimension, rank, bias=False, **layer_options)
        self.up = torch.nn.Linear(rank, dimension, bias=False, **layer_options)
        self.bias = torch.nn.Parameter(mean_shift.detach().clone())
        torch.nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        values = states.to(device=self.bias.device, dtype=self.bias.dtype)
        return self.up(self.down(values)) + self.bias

    def predict(self, states: torch.Tensor) -> torch.Tensor:
        return self(states)

    def state_prediction(self, states: torch.Tensor) -> torch.Tensor:
        values = states.to(device=self.bias.device, dtype=self.bias.dtype)
        return values + self.predict(values)

    def effective_weight(self) -> torch.Tensor:
        """Materialize ``U V`` for diagnostics, not for inference."""
        return self.up.weight @ self.down.weight


def forward_kl(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> torch.Tensor:
    reference_logp = reference_logits.float().log_softmax(-1)
    candidate_logp = candidate_logits.float().log_softmax(-1)
    return (
        reference_logp.exp() * (reference_logp - candidate_logp)
    ).sum(-1).clamp_min(0)
