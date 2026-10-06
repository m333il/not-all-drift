"""Metrics for prompt-induced residual shifts expressed in shared SAE coordinates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch.nn import functional


@dataclass(frozen=True)
class ShiftCoverageMetrics:
    """Aggregate fidelity of an SAE-decoded shift relative to a dense shift."""

    states: int
    hidden_size: int
    shift_energy_recovered: float | None
    centered_shift_r_squared: float | None
    explained_variance: float | None
    mean_bias_fraction: float | None
    mean_cosine: float | None
    mean_angle_degrees: float | None
    mean_relative_l2_error: float | None
    mean_norm_ratio: float | None
    dense_shift_rms: float
    decoded_shift_rms: float
    residual_shift_rms: float
    valid_direction_states: int

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def decode_feature_shift(
    manual_features: torch.Tensor,
    method_features: torch.Tensor,
    decoder_directions: torch.Tensor,
) -> torch.Tensor:
    """Decode a paired feature difference without adding the decoder bias."""
    if manual_features.shape != method_features.shape:
        raise ValueError("manual_features and method_features must have equal shapes")
    if manual_features.ndim != 2:
        raise ValueError("feature tensors must have shape [states, features]")
    if decoder_directions.ndim != 2:
        raise ValueError("decoder_directions must have shape [features, hidden_size]")
    if manual_features.shape[1] != decoder_directions.shape[0]:
        raise ValueError("feature width does not match decoder directions")
    return (method_features - manual_features) @ decoder_directions


def per_state_shift_metrics(
    dense_shift: torch.Tensor,
    decoded_shift: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Return per-state geometry, using NaN when a direction is undefined."""
    if dense_shift.shape != decoded_shift.shape:
        raise ValueError("dense_shift and decoded_shift must have equal shapes")
    if dense_shift.ndim != 2:
        raise ValueError("shift tensors must have shape [states, hidden_size]")
    if eps <= 0:
        raise ValueError("eps must be positive")

    dense = dense_shift.detach().double()
    decoded = decoded_shift.detach().double()
    residual = dense - decoded
    dense_norm = dense.norm(dim=-1)
    decoded_norm = decoded.norm(dim=-1)
    residual_norm = residual.norm(dim=-1)
    valid_dense = dense_norm > eps
    valid_direction = valid_dense & (decoded_norm > eps)
    nan = torch.full_like(dense_norm, float("nan"))

    cosine = functional.cosine_similarity(dense, decoded, dim=-1, eps=eps).clamp(-1, 1)
    cosine = torch.where(valid_direction, cosine, nan)
    angle = torch.where(valid_direction, torch.rad2deg(torch.acos(cosine)), nan)
    relative_l2_error = torch.where(valid_dense, residual_norm / dense_norm, nan)
    norm_ratio = torch.where(valid_dense, decoded_norm / dense_norm, nan)
    energy_recovered = torch.where(
        valid_dense,
        1.0 - residual_norm.square() / dense_norm.square(),
        nan,
    )
    return {
        "dense_norm": dense_norm,
        "decoded_norm": decoded_norm,
        "residual_norm": residual_norm,
        "cosine": cosine,
        "angle_degrees": angle,
        "relative_l2_error": relative_l2_error,
        "norm_ratio": norm_ratio,
        "shift_energy_recovered": energy_recovered,
        "valid_direction": valid_direction,
    }


def _finite_mean(values: torch.Tensor) -> float | None:
    finite = values[torch.isfinite(values)]
    return float(finite.mean().item()) if len(finite) else None


def compute_shift_coverage(
    dense_shift: torch.Tensor,
    decoded_shift: torch.Tensor,
) -> ShiftCoverageMetrics:
    """Aggregate zero-baseline and centered coverage of a decoded SAE shift."""
    per_state = per_state_shift_metrics(dense_shift, decoded_shift)
    # Centered shift variance can be much smaller than the common shift mean.
    # Accumulate aggregate sums in float64 so R2 = EV - bias remains stable.
    dense = dense_shift.detach().double()
    decoded = decoded_shift.detach().double()
    residual = dense - decoded
    states, hidden_size = dense.shape
    if states == 0:
        raise ValueError("shift tensors must contain at least one state")

    dense_sum_squares = float(dense.square().sum().item())
    decoded_sum_squares = float(decoded.square().sum().item())
    residual_sum_squares = float(residual.square().sum().item())
    centered_dense = dense - dense.mean(dim=0, keepdim=True)
    centered_residual = residual - residual.mean(dim=0, keepdim=True)
    centered_sum_squares = float(centered_dense.square().sum().item())
    centered_residual_sum_squares = float(centered_residual.square().sum().item())
    mean_residual = residual.mean(dim=0)
    mean_bias_sum_squares = float(mean_residual.square().sum().item()) * states

    energy_recovered = (
        1.0 - residual_sum_squares / dense_sum_squares
        if dense_sum_squares > 0
        else None
    )
    centered_r_squared = (
        1.0 - residual_sum_squares / centered_sum_squares
        if centered_sum_squares > 0
        else None
    )
    explained_variance = (
        1.0 - centered_residual_sum_squares / centered_sum_squares
        if centered_sum_squares > 0
        else None
    )
    mean_bias_fraction = (
        mean_bias_sum_squares / centered_sum_squares
        if centered_sum_squares > 0
        else None
    )
    denominator = states * hidden_size
    return ShiftCoverageMetrics(
        states=states,
        hidden_size=hidden_size,
        shift_energy_recovered=energy_recovered,
        centered_shift_r_squared=centered_r_squared,
        explained_variance=explained_variance,
        mean_bias_fraction=mean_bias_fraction,
        mean_cosine=_finite_mean(per_state["cosine"]),
        mean_angle_degrees=_finite_mean(per_state["angle_degrees"]),
        mean_relative_l2_error=_finite_mean(per_state["relative_l2_error"]),
        mean_norm_ratio=_finite_mean(per_state["norm_ratio"]),
        dense_shift_rms=(dense_sum_squares / denominator) ** 0.5,
        decoded_shift_rms=(decoded_sum_squares / denominator) ** 0.5,
        residual_shift_rms=(residual_sum_squares / denominator) ** 0.5,
        valid_direction_states=int(per_state["valid_direction"].sum().item()),
    )
