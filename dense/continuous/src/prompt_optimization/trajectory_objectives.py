"""Trajectory-level objectives for residual-shift predictors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch


TrainingMode = Literal[
    "prefill_once",
    "fixed_recurrent",
    "repredict_recurrent",
]

OBJECTIVE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "dense": ("dense",),
    "dense_plus_kl": ("dense", "kl"),
    "dense_sae_kl": ("dense", "enc", "dec", "kl"),
}


def validate_packed_trajectories(states: torch.Tensor, offsets: torch.Tensor) -> None:
    if states.ndim != 2:
        raise ValueError("states must have shape [positions, hidden_size]")
    if offsets.ndim != 1 or len(offsets) < 2:
        raise ValueError("offsets must delimit at least one trajectory")
    values = offsets.to(dtype=torch.long)
    if int(values[0]) != 0 or int(values[-1]) != len(states):
        raise ValueError("offsets must cover every packed position")
    if bool(((values[1:] - values[:-1]) < 1).any()):
        raise ValueError("packed trajectories cannot be empty")


def apply_packed_predictor(
    predictor: torch.nn.Module,
    baseline: torch.Tensor,
    offsets: torch.Tensor,
    mode: TrainingMode,
) -> torch.Tensor:
    """Apply a predictor to packed trajectories as it is used in generation."""
    validate_packed_trajectories(baseline, offsets)
    values = offsets.to(device=baseline.device, dtype=torch.long)
    anchors = values[:-1]
    anchor_states = baseline[anchors].float()
    if mode == "prefill_once":
        predicted = baseline.clone()
        predicted[anchors] = baseline[anchors] + predictor(anchor_states).to(baseline)
        return predicted
    if mode == "fixed_recurrent":
        lengths = values[1:] - values[:-1]
        delta = predictor(anchor_states).repeat_interleave(lengths, dim=0)
        return baseline + delta.to(baseline)
    if mode == "repredict_recurrent":
        return baseline + predictor(baseline.float()).to(baseline)
    raise ValueError(f"Unsupported predictor mode: {mode}")


def trajectory_objective_parts(
    predicted: torch.Tensor,
    adapted: torch.Tensor,
    *,
    readout: Callable[[torch.Tensor], torch.Tensor] | None = None,
    sae: torch.nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    """Compute per-position-mean DENSE, ENC, DEC, and forward-KL losses."""
    if predicted.shape != adapted.shape or predicted.ndim != 2:
        raise ValueError("predicted and adapted states must align as [positions, hidden_size]")
    difference = predicted.float() - adapted.detach().float()
    parts = {"dense": difference.square().mean()}
    if sae is not None:
        predicted_features = sae.encode(predicted.float())
        with torch.no_grad():
            adapted_features = sae.encode(adapted.float()).detach()
        feature_error = predicted_features - adapted_features
        parts["enc"] = feature_error.square().mean()
        parts["dec"] = (feature_error @ sae.W_dec).square().mean()
    if readout is not None:
        with torch.no_grad():
            adapted_logp = readout(adapted.float()).float().log_softmax(-1)
        predicted_logp = readout(predicted.float()).float().log_softmax(-1)
        parts["kl"] = (
            adapted_logp.exp() * (adapted_logp - predicted_logp)
        ).sum(-1).mean()
    return parts


def normalized_objective(
    parts: dict[str, torch.Tensor],
    scales: dict[str, float],
    objective: str,
) -> torch.Tensor:
    try:
        components = OBJECTIVE_COMPONENTS[objective]
    except KeyError as error:
        raise ValueError(f"Unknown objective: {objective}") from error
    missing = [name for name in components if name not in parts or name not in scales]
    if missing:
        raise ValueError(f"Missing objective components: {missing}")
    return sum(
        (parts[name] / scales[name] for name in components),
        parts[components[0]].new_zeros(()),
    )


def mean_shift_statistics(
    baseline: torch.Tensor,
    adapted: torch.Tensor,
    offsets: torch.Tensor,
    mode: TrainingMode,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return predictor initialization and input normalization statistics."""
    validate_packed_trajectories(baseline, offsets)
    values = offsets.to(dtype=torch.long)
    if mode == "repredict_recurrent":
        inputs = baseline.float()
        deltas = adapted.float() - inputs
    elif mode in ("prefill_once", "fixed_recurrent"):
        anchors = values[:-1]
        inputs = baseline[anchors].float()
        deltas = adapted[anchors].float() - inputs
    else:
        raise ValueError(f"Unsupported predictor mode: {mode}")
    input_mean = inputs.mean(0)
    input_scale = (inputs - input_mean).square().mean().sqrt().clamp_min(1e-6)
    return deltas.mean(0), input_mean, input_scale
