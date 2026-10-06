"""Minimal Gemma Scope JumpReLU inference and reconstruction helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional

ReconstructionScope = Literal["anchor", "all_valid"]


@dataclass(frozen=True)
class ReconstructionMetrics:
    """Aggregate SAE reconstruction statistics over token states."""

    states: int
    hidden_size: int
    mse: float
    normalized_mse: float | None
    fvu: float | None
    r_squared: float | None
    explained_variance: float | None
    mean_bias_fraction: float | None
    mean_cosine: float
    mean_relative_l2_error: float
    mean_norm_ratio: float
    mean_l0: float

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "states": self.states,
            "hidden_size": self.hidden_size,
            "mse": self.mse,
            "normalized_mse": self.normalized_mse,
            "fvu": self.fvu,
            "r_squared": self.r_squared,
            "explained_variance": self.explained_variance,
            "mean_bias_fraction": self.mean_bias_fraction,
            "mean_cosine": self.mean_cosine,
            "mean_relative_l2_error": self.mean_relative_l2_error,
            "mean_norm_ratio": self.mean_norm_ratio,
            "mean_l0": self.mean_l0,
        }


class ReconstructionAccumulator:
    """Streaming statistics without retaining dense feature activations."""

    def __init__(self, *, retain_state_metrics: bool = False) -> None:
        self.states = 0
        self.hidden_size: int | None = None
        self.error_sum_squares = 0.0
        self.input_sum_squares = 0.0
        self.input_sum: torch.Tensor | None = None
        self.error_sum: torch.Tensor | None = None
        self.cosine_sum = 0.0
        self.relative_l2_error_sum = 0.0
        self.norm_ratio_sum = 0.0
        self.l0_sum = 0.0
        self.retain_state_metrics = retain_state_metrics
        self._state_metrics: dict[str, list[torch.Tensor]] = {
            name: []
            for name in (
                "mse",
                "input_rms",
                "reconstruction_rms",
                "cosine",
                "relative_l2_error",
                "norm_ratio",
                "l0",
            )
        }

    def update(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        feature_activations: torch.Tensor,
    ) -> None:
        if inputs.shape != reconstructions.shape:
            raise ValueError("inputs and reconstructions must have identical shapes")
        if inputs.ndim < 2:
            raise ValueError("inputs must have at least a state and hidden dimension")
        if feature_activations.shape[:-1] != inputs.shape[:-1]:
            raise ValueError("feature activations must match the input state dimensions")

        flattened_inputs = inputs.detach().float().reshape(-1, inputs.shape[-1])
        flattened_reconstructions = reconstructions.detach().float().reshape_as(
            flattened_inputs
        )
        flattened_features = feature_activations.detach().reshape(
            -1, feature_activations.shape[-1]
        )
        if flattened_inputs.shape[0] == 0:
            return

        hidden_size = int(flattened_inputs.shape[1])
        if self.hidden_size is None:
            self.hidden_size = hidden_size
            self.input_sum = torch.zeros(hidden_size, dtype=torch.float64)
            self.error_sum = torch.zeros(hidden_size, dtype=torch.float64)
        elif self.hidden_size != hidden_size:
            raise ValueError("hidden size changed between accumulator updates")

        self.states += int(flattened_inputs.shape[0])
        error = flattened_inputs - flattened_reconstructions
        self.error_sum_squares += float(error.square().sum().item())
        self.input_sum_squares += float(flattened_inputs.square().sum().item())
        if self.input_sum is None or self.error_sum is None:
            raise RuntimeError("streaming sums were not initialized")
        self.input_sum += flattened_inputs.sum(dim=0, dtype=torch.float64).cpu()
        self.error_sum += error.sum(dim=0, dtype=torch.float64).cpu()
        cosine = functional.cosine_similarity(
            flattened_inputs,
            flattened_reconstructions,
            dim=-1,
            eps=1e-12,
        )
        self.cosine_sum += float(cosine.sum().item())
        input_norm = flattened_inputs.norm(dim=-1).clamp_min(1e-12)
        relative_l2_error = error.norm(dim=-1) / input_norm
        norm_ratio = flattened_reconstructions.norm(dim=-1) / input_norm
        self.relative_l2_error_sum += float(relative_l2_error.sum().item())
        self.norm_ratio_sum += float(norm_ratio.sum().item())
        per_state_l0 = (flattened_features != 0).sum(dim=-1)
        self.l0_sum += float(per_state_l0.sum().item())
        if self.retain_state_metrics:
            retained = {
                "mse": error.square().mean(dim=-1),
                "input_rms": flattened_inputs.square().mean(dim=-1).sqrt(),
                "reconstruction_rms": flattened_reconstructions.square().mean(dim=-1).sqrt(),
                "cosine": cosine,
                "relative_l2_error": relative_l2_error,
                "norm_ratio": norm_ratio,
                "l0": per_state_l0.float(),
            }
            for name, values in retained.items():
                self._state_metrics[name].append(values.cpu())

    def compute(self) -> ReconstructionMetrics:
        if (
            self.states == 0
            or self.hidden_size is None
            or self.input_sum is None
            or self.error_sum is None
        ):
            raise RuntimeError("No reconstruction states were accumulated")
        centered_sum_squares = self.input_sum_squares - float(
            self.input_sum.square().sum().item()
        ) / self.states
        fvu = (
            self.error_sum_squares / centered_sum_squares
            if centered_sum_squares > 0
            else None
        )
        centered_error_sum_squares = self.error_sum_squares - float(
            self.error_sum.square().sum().item()
        ) / self.states
        r_squared = 1.0 - fvu if fvu is not None else None
        explained_variance = (
            1.0 - centered_error_sum_squares / centered_sum_squares
            if centered_sum_squares > 0
            else None
        )
        mean_bias_fraction = (
            float(self.error_sum.square().sum().item())
            / self.states
            / centered_sum_squares
            if centered_sum_squares > 0
            else None
        )
        return ReconstructionMetrics(
            states=self.states,
            hidden_size=self.hidden_size,
            mse=self.error_sum_squares / (self.states * self.hidden_size),
            normalized_mse=(
                self.error_sum_squares / self.input_sum_squares
                if self.input_sum_squares > 0
                else None
            ),
            fvu=fvu,
            r_squared=r_squared,
            explained_variance=explained_variance,
            mean_bias_fraction=mean_bias_fraction,
            mean_cosine=self.cosine_sum / self.states,
            mean_relative_l2_error=self.relative_l2_error_sum / self.states,
            mean_norm_ratio=self.norm_ratio_sum / self.states,
            mean_l0=self.l0_sum / self.states,
        )

    def distribution_summary(self) -> dict[str, dict[str, float]]:
        """Return deterministic per-state quantiles when retention was requested."""
        if not self.retain_state_metrics:
            raise RuntimeError("State metric retention was not enabled")
        if self.states == 0:
            raise RuntimeError("No reconstruction states were accumulated")
        quantiles = torch.tensor((0.0, 0.5, 0.9, 0.95, 0.99, 1.0), dtype=torch.float64)
        names = ("min", "p50", "p90", "p95", "p99", "max")
        summary: dict[str, dict[str, float]] = {}
        for metric, chunks in self._state_metrics.items():
            values = torch.cat(chunks).to(dtype=torch.float64)
            observed = torch.quantile(values, quantiles)
            summary[metric] = {
                name: float(value.item())
                for name, value in zip(names, observed, strict=True)
            }
        return summary


class GemmaScopeJumpReLU(nn.Module):
    """Inference-only Gemma Scope SAE matching the official SAELens formula."""

    REQUIRED_KEYS = frozenset({"W_enc", "W_dec", "b_enc", "b_dec", "threshold"})

    def __init__(
        self,
        *,
        W_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_enc: torch.Tensor,
        b_dec: torch.Tensor,
        threshold: torch.Tensor,
    ) -> None:
        super().__init__()
        self._validate_shapes(W_enc, W_dec, b_enc, b_dec, threshold)
        self.register_buffer("W_enc", W_enc.contiguous())
        self.register_buffer("W_dec", W_dec.contiguous())
        self.register_buffer("b_enc", b_enc.contiguous())
        self.register_buffer("b_dec", b_dec.contiguous())
        self.register_buffer("threshold", threshold.contiguous())

    @staticmethod
    def _validate_shapes(
        W_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_enc: torch.Tensor,
        b_dec: torch.Tensor,
        threshold: torch.Tensor,
    ) -> None:
        if W_enc.ndim != 2 or W_dec.ndim != 2:
            raise ValueError("W_enc and W_dec must be matrices")
        d_in, d_sae = W_enc.shape
        if W_dec.shape != (d_sae, d_in):
            raise ValueError("W_dec must have shape [d_sae, d_in]")
        if b_enc.shape != (d_sae,) or threshold.shape != (d_sae,):
            raise ValueError("b_enc and threshold must have shape [d_sae]")
        if b_dec.shape != (d_in,):
            raise ValueError("b_dec must have shape [d_in]")
        dtypes = {tensor.dtype for tensor in (W_enc, W_dec, b_enc, b_dec, threshold)}
        if len(dtypes) != 1:
            raise ValueError("All SAE tensors must use the same dtype")

    @classmethod
    def from_arrays(
        cls,
        arrays: Mapping[str, np.ndarray],
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> GemmaScopeJumpReLU:
        missing = cls.REQUIRED_KEYS - set(arrays)
        if missing:
            raise ValueError(f"Gemma Scope weights miss arrays: {sorted(missing)}")
        tensors = {
            key: torch.from_numpy(np.array(arrays[key], copy=True)).to(
                device=device,
                dtype=dtype,
            )
            for key in cls.REQUIRED_KEYS
        }
        return cls(**tensors)

    @classmethod
    def from_npz(
        cls,
        path: Path,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> GemmaScopeJumpReLU:
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as arrays:
            return cls.from_arrays(arrays, device=device, dtype=dtype)

    @property
    def d_in(self) -> int:
        return int(self.W_enc.shape[0])

    @property
    def d_sae(self) -> int:
        return int(self.W_enc.shape[1])

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.d_in:
            raise ValueError(f"Expected hidden size {self.d_in}, found {inputs.shape[-1]}")
        inputs = inputs.to(device=self.W_enc.device, dtype=self.W_enc.dtype)
        hidden_pre = inputs @ self.W_enc + self.b_enc
        return torch.relu(hidden_pre) * (hidden_pre > self.threshold)

    def decode(self, feature_activations: torch.Tensor) -> torch.Tensor:
        if feature_activations.shape[-1] != self.d_sae:
            raise ValueError(
                f"Expected SAE width {self.d_sae}, found {feature_activations.shape[-1]}"
            )
        return feature_activations @ self.W_dec + self.b_dec

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_activations = self.encode(inputs)
        return self.decode(feature_activations), feature_activations

    def reconstruct_chunked(
        self,
        inputs: torch.Tensor,
        *,
        chunk_size: int,
        accumulator: ReconstructionAccumulator | None = None,
        subgroup_masks: Mapping[str, torch.Tensor] | None = None,
        subgroup_accumulators: Mapping[str, ReconstructionAccumulator] | None = None,
    ) -> tuple[torch.Tensor, ReconstructionMetrics]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if inputs.shape[-1] != self.d_in:
            raise ValueError(f"Expected hidden size {self.d_in}, found {inputs.shape[-1]}")
        original_shape = inputs.shape
        flattened = inputs.reshape(-1, self.d_in)
        flattened_subgroups: dict[str, torch.Tensor] = {}
        if (subgroup_masks is None) != (subgroup_accumulators is None):
            raise ValueError(
                "subgroup_masks and subgroup_accumulators must be provided together"
            )
        if subgroup_masks is not None and subgroup_accumulators is not None:
            if set(subgroup_masks) != set(subgroup_accumulators):
                raise ValueError("Subgroup masks and accumulators must have equal keys")
            for name, mask in subgroup_masks.items():
                if mask.shape != original_shape[:-1]:
                    raise ValueError(
                        f"Subgroup {name!r} has shape {tuple(mask.shape)}, expected "
                        f"{tuple(original_shape[:-1])}"
                    )
                flattened_subgroups[name] = mask.to(
                    device=flattened.device,
                    dtype=torch.bool,
                ).reshape(-1)
        reconstructed_chunks: list[torch.Tensor] = []
        active_accumulator = accumulator or ReconstructionAccumulator()
        for start in range(0, len(flattened), chunk_size):
            current = flattened[start : start + chunk_size]
            reconstruction, features = self(current)
            active_accumulator.update(current, reconstruction, features)
            if subgroup_accumulators is not None:
                for name, subgroup_accumulator in subgroup_accumulators.items():
                    members = flattened_subgroups[name][start : start + chunk_size]
                    if bool(members.any()):
                        subgroup_accumulator.update(
                            current[members],
                            reconstruction[members],
                            features[members],
                        )
            reconstructed_chunks.append(reconstruction.to(dtype=inputs.dtype))
        reconstructed = torch.cat(reconstructed_chunks, dim=0).reshape(original_shape)
        return reconstructed, active_accumulator.compute()


def resid_post_hidden_state_index(layer: int) -> int:
    """Map a zero-based Gemma block index to Transformers hidden_states index."""
    if layer < 0:
        raise ValueError("layer must be non-negative")
    return layer + 1


def build_reconstruction_mask(
    attention_mask: torch.Tensor,
    *,
    prepended_virtual_tokens: int,
    scope: ReconstructionScope,
) -> torch.Tensor:
    """Select the aligned prompt anchor or every valid prompt state."""
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    if prepended_virtual_tokens < 0:
        raise ValueError("prepended_virtual_tokens must be non-negative")
    valid_text = attention_mask.to(dtype=torch.bool)
    if not bool(valid_text.any(dim=1).all()):
        raise ValueError("Every prompt must contain at least one non-padding token")
    if prepended_virtual_tokens:
        virtual = torch.ones(
            (len(valid_text), prepended_virtual_tokens),
            dtype=torch.bool,
            device=valid_text.device,
        )
        valid_states = torch.cat((virtual, valid_text), dim=1)
    else:
        valid_states = valid_text

    if scope == "all_valid":
        return valid_states
    if scope != "anchor":
        raise ValueError(f"Unknown reconstruction scope: {scope}")

    text_positions = torch.arange(
        valid_text.shape[1], device=valid_text.device, dtype=torch.long
    ).expand_as(valid_text)
    final_text_positions = text_positions.masked_fill(~valid_text, -1).max(dim=1).values
    final_state_positions = final_text_positions + prepended_virtual_tokens
    selected = torch.zeros_like(valid_states)
    selected.scatter_(1, final_state_positions[:, None], True)
    return selected


def hidden_from_decoder_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("Expected decoder layer output tensor or tuple beginning with a tensor")


def replace_decoder_hidden(output: Any, replacement: torch.Tensor) -> Any:
    """Preserve the decoder layer output container while replacing hidden states."""
    if isinstance(output, torch.Tensor):
        return replacement
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return (replacement, *output[1:])
    raise TypeError("Expected decoder layer output tensor or tuple beginning with a tensor")


def resolve_gemma_decoder_layer(model: Any, layer: int) -> nn.Module:
    """Resolve a Gemma decoder block from a Transformers or PEFT causal LM."""
    if layer < 0:
        raise ValueError("layer must be non-negative")
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(base_model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise TypeError("Expected a causal LM with model.layers")
    if layer >= len(layers):
        raise ValueError(f"Layer {layer} is outside model with {len(layers)} blocks")
    return layers[layer]


def resolve_gemma_final_norm(model: Any) -> nn.Module:
    """Resolve the final Gemma normalization following the last decoder block."""
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(base_model, "model", None)
    final_norm = getattr(backbone, "norm", None)
    if not isinstance(final_norm, nn.Module):
        raise TypeError("Expected a causal LM with model.norm")
    return final_norm


class OneShotReconstructionHook:
    """Reconstruct selected states on the first decoder call of one generation batch."""

    def __init__(
        self,
        sae: GemmaScopeJumpReLU,
        selection_mask: torch.Tensor,
        *,
        chunk_size: int,
        accumulator: ReconstructionAccumulator | None = None,
        diagnostic_top_k: int = 0,
        metric_group_masks: Mapping[str, torch.Tensor] | None = None,
        metric_group_accumulators: Mapping[str, ReconstructionAccumulator] | None = None,
    ) -> None:
        if selection_mask.ndim != 2:
            raise ValueError("selection_mask must have shape [batch, sequence]")
        self.sae = sae
        self.selection_mask = selection_mask.to(dtype=torch.bool)
        self.chunk_size = chunk_size
        self.accumulator = accumulator or ReconstructionAccumulator()
        if diagnostic_top_k < 0:
            raise ValueError("diagnostic_top_k must be non-negative")
        self.diagnostic_top_k = diagnostic_top_k
        self.metric_group_masks = {
            name: mask.to(dtype=torch.bool)
            for name, mask in (metric_group_masks or {}).items()
        }
        for name, mask in self.metric_group_masks.items():
            if mask.shape != selection_mask.shape:
                raise ValueError(
                    f"Metric group {name!r} has shape {tuple(mask.shape)}, expected "
                    f"{tuple(selection_mask.shape)}"
                )
            if bool((mask & ~self.selection_mask).any()):
                raise ValueError(f"Metric group {name!r} is not a selection subset")
        if metric_group_accumulators is not None and set(metric_group_accumulators) != set(
            self.metric_group_masks
        ):
            raise ValueError("Metric group masks and accumulators must have equal keys")
        self.group_accumulators = dict(
            metric_group_accumulators
            or {
                name: ReconstructionAccumulator(
                    retain_state_metrics=self.accumulator.retain_state_metrics
                )
                for name in self.metric_group_masks
            }
        )
        self.applied = False
        self.metrics: ReconstructionMetrics | None = None
        self.group_metrics: dict[str, ReconstructionMetrics] = {}
        self.top_error_states: list[dict[str, float | int]] = []

    def __call__(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if self.applied:
            return output
        hidden = hidden_from_decoder_output(output)
        if hidden.shape[:2] != self.selection_mask.shape:
            raise ValueError(
                "Decoder output and reconstruction mask disagree: "
                f"{tuple(hidden.shape[:2])} vs {tuple(self.selection_mask.shape)}"
            )
        mask = self.selection_mask.to(device=hidden.device)
        selected = hidden[mask]
        reconstruction, self.metrics = self.sae.reconstruct_chunked(
            selected,
            chunk_size=self.chunk_size,
            accumulator=self.accumulator,
            subgroup_masks={
                name: group_mask.to(device=hidden.device)[mask]
                for name, group_mask in self.metric_group_masks.items()
                if name in self.group_accumulators
            },
            subgroup_accumulators=self.group_accumulators,
        )
        self.group_metrics = {
            name: accumulator.compute()
            for name, accumulator in self.group_accumulators.items()
            if accumulator.states
        }
        if self.diagnostic_top_k:
            selected_float = selected.detach().float()
            reconstructed_float = reconstruction.detach().float()
            per_state_mse = (selected_float - reconstructed_float).square().mean(dim=-1)
            count = min(self.diagnostic_top_k, len(per_state_mse))
            values, indices = per_state_mse.topk(count)
            top_features = self.sae.encode(selected[indices])
            top_cosines = functional.cosine_similarity(
                selected_float[indices],
                reconstructed_float[indices],
                dim=-1,
                eps=1e-12,
            )
            coordinates = mask.nonzero(as_tuple=False).cpu()
            for rank, (value, selected_index) in enumerate(
                zip(values, indices, strict=True),
                start=1,
            ):
                index = int(selected_index.item())
                coordinate = coordinates[index]
                self.top_error_states.append(
                    {
                        "batch_rank": rank,
                        "selected_index": index,
                        "batch_index": int(coordinate[0].item()),
                        "sequence_position": int(coordinate[1].item()),
                        "mse": float(value.item()),
                        "input_rms": float(selected_float[index].square().mean().sqrt().item()),
                        "reconstruction_rms": float(
                            reconstructed_float[index].square().mean().sqrt().item()
                        ),
                        "cosine": float(top_cosines[rank - 1].item()),
                        "l0": int((top_features[rank - 1] != 0).sum().item()),
                    }
                )
        modified = hidden.clone()
        modified[mask] = reconstruction.to(device=hidden.device, dtype=hidden.dtype)
        self.applied = True
        return replace_decoder_hidden(output, modified)
