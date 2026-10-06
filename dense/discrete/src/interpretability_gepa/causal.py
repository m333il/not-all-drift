from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .geometry import orthonormal_basis, project


def mean_direction(target: np.ndarray, seed: np.ndarray) -> np.ndarray:
    if target.shape != seed.shape:
        raise ValueError("paired activations must have identical shapes")
    return (target.astype(np.float32) - seed.astype(np.float32)).mean(axis=0)


def norm_matched_random(vector: np.ndarray, draws: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random = rng.normal(size=(draws, vector.size))
    random /= np.maximum(np.linalg.norm(random, axis=1, keepdims=True), 1e-12)
    return random * np.linalg.norm(vector)


def relative_alpha(vector: np.ndarray, reference_states: np.ndarray, *, fraction: float) -> float:
    if fraction <= 0:
        raise ValueError("relative intervention fraction must be positive")
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm == 0:
        raise ValueError("cannot scale a zero intervention vector")
    if reference_states.ndim != 2:
        raise ValueError("reference states must have shape [examples,hidden]")
    reference_norm = float(np.linalg.norm(reference_states, axis=1).mean())
    return fraction * reference_norm / vector_norm


@dataclass(frozen=True, slots=True)
class InterventionCell:
    layer: int
    alpha: float
    validation_recovery: float
    parse_failure_rate: float
    avg_predictions: float
    id: str


def select_intervention_cell(
    cells: tuple[InterventionCell, ...] | list[InterventionCell],
    *,
    parse_limit: float,
    baseline_avg_predictions: float,
) -> InterventionCell:
    eligible = [
        cell
        for cell in cells
        if cell.parse_failure_rate <= parse_limit
        and cell.avg_predictions <= 2 * baseline_avg_predictions
    ]
    if not eligible:
        raise ValueError("no intervention cell passed preregistered degradation gates")
    return max(eligible, key=lambda cell: (cell.validation_recovery, -cell.parse_failure_rate))


def shuffled_pair_deltas(target: np.ndarray, seed: np.ndarray, random_seed: int = 0) -> np.ndarray:
    """Negative control for per-input patching: per-example deltas with shuffled pairing."""
    if target.shape != seed.shape or target.ndim != 2:
        raise ValueError("paired states must share [examples,hidden]")
    permutation = np.random.default_rng(random_seed).permutation(len(seed))
    return target - seed[permutation]


def mismatched_pair_indices(labels: np.ndarray, *, same_label: bool, seed: int = 0) -> np.ndarray:
    if labels.ndim != 2:
        raise ValueError("labels must have shape [examples,labels]")
    rng = np.random.default_rng(seed)
    result = np.empty(len(labels), dtype=int)
    for index, row in enumerate(labels.astype(bool)):
        overlap = np.logical_and(labels.astype(bool), row).any(axis=1)
        eligible = overlap if same_label else ~overlap
        eligible[index] = False
        candidates = np.flatnonzero(eligible)
        if not len(candidates):
            kind = "same-label" if same_label else "different-label"
            raise ValueError(f"no {kind} mismatch candidate for example {index}")
        result[index] = int(rng.choice(candidates))
    return result


def patch_values(
    seed_states: np.ndarray,
    target_states: np.ndarray,
    *,
    mode: Literal["state", "delta", "mean"],
) -> np.ndarray:
    if seed_states.shape != target_states.shape or seed_states.ndim != 2:
        raise ValueError("patch states must share [examples,hidden]")
    deltas = target_states - seed_states
    if mode == "state":
        return target_states.copy()
    if mode == "delta":
        return deltas
    if mode == "mean":
        return np.repeat(deltas.mean(axis=0, keepdims=True), len(deltas), axis=0)
    raise ValueError(f"unknown patch mode: {mode}")


@dataclass(frozen=True)
class LeaceEraser:
    matrix: np.ndarray
    bias: np.ndarray


def fit_leace(x: np.ndarray, concepts: np.ndarray, ridge: float = 1e-8) -> LeaceEraser:
    """Fit the closed-form covariance-whitened affine LEACE operator."""
    centered_x = x - x.mean(0)
    centered_c = concepts - concepts.mean(0)
    covariance = centered_x.T @ centered_x / len(x)
    cross = centered_x.T @ centered_c / len(x)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = ridge * max(float(eigenvalues.max()), 1.0)
    safe = np.maximum(eigenvalues, floor)
    whitening = (eigenvectors * (1.0 / np.sqrt(safe))) @ eigenvectors.T
    whitening_inverse = (eigenvectors * np.sqrt(safe)) @ eigenvectors.T
    concept_basis = orthonormal_basis((whitening @ cross).T)
    matrix = np.eye(x.shape[1]) - whitening_inverse @ concept_basis @ concept_basis.T @ whitening
    mean = x.mean(0)
    return LeaceEraser(matrix, mean - matrix @ mean)


def apply_leace(x: np.ndarray, eraser: LeaceEraser) -> np.ndarray:
    return x @ eraser.matrix.T + eraser.bias


def erase_subspace(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Orthogonal matched-control erasure for a supplied subspace."""
    return x - (x @ basis) @ basis.T if basis.size else x.copy()


def decompose_direction(
    vector: np.ndarray, probe_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return project(vector, orthonormal_basis(probe_weights))


def block_for_residual_index(index: int) -> int:
    """Map an activation-store index to the block that writes it.

    Store index 0 is the embedding output and index i the output of block i-1.
    Index 0 has no block to hook.
    """
    if index <= 0:
        raise ValueError(
            "store index 0 is the embedding output and has no block whose forward hook "
            "writes it; edit at index 1 or deeper"
        )
    return index - 1


@dataclass(frozen=True)
class EditSpec:
    layer: int
    mode: Literal["add", "replace", "erase"]
    positions: Literal["last", "prompt_last", "all"] = "last"
    alpha: float = 1.0


def transformer_layers(model: Any) -> Any:
    base = getattr(model, "model", model)
    for candidate in (
        getattr(base, "layers", None),
        getattr(getattr(base, "model", None), "layers", None),
        getattr(getattr(base, "transformer", None), "h", None),
    ):
        if candidate is not None:
            return candidate
    raise TypeError("unsupported decoder architecture: cannot locate transformer blocks")


class ResidualEditor(AbstractContextManager["ResidualEditor"]):
    def __init__(self, model: Any, spec: EditSpec, value: Any, attention_mask: Any | None = None):
        self.spec, self.value, self.attention_mask = spec, value, attention_mask
        self._edited_prefill = False
        self._handle = transformer_layers(model)[spec.layer].register_forward_hook(self._hook)

    def _hook(self, module: Any, inputs: Any, output: Any) -> Any:
        del module, inputs
        hidden = output[0] if isinstance(output, tuple) else output
        if self.spec.positions == "prompt_last" and self._edited_prefill:
            return output
        edited = hidden.clone()
        if self.spec.positions == "all":
            positions: Any = slice(None)
        else:
            # With KV cache the sequence length is 1, so resolve "last" per forward.
            positions = -1
        if self.spec.mode == "erase":
            if not isinstance(self.value, tuple) or len(self.value) != 2:
                raise TypeError("erase intervention requires a (matrix, bias) tuple")
            matrix = self.value[0].to(hidden.device, hidden.dtype)
            bias = self.value[1].to(hidden.device, hidden.dtype)
            value = matrix
        else:
            value = self.value.to(hidden.device, hidden.dtype)
            if value.ndim not in {1, 2}:
                raise ValueError("intervention value must be [hidden] or [batch,hidden]")
            if value.ndim == 2 and value.shape[0] != hidden.shape[0]:
                raise ValueError("paired intervention batch does not match hidden states")
            if self.spec.positions == "all" and value.ndim == 2:
                value = value.unsqueeze(1)
        if self.spec.positions != "all" and value.ndim > 1 and self.spec.mode != "erase":
            import torch

            rows = torch.arange(hidden.shape[0], device=hidden.device)
            if self.spec.mode == "add":
                edited[rows, positions] += self.spec.alpha * value
            elif self.spec.mode == "replace":
                edited[rows, positions] = value
            else:
                edited[rows, positions] = edited[rows, positions] @ matrix.T + bias
        elif self.spec.mode == "add":
            edited[:, positions] += self.spec.alpha * value
        elif self.spec.mode == "replace":
            edited[:, positions] = value
        else:
            edited[:, positions] = edited[:, positions] @ matrix.T + bias
        if self.spec.positions == "prompt_last":
            self._edited_prefill = True
        return (edited, *output[1:]) if isinstance(output, tuple) else edited

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Literal[False]:
        self._handle.remove()
        return False


def logit_lens(
    hidden_states: list[Any] | tuple[Any, ...], model: Any, yes_token: int, no_token: int
) -> np.ndarray:
    import torch

    base = getattr(model, "model", model)
    norm = getattr(base, "norm", None)
    if norm is None:
        norm = getattr(base, "final_layernorm", None)
    values = []
    with torch.no_grad():
        for hidden in hidden_states:
            last = hidden[:, -1]
            logits = model.lm_head(norm(last) if norm is not None else last)
            values.append((logits[:, yes_token] - logits[:, no_token]).float().cpu().numpy())
    return np.stack(values, axis=1)


def generate_with_edit(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    spec: EditSpec,
    value: Any,
    max_new_tokens: int = 128,
    batch_size: int = 8,
) -> list[str]:
    """Greedy HF generation under a residual add/replace/erase intervention."""
    outputs: list[str] = []
    tokenizer.padding_side = "left"
    for start in range(0, len(prompts), batch_size):
        # The chat template already emits <bos>.
        encoded = tokenizer(
            prompts[start : start + batch_size],
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        device = next(model.parameters()).device
        encoded = {key: tensor.to(device) for key, tensor in encoded.items()}
        batch_value = value
        if getattr(value, "ndim", 1) > 1 and spec.mode != "erase":
            batch_value = value[start : start + batch_size]
        with ResidualEditor(model, spec, batch_value, encoded["attention_mask"]):
            generated = model.generate(
                **encoded,
                do_sample=False,
                temperature=None,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        continuation = generated[:, encoded["input_ids"].shape[1] :]
        outputs.extend(tokenizer.batch_decode(continuation, skip_special_tokens=True))
    return outputs
