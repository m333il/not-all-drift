"""Pure helpers for residual-stream probing experiments."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch.nn import functional

@dataclass(frozen=True)
class ProbeBank:
    """Layerwise one-vs-rest linear probes in standardized coordinates."""

    class_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    weights: np.ndarray
    intercepts: np.ndarray
    selected_regularization: np.ndarray


def select_last_prompt_states(
    hidden_states: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    *,
    num_virtual_tokens: int,
    hidden_state_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """Select the final non-padding prompt position from every layer.

    The tokenizer inputs must be right padded. PEFT prompt tuning prepends virtual
    embeddings, so their count is added to each unpadded tokenizer length.
    """
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    if num_virtual_tokens < 0:
        raise ValueError("num_virtual_tokens must be non-negative")
    if not hidden_states:
        raise ValueError("hidden_states must not be empty")

    if hidden_state_indices is None:
        selected_layers = tuple(hidden_states)
    else:
        normalized_indices = tuple(int(index) for index in hidden_state_indices)
        if not normalized_indices:
            raise ValueError("hidden_state_indices must not be empty")
        if len(set(normalized_indices)) != len(normalized_indices):
            raise ValueError("hidden_state_indices must be unique")
        if min(normalized_indices) < 0 or max(normalized_indices) >= len(hidden_states):
            raise ValueError(
                f"hidden_state_indices must be in [0, {len(hidden_states) - 1}]"
            )
        selected_layers = tuple(hidden_states[index] for index in normalized_indices)

    positions = attention_mask.sum(dim=1, dtype=torch.long) + num_virtual_tokens - 1
    batch_indices = torch.arange(attention_mask.shape[0], device=attention_mask.device)
    selected = [
        layer[batch_indices, positions.to(layer.device)]
        for layer in selected_layers
    ]
    return torch.stack(selected, dim=1)


def _validate_probe_arrays(states: np.ndarray, labels: np.ndarray) -> None:
    if states.ndim != 3:
        raise ValueError("states must have shape [samples, layers, hidden_size]")
    if labels.ndim != 1 or len(labels) != len(states):
        raise ValueError("labels must have shape [samples]")


def fit_probe_bank(
    train_states: np.ndarray,
    train_labels: np.ndarray,
    validation_states: np.ndarray,
    validation_labels: np.ndarray,
    *,
    class_names: Sequence[str],
    c_values: Sequence[float],
    seed: int,
) -> ProbeBank:
    """Fit one L2 logistic-regression probe per layer and one-vs-rest class."""
    _validate_probe_arrays(train_states, train_labels)
    _validate_probe_arrays(validation_states, validation_labels)
    if train_states.shape[1:] != validation_states.shape[1:]:
        raise ValueError("train and validation state shapes must agree")
    normalized_c = tuple(sorted({float(value) for value in c_values}))
    if not normalized_c or normalized_c[0] <= 0:
        raise ValueError("c_values must contain positive values")
    if len(class_names) < 2:
        raise ValueError("class_names must contain at least two classes")

    _, num_layers, hidden_size = train_states.shape
    num_classes = len(class_names)
    means = train_states.mean(axis=0, dtype=np.float64).astype(np.float32)
    scales = train_states.std(axis=0, dtype=np.float64).astype(np.float32)
    scales[scales == 0] = 1.0
    weights = np.empty((num_layers, num_classes, hidden_size), dtype=np.float32)
    intercepts = np.empty((num_layers, num_classes), dtype=np.float32)
    selected_c = np.empty((num_layers, num_classes), dtype=np.float32)

    for layer in range(num_layers):
        train_x = (train_states[:, layer] - means[layer]) / scales[layer]
        validation_x = (
            validation_states[:, layer] - means[layer]
        ) / scales[layer]
        for class_index in range(num_classes):
            train_y = (train_labels == class_index).astype(np.int64)
            validation_y = (validation_labels == class_index).astype(np.int64)
            best_model: LogisticRegression | None = None
            best_score = float("-inf")
            best_c = normalized_c[0]
            for c_value in normalized_c:
                candidate = LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=seed,
                    solver="liblinear",
                )
                candidate.fit(train_x, train_y)
                score = f1_score(
                    validation_y,
                    candidate.predict(validation_x),
                    zero_division=0,
                )
                if score > best_score:
                    best_model = candidate
                    best_score = float(score)
                    best_c = c_value
            if best_model is None:
                raise RuntimeError("probe selection failed")
            weights[layer, class_index] = best_model.coef_[0]
            intercepts[layer, class_index] = best_model.intercept_[0]
            selected_c[layer, class_index] = best_c

    return ProbeBank(
        class_names=tuple(class_names),
        means=means,
        scales=scales,
        weights=weights,
        intercepts=intercepts,
        selected_regularization=selected_c,
    )


def score_probe_bank(
    bank: ProbeBank,
    states: np.ndarray,
    labels: np.ndarray,
) -> dict[str, list[dict[str, object]]]:
    """Score a fitted probe bank on held-out or cross-condition states."""
    _validate_probe_arrays(states, labels)
    if states.shape[1:] != bank.means.shape:
        raise ValueError("state shape does not match probe bank")

    layer_metrics: list[dict[str, object]] = []
    for layer in range(states.shape[1]):
        standardized = (states[:, layer] - bank.means[layer]) / bank.scales[layer]
        logits = standardized @ bank.weights[layer].T + bank.intercepts[layer]
        predictions = logits >= 0
        per_class = {
            class_name: float(
                f1_score(
                    labels == class_index,
                    predictions[:, class_index],
                    zero_division=0,
                )
            )
            for class_index, class_name in enumerate(bank.class_names)
        }
        layer_metrics.append(
            {
                "layer": layer,
                "macro_f1": float(np.mean(list(per_class.values()))),
                "per_class_f1": per_class,
            }
        )
    return {"layers": layer_metrics}


def fit_torch_probe_bank(
    train_states: torch.Tensor,
    train_labels: torch.Tensor,
    validation_states: torch.Tensor,
    validation_labels: torch.Tensor,
    *,
    class_names: Sequence[str],
    l2_values: Sequence[float],
    device: torch.device,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> ProbeBank:
    """Fit layerwise logistic probes on GPU and select L2 by validation F1."""
    if train_states.ndim != 3 or validation_states.ndim != 3:
        raise ValueError("states must have shape [samples, layers, hidden_size]")
    if train_states.shape[1:] != validation_states.shape[1:]:
        raise ValueError("train and validation state shapes must agree")
    if len(train_labels) != len(train_states) or len(validation_labels) != len(
        validation_states
    ):
        raise ValueError("labels must align with states")
    normalized_l2 = tuple(sorted({float(value) for value in l2_values}))
    if not normalized_l2 or normalized_l2[0] < 0:
        raise ValueError("l2_values must contain non-negative values")
    if steps <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("steps, batch_size, and learning_rate must be positive")

    num_layers = train_states.shape[1]
    hidden_size = train_states.shape[2]
    num_classes = len(class_names)
    means = np.empty((num_layers, hidden_size), dtype=np.float32)
    scales = np.empty((num_layers, hidden_size), dtype=np.float32)
    weights_out = np.empty((num_layers, num_classes, hidden_size), dtype=np.float32)
    intercepts_out = np.empty((num_layers, num_classes), dtype=np.float32)
    selected_out = np.empty((num_layers, num_classes), dtype=np.float32)
    l2_tensor = torch.tensor(normalized_l2, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    train_labels_device = train_labels.to(device=device, dtype=torch.long)
    validation_labels_numpy = validation_labels.cpu().numpy()
    class_indices = torch.arange(num_classes, device=device)
    binary_train_labels = (
        train_labels_device[:, None] == class_indices[None, :]
    ).to(torch.float32)
    positive_counts = binary_train_labels.sum(dim=0).clamp_min(1)
    negative_counts = len(train_labels) - positive_counts
    positive_weights = negative_counts / positive_counts

    for layer in range(num_layers):
        train_x = train_states[:, layer].to(device=device, dtype=torch.float32)
        validation_x = validation_states[:, layer].to(
            device=device,
            dtype=torch.float32,
        )
        mean = train_x.mean(dim=0)
        scale = train_x.std(dim=0).clamp_min(1e-6)
        train_x = (train_x - mean) / scale
        validation_x = (validation_x - mean) / scale
        means[layer] = mean.cpu().numpy()
        scales[layer] = scale.cpu().numpy()

        weights = torch.zeros(
            (len(normalized_l2), num_classes, hidden_size),
            device=device,
            requires_grad=True,
        )
        intercepts = torch.zeros(
            (len(normalized_l2), num_classes),
            device=device,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam((weights, intercepts), lr=learning_rate)
        for _ in range(steps):
            indices = torch.randint(
                len(train_x),
                (min(batch_size, len(train_x)),),
                generator=generator,
            ).to(device)
            batch_x = train_x[indices]
            batch_y = binary_train_labels[indices]
            logits = torch.einsum("bd,rkd->brk", batch_x, weights) + intercepts
            targets = batch_y[:, None, :].expand_as(logits)
            elementwise = functional.binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction="none",
            )
            balance = torch.where(
                targets > 0,
                positive_weights[None, None, :],
                1.0,
            )
            data_loss = (elementwise * balance).mean(dim=(0, 2))
            regularization = l2_tensor * weights.square().mean(dim=(1, 2))
            loss = (data_loss + regularization).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.inference_mode():
            validation_logits = (
                torch.einsum("bd,rkd->brk", validation_x, weights) + intercepts
            )
            validation_predictions = (validation_logits >= 0).cpu().numpy()
        weights_numpy = weights.detach().cpu().numpy()
        intercepts_numpy = intercepts.detach().cpu().numpy()
        for class_index in range(num_classes):
            target = validation_labels_numpy == class_index
            candidate_scores = [
                f1_score(
                    target,
                    validation_predictions[:, regularization_index, class_index],
                    zero_division=0,
                )
                for regularization_index in range(len(normalized_l2))
            ]
            best_index = int(np.argmax(candidate_scores))
            weights_out[layer, class_index] = weights_numpy[best_index, class_index]
            intercepts_out[layer, class_index] = intercepts_numpy[
                best_index,
                class_index,
            ]
            selected_out[layer, class_index] = normalized_l2[best_index]
        del train_x, validation_x, weights, intercepts, optimizer

    return ProbeBank(
        class_names=tuple(class_names),
        means=means,
        scales=scales,
        weights=weights_out,
        intercepts=intercepts_out,
        selected_regularization=selected_out,
    )
