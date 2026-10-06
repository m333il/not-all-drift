"""Layerwise one-vs-rest probes for true multilabel targets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.nn import functional

from prompt_optimization.probing import ProbeBank


def _validate(states: torch.Tensor, labels: torch.Tensor) -> None:
    if states.ndim != 3:
        raise ValueError("states must have shape [samples, layers, hidden_size]")
    if labels.ndim != 2 or len(labels) != len(states):
        raise ValueError("labels must have shape [samples, classes]")


def _samplewise_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Return multilabel sample F1, including its binary single-label limit.

    For one label, per-sample F1 with ``zero_division=1`` is exactly correctness:
    both the positive-positive and negative-negative cases score one, while a
    mismatch scores zero. Scikit-learn classifies ``[N, 1]`` targets as binary
    rather than multilabel and rejects ``average=\"samples\"``, so compute that
    equivalent limit directly.
    """
    if labels.shape[1] == 1:
        return float(np.mean(labels[:, 0] == predictions[:, 0]))
    return float(f1_score(labels, predictions, average="samples", zero_division=1))


def fit_torch_multilabel_probe_bank(
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
    """Fit one balanced binary logistic probe per class and layer."""
    _validate(train_states, train_labels)
    _validate(validation_states, validation_labels)
    if train_states.shape[1:] != validation_states.shape[1:]:
        raise ValueError("train and validation state shapes must agree")
    if train_labels.shape[1] != len(class_names) or validation_labels.shape[1] != len(class_names):
        raise ValueError("label width must match class_names")
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

    train_targets = train_labels.to(device=device, dtype=torch.float32)
    validation_targets = validation_labels.cpu().numpy().astype(np.int64)
    positive_counts = train_targets.sum(dim=0).clamp_min(1)
    negative_counts = len(train_targets) - positive_counts
    positive_weights = negative_counts / positive_counts

    for layer in range(num_layers):
        train_x = train_states[:, layer].to(device=device, dtype=torch.float32)
        validation_x = validation_states[:, layer].to(device=device, dtype=torch.float32)
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
            targets = train_targets[indices]
            logits = torch.einsum("bd,rkd->brk", batch_x, weights) + intercepts
            expanded_targets = targets[:, None, :].expand_as(logits)
            elementwise = functional.binary_cross_entropy_with_logits(
                logits,
                expanded_targets,
                reduction="none",
            )
            balance = torch.where(
                expanded_targets > 0,
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
            validation_logits = torch.einsum("bd,rkd->brk", validation_x, weights) + intercepts
            validation_predictions = (validation_logits >= 0).cpu().numpy()
        weights_numpy = weights.detach().cpu().numpy()
        intercepts_numpy = intercepts.detach().cpu().numpy()
        for class_index in range(num_classes):
            candidate_scores = [
                f1_score(
                    validation_targets[:, class_index],
                    validation_predictions[:, regularization_index, class_index],
                    zero_division=0,
                )
                for regularization_index in range(len(normalized_l2))
            ]
            best_index = int(np.argmax(candidate_scores))
            weights_out[layer, class_index] = weights_numpy[best_index, class_index]
            intercepts_out[layer, class_index] = intercepts_numpy[best_index, class_index]
            selected_out[layer, class_index] = normalized_l2[best_index]

    return ProbeBank(
        class_names=tuple(class_names),
        means=means,
        scales=scales,
        weights=weights_out,
        intercepts=intercepts_out,
        selected_regularization=selected_out,
    )


def score_multilabel_probe_bank(
    bank: ProbeBank,
    states: np.ndarray,
    labels: np.ndarray,
) -> dict[str, list[dict[str, object]]]:
    if states.ndim != 3:
        raise ValueError("states must have shape [samples, layers, hidden_size]")
    if labels.ndim != 2 or len(labels) != len(states):
        raise ValueError("labels must have shape [samples, classes]")
    if states.shape[1:] != bank.means.shape:
        raise ValueError("state shape does not match probe bank")
    if labels.shape[1] != len(bank.class_names):
        raise ValueError("label width does not match probe bank")

    layer_metrics: list[dict[str, object]] = []
    for layer in range(states.shape[1]):
        standardized = (states[:, layer] - bank.means[layer]) / bank.scales[layer]
        logits = standardized @ bank.weights[layer].T + bank.intercepts[layer]
        predictions = logits >= 0
        per_class = {
            class_name: float(
                f1_score(labels[:, class_index], predictions[:, class_index], zero_division=0)
            )
            for class_index, class_name in enumerate(bank.class_names)
        }
        layer_metrics.append(
            {
                "layer": layer,
                "macro_f1": float(np.mean(list(per_class.values()))),
                "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
                "samples_f1": _samplewise_f1(labels, predictions),
                "subset_accuracy": float(np.mean(np.all(labels == predictions, axis=1))),
                "per_class_f1": per_class,
            }
        )
    return {"layers": layer_metrics}


def score_multilabel_probe_bank_torch(
    bank: ProbeBank,
    states: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, list[dict[str, object]]]:
    """Score a probe bank with linear algebra on the requested Torch device."""
    _validate(states, labels)
    if tuple(states.shape[1:]) != tuple(bank.means.shape):
        raise ValueError("state shape does not match probe bank")
    if labels.shape[1] != len(bank.class_names):
        raise ValueError("label width does not match probe bank")

    targets = labels.cpu().numpy().astype(np.int64)
    means = torch.from_numpy(bank.means).to(device=device, dtype=torch.float32)
    scales = torch.from_numpy(bank.scales).to(device=device, dtype=torch.float32)
    weights = torch.from_numpy(bank.weights).to(device=device, dtype=torch.float32)
    intercepts = torch.from_numpy(bank.intercepts).to(device=device, dtype=torch.float32)
    result: list[dict[str, object]] = []
    with torch.inference_mode():
        for layer in range(states.shape[1]):
            x = states[:, layer].to(device=device, dtype=torch.float32)
            x = (x - means[layer]) / scales[layer]
            logits = x @ weights[layer].T + intercepts[layer]
            predictions = (logits >= 0).cpu().numpy()
            per_class = {
                name: float(f1_score(targets[:, i], predictions[:, i], zero_division=0))
                for i, name in enumerate(bank.class_names)
            }
            result.append(
                {
                    "layer": layer,
                    "macro_f1": float(np.mean(list(per_class.values()))),
                    "micro_f1": float(
                        f1_score(targets, predictions, average="micro", zero_division=0)
                    ),
                    "samples_f1": _samplewise_f1(targets, predictions),
                    "subset_accuracy": float(np.mean(np.all(targets == predictions, axis=1))),
                    "per_class_f1": per_class,
                }
            )
    return {"layers": result}
