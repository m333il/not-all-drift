from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import f1_score, roc_auc_score

from .metrics import empty_aware_sample_f1_rows


@dataclass(frozen=True)
class LayerProbeResult:
    layer: int
    c: float
    weights: np.ndarray
    intercepts: np.ndarray
    thresholds: np.ndarray
    validation_f1: float
    validation_aurocs: tuple[float | None, ...]


def _validation_f1(targets: np.ndarray, predictions: np.ndarray) -> float:
    if targets.shape != predictions.shape or targets.ndim != 2:
        raise ValueError("probe predictions must match [examples, labels] targets")
    if targets.shape[1] == 1:
        return float(f1_score(targets[:, 0], predictions[:, 0], zero_division=0))
    # Empty-aware samples-F1.
    return float(empty_aware_sample_f1_rows(targets, predictions).mean())


def _fit_layer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    c_values: Sequence[float],
    default_threshold: float,
) -> LayerProbeResult:
    best: LayerProbeResult | None = None
    for c in c_values:
        weights: list[np.ndarray] = []
        intercepts: list[float] = []
        probabilities: list[np.ndarray] = []
        aurocs: list[float | None] = []
        for label in range(y_train.shape[1]):
            if len(np.unique(y_train[:, label])) < 2:
                weights.append(np.zeros(x_train.shape[1]))
                intercepts.append(-100.0)
                probabilities.append(np.zeros(len(x_val)))
                aurocs.append(None)
                continue
            model = LogisticRegression(C=c, max_iter=1000, random_state=0)
            model.fit(x_train, y_train[:, label])
            prob = model.predict_proba(x_val)[:, 1]
            weights.append(model.coef_[0])
            intercepts.append(model.intercept_[0])
            probabilities.append(prob)
            aurocs.append(
                None
                if len(np.unique(y_val[:, label])) < 2
                else float(roc_auc_score(y_val[:, label], prob))
            )
        probability_matrix = np.stack(probabilities, axis=1)
        thresholds = np.full(y_train.shape[1], default_threshold)
        score = _validation_f1(y_val, probability_matrix >= thresholds)
        result = LayerProbeResult(
            -1, c, np.stack(weights), np.asarray(intercepts), thresholds, score, tuple(aurocs)
        )
        if best is None or result.validation_f1 > best.validation_f1:
            best = result
    assert best is not None
    return best


def train_layerwise_probes(
    train_activations: np.ndarray,
    train_targets: np.ndarray,
    val_activations: np.ndarray,
    val_targets: np.ndarray,
    *,
    c_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    threshold: float = 0.5,
) -> tuple[LayerProbeResult, ...]:
    if train_activations.ndim != 3 or val_activations.shape[1:] != train_activations.shape[1:]:
        raise ValueError("activations must align as [examples,layers,hidden]")
    results = []
    for layer in range(train_activations.shape[1]):
        fitted = _fit_layer(
            train_activations[:, layer],
            train_targets,
            val_activations[:, layer],
            val_targets,
            c_values,
            threshold,
        )
        results.append(
            LayerProbeResult(
                layer,
                fitted.c,
                fitted.weights,
                fitted.intercepts,
                fitted.thresholds,
                fitted.validation_f1,
                fitted.validation_aurocs,
            )
        )
    return tuple(results)


def predict_probe_probabilities(
    result: LayerProbeResult, activations: np.ndarray
) -> np.ndarray:
    logits = activations @ result.weights.T + result.intercepts
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))


def predict_probe(result: LayerProbeResult, activations: np.ndarray) -> np.ndarray:
    return predict_probe_probabilities(result, activations) >= result.thresholds


def optimize_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    grid: Sequence[float] = tuple(np.linspace(0.1, 0.9, 17)),
) -> np.ndarray:
    """Select each label threshold on probe-val only for robustness analyses."""
    thresholds = np.full(targets.shape[1], 0.5)
    for label in range(targets.shape[1]):
        best_score = -1.0
        for threshold in grid:
            score = f1_score(
                targets[:, label], probabilities[:, label] >= threshold, zero_division=0
            )
            if score > best_score:
                best_score, thresholds[label] = float(score), threshold
    return thresholds


def difference_of_means(x: np.ndarray, targets: np.ndarray) -> np.ndarray:
    directions = []
    for column in targets.T:
        if not np.any(column) or np.all(column):
            directions.append(np.zeros(x.shape[1]))
        else:
            directions.append(x[column.astype(bool)].mean(0) - x[~column.astype(bool)].mean(0))
    return np.stack(directions)


def residualize_covariates(
    train_x: np.ndarray,
    eval_x: np.ndarray,
    train_covariates: np.ndarray,
    eval_covariates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit nuisance regression on train only and apply it cross-conditionally."""
    train_design = np.column_stack([np.ones(len(train_covariates)), train_covariates])
    eval_design = np.column_stack([np.ones(len(eval_covariates)), eval_covariates])
    coefficients = np.linalg.lstsq(train_design, train_x, rcond=None)[0]
    return train_x - train_design @ coefficients, eval_x - eval_design @ coefficients


@dataclass(frozen=True)
class CountProbeResult:
    direction: np.ndarray
    intercept: float
    validation_r2: float


def train_count_probe(
    x_train: np.ndarray,
    predicted_counts_train: np.ndarray,
    x_val: np.ndarray,
    predicted_counts_val: np.ndarray,
) -> CountProbeResult:
    model = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0))
    model.fit(x_train, predicted_counts_train)
    return CountProbeResult(
        np.asarray(model.coef_),
        float(model.intercept_),
        float(model.score(x_val, predicted_counts_val)),
    )


@dataclass(frozen=True)
class ProbeSelectivityResult:
    label_f1: float
    control_f1_mean: float
    selectivity: float
    permutation_p: float
    control_scores: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ProbeTrainSizePoint:
    train_size: int
    validation_f1: float
    selected_c: float


def probe_train_size_curve(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    train_sizes: Sequence[int] = (250, 500, 1000, 2000, 4000),
    c_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    threshold: float = 0.5,
    seed: int = 0,
) -> tuple[ProbeTrainSizePoint, ...]:
    """Fit a nested train-size curve from one preregistered permutation."""
    if x_train.ndim != 2 or y_train.ndim != 2 or len(x_train) != len(y_train):
        raise ValueError("probe training arrays must align on the example axis")
    sizes = tuple(train_sizes)
    if not sizes or any(size <= 0 or size > len(x_train) for size in sizes):
        raise ValueError("train sizes must be positive and fit the training pool")
    if tuple(sorted(set(sizes))) != sizes:
        raise ValueError("train sizes must be unique and increasing")
    order = np.random.default_rng(seed).permutation(len(x_train))
    points: list[ProbeTrainSizePoint] = []
    for size in sizes:
        selected = order[:size]
        fitted = _fit_layer(
            x_train[selected],
            y_train[selected],
            x_val,
            y_val,
            c_values,
            threshold,
        )
        points.append(ProbeTrainSizePoint(size, fitted.validation_f1, fitted.c))
    return tuple(points)


def evaluate_probe_selectivity(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    c_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    threshold: float = 0.5,
    permutations: int = 99,
    seed: int = 0,
) -> ProbeSelectivityResult:
    if permutations < 1:
        raise ValueError("permutations must be positive")
    actual = _fit_layer(x_train, y_train, x_val, y_val, c_values, threshold)
    rng = np.random.default_rng(seed)
    controls: list[float] = []
    for _ in range(permutations):
        shuffled = y_train[rng.permutation(len(y_train))]
        fitted = _fit_layer(x_train, shuffled, x_val, y_val, c_values, threshold)
        controls.append(fitted.validation_f1)
    control_array = np.asarray(controls)
    p_value = (1 + int(np.sum(control_array >= actual.validation_f1))) / (permutations + 1)
    control_mean = float(control_array.mean())
    return ProbeSelectivityResult(
        actual.validation_f1,
        control_mean,
        actual.validation_f1 - control_mean,
        p_value,
        tuple(float(value) for value in controls),
    )


def run_probe_protocol(
    train_activations: np.ndarray,
    train_targets: np.ndarray,
    val_activations: np.ndarray,
    val_targets: np.ndarray,
    *,
    controls: dict[str, tuple[np.ndarray, np.ndarray]],
    subset_indices: Mapping[int, np.ndarray],
    subset_hashes: Mapping[int, str] | None,
    train_sizes: Sequence[int],
    c_values: Sequence[float],
    threshold: float,
    permutations: int,
) -> list[dict[str, object]]:
    """Run E1/E2 using the externally frozen probe-training subsets."""
    required = {"length", "label_count"}
    if missing := required - set(controls):
        raise ValueError(f"missing matched probe controls: {sorted(missing)}")
    for name in ("length", "label_count"):
        train_control, val_control = controls[name]
        if train_control.ndim != 1 or val_control.ndim != 1:
            raise ValueError(f"{name} control must be a scalar vector")
    condition_train: np.ndarray | None = None
    condition_val: np.ndarray | None = None
    if "condition" in controls:
        condition_train, condition_val = controls["condition"]
        if condition_train.ndim == 1:
            condition_train = condition_train[:, None]
        if condition_val.ndim == 1:
            condition_val = condition_val[:, None]
        if condition_train.ndim != 2 or condition_val.ndim != 2:
            raise ValueError("condition control must be binary labels")
    if not subset_indices:
        raise ValueError("at least one frozen probe subset is required")
    rows: list[dict[str, object]] = []
    for seed, raw_indices in sorted(subset_indices.items()):
        selected = np.asarray(raw_indices, dtype=int)
        if selected.ndim != 1 or not len(selected):
            raise ValueError(f"probe subset {seed} must be a non-empty index vector")
        if len(np.unique(selected)) != len(selected):
            raise ValueError(f"probe subset {seed} contains duplicate indices")
        if np.any(selected < 0) or np.any(selected >= len(train_activations)):
            raise ValueError(f"probe subset {seed} contains out-of-range indices")
        if any(size > len(selected) for size in train_sizes):
            raise ValueError(f"train sizes do not fit frozen probe subset {seed}")
        for layer in range(train_activations.shape[1]):
            actual = evaluate_probe_selectivity(
                train_activations[selected, layer],
                train_targets[selected],
                val_activations[:, layer],
                val_targets,
                c_values=c_values,
                threshold=threshold,
                permutations=permutations,
                seed=seed,
            )
            control_scores: dict[str, dict[str, float | str]] = {}
            if condition_train is not None and condition_val is not None:
                condition_fitted = _fit_layer(
                    train_activations[selected, layer],
                    condition_train[selected],
                    val_activations[:, layer],
                    condition_val,
                    c_values,
                    threshold,
                )
                control_scores["condition"] = {
                    "metric": "f1_binary",
                    "value": condition_fitted.validation_f1,
                }
            for name in ("length", "label_count"):
                control_train, control_val = controls[name]
                regression = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0))
                regression.fit(train_activations[selected, layer], control_train[selected])
                control_scores[name] = {
                    "metric": "r2",
                    "value": float(regression.score(val_activations[:, layer], control_val)),
                }
            curve = probe_train_size_curve(
                train_activations[selected, layer],
                train_targets[selected],
                val_activations[:, layer],
                val_targets,
                train_sizes=train_sizes,
                c_values=c_values,
                threshold=threshold,
                seed=seed,
            )
            rows.append(
                {
                    "seed": seed,
                    "subset_size": len(selected),
                    "subset_hash": None if subset_hashes is None else subset_hashes.get(seed),
                    "layer": layer,
                    "label_f1": actual.label_f1,
                    "permutation_control_f1": actual.control_f1_mean,
                    "permutation_p": actual.permutation_p,
                    "selectivity": actual.selectivity,
                    "matched_controls": control_scores,
                    "condition_control_status": (
                        "computed" if condition_train is not None else "omitted"
                    ),
                    "train_size_curve": [
                        {
                            "train_size": point.train_size,
                            "validation_f1": point.validation_f1,
                            "selected_c": point.selected_c,
                        }
                        for point in curve
                    ],
                }
            )
    return rows
