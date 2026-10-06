from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from interpretability_gepa.activations import (
    ActivationBatch,
    forward_with_residuals,
    induced_shift,
    induced_shift_batches,
    load_activation_store,
    logical_last_positions,
    save_activation_store,
    verify_final_logits,
)
from interpretability_gepa.causal import (
    InterventionCell,
    apply_leace,
    decompose_direction,
    fit_leace,
    mean_direction,
    mismatched_pair_indices,
    norm_matched_random,
    patch_values,
    relative_alpha,
    select_intervention_cell,
)
from interpretability_gepa.geometry import (
    cosine,
    cosine_matrix,
    linear_cka,
    orthonormal_basis,
    procrustes_distance,
    projection_energy,
    projection_null_test,
    rbf_cka,
    solution_consensus_correlation,
)
from interpretability_gepa.logit_lens import token_sequence_logprobs
from interpretability_gepa.probes import (
    evaluate_probe_selectivity,
    predict_probe,
    probe_train_size_curve,
    train_count_probe,
    train_layerwise_probes,
)
from interpretability_gepa.statistics import grouped_bootstrap_difference, holm_adjust, mcnemar_test


def test_activation_store_and_shift(tmp_path: Path) -> None:
    seed = np.zeros((4, 3, 5), dtype=np.float16)
    target = seed + 2
    batch = ActivationBatch(
        tuple(f"x{i}" for i in range(4)), target, target + 1, {"condition": "a"}
    )
    save_activation_store(tmp_path / "store", batch)
    restored = load_activation_store(tmp_path / "store")
    assert np.array_equal(restored.last_prompt, target)
    per_example, average = induced_shift(target, seed)
    assert per_example.shape == target.shape
    assert np.all(average == 2)
    assert np.array_equal(logical_last_positions(np.array([[1, 1, 0], [1, 1, 1]])), [1, 2])

    reordered = ActivationBatch(tuple(reversed(batch.example_ids)), target, target, {})
    with pytest.raises(ValueError, match="example ordering"):
        induced_shift_batches(batch, reordered)


def test_forward_with_residuals_captures_embedding_and_post_blocks() -> None:
    class Handle:
        def __init__(self, hooks: list[object], hook: object) -> None:
            self.hooks, self.hook = hooks, hook

        def remove(self) -> None:
            self.hooks.remove(self.hook)

    class Module:
        def __init__(self) -> None:
            self.hooks: list[object] = []
            self.pre_hooks: list[object] = []

        def register_forward_hook(self, hook: object) -> Handle:
            self.hooks.append(hook)
            return Handle(self.hooks, hook)

        def register_forward_pre_hook(self, hook: object) -> Handle:
            self.pre_hooks.append(hook)
            return Handle(self.pre_hooks, hook)

        def emit(self, inputs: object, output: object) -> object:
            for hook in self.hooks:
                hook(self, inputs, output)  # type: ignore[operator]
            return output

    class Embedding(Module):
        def __call__(self, input_ids: np.ndarray) -> np.ndarray:
            output = np.repeat(input_ids[..., None], 3, axis=-1).astype(float)
            return self.emit((input_ids,), output)  # type: ignore[return-value]

    class Block(Module):
        def __call__(self, hidden: np.ndarray) -> tuple[np.ndarray]:
            for hook in self.pre_hooks:
                hook(self, (hidden,))  # type: ignore[operator]
            return self.emit((hidden,), (hidden + 1,))  # type: ignore[return-value]

    class Decoder:
        def __init__(self) -> None:
            self.embed_tokens = Embedding()
            self.layers = [Block(), Block()]

    class CausalModel:
        def __init__(self) -> None:
            self.model = Decoder()

        def get_input_embeddings(self) -> Embedding:
            return self.model.embed_tokens

        def __call__(self, input_ids: np.ndarray) -> object:
            hidden = self.model.embed_tokens(input_ids)
            for layer in self.model.layers:
                hidden = layer(hidden)[0]
            return type("Output", (), {"logits": hidden.sum(axis=-1)})()

    model = CausalModel()
    output, residuals = forward_with_residuals(model, input_ids=np.array([[0, 1]]))

    assert output.logits.shape == (1, 2)
    assert len(residuals) == 3
    assert np.allclose(residuals[1], residuals[0] + 1)
    assert np.allclose(residuals[2], residuals[0] + 2)


def test_verify_final_logits_applies_gemma2_softcap() -> None:
    torch = pytest.importorskip("torch")

    class Base(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = torch.nn.Identity()

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Base()
            self.lm_head = torch.nn.Linear(2, 3, bias=False)
            self.config = type("Config", (), {"final_logit_softcapping": 30.0})()

    model = Model()
    hidden = torch.tensor([[[1.0, -2.0]]])
    projected = model.lm_head(hidden)
    expected = torch.tanh(projected / 30.0) * 30.0

    verify_final_logits(model, hidden, expected)
    with pytest.raises(AssertionError, match="does not reproduce"):
        verify_final_logits(model, hidden, expected + 1.0)


def test_geometry_invariants() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 6))
    rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    y = x @ rotation
    assert linear_cka(x, y) == pytest.approx(1.0)
    assert rbf_cka(x, x) == pytest.approx(1.0)
    assert procrustes_distance(x, y) < 1e-10
    assert cosine(x[0], x[0]) == pytest.approx(1.0)
    matrix = cosine_matrix(x[:3])
    assert np.allclose(np.diag(matrix), 1)


def test_projection_and_leace_remove_planted_concept() -> None:
    rng = np.random.default_rng(4)
    concept = rng.integers(0, 2, size=(200, 1)).astype(float)
    direction = np.array([1.0, 0, 0, 0])
    x = rng.normal(scale=0.1, size=(200, 4)) + concept @ direction[None]
    eraser = fit_leace(x, concept)
    removed = apply_leace(x, eraser)
    before = abs(np.corrcoef(x[:, 0], concept[:, 0])[0, 1])
    after = np.max(np.abs((removed - removed.mean(0)).T @ (concept - concept.mean(0))))
    assert before > 0.8
    assert after < 1e-8
    parallel, perpendicular = decompose_direction(np.ones(4), direction[None])
    assert np.allclose(parallel + perpendicular, 1)
    assert projection_energy(np.ones(4), orthonormal_basis(direction[None])) == pytest.approx(0.25)


def test_directions_and_controls() -> None:
    target = np.ones((20, 5))
    seed = np.zeros_like(target)
    direction = mean_direction(target, seed)
    controls = norm_matched_random(direction, 5, seed=2)
    assert np.allclose(np.linalg.norm(controls, axis=1), np.linalg.norm(direction))


def test_controls_survive_a_direction_left_in_float16() -> None:
    """A float16 direction whose squares leave the float16 range must not size to inf.

    Activation stores are float16 and the deep-layer shifts have norms near 280, whose
    squares sum past the 65504 the dtype can hold. A caller that forgets to widen gets
    inf-scaled controls that score zero and look like a catastrophic intervention.
    """
    direction = np.full(2304, 6.0, dtype=np.float16)
    assert np.isinf(np.linalg.norm(direction))
    controls = norm_matched_random(direction.astype(np.float32), 3, seed=0)
    assert np.isfinite(controls).all()


def test_layerwise_probe_recovers_signal() -> None:
    rng = np.random.default_rng(5)
    first_train = rng.integers(0, 2, size=200)
    first_val = rng.integers(0, 2, size=80)
    y_train = np.stack([first_train, 1 - first_train], axis=1)
    y_val = np.stack([first_val, 1 - first_val], axis=1)
    x_train = rng.normal(scale=0.2, size=(200, 3, 5))
    x_val = rng.normal(scale=0.2, size=(80, 3, 5))
    x_train[:, 1, :2] += y_train * 3
    x_val[:, 1, :2] += y_val * 3
    results = train_layerwise_probes(x_train, y_train, x_val, y_val, c_values=(1,))
    assert results[1].validation_f1 > 0.9
    assert predict_probe(results[1], x_val[:, 1]).shape == y_val.shape


def test_count_probe_recovers_reporting_axis() -> None:
    rng = np.random.default_rng(8)
    train_counts = rng.integers(0, 4, size=100)
    val_counts = rng.integers(0, 4, size=40)
    x_train = rng.normal(scale=0.05, size=(100, 4))
    x_train[:, 0] += train_counts
    x_val = rng.normal(scale=0.05, size=(40, 4))
    x_val[:, 0] += val_counts
    result = train_count_probe(x_train, train_counts, x_val, val_counts)
    assert result.validation_r2 > 0.95
    assert abs(result.direction[0]) > abs(result.direction[1:]).max()


def test_grouped_bootstrap_resamples_whole_pmids() -> None:
    first = np.array([1.0, 1.0, 0.0, 0.0])
    second = np.zeros(4)
    groups = np.array(["a", "a", "b", "b"])

    interval = grouped_bootstrap_difference(first, second, groups, samples=200, seed=7)

    assert interval.estimate == pytest.approx(0.5)
    assert interval.low == pytest.approx(0.0)
    assert interval.high == pytest.approx(1.0)


def test_paired_tests_and_holm_correction() -> None:
    result = mcnemar_test(
        np.array([True, True, False, False]),
        np.array([True, False, True, True]),
    )
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.2]))

    assert result.discordant_first == 1
    assert result.discordant_second == 2
    assert np.all(adjusted >= np.array([0.01, 0.04, 0.2]))
    assert np.all(adjusted <= 1)


def test_probe_selectivity_rejects_shuffled_label_controls() -> None:
    rng = np.random.default_rng(21)
    train_targets = rng.integers(0, 2, size=(240, 2))
    val_targets = rng.integers(0, 2, size=(100, 2))
    train = rng.normal(scale=0.2, size=(240, 6))
    val = rng.normal(scale=0.2, size=(100, 6))
    train[:, :2] += train_targets * 2
    val[:, :2] += val_targets * 2

    result = evaluate_probe_selectivity(
        train,
        train_targets,
        val,
        val_targets,
        c_values=(1.0,),
        permutations=19,
        seed=5,
    )

    assert result.label_f1 > 0.7
    assert result.selectivity > 0.4
    assert result.permutation_p <= 0.05


def test_probe_train_size_curve_is_deterministic_and_nested() -> None:
    rng = np.random.default_rng(14)
    x = rng.normal(size=(40, 5))
    y = (x[:, :2] > 0).astype(int)

    first = probe_train_size_curve(x, y, x, y, train_sizes=(10, 20, 40), seed=7)
    second = probe_train_size_curve(x, y, x, y, train_sizes=(10, 20, 40), seed=7)

    assert first == second
    assert [point.train_size for point in first] == [10, 20, 40]
    assert all(0 <= point.validation_f1 <= 1 for point in first)


def test_full_string_logit_lens_sums_all_label_tokens() -> None:
    logits = np.array(
        [
            [[3.0, 0.0, -1.0], [0.0, 2.0, -1.0]],
            [[0.0, 3.0, -1.0], [2.0, 0.0, -1.0]],
        ]
    )
    tokens = np.array([[0, 1], [1, 0]])
    mask = np.array([[True, True], [True, False]])

    scores = token_sequence_logprobs(logits, tokens, mask)

    assert scores.shape == (2,)
    assert scores[0] < scores[1]


def test_intervention_grid_selection_applies_degradation_gates() -> None:
    vector = np.array([3.0, 4.0])
    states = np.array([[6.0, 8.0], [6.0, 8.0]])
    assert relative_alpha(vector, states, fraction=0.1) == pytest.approx(0.2)

    cells = (
        InterventionCell(5, 1.0, 0.4, 0.02, 1.0, "valid"),
        InterventionCell(6, 2.0, 0.8, 0.2, 1.0, "parse-broken"),
        InterventionCell(7, 4.0, 0.9, 0.01, 3.0, "count-broken"),
    )
    selected = select_intervention_cell(cells, parse_limit=0.1, baseline_avg_predictions=1.0)
    assert selected.id == "valid"


def test_paired_patch_controls_destroy_input_matching() -> None:
    labels = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    same = mismatched_pair_indices(labels, same_label=True, seed=0)
    different = mismatched_pair_indices(labels, same_label=False, seed=0)
    assert np.all(np.logical_and(labels, labels[same]).any(axis=1))
    assert not np.any(np.logical_and(labels, labels[different]).any(axis=1))

    seed = np.zeros((4, 3))
    target = np.arange(12).reshape(4, 3)
    assert np.array_equal(patch_values(seed, target, mode="state"), target)
    assert np.allclose(patch_values(seed, target, mode="mean")[0], target.mean(0))


def test_geometry_null_and_good_solution_consensus() -> None:
    basis = orthonormal_basis(np.array([[1.0, 0.0, 0.0, 0.0]]))
    observed, expected, p_value = projection_null_test(
        np.array([1.0, 0.0, 0.0, 0.0]), basis, draws=199, seed=2
    )
    assert observed == pytest.approx(1.0)
    assert expected == pytest.approx(0.25, abs=0.08)
    assert p_value <= 0.01

    scores = np.array([0.1, 0.2, 0.8, 0.9])
    vectors = np.array([[0.0, 1.0], [0.2, 1.0], [1.0, 0.2], [1.0, 0.0]])
    correlation, alignments = solution_consensus_correlation(scores, vectors, top_k=2)
    assert correlation > 0.79
    assert alignments[-1] > alignments[0]


def test_correct_abstention_is_a_perfect_match_not_a_zero() -> None:
    from interpretability_gepa.metrics import evaluate_multilabel

    labels = ("toxicity", "obscene", "threat", "insult")
    gold = [(), ("toxicity",), (), ("toxicity", "insult")]

    perfect = evaluate_multilabel(gold, gold, labels)
    assert perfect["f1_samples"] == 1.0
    # The sklearn convention caps a perfect classifier at the labelled fraction.
    assert perfect["f1_samples_legacy"] == 0.5

    # Per example, which is how the optimizer consumes it: abstaining correctly must
    # beat over-predicting, or there is no gradient on the empty half.
    assert evaluate_multilabel([()], [()], labels)["f1_samples"] == 1.0
    assert evaluate_multilabel([()], [("toxicity",)], labels)["f1_samples"] == 0.0


def test_headline_metric_splits_out_the_always_none_policy() -> None:
    from interpretability_gepa.metrics import evaluate_multilabel

    labels = ("toxicity", "obscene", "threat", "insult")
    gold = [(), ("toxicity",), (), ("toxicity", "insult")]

    always_none = evaluate_multilabel(gold, [() for _ in gold], labels)

    # Half-empty splits make blanket abstention worth 0.5, so the decomposition has
    # to make that visible rather than let it hide in the headline number.
    assert always_none["f1_samples"] == 0.5
    assert always_none["no_label_accuracy"] == 1.0
    assert always_none["f1_samples_positive"] == 0.0
    assert always_none["avg_predictions"] == 0.0
