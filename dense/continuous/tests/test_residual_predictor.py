from __future__ import annotations

import torch

from prompt_optimization.residual_predictor import (
    BiasOnlyResidualPredictor,
    LinearResidualPredictor,
    LowRankResidualPredictor,
    MLPResidualPredictor,
    forward_kl,
)
from prompt_optimization.trajectory_objectives import (
    apply_packed_predictor,
    mean_shift_statistics,
    normalized_objective,
)


def test_bias_only_predictor_is_input_independent_and_trainable() -> None:
    mean_shift = torch.tensor([1.0, -2.0, 0.5])
    predictor = BiasOnlyResidualPredictor(mean_shift)
    states = torch.randn(4, 3, generator=torch.Generator().manual_seed(5))

    assert torch.allclose(predictor(states), mean_shift.expand_as(states))
    assert torch.allclose(predictor(states + 100.0), predictor(states))
    predictor(states).square().mean().backward()
    assert predictor.bias.grad is not None
    assert sum(parameter.numel() for parameter in predictor.parameters()) == 3


def test_bias_only_predictor_rejects_wrong_hidden_size() -> None:
    predictor = BiasOnlyResidualPredictor(torch.zeros(3))
    try:
        predictor(torch.zeros(2, 4))
    except ValueError:
        return
    raise AssertionError("wrong hidden size was accepted")


def test_linear_predictor_starts_as_mean_shift() -> None:
    mean_shift = torch.tensor([1.0, -2.0, 0.5])
    predictor = LinearResidualPredictor(mean_shift)
    states = torch.randn(4, 3, generator=torch.Generator().manual_seed(7))

    assert torch.allclose(predictor.predict(states), mean_shift.expand_as(states))
    assert torch.allclose(predictor.state_prediction(states), states + mean_shift)


def test_mlp_predictor_starts_as_mean_shift() -> None:
    mean_shift = torch.tensor([0.25, -0.5, 1.0])
    predictor = MLPResidualPredictor(
        dimension=3,
        width=5,
        input_mean=torch.zeros(3),
        input_scale=torch.ones(()),
        mean_shift=mean_shift,
    )
    states = torch.randn(4, 3, generator=torch.Generator().manual_seed(11))

    assert torch.allclose(predictor.predict(states), mean_shift.expand_as(states))
    assert torch.allclose(predictor(states), mean_shift.expand_as(states))
    assert torch.allclose(predictor.state_prediction(states), states + mean_shift)


def test_forward_kl_is_zero_for_equal_logits() -> None:
    logits = torch.randn(6, 9, generator=torch.Generator().manual_seed(13))
    assert torch.allclose(forward_kl(logits, logits), torch.zeros(6), atol=1e-6)


def test_low_rank_predictor_starts_as_mean_shift_and_has_gradient_path() -> None:
    mean_shift = torch.tensor([1.0, -2.0, 0.5])
    predictor = LowRankResidualPredictor(mean_shift, rank=1)
    states = torch.randn(4, 3, generator=torch.Generator().manual_seed(19))

    assert torch.allclose(predictor(states), mean_shift.expand_as(states))
    predictor(states).square().mean().backward()
    assert predictor.up.weight.grad is not None
    assert torch.count_nonzero(predictor.up.weight.grad) > 0
    assert sum(parameter.numel() for parameter in predictor.parameters()) == 9
    assert predictor.effective_weight().shape == (3, 3)


def test_low_rank_predictor_rejects_invalid_rank() -> None:
    with torch.no_grad():
        for rank in (0, 4):
            try:
                LowRankResidualPredictor(torch.zeros(3), rank=rank)
            except ValueError:
                continue
            raise AssertionError("invalid rank was accepted")


class _IdentityDelta(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


def test_packed_predictor_modes_match_generation_contract() -> None:
    baseline = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    offsets = torch.tensor([0, 2, 5])
    predictor = _IdentityDelta()

    once = apply_packed_predictor(predictor, baseline, offsets, "prefill_once")
    fixed = apply_packed_predictor(predictor, baseline, offsets, "fixed_recurrent")
    recurrent = apply_packed_predictor(
        predictor, baseline, offsets, "repredict_recurrent"
    )

    assert once.squeeze().tolist() == [2.0, 2.0, 6.0, 4.0, 5.0]
    assert fixed.squeeze().tolist() == [2.0, 3.0, 6.0, 7.0, 8.0]
    assert recurrent.squeeze().tolist() == [2.0, 4.0, 6.0, 8.0, 10.0]


def test_mean_shift_statistics_use_anchor_inputs_for_fixed_modes() -> None:
    baseline = torch.tensor([[1.0], [100.0], [3.0], [200.0]])
    adapted = baseline + torch.tensor([[2.0], [8.0], [4.0], [16.0]])
    offsets = torch.tensor([0, 2, 4])

    mean_shift, input_mean, _ = mean_shift_statistics(
        baseline, adapted, offsets, "fixed_recurrent"
    )
    assert mean_shift.item() == 3.0
    assert input_mean.item() == 2.0


def test_normalized_objective_uses_requested_components() -> None:
    parts = {
        "dense": torch.tensor(2.0),
        "enc": torch.tensor(6.0),
        "dec": torch.tensor(8.0),
        "kl": torch.tensor(3.0),
    }
    scales = {"dense": 2.0, "enc": 3.0, "dec": 4.0, "kl": 1.5}
    assert normalized_objective(parts, scales, "dense").item() == 1.0
    assert normalized_objective(parts, scales, "dense_plus_kl").item() == 3.0
    assert normalized_objective(parts, scales, "dense_sae_kl").item() == 7.0
