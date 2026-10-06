"""Behavioral tests for vanilla and tuned logit lens primitives."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from scripts.lenses.evaluate_civil_comments_tuned_lens import load_and_validate_states
from prompt_optimization.tuned_lens import (
    ResidualTranslator,
    distillation_kl,
    evaluate_lens_layer,
    fit_tuned_lens_layer,
    load_translator,
    project_final_states,
    project_intermediate_states,
    save_translator,
)
from prompt_optimization.logit_lens import direct_token_probabilities


def test_zero_initialized_translator_equals_vanilla_lens() -> None:
    generator = torch.Generator().manual_seed(3)
    states = torch.randn(8, 5, generator=generator)
    norm = nn.LayerNorm(5)
    head = nn.Linear(5, 11, bias=False)
    translator = ResidualTranslator(5)

    direct = project_intermediate_states(states, final_norm=norm, lm_head=head)
    translated = project_intermediate_states(
        states,
        final_norm=norm,
        lm_head=head,
        translator=translator,
    )

    assert torch.equal(translator(states), states)
    assert torch.equal(direct, translated)


def test_final_projection_does_not_apply_norm_twice() -> None:
    final_states = torch.tensor([[2.0, 0.0]])
    norm = nn.LayerNorm(2, elementwise_affine=False)
    head = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(2))

    actual = project_final_states(final_states, head)
    incorrectly_renormalized = project_intermediate_states(
        final_states,
        final_norm=norm,
        lm_head=head,
    )

    assert torch.equal(actual, final_states)
    assert not torch.allclose(actual, incorrectly_renormalized)


def test_direct_logit_lens_does_not_renormalize_terminal_state() -> None:
    states = torch.tensor([[[1.0, 3.0], [2.0, 0.0]]])
    norm = nn.LayerNorm(2, elementwise_affine=False)
    head = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(2))

    probabilities = direct_token_probabilities(
        states,
        token_ids=[0, 1],
        final_norm=norm,
        lm_head=head,
        device=torch.device("cpu"),
        batch_size=1,
    )

    expected_terminal = states[:, -1].softmax(-1)
    torch.testing.assert_close(probabilities[:, -1], expected_terminal)


def test_tuned_lens_reduces_kl_for_synthetic_affine_drift() -> None:
    generator = torch.Generator().manual_seed(17)
    layer_states = torch.randn(180, 4, generator=generator)
    transform = torch.tensor(
        [
            [1.2, 0.4, 0.0, 0.0],
            [-0.3, 0.9, 0.2, 0.0],
            [0.0, 0.1, 1.1, 0.3],
            [0.2, 0.0, -0.2, 0.8],
        ]
    )
    bias = torch.tensor([0.3, -0.2, 0.1, 0.4])
    final_states = layer_states @ transform.T + bias
    norm = nn.Identity()
    head = nn.Linear(4, 9, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.randn(9, 4, generator=generator))

    fit = fit_tuned_lens_layer(
        layer_states[:120],
        final_states[:120],
        layer_states[120:150],
        final_states[120:150],
        final_norm=norm,
        lm_head=head,
        device=torch.device("cpu"),
        steps=250,
        batch_size=64,
        evaluation_batch_size=30,
        learning_rate=0.05,
        weight_decay=0.0,
        temperature=1.0,
        evaluation_interval=10,
        patience=8,
        seed=23,
    )
    direct_test = evaluate_lens_layer(
        layer_states[150:],
        final_states[150:],
        final_norm=norm,
        lm_head=head,
        device=torch.device("cpu"),
        translator=None,
        batch_size=30,
    )
    tuned_test = evaluate_lens_layer(
        layer_states[150:],
        final_states[150:],
        final_norm=norm,
        lm_head=head,
        device=torch.device("cpu"),
        translator=fit.translator,
        batch_size=30,
    )

    assert tuned_test.kl_to_final < direct_test.kl_to_final * 0.15
    assert tuned_test.top1_agreement > direct_test.top1_agreement
    assert fit.best_step > 0
    assert fit.history


def test_translator_safetensors_round_trip(tmp_path: Path) -> None:
    translator = ResidualTranslator(3)
    with torch.no_grad():
        translator.delta.weight.copy_(torch.arange(9).reshape(3, 3) / 10)
        translator.delta.bias.copy_(torch.tensor([0.1, 0.2, 0.3]))
    path = tmp_path / "layer_00.safetensors"

    save_translator(path, translator, metadata={"depth_point": "0"})
    restored = load_translator(path, hidden_size=3, device=torch.device("cpu"))

    states = torch.randn(4, 3, generator=torch.Generator().manual_seed(5))
    assert torch.equal(translator(states), restored(states))


def test_distillation_kl_is_zero_for_identical_logits() -> None:
    logits = torch.randn(5, 7, generator=torch.Generator().manual_seed(13))
    loss = distillation_kl(logits, logits, temperature=2.0)
    assert abs(float(loss)) < 1e-6


def test_adapted_activations_accept_generic_summary(tmp_path: Path) -> None:
    states = torch.randn(3, 4, 5)
    save_file({"states": states}, tmp_path / "adapted_test.safetensors")
    (tmp_path / "summary.json").write_text(
        '{"status":"done","condition":"adapted","splits":{"test":{"shape":[3,4,5]}}}',
        encoding="utf-8",
    )

    restored, summary = load_and_validate_states(
        tmp_path,
        condition="adapted",
        split="test",
    )

    assert torch.equal(restored, states)
    assert summary["condition"] == "adapted"
