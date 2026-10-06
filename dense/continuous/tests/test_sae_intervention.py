from __future__ import annotations

import torch

from prompt_optimization.sae_intervention import (
    js_from_logits,
    kl_from_logits,
    normalized_recovery,
)


def test_distribution_distances_are_zero_for_equal_logits() -> None:
    logits = torch.randn(5, 7, generator=torch.Generator().manual_seed(17))
    zeros = torch.zeros(5)
    assert torch.allclose(kl_from_logits(logits, logits), zeros, atol=1e-6)
    assert torch.allclose(js_from_logits(logits, logits), zeros, atol=1e-6)


def test_normalized_recovery_uses_baseline_denominator() -> None:
    baseline = torch.tensor([4.0, 2.0, 0.0])
    candidate = torch.tensor([1.0, 2.0, 1.0])
    recovery = normalized_recovery(baseline, candidate)

    assert torch.allclose(recovery[:2], torch.tensor([0.75, 0.0]))
    assert torch.isnan(recovery[2])
