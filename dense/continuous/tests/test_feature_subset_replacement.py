from __future__ import annotations

import torch

from prompt_optimization.feature_subset_replacement import (
    build_pair_feature_sets,
    deterministic_topk,
    masked_feature_delta,
    norm_match_rows,
)


def test_deterministic_topk_breaks_ties_by_feature_id() -> None:
    assert deterministic_topk(torch.tensor([1.0, 3.0, 3.0, 2.0]), 3) == [1, 2, 3]


def test_pair_feature_sets_separate_aligned_and_unique_support() -> None:
    sets = build_pair_feature_sets(
        {
            "prefix": torch.tensor([4.0, 0.0, -3.0, 0.0, 2.5, 0.0]),
            "prompt": torch.tensor([5.0, 4.0, -2.0, 0.0, 0.0, 0.0]),
        },
        torch.ones(6),
        top_ks=(3,),
        sign_epsilon=1e-8,
        random_seed=7,
        random_k=1,
    )["prefix__prompt"]

    assert sets["aligned_shared_top3"] == [0, 2]
    assert sets["prefix_unique_top3"] == [4]
    assert sets["prompt_unique_top3"] == [1]
    assert len(sets["direct_top3"]) == 3


def test_masked_feature_delta_omits_decoder_bias() -> None:
    delta = torch.tensor([[2.0, 3.0, 4.0]])
    decoder = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])

    assert torch.equal(masked_feature_delta(delta, decoder, [1]), torch.tensor([[0.0, 6.0]]))
    assert torch.equal(masked_feature_delta(delta, decoder, []), torch.zeros(1, 2))


def test_norm_match_rows_matches_nonzero_reference_norms() -> None:
    values = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    reference = torch.tensor([[0.0, 10.0], [1.0, 0.0]])
    matched = norm_match_rows(values, reference)

    assert torch.allclose(matched[0].norm(), reference[0].norm())
    assert torch.equal(matched[1], torch.zeros(2))
