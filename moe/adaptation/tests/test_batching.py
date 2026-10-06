"""Correctness of the batched training path.

Batching is where a silent bug is most likely and most costly: if the padding
mask or the -100 label padding is wrong, the model trains partly on pad tokens
and nothing visibly fails -- the loss just means something slightly different
and the adapter quietly learns the wrong thing. So the load-bearing test here
is numerical: a batched forward must produce the same loss as running the same
examples one at a time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mrd.peft_training import (  # noqa: E402
    _length_bucketed_batches,
    _pad_batch,
)

PAD = 63   # must be a valid id for the tiny model's vocab (64)


def test_pad_batch_shapes_and_padding_values():
    encoded = [
        ([1, 2, 3], [-100, 2, 3]),
        ([4, 5], [-100, 5]),
        ([6, 7, 8, 9], [-100, -100, 8, 9]),
    ]
    ids, mask, labels = _pad_batch(encoded, PAD, device="cpu")

    assert ids.shape == (3, 4)          # padded to the longest member
    assert mask.shape == (3, 4)
    assert labels.shape == (3, 4)

    # Row 1 (len 2) gets two pad columns.
    assert ids[1].tolist() == [4, 5, PAD, PAD]
    assert mask[1].tolist() == [1, 1, 0, 0]
    assert labels[1].tolist() == [-100, 5, -100, -100]

    # The longest row is untouched.
    assert ids[2].tolist() == [6, 7, 8, 9]
    assert mask[2].tolist() == [1, 1, 1, 1]


def test_padding_never_contributes_to_loss_or_attention():
    """Every padded position must be masked out *and* ignored by the loss."""
    encoded = [([1, 2, 3, 4], [-100, 2, 3, 4]), ([5], [-100])]
    ids, mask, labels = _pad_batch(encoded, PAD, device="cpu")

    pad_positions = mask == 0
    assert pad_positions.any(), "test is vacuous without real padding"
    # -100 is torch's ignore_index: these positions cannot enter the loss.
    assert (labels[pad_positions] == -100).all()
    # and the pad token id is only ever where the mask is 0
    assert (ids[pad_positions] == PAD).all()
    assert (ids[~pad_positions] != PAD).all()


def test_length_bucketing_groups_similar_lengths():
    encoded = [([0] * n, [0] * n) for n in (10, 1, 9, 2, 8, 3)]
    batches = _length_bucketed_batches(encoded, batch_size=2, rng=__import__("random").Random(0))

    assert sorted(i for b in batches for i in b) == list(range(6)), "every example used exactly once"
    # Within each batch the length spread must be small -- that is the whole
    # point (padding waste is proportional to the spread).
    # Batches are formed from length-adjacent examples, so the total padding
    # waste must beat what random grouping would give on the same data.
    waste = sum(
        max(len(encoded[i][0]) for i in b) * len(b) - sum(len(encoded[i][0]) for i in b)
        for b in batches
    )
    # Optimal grouping here is (1,2),(3,8),(9,10) -> waste 1 + 5 + 1 = 7.
    assert waste == 7


def test_bucketing_handles_ragged_last_batch():
    encoded = [([0] * n, [0] * n) for n in range(1, 6)]   # 5 examples, batch 2
    batches = _length_bucketed_batches(encoded, batch_size=2, rng=__import__("random").Random(0))
    assert sorted(len(b) for b in batches) == [1, 2, 2]
    assert sorted(i for b in batches for i in b) == list(range(5))


# ── the numerical test: batched == unbatched ────────────────────────────────

transformers = pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402


def _tiny_model():
    torch.manual_seed(0)
    m = GPT2LMHeadModel(GPT2Config(
        n_layer=2, n_head=2, n_embd=32, vocab_size=64, n_positions=64,
    ))
    m.eval()
    return m


def test_batched_loss_matches_token_weighted_unbatched_loss():
    """The real correctness check.

    HF reduces cross-entropy over all non-ignored tokens in the batch, so the
    batched loss equals the *token-weighted* mean of the per-example losses --
    not their plain mean. Reproducing that identity is what proves the padding
    is genuinely inert: any leak (pad attended to, or pad counted in the loss)
    shifts the batched number away from it.
    """
    model = _tiny_model()
    examples = [
        ([1, 2, 3, 4, 5, 6], [-100, -100, 3, 4, 5, 6]),   # 4 target tokens
        ([7, 8, 9], [-100, 8, 9]),                        # 2 target tokens
        ([10, 11, 12, 13, 14], [-100, -100, -100, 13, 14]),  # 2 target tokens
    ]

    per_example = []
    with torch.no_grad():
        for ids, labels in examples:
            t_ids = torch.tensor([ids])
            out = model(input_ids=t_ids, attention_mask=torch.ones_like(t_ids),
                        labels=torch.tensor([labels]), use_cache=False)
            n_tokens = sum(1 for x in labels if x != -100)
            per_example.append((float(out.loss), n_tokens))

    ids, mask, labels = _pad_batch(examples, PAD, device="cpu")
    with torch.no_grad():
        batched = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)

    expected = sum(loss * n for loss, n in per_example) / sum(n for _, n in per_example)
    assert float(batched.loss) == pytest.approx(expected, rel=1e-4), (
        "batched loss diverges from the token-weighted unbatched loss -- padding "
        "is leaking into attention or into the loss"
    )


def test_padding_amount_does_not_change_the_loss():
    """Same examples, different amounts of padding -> identical loss.

    Guards the failure mode the previous test could miss: a mask that is wrong
    in a way that happens to cancel out at one particular padding width.
    """
    model = _tiny_model()
    real = [([1, 2, 3, 4], [-100, 2, 3, 4]), ([5, 6], [-100, 6])]

    _, _, _ = _pad_batch(real, PAD, device="cpu")
    ids_a, mask_a, labels_a = _pad_batch(real, PAD, device="cpu")
    # Force a much wider pad by adding a long throwaway row, then drop that row
    # from the comparison.
    wide = real + [([7] * 20, [-100] * 20)]
    ids_b, mask_b, labels_b = _pad_batch(wide, PAD, device="cpu")

    with torch.no_grad():
        a = model(input_ids=ids_a, attention_mask=mask_a, labels=labels_a, use_cache=False)
        b = model(input_ids=ids_b[:2], attention_mask=mask_b[:2], labels=labels_b[:2],
                  use_cache=False)

    assert float(a.loss) == pytest.approx(float(b.loss), rel=1e-4), (
        "loss depends on how much padding was appended -- the mask is not inert"
    )
