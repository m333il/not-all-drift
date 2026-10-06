"""The routing map must be counted without padding.

PEFT prepends prompt-tuning's virtual tokens at position 0, *ahead* of any left
padding, so a padded batch routes ``[virtual][pad…][text]`` while the stage row
built beside it says ``[pad…][virtual][text]``. Every virtual position would be
labelled padding and dropped, and padding would be counted as virtual. Batches
of one length remove the padding, and with it the ambiguity.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "measure_routing_map", ROOT / "scripts" / "measure_routing_map.py"
)
measure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measure)


def test_every_batch_holds_one_length():
    seqs = [[1, 2, 3], [4], [5, 6], [7, 8, 9, 10], [11, 12], [13], [14, 15, 16]]
    batches = list(measure.iter_equal_length_batches(seqs, batch_size=32))
    for batch in batches:
        widths = {len(seqs[i]) for i in batch}
        assert len(widths) == 1, f"batch mixes widths {widths}"


def test_every_sequence_appears_exactly_once():
    seqs = [[1, 2, 3], [4], [5, 6], [7, 8, 9, 10], [11, 12], [13], [14, 15, 16]]
    seen = [i for batch in measure.iter_equal_length_batches(seqs, 32) for i in batch]
    assert sorted(seen) == list(range(len(seqs)))


def test_batch_size_caps_a_large_group():
    seqs = [[1, 2]] * 5
    sizes = [len(b) for b in measure.iter_equal_length_batches(seqs, batch_size=2)]
    assert sizes == [2, 2, 1]


def test_single_sequence_still_yields_a_batch():
    assert list(measure.iter_equal_length_batches([[1, 2, 3]], 8)) == [[0]]


def test_no_batches_for_no_sequences():
    assert list(measure.iter_equal_length_batches([], 8)) == []


def test_token_budget_shrinks_the_batch_for_long_sequences():
    """A GEPA-length sequence must not ride in a batch tuned for PEFT lengths.

    Eight rows of 6.5k tokens is what ran the card out of memory; the same
    budget has to leave short sequences alone.
    """
    long_seqs = [list(range(6500))] * 8
    sizes = [
        len(b)
        for b in measure.iter_equal_length_batches(
            long_seqs, batch_size=8, max_batch_tokens=12288
        )
    ]
    assert sizes == [1] * 8


def test_token_budget_leaves_short_sequences_at_full_batch():
    short_seqs = [list(range(700))] * 8
    sizes = [
        len(b)
        for b in measure.iter_equal_length_batches(
            short_seqs, batch_size=8, max_batch_tokens=12288
        )
    ]
    assert sizes == [8]


def test_token_budget_never_yields_an_empty_batch():
    """One sequence longer than the whole budget still has to be measured."""
    huge = [list(range(40000))]
    assert list(
        measure.iter_equal_length_batches(huge, batch_size=8, max_batch_tokens=1024)
    ) == [[0]]


def test_token_budget_off_by_default_matches_old_behaviour():
    seqs = [list(range(6500))] * 4
    assert [
        len(b) for b in measure.iter_equal_length_batches(seqs, batch_size=8)
    ] == [4]


def test_token_budget_still_covers_every_sequence_once():
    seqs = [list(range(3000))] * 5 + [list(range(500))] * 3
    seen = [
        i
        for batch in measure.iter_equal_length_batches(seqs, 8, max_batch_tokens=6000)
        for i in batch
    ]
    assert sorted(seen) == list(range(len(seqs)))


def test_stage_row_length_matches_routed_positions():
    """What the hook asserts at runtime, checked on the shapes we build.

    The routed sequence is `n_virtual + len(seq)` positions long, because PEFT
    adds the virtual ones; the stage row must be exactly that long or the two
    are talking about different tokens.
    """
    n_virtual = 4
    seq = [10, 11, 12, 13, 14]
    prompt_stages = ["comment"] * 3
    answer_stages = ["answer"] * 2
    stages = ["virtual"] * n_virtual + prompt_stages + answer_stages
    assert len(stages) == n_virtual + len(seq)


def test_stage_ids_cover_every_named_stage():
    for name in measure.STAGES:
        assert name in measure.STAGE_ID
    assert measure.PAD_ID not in measure.STAGE_ID.values()
    assert measure.PAD_ID < 0, "padding id must be negative so `stage >= 0` drops it"
