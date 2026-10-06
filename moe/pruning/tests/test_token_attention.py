"""Per-token attention has to survive the things that quietly fake a sink.

The measurement is a means to an end: a super expert is supposed to announce
itself as an attention sink on the token carrying its massive activation. That
inference only holds if the column mass is real, so the failure modes worth
pinning are the ones that manufacture a convincing sink out of nothing:

  * padding that keeps its softmax mass - a pad column collecting 40% of the
    attention would be the most impressive sink in the run, and meaningless;
  * a peak that is merely peaked rather than *placed*, which the top-1 index
    distinguishes;
  * an L2 norm over the hidden width, which averages away the single runaway
    coordinate that is the whole definition of a massive activation.

The arithmetic is checked against hand-built matrices rather than a model, so a
failure names the defect instead of a number that drifted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from mrd_pruning.token_attention import (TokenAttention,  # noqa: E402
                                         causal_null_top1_share, describe_edges,
                                         massive_activations, position_mode,
                                         segment_mass, sink_metrics)


class _Attn:
    """Stands in for `self_attn`: returns (hidden, weights) like eager does."""

    def __init__(self, weights):
        self.weights = weights
        self._hooks = []

    def register_forward_hook(self, fn):
        self._hooks.append(fn)

        class H:
            def remove(_self):
                pass
        return H()

    def fire(self):
        hidden = torch.zeros(self.weights.shape[0], self.weights.shape[2], 4)
        for fn in self._hooks:
            fn(self, (), (hidden, self.weights))


class _Layer:
    def __init__(self, weights):
        self.self_attn = _Attn(weights)
        self._hooks = []

    def register_forward_hook(self, fn):
        self._hooks.append(fn)

        class H:
            def remove(_self):
                pass
        return H()


class _Model:
    def __init__(self, layers):
        self.layers = layers


def run_probe(weights, mask=None, capture_hidden=False):
    layer = _Layer(weights)
    probe = TokenAttention(_Model([layer]), capture_hidden=capture_hidden)
    if mask is not None:
        probe.set_attention_mask(mask)
    with probe:
        layer.self_attn.fire()
    return probe.profile


def test_column_sums_are_what_each_position_received():
    """Row i attends; column j is attended to. The hook keeps the columns."""
    # One example, one head, three positions, lower-triangular and normalised.
    w = torch.tensor([[[[1.0, 0.0, 0.0],
                        [0.5, 0.5, 0.0],
                        [0.2, 0.3, 0.5]]]])
    prof = run_probe(w)
    got = prof.received[0][0]
    assert np.allclose(got, [1.7, 0.8, 0.5])


def test_padding_columns_do_not_collect_mass():
    """A pad column holding the mass would be the most convincing fake sink."""
    # Position 0 is padding, and the model happily attends to it.
    w = torch.tensor([[[[1.0, 0.0, 0.0],
                        [0.9, 0.1, 0.0],
                        [0.8, 0.1, 0.1]]]])
    mask = torch.tensor([[0, 1, 1]])
    prof = run_probe(w, mask=mask)
    got = prof.received[0][0]
    assert got[0] == 0.0, f"The rubbish column kept the mass: {got}"
    assert got[1] > 0 and got[2] > 0


def test_padding_rows_do_not_ask():
    """A pad row's attention is not a real read and must not count."""
    w = torch.tensor([[[[0.0, 0.0, 1.0],   # pad row, all on position 2
                        [0.0, 1.0, 0.0],
                        [0.0, 0.5, 0.5]]]])
    mask = torch.tensor([[0, 1, 1]])
    prof = run_probe(w, mask=mask)
    got = prof.received[0][0]
    # Position 2 keeps only the 0.5 from the real row, not the pad row's 1.0.
    assert got[2] == pytest.approx(0.5)


def test_heads_are_averaged_not_summed():
    """Summing heads would scale every number by the head count."""
    one = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    w = torch.stack([one, one], dim=0)[None]      # [1, 2 heads, 2, 2]
    prof = run_probe(w)
    assert np.allclose(prof.received[0][0], [1.5, 0.5])


def test_sink_shows_as_a_large_top1_share():
    received = np.array([9.0, 0.3, 0.3, 0.4])
    m = sink_metrics(received)
    assert m["top1_share"] == pytest.approx(0.9)
    assert m["top1_index"] == 0
    assert m["entropy_ratio"] < 0.5


def test_flat_attention_reads_as_flat():
    received = np.ones(8)
    m = sink_metrics(received)
    assert m["top1_share"] == pytest.approx(0.125)
    assert m["entropy_ratio"] == pytest.approx(1.0)


def test_top1_share_of_a_sink_does_not_shrink_with_the_prompt():
    """The property that separates a sink from ordinary peakedness.

    Spreading the same non-sink mass over more positions leaves a uniform
    distribution's top share falling as 1/n; a sink's does not move.
    """
    short = np.concatenate([[9.0], np.full(9, 1.0 / 9)])
    long = np.concatenate([[9.0], np.full(99, 1.0 / 99)])
    assert sink_metrics(short)["top1_share"] == pytest.approx(
        sink_metrics(long)["top1_share"], abs=1e-9)


def test_empty_attention_does_not_divide_by_zero():
    m = sink_metrics(np.zeros(5))
    assert m["top1_share"] == 0.0 and m["top1_index"] is None


def test_massive_activation_is_relative_to_the_sequence():
    """Layers differ by orders of magnitude, so the threshold has to be local."""
    hidden = np.array([1.0, 1.1, 0.9, 50.0, 1.0])
    m = massive_activations(hidden)
    assert m["argmax"] == 3
    assert m["ratio"] == pytest.approx(50.0)
    assert m["n_massive"] == 1


def test_a_flat_layer_reports_no_massive_activation():
    m = massive_activations(np.array([1.0, 1.02, 0.98, 1.01]))
    assert m["n_massive"] == 0
    assert m["ratio"] < 1.1


def test_hidden_norm_is_the_max_coordinate_not_the_mean():
    """One runaway coordinate is the definition; the mean would hide it."""
    layer = _Layer(torch.tensor([[[[1.0, 0.0], [0.5, 0.5]]]]))
    probe = TokenAttention(_Model([layer]), capture_hidden=True)
    with probe:
        hidden = torch.zeros(1, 2, 4)
        hidden[0, 1, 2] = 80.0            # one coordinate, one position
        for fn in layer._hooks:
            fn(layer, (), (hidden,))
        layer.self_attn.fire()
    norm = probe.profile.hidden_norm[0][0]
    assert norm[1] == pytest.approx(80.0), "Averaging by width concealed the spike"
    assert norm[0] == pytest.approx(0.0)


def test_fused_attention_is_an_error_not_a_silent_zero():
    """SDPA returns None for the weights; a silent miss would read as 'no sink'."""
    class _Fused(_Attn):
        def fire(self):
            for fn in self._hooks:
                fn(self, (), (torch.zeros(1, 2, 4), None))

    layer = _Layer(torch.zeros(1, 1, 2, 2))
    layer.self_attn = _Fused(torch.zeros(1, 1, 2, 2))
    probe = TokenAttention(_Model([layer]), capture_hidden=False)
    with probe:
        layer.self_attn.fire()
    with pytest.raises(RuntimeError, match="eager"):
        probe.assert_captured()


def test_hooks_come_off_even_on_an_exception():
    """An instrumented model left behind would poison every later measurement."""
    layer = _Layer(torch.tensor([[[[1.0, 0.0], [0.5, 0.5]]]]))
    probe = TokenAttention(_Model([layer]), capture_hidden=False)
    try:
        with probe:
            raise ValueError("glitch")
    except ValueError:
        pass
    assert probe._handles == []


# Head-level sinks and padding

def test_head_specific_sink_survives_the_head_collapse():
    """Averaging thirty-two heads buries a sink that lives in two of them.

    This is how the measurement would have reported "no sink" on exactly the
    models it was written to find one in: the reported sinks are concentrated
    in a minority of heads, and a column holding 80% of two heads' mass averages
    down to 5% across thirty-two and reads as ordinary.
    """
    n_heads = 32
    # One head pours everything onto position 0; the rest are flat.
    sink_head = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    flat_head = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    heads = [sink_head] + [flat_head] * (n_heads - 1)
    w = torch.stack(heads, dim=0)[None]           # [1, 32, 2, 2]

    prof = run_probe(w)
    averaged = sink_metrics(prof.received[0][0])
    top_head = sink_metrics(prof.received_top_head[0][0])

    # The mean dilutes it; the most concentrated head does not.
    assert averaged["top1_share"] < 0.8
    assert top_head["top1_share"] == pytest.approx(1.0)


def test_the_kept_head_is_a_whole_head_not_a_column_wise_maximum():
    """Taking the maximum column by column stitches a vector from many heads.

    Each column would come from whichever head read it hardest, so the result is
    no head's distribution and its "share" is a ratio between numbers that never
    coexisted. Here head 0 owns column 0 and head 1 owns column 1; a column-wise
    maximum reports both at full strength, which no head ever did.
    """
    head_a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])    # all on column 0
    head_b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])    # column 1 as well
    w = torch.stack([head_a, head_b], dim=0)[None]     # [1, 2 heads, 2, 2]

    prof = run_probe(w)
    kept = prof.received_top_head[0][0]
    per_head_columns = [[2.0, 0.0], [1.0, 1.0]]

    assert list(kept) in [list(h) for h in per_head_columns], \
        f"The vector is assembled from different heads: {kept}"
    # And it is the concentrated one, not merely the first.
    assert sink_metrics(kept)["top1_share"] == pytest.approx(1.0)


def test_positions_are_comparable_across_uneven_padding():
    """Absolute column indices mean different things in a left-padded batch.

    Two examples whose sink sits on their own first real token land on columns
    2 and 0. A mode over raw indices mixes them into a number that describes
    neither; the offset from the first real token is the same for both.
    """
    # Example A: two pad positions then two real ones, sink on the first real.
    a = torch.tensor([[0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.9, 0.1]])
    # Example B: no padding, sink on its first real position (column 0).
    b = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                      [0.9, 0.1, 0.0, 0.0],
                      [0.9, 0.05, 0.05, 0.0],
                      [0.9, 0.04, 0.03, 0.03]])
    w = torch.stack([a, b], dim=0)[:, None]       # [2, 1 head, 4, 4]
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])

    prof = run_probe(w, mask=mask)
    first = prof.first_real
    assert list(first) == [2, 0]

    m_a = sink_metrics(prof.received[0][0], first_real=first[0])
    m_b = sink_metrics(prof.received[0][1], first_real=first[1])

    assert m_a["top1_index"] != m_b["top1_index"], "fixture doesn't check displacement"
    assert m_a["top1_offset"] == m_b["top1_offset"] == 0


def test_activation_position_uses_the_same_coordinates_as_the_sink():
    """The pairing is the evidence, so both have to count from the same origin."""
    hidden = np.array([99.0, 99.0, 1.0, 40.0])    # first two are padding
    m = massive_activations(hidden, first_real=2)
    assert m["argmax"] == 3
    assert m["argmax_offset"] == 1
    # Padding excluded from the median, so the ratio is 40/median(1, 40).
    assert m["median"] == pytest.approx(20.5)


def test_padding_does_not_manufacture_an_activation_ratio():
    """A pad token's hidden state is not part of the sequence's scale."""
    hidden = np.array([0.001, 0.001, 5.0, 5.2, 4.8])
    without = massive_activations(hidden, first_real=2)
    assert without["n_massive"] == 0
    assert without["ratio"] < 1.2


def test_a_mode_without_its_share_cannot_be_read():
    """Three wins out of sixty-four and sixty-four out of sixty-four look alike.

    The mode alone always names some position, so the share is what separates a
    sink standing on a fixed token from a peak that wanders.
    """
    fixed = [{"top1_offset": 0} for _ in range(64)]
    wandering = [{"top1_offset": i} for i in range(64)]

    assert position_mode("top1_offset", fixed)["top1_offset_mode_frac"] == 1.0
    assert position_mode("top1_offset", wandering)["top1_offset_mode_frac"] < 0.05


def test_failed_examples_do_not_win_the_vote():
    """`valid` marks a layer whose metric could not be computed, not the sign."""
    metrics = [{"top1_offset": None, "valid": False}] * 5 + \
              [{"top1_offset": 3, "valid": True}] * 2
    got = position_mode("top1_offset", metrics)
    assert got["top1_offset_mode"] == 3
    assert got["top1_offset_mode_frac"] == pytest.approx(1.0)


def test_an_empty_layer_reports_no_position():
    got = position_mode("top1_offset", [])
    assert got["top1_offset_mode"] is None and got["top1_offset_mode_frac"] == 0.0


# Detailed layer: where exactly did you look?

def test_every_head_is_kept_when_asked_for():
    """The collapse is a convenience; super-expert work needs the heads named."""
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    w = torch.stack([a, b], dim=0)[None]

    layer = _Layer(w)
    probe = TokenAttention(_Model([layer]), capture_hidden=False,
                           keep_per_head=True)
    with probe:
        layer.self_attn.fire()

    per_head = probe.profile.received_per_head[0]
    assert per_head.shape == (1, 2, 2)
    assert np.allclose(per_head[0, 0], [2.0, 0.0])
    assert np.allclose(per_head[0, 1], [1.0, 1.0])


def test_edges_name_the_reader_and_the_read():
    """A column sum says a position is popular; an edge says who made it so."""
    w = torch.tensor([[[[1.0, 0.0, 0.0],
                        [0.1, 0.9, 0.0],
                        [0.7, 0.2, 0.1]]]])

    layer = _Layer(w)
    # min_context=1 keeps every row: this test is about the pairing itself,
    # and the opening-row cut has its own test below.
    probe = TokenAttention(_Model([layer]), capture_hidden=False, top_edges=2,
                           edge_min_context=1)
    with probe:
        layer.self_attn.fire()

    edges = probe.profile.edges[0][0]              # [heads, n, 3]
    assert edges.shape == (1, 2, 3)
    pairs = {(int(q), int(k)): float(v) for q, k, v in edges[0]}
    assert pairs == {(0, 0): pytest.approx(1.0), (1, 1): pytest.approx(0.9)}


def test_edges_cannot_come_from_padding():
    """A pad row reading a pad column would be the loudest edge in the dump."""
    w = torch.tensor([[[[1.0, 0.0, 0.0],      # pad row, all on the pad column
                        [0.95, 0.05, 0.0],
                        [0.9, 0.05, 0.05]]]])
    mask = torch.tensor([[0, 1, 1]])

    layer = _Layer(w)
    probe = TokenAttention(_Model([layer]), capture_hidden=False, top_edges=3,
                           edge_min_context=1)
    probe.set_attention_mask(mask)
    with probe:
        layer.self_attn.fire()

    edges = probe.profile.edges[0][0][0]
    for q_i, k_i, weight in edges:
        assert not (int(q_i) == 0 or int(k_i) == 0) or weight == 0.0, \
            f"Padding rib: {q_i}->{k_i} weight {weight}"


def test_detail_is_off_unless_asked_for():
    """Sixteen cells at three levels each cannot afford it by default."""
    prof = run_probe(torch.tensor([[[[1.0, 0.0], [0.5, 0.5]]]]))
    assert prof.received_per_head == {} and prof.edges == {}


def test_described_edges_are_readable_and_use_offsets():
    edges = np.array([[[2.0, 2.0, 0.90],
                       [3.0, 2.0, 0.02]]])        # one head, two edges
    tokens = ["<pad>", "<pad>", "You", " fool"]
    lines = describe_edges(edges, tokens, layer=23, first_real=2)

    assert len(lines) == 1, "ligature"
    assert "layer 23 head  0" in lines[0]
    assert "+0 'You'" in lines[0] and "0.900" in lines[0]


# causal base: without it, "stock" gives any model

def test_a_model_with_no_sink_at_all_scores_exactly_one():
    """The number that makes a share readable.

    Under a causal mask row i sees positions <= i, so column 0 is summed over
    all n rows and the last over one. Perfectly even attention therefore still
    puts H_n/n on position 0 - 0.024 at a 256-token window, six times the 1/n
    an uncorrected reading would call flat. Comparing to 1/n invents a sink in
    every model ever measured.
    """
    n = 256
    received = np.array([sum(1.0 / (i + 1) for i in range(j, n)) for j in range(n)])
    m = sink_metrics(received)

    assert m["top1_share"] > 5 * (1 / n), "The flat causal model is not equal to 1/n"
    assert m["top1_over_null"] == pytest.approx(1.0, abs=0.01)


def test_a_real_sink_still_towers_over_the_null():
    n = 256
    received = np.array([sum(1.0 / (i + 1) for i in range(j, n)) for j in range(n)])
    received[7] = received.sum()                  # one position takes half
    m = sink_metrics(received)
    assert m["top1_over_null"] > 10
    assert m["top1_offset"] == 7


def test_the_null_tracks_the_window():
    """Shares are only comparable between layers measured at the same width."""
    assert causal_null_top1_share(64) > causal_null_top1_share(256)
    assert causal_null_top1_share(256) == pytest.approx(0.0239, abs=1e-3)
    assert causal_null_top1_share(1) == 1.0


def test_forced_edges_from_the_opening_rows_are_dropped():
    """The first real row attends to one position, so its weight there is 1.0.

    Without the cut, every head's strongest edge in every layer of every model
    is one of the opening rows, and the dump describes arithmetic rather than
    the model.
    """
    n = 6
    w = torch.zeros(1, 1, n, n)
    for i in range(n):
        w[0, 0, i, :i + 1] = 1.0 / (i + 1)        # perfectly even, no sink
    w[0, 0, 5, 2] = 0.9                            # the one deliberate read

    layer = _Layer(w)
    probe = TokenAttention(_Model([layer]), capture_hidden=False, top_edges=3,
                           edge_min_context=1)
    with probe:
        layer.self_attn.fire()
    strongest = probe.profile.edges[0][0, 0, 0]
    assert int(strongest[0]) == 0, "Fixture does not reproduce the trap"

    layer = _Layer(w)
    probe = TokenAttention(_Model([layer]), capture_hidden=False, top_edges=3,
                           edge_min_context=4)
    with probe:
        layer.self_attn.fire()
    strongest = probe.profile.edges[0][0, 0, 0]
    assert (int(strongest[0]), int(strongest[1])) == (5, 2)
    assert strongest[2] == pytest.approx(0.9)


# PEFT: virtual tokens face paddling

def test_a_prefix_arm_is_measurable_at_all():
    """PEFT puts the virtual tokens at position 0, ahead of the padding.

    The routed sequence is [virtual][pad][text] while the tokenizer's mask
    covers only the text, so the key axis is wider than the mask by the virtual
    count. Before this was handled the multiplication raised and the script
    could not run on a single one of the arms it exists for.
    """
    vt, width = 3, 4
    q, k = width, vt + width               # prefix: virtual are keys, not queries
    w = torch.zeros(1, 2, q, k)
    w[0, :, :, :] = 1.0 / k
    mask = torch.tensor([[0, 1, 1, 1]])    # one pad, three real

    layer = _Layer(w)
    probe = TokenAttention(_Model([layer]), capture_hidden=False)
    probe.set_attention_mask(mask, n_virtual=vt)
    with probe:
        layer.self_attn.fire()

    received = probe.profile.received[0][0]
    assert received.shape == (k,)
    # Virtual positions are real and keep their mass; the pad column does not.
    assert received[0] > 0 and received[vt] == 0.0
    assert received[vt + 1] > 0


def test_a_prompt_arm_carries_the_virtual_tokens_on_both_axes():
    vt, width = 2, 3
    k = vt + width
    w = torch.full((1, 1, k, k), 1.0 / k)
    mask = torch.tensor([[0, 1, 1]])

    layer = _Layer(w)
    probe = TokenAttention(_Model([layer]), capture_hidden=False)
    probe.set_attention_mask(mask, n_virtual=vt)
    with probe:
        layer.self_attn.fire()
    assert probe.profile.received[0].shape == (1, k)


def test_offsets_are_counted_from_the_prompt_not_the_virtual_block():
    """Otherwise vt=20 and vt=500 arms cannot be put in the same table."""
    vt, width = 5, 4
    w = torch.full((1, 1, width, vt + width), 0.1)
    mask = torch.tensor([[0, 1, 1, 1]])

    layer = _Layer(w)
    probe = TokenAttention(_Model([layer]), capture_hidden=False)
    probe.set_attention_mask(mask, n_virtual=vt)
    with probe:
        layer.self_attn.fire()
    # First prompt token sits at vt + pad = 6; virtual tokens are negative.
    assert int(probe.profile.first_real[0]) == vt + 1


def test_a_wrong_virtual_count_is_an_error_not_a_shift():
    """Off by the virtual count, every position in the run is wrong."""
    w = torch.full((1, 1, 4, 9), 0.1)      # k = 9 means vt = 5
    layer = _Layer(w)
    probe = TokenAttention(_Model([layer]), capture_hidden=False)
    probe.set_attention_mask(torch.ones(1, 4), n_virtual=20)
    with pytest.raises(RuntimeError, match="virtual"):
        with probe:
            layer.self_attn.fire()


# The read position's row and its mass by segment

def test_the_read_position_row_is_kept():
    """What the answer position looks at, as opposed to what is looked at."""
    w = torch.tensor([[[[1.0, 0.0, 0.0],
                        [0.5, 0.5, 0.0],
                        [0.2, 0.3, 0.5]]]])
    prof = run_probe(w)
    assert np.allclose(prof.last_row[0][0, 0], [0.2, 0.3, 0.5])


def test_row_entropy_separates_a_focused_head_from_a_diffuse_one():
    focused = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0], [0.97, 0.01, 0.01, 0.01]])
    diffuse = torch.full((4, 4), 0.25)
    w = torch.stack([focused, diffuse], dim=0)[None]

    prof = run_probe(w)
    ent = prof.row_entropy[0][0]
    assert ent[0] < ent[1]
    assert ent[1] == pytest.approx(np.log(4), abs=1e-4)


def test_segment_mass_answers_what_share_the_soft_prompt_takes():
    """Four numbers that say which mechanism an arm is using."""
    row = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
    seg = np.array([0, 0, 1, 2, -1])          # two virtual, one instruction, one text, one pad
    mass = segment_mass(row, seg, 3)
    assert mass.tolist() == pytest.approx([0.7, 0.2, 0.05])


def test_unlabelled_positions_are_dropped_not_folded_into_a_neighbour():
    row = np.array([0.5, 0.5])
    assert segment_mass(row, np.array([-1, 0]), 1).tolist() == pytest.approx([0.5])


def test_segment_mass_keeps_the_layer_and_head_axes():
    rows = np.ones((2, 3, 4))                  # [layers, heads, keys]
    mass = segment_mass(rows, np.array([0, 0, 1, 1]), 2)
    assert mass.shape == (2, 3, 2)
    assert np.allclose(mass, 2.0)


def test_a_labelling_of_the_wrong_width_is_refused():
    """Silently misaligned segments would put the comment's mass on the prompt."""
    with pytest.raises(ValueError, match="Segment mask"):
        segment_mass(np.ones(5), np.zeros(4, dtype=int), 1)


def test_the_fast_path_agrees_with_the_slow_one_element_by_element():
    """The table is built by the vectorised version; both must be one function.

    Checked against the scalar implementation rather than by re-deriving the
    arithmetic, because two derivations that drift agree with nothing.
    """
    from mrd_pruning.token_attention import sink_metrics_batch

    rng = np.random.default_rng(0)
    rows = np.abs(rng.normal(size=(3, 5, 32)))
    rows[0, 0] = 0.0                              # a dead layer
    rows[1, 2, 7] = 500.0                         # a planted sink
    first = np.array([0, 3, 11])

    fast = sink_metrics_batch(rows, first)
    for i in range(rows.shape[0]):
        for j in range(rows.shape[1]):
            slow = sink_metrics(rows[i, j], first_real=first[i])
            assert bool(fast["valid"][i, j]) == slow["valid"]
            if not slow["valid"]:
                continue          # a dead row has nothing to agree about
            for key in ("top1_share", "top1_over_null", "top4_share",
                        "top1_index", "top1_offset", "top1_from_end",
                        "entropy_ratio", "n_positions"):
                assert fast[key][i, j] == pytest.approx(slow[key], rel=1e-9), \
                    f"{key} spread out into{i},{j}]"


def test_a_batch_of_origins_that_is_not_this_batch_is_refused():
    """Silently broadcasting the wrong origins makes every offset someone else's."""
    from mrd_pruning.token_attention import sink_metrics_batch

    with pytest.raises(ValueError, match="Origins"):
        sink_metrics_batch(np.ones((2, 4, 8)), np.array([0, 1, 2]))


# Export of the sixteen cells

def test_a_sink_inside_the_virtual_block_keeps_its_position():
    """Offsets run from the first prompt token, so the soft prompt is negative.

    Treating `< 0` as "could not be computed" reported mode -1 with a share of
    zero for ten cells out of sixteen - every arm whose sink sits on its own
    virtual tokens, which is the case the measurement exists for.
    """
    from mrd_pruning.token_attention import position_mode

    metrics = [{"top1_offset": -7, "valid": True} for _ in range(30)]
    metrics += [{"top1_offset": 3, "valid": True} for _ in range(10)]
    got = position_mode("top1_offset", metrics)
    assert got["top1_offset_mode"] == -7
    assert got["top1_offset_mode_frac"] == pytest.approx(0.75)


def test_a_layer_with_nothing_in_it_is_still_skipped():
    """The sign is a place; `valid` is what marks a failure."""
    from mrd_pruning.token_attention import position_mode

    metrics = [{"top1_offset": None, "valid": False}] * 5 + \
              [{"top1_offset": -2, "valid": True}] * 3
    got = position_mode("top1_offset", metrics)
    assert got["top1_offset_mode"] == -2
    assert got["top1_offset_mode_frac"] == pytest.approx(1.0)


def test_an_empty_metric_set_reports_no_position_rather_than_zero():
    from mrd_pruning.token_attention import position_mode

    got = position_mode("top1_offset", [{"top1_offset": None, "valid": False}])
    assert got["top1_offset_mode"] is None


def test_a_dead_row_is_flagged_not_encoded_as_minus_one():
    m = sink_metrics(np.zeros(5))
    assert m["valid"] is False and m["top1_offset"] is None
    live = sink_metrics(np.array([1.0, 2.0, 3.0]))
    assert live["valid"] is True


def test_the_batch_path_keeps_negative_offsets_too():
    from mrd_pruning.token_attention import sink_metrics_batch

    rows = np.zeros((2, 6))
    rows[0, 1] = 5.0          # wins at index 1
    rows[1] = 0.0             # dead
    got = sink_metrics_batch(rows, np.array([4, 4]))
    assert got["top1_offset"][0] == -3
    assert bool(got["valid"][0]) is True and bool(got["valid"][1]) is False


def test_an_all_zero_hidden_row_has_no_activation():
    """Otherwise the argmax of nothing is reported as a measured position."""
    m = massive_activations(np.zeros(6), first_real=2)
    assert m["valid"] is False
    assert m["argmax"] is None and m["argmax_offset"] is None


# Checks of the exported arrays

def test_the_hidden_axis_is_translated_not_assumed():
    """Prefix virtual tokens are keys that were never positions.

    Hidden states are then narrower than the key axis by exactly that block, so
    an origin counted on the key axis runs past the end of the hidden row. The
    delivered prefix cells took the median over padding and reported an argmax
    in padded-text coordinates - a different coordinate system from the sink it
    is supposed to be paired with, which makes the mechanism claim untestable.
    """
    from mrd_pruning.token_attention import hidden_origin

    assert hidden_origin(622, 256, 756, 500) == 122      # prefix
    assert hidden_origin(622, 756, 756, 500) == 622      # prompt tuning
    assert hidden_origin(122, 256, 256, 0) == 122        # no adapter


def test_an_unfamiliar_pair_of_widths_is_refused():
    from mrd_pruning.token_attention import hidden_origin

    with pytest.raises(ValueError, match="unfamiliar"):
        hidden_origin(10, 100, 756, 500)


def test_an_origin_off_the_end_is_an_error_not_a_silent_whole_row():
    """Falling back to the whole row is what put the median over padding."""
    with pytest.raises(ValueError, match="different axes"):
        massive_activations(np.ones(256), first_real=622)


def test_the_translated_origin_excludes_padding_again():
    hidden = np.array([0.001, 0.001, 5.0, 40.0])   # two pads, then the prompt
    m = massive_activations(hidden, first_real=2)
    assert m["median"] == pytest.approx(22.5)
    assert m["argmax_offset"] == 1


def test_the_null_is_measured_from_the_mask_not_assumed_causal():
    """A prefix arm's virtual keys are visible to every row.

    The causal formula H_n/n assumes a square triangle. Here four query rows all
    see two always-visible keys plus their own causal tail, so an even spread
    puts far more on the shared keys than the formula predicts - the delivered
    numbers were about sevenfold too generous on prefix arms, which hides sinks.
    """
    q, vt = 4, 2
    k = vt + q
    w = torch.zeros(1, 1, q, k)
    for i in range(q):
        w[0, 0, i, :vt] = 1.0 / (vt + i + 1)          # the shared prefix
        w[0, 0, i, vt:vt + i + 1] = 1.0 / (vt + i + 1)
    layer = _Layer(w)
    probe = TokenAttention(_Model([layer]), capture_hidden=False)
    with probe:
        layer.self_attn.fire()

    measured = float(probe.profile.null_top1[0][0])
    assumed = causal_null_top1_share(k)

    # The invariant that matters: a model which spreads evenly scores exactly
    # one against its own null, whatever shape the mask has.
    m = sink_metrics(probe.profile.received[0][0], null=measured)
    assert m["top1_over_null"] == pytest.approx(1.0, abs=0.02)

    # And the formula does not describe this mask - it assumes a square causal
    # triangle, while here four rows share two always-visible keys.
    assert abs(measured - assumed) > 0.1
    formula = sink_metrics(probe.profile.received[0][0])
    assert abs(formula["top1_over_null"] - 1.0) > 0.2, \
        "formula zero coincides accidentally - fixture does not check for discrepancy"


def test_a_plain_causal_layer_still_matches_the_formula():
    """Where the assumption holds, the measured null must agree with it."""
    n = 32
    w = torch.zeros(1, 1, n, n)
    for i in range(n):
        w[0, 0, i, :i + 1] = 1.0 / (i + 1)
    layer = _Layer(w)
    probe = TokenAttention(_Model([layer]), capture_hidden=False)
    with probe:
        layer.self_attn.fire()
    assert float(probe.profile.null_top1[0][0]) == pytest.approx(
        causal_null_top1_share(n), rel=0.05)


def test_the_row_count_follows_the_key_axis_not_the_query_axis():
    """They are different widths on a prefix arm, and the mean divides by this."""
    q, k = 3, 5
    w = torch.zeros(1, 1, q, k)
    w[0, 0, :, :] = 0.2
    layer = _Layer(w)
    probe = TokenAttention(_Model([layer]), capture_hidden=False)
    with probe:
        layer.self_attn.fire()
    assert probe.profile.received_mean[0].shape == (1, k)
