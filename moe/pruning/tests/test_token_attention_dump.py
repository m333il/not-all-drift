"""The detailed dump is the deliverable, so its shape is pinned here.

The summary answers "is there a sink and where". The dump answers the question
the super-expert work actually asks: *which head of which layer read which
token, and how hard*. That only survives the trip to disk if three things hold -
the arrays keep their layer and head axes apart, the decoded tokens travel with
the numbers, and every position is written as an offset from the first real
token rather than as a raw column of a left-padded batch.

The helpers are exercised against a hand-built profile: no model, no card, and a
failure names the defect instead of a number that moved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

import measure_token_attention as mta  # noqa: E402

from mrd_pruning.token_attention import AttentionProfile  # noqa: E402


class _Tokenizer:
    """Just the one method the dump needs, with a legible vocabulary."""

    def convert_ids_to_tokens(self, ids):
        table = {0: "<pad>", 1: "You", 2: " fool", 3: "<|im_end|>"}
        return [table.get(int(i), f"<{int(i)}>") for i in ids]


def _profile(n_layers=2, n_heads=3, n_pos=4, batch=2):
    prof = AttentionProfile()
    prof.first_real = np.array([2, 0])[:batch]  # uneven left padding
    for l in range(n_layers):
        prof.received[l] = np.zeros((batch, n_pos))
        prof.received_per_head[l] = np.arange(
            batch * n_heads * n_pos, dtype=np.float32).reshape(batch, n_heads, n_pos)
        prof.top_head_index[l] = np.array([1, 0])
        # Two edges per head: one strong, one below any sane floor.
        edges = np.zeros((batch, n_heads, 2, 3), dtype=np.float32)
        edges[..., 0] = 3          # query index
        edges[..., 1] = 2          # key index
        edges[:, :, 0, 2] = 0.9    # strong
        edges[:, :, 1, 2] = 0.01   # noise
        prof.edges[l] = edges
        prof.hidden_norm[l] = np.ones((batch, n_pos))
    return prof


def test_a_row_carries_its_tokens_and_its_origin():
    row = mta._detail_row(_profile(), 0, [0, 0, 1, 2], _Tokenizer(), index=7)
    assert row["example"] == 7
    assert row["first_real"] == 2
    assert row["tokens"] == ["<pad>", "<pad>", "You", " fool"]
    # Offsets, not raw columns: the first real token is position 0.
    assert row["offsets"] == [-2, -1, 0, 1]


def test_heads_stay_separate_in_the_dump():
    """Collapsing them on the way out would throw away the reason for the dump."""
    row = mta._detail_row(_profile(n_heads=3, n_pos=4), 0, [0, 0, 1, 2],
                          _Tokenizer(), index=0)
    assert row["per_head"]["0"].shape == (3, 4)
    assert not np.allclose(row["per_head"]["0"][0], row["per_head"]["0"][1])


def test_the_second_example_is_not_the_first(tmp_path):
    """Row indexing has to follow the batch row, padding offset included."""
    prof = _profile()
    a = mta._detail_row(prof, 0, [0, 0, 1, 2], _Tokenizer(), index=0)
    b = mta._detail_row(prof, 1, [1, 2, 3, 3], _Tokenizer(), index=1)
    assert a["first_real"] == 2 and b["first_real"] == 0
    assert not np.allclose(a["per_head"]["0"], b["per_head"]["0"])


def test_written_dump_keeps_layer_and_example_apart(tmp_path):
    prof = _profile(n_layers=2)
    rows = [mta._detail_row(prof, 0, [0, 0, 1, 2], _Tokenizer(), index=0),
            mta._detail_row(prof, 1, [1, 2, 3, 3], _Tokenizer(), index=1)]
    mta._write_detail(tmp_path, 64, rows, 0.05)

    with np.load(tmp_path / "detail_level064.npz") as z:
        keys = set(z.files)
        assert "ex000_per_head_layer00" in keys
        assert "ex000_per_head_layer01" in keys
        assert "ex001_per_head_layer00" in keys
        assert "ex000_edges_layer00" in keys
        assert "ex000_hidden_layer00" in keys
        # Two examples, two layers, three kinds of array, plus the origins.
        assert len(keys) == 2 * 2 * 3 + 2


def test_tokens_are_written_beside_the_numbers(tmp_path):
    rows = [mta._detail_row(_profile(), 0, [0, 0, 1, 2], _Tokenizer(), index=0)]
    mta._write_detail(tmp_path, 0, rows, 0.05)

    got = json.loads((tmp_path / "tokens_level000.json").read_text())
    assert got[0]["tokens"][2] == "You"
    assert got[0]["first_real"] == 2


def test_readable_dump_names_the_head_and_drops_the_noise(tmp_path):
    rows = [mta._detail_row(_profile(n_layers=1, n_heads=3), 0, [0, 0, 1, 2],
                            _Tokenizer(), index=0)]
    mta._write_detail(tmp_path, 0, rows, 0.05)

    text = (tmp_path / "edges_level000.txt").read_text()
    lines = [l for l in text.splitlines() if l.startswith("layer")]
    # One surviving edge per head, the 0.01 ones filtered out.
    assert len(lines) == 3
    assert "head" in text
    assert "0.900" in text and "0.010" not in text
    # Positions as offsets: the key is column 2, which is offset 0 here.
    assert "+0 'You'" in text


def test_nothing_is_written_when_the_dump_is_switched_off(tmp_path):
    mta._write_detail(tmp_path, 0, [], 0.05)
    assert list(tmp_path.iterdir()) == []


# The key axis is [virtual][pad][prompt] ---

def test_virtual_tokens_come_before_the_padding_not_after():
    """PEFT prepends them at position 0, ahead of the pad, and that is the trap.

    A labelling built as [pad][virtual] is off by the padding width, and the
    soft prompt's attention mass is then credited to whatever token sits there.
    """
    seg = mta._segment_ids([[4, 4, 2, 2]], n_virtual=3, width=6, virtual_id=1)
    assert seg.shape == (1, 9)
    assert seg[0, :3].tolist() == [1, 1, 1]          # virtual, at the front
    assert seg[0, 3:5].tolist() == [-1, -1]          # then the padding
    assert seg[0, 5:].tolist() == [4, 4, 2, 2]       # then the prompt


def test_an_arm_without_virtual_tokens_is_just_pad_and_prompt():
    seg = mta._segment_ids([[4, 2, 2]], n_virtual=0, width=5, virtual_id=1)
    assert seg[0].tolist() == [-1, -1, 4, 2, 2]


def test_each_example_keeps_its_own_padding_width():
    seg = mta._segment_ids([[2, 2], [4, 4, 4, 4]], n_virtual=1, width=4,
                           virtual_id=1)
    assert seg[0].tolist() == [1, -1, -1, 2, 2]
    assert seg[1].tolist() == [1, 4, 4, 4, 4]


def test_the_mass_of_a_labelled_row_adds_up_to_the_row():
    """Segments partition the real positions, so nothing may go missing."""
    from mrd_pruning.token_attention import segment_mass

    seg = mta._segment_ids([[4, 2, 2, 3]], n_virtual=2, width=6, virtual_id=1)
    row = np.full(8, 0.125)
    mass = segment_mass(row, seg[0], 7)
    # Two pad positions are dropped; everything else is accounted for once.
    assert mass.sum() == pytest.approx(0.75)
    assert mass[1] == pytest.approx(0.25)            # the virtual block


# Full tables: what is then analyzed

def _full_profile(n_layers=2, n_heads=3, n_pos=5, batch=2):
    prof = _profile(n_layers=n_layers, n_heads=n_heads, n_pos=n_pos, batch=batch)
    rng = np.random.default_rng(1)
    for l in range(n_layers):
        prof.received[l] = np.abs(rng.normal(size=(batch, n_pos))) + 0.1
        prof.received_top_head[l] = prof.received[l] * 2
        prof.last_row[l] = np.abs(rng.normal(size=(batch, n_heads, n_pos))) + 0.1
        prof.row_entropy[l] = np.abs(rng.normal(size=(batch, n_heads)))
        prof.hidden_norm[l] = np.abs(rng.normal(size=(batch, n_pos))) + 1.0
    prof.n_positions = n_pos          # key axis needed to translate axes
    return prof


def _read_csv(path):
    import csv
    with path.open(encoding="utf-8") as h:
        return list(csv.DictReader(h))


def test_the_head_table_has_a_row_for_every_head_of_every_layer(tmp_path):
    """The analysis table is per head; a layer mean cannot be un-averaged."""
    mta.SEGMENT_COLUMNS = ["system", "virtual", "comment"]
    prof = _full_profile(n_layers=2, n_heads=3, n_pos=5, batch=2)
    seg = mta._segment_ids([[2, 2, 2], [2, 2, 2]], n_virtual=1, width=4, virtual_id=1)

    handle, writer = mta._open_table(tmp_path / "heads.csv", mta._head_columns())
    mta._write_head_rows(writer, prof, seg, level=64, first_index=8)
    handle.close()

    rows = _read_csv(tmp_path / "heads.csv")
    assert len(rows) == 2 * 2 * 3                       # examples x layers x heads
    assert {r["example"] for r in rows} == {"8", "9"}   # numbered from the batch start
    assert {r["head"] for r in rows} == {"0", "1", "2"}
    assert all(r["level"] == "64" for r in rows)


def test_the_head_table_carries_the_segments_and_the_entropy(tmp_path):
    mta.SEGMENT_COLUMNS = ["system", "virtual", "comment"]
    prof = _full_profile(n_layers=1, n_heads=2, n_pos=5, batch=1)
    seg = mta._segment_ids([[2, 2, 2]], n_virtual=1, width=4, virtual_id=1)

    handle, writer = mta._open_table(tmp_path / "heads.csv", mta._head_columns())
    mta._write_head_rows(writer, prof, seg, level=0, first_index=0)
    handle.close()

    rows = _read_csv(tmp_path / "heads.csv")
    assert "seg_comment" in rows[0] and "seg_virtual" in rows[0]
    assert float(rows[0]["row_entropy"]) > 0
    # The row's mass is split between the labelled segments, pad excluded.
    # Six significant digits is what the table stores, so that is the tolerance.
    total = sum(float(rows[0][f"seg_{s}"]) for s in mta.SEGMENT_COLUMNS)
    assert total == pytest.approx(prof.last_row[0][0, 0][[0, 2, 3, 4]].sum(),
                                  rel=1e-5)


def test_head_shares_are_per_head_not_the_layer_mean_repeated(tmp_path):
    mta.SEGMENT_COLUMNS = ["system"]
    prof = _full_profile(n_layers=1, n_heads=3, n_pos=8, batch=1)
    prof.received_per_head[0][0, 1] = np.array([50.0] + [0.1] * 7)   # one sinky head

    handle, writer = mta._open_table(tmp_path / "heads.csv", mta._head_columns())
    mta._write_head_rows(writer, prof, np.zeros((1, 8), dtype=np.int16),
                         level=0, first_index=0)
    handle.close()

    shares = [float(r["top1_share"]) for r in _read_csv(tmp_path / "heads.csv")]
    assert shares[1] > 0.9 and max(shares[0], shares[2]) < 0.9


def test_the_layer_table_carries_the_activations(tmp_path):
    prof = _full_profile(n_layers=2, n_heads=3, n_pos=5, batch=2)
    handle, writer = mta._open_table(tmp_path / "layers.csv", mta._layer_columns())
    mta._write_layer_rows(writer, prof, level=96, first_index=0)
    handle.close()

    rows = _read_csv(tmp_path / "layers.csv")
    assert len(rows) == 2 * 2
    assert float(rows[0]["activation_ratio"]) > 0
    assert rows[0]["top_head"] in {"0", "1", "2"}
    assert float(rows[0]["top1_over_null"]) > 0


def test_arrays_are_written_one_file_per_example_per_level(tmp_path):
    prof = _full_profile(n_layers=2, n_heads=3, n_pos=5, batch=2)
    seg = np.zeros((2, 5), dtype=np.int16)
    mta._write_per_example(tmp_path, prof, seg, level=64, first_index=4)

    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["level064_ex0004.npz", "level064_ex0005.npz"]
    with np.load(tmp_path / "level064_ex0004.npz") as z:
        assert "received_per_head_layer00" in z.files
        assert "last_row_layer01" in z.files
        assert "hidden_norm_layer00" in z.files
        assert "segment_ids" in z.files
        assert z["received_per_head_layer00"].shape == (3, 5)


def test_the_saved_arrays_belong_to_the_example_they_are_named_for(tmp_path):
    """Off by one row here and every per-example conclusion is another's."""
    prof = _full_profile(n_layers=1, n_heads=2, n_pos=4, batch=2)
    mta._write_per_example(tmp_path, prof, np.zeros((2, 4), dtype=np.int16),
                           level=0, first_index=0)
    with np.load(tmp_path / "level000_ex0001.npz") as z:
        assert np.allclose(z["hidden_norm_layer00"], prof.hidden_norm[0][1])


# Axis of tokens: the dump must live where attention

def test_the_token_list_spans_the_key_axis_not_just_the_text():
    """On a PEFT arm the attention axis is wider than the tokenizer's output.

    Taking the token list from the text while taking `first_real` from the key
    axis put every offset inside the virtual block: the delivered dump for
    prefix-m500 showed first_real=622 against 256 tokens and offsets −622..−367,
    and the readable edge file indexed the wrong token for every edge.
    """
    prof = _profile(n_pos=7, batch=1)
    prof.first_real = np.array([5])            # 3 virtual + 2 pad
    row = mta._detail_row(prof, 0, [0, 0, 1, 2], _Tokenizer(), index=0,
                          n_virtual=3)

    assert len(row["tokens"]) == 3 + 4
    assert row["tokens"][:3] == ["<vt0>", "<vt1>", "<vt2>"]
    assert row["tokens"][5] == "You"            # the first real prompt token
    assert row["offsets"][5] == 0              # ...and it is offset zero
    assert row["offsets"][0] == -5             # the virtual block is negative
    assert row["n_virtual"] == 3


def test_an_arm_without_virtual_tokens_is_unchanged():
    prof = _profile(n_pos=4, batch=1)
    prof.first_real = np.array([2])
    row = mta._detail_row(prof, 0, [0, 0, 1, 2], _Tokenizer(), index=0,
                          n_virtual=0)
    assert row["tokens"] == ["<pad>", "<pad>", "You", " fool"]
    assert row["offsets"] == [-2, -1, 0, 1]


def test_edges_now_name_the_token_the_key_index_points_at():
    """The whole point of the dump: key index k must decode to tokens[k]."""
    from mrd_pruning.token_attention import describe_edges

    prof = _profile(n_pos=7, batch=1)
    prof.first_real = np.array([5])
    row = mta._detail_row(prof, 0, [0, 0, 1, 2], _Tokenizer(), index=0,
                          n_virtual=3)
    edges = np.array([[[6.0, 1.0, 0.9]]])      # query 6 read key 1 - a vt slot
    line = describe_edges(edges, row["tokens"], layer=0,
                          first_real=row["first_real"])[0]
    assert "'<vt1>'" in line
    assert "+1" in line and "-4" in line       # query at +1, key at -4


# A dead line should not look like a measured position.

def _mixed_profile():
    """Two examples, the second with nothing in it at all."""
    prof = _full_profile(n_layers=1, n_heads=2, n_pos=8, batch=2)
    prof.first_real = np.array([3, 0])
    prof.received[0][1] = 0.0
    prof.received_per_head[0][1] = 0.0
    prof.received_top_head[0][1] = 0.0
    prof.hidden_norm[0][1] = 0.0
    return prof


def test_a_dead_row_is_marked_and_its_position_left_blank(tmp_path):
    """An all-zero layer still has an argmax, and it means nothing.

    Before the column existed, such a row wrote a position like 7 that could not
    be told apart from a measured one.
    """
    mta.SEGMENT_COLUMNS = ["system"]
    prof = _mixed_profile()
    seg = np.zeros((2, 8), dtype=np.int16)

    handle, writer = mta._open_table(tmp_path / "heads.csv", mta._head_columns())
    mta._write_head_rows(writer, prof, seg, level=0, first_index=0)
    handle.close()

    rows = _read_csv(tmp_path / "heads.csv")
    live = [r for r in rows if r["example"] == "0"]
    dead = [r for r in rows if r["example"] == "1"]
    assert all(r["valid"] == "1" for r in live)
    assert all(r["valid"] == "0" for r in dead)
    assert all(r["top1_offset"] == "" for r in dead), "deadline gives out position"
    assert all(r["top1_offset"] != "" for r in live)


def test_the_layer_table_blanks_a_dead_activation_too(tmp_path):
    prof = _mixed_profile()
    handle, writer = mta._open_table(tmp_path / "layers.csv", mta._layer_columns())
    mta._write_layer_rows(writer, prof, level=0, first_index=0)
    handle.close()

    rows = _read_csv(tmp_path / "layers.csv")
    dead = [r for r in rows if r["example"] == "1"][0]
    assert dead["valid"] == "0"
    assert dead["top1_offset"] == "" and dead["activation_offset"] == ""
    assert dead["activation_ratio"] == ""


def test_live_rows_still_carry_every_number(tmp_path):
    prof = _mixed_profile()
    handle, writer = mta._open_table(tmp_path / "layers.csv", mta._layer_columns())
    mta._write_layer_rows(writer, prof, level=0, first_index=0)
    handle.close()
    live = [r for r in _read_csv(tmp_path / "layers.csv") if r["example"] == "0"][0]
    assert live["valid"] == "1"
    assert float(live["activation_ratio"]) > 0
    assert live["top1_offset"] != ""
