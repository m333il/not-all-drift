"""Padding is a speedup for most arms and a correctness bug for one kind.

Equal-length batches exist because PEFT puts prompt-tuning's virtual tokens at
position 0, ahead of any padding: the gap between the arm's tokens and the text
they condition then differs from row to row, which measurably depressed the
arms and left the base model alone. Everything without virtual tokens can be
batched the ordinary way, and on the 2000-example test set that is 63 batches
instead of 232.

These tests pin both halves: the speedup happens where it is allowed, and the
attempt is refused where it is not.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mrd_evaluate", ROOT / "src" / "mrd_pruning" / "evaluate.py"
)


def _module():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from mrd_pruning import evaluate
    return evaluate


ev = _module()


class _Cfg:
    def __init__(self, n_virtual):
        self.num_virtual_tokens = n_virtual


class _Model:
    def __init__(self, n_virtual=0):
        if n_virtual:
            self.peft_config = {"default": _Cfg(n_virtual)}
        self.device = "cpu"


def _lengths():
    # Mirrors the real shape: many distinct lengths, few rows each.
    return [[0] * n for n in (5, 5, 3, 9, 3, 7, 5, 9, 1, 3)]


def test_virtual_tokens_detected_from_the_adapter_config():
    assert ev.has_virtual_tokens(_Model(n_virtual=100)) is True


def test_no_adapter_means_no_virtual_tokens():
    assert ev.has_virtual_tokens(_Model()) is False


def test_adapter_with_zero_virtual_tokens_is_not_flagged():
    """A LoRA-style adapter has a config but adds no positions."""
    assert ev.has_virtual_tokens(_Model(n_virtual=0)) is False


def test_equal_length_batches_hold_exactly_one_length():
    for batch in ev._equal_length_batches(_lengths(), 32):
        assert len({len(_lengths()[i]) for i in batch}) == 1


def test_padded_batches_cover_every_prompt_once():
    seen = [i for b in ev._padded_batches(_lengths(), 4) for i in b]
    assert sorted(seen) == list(range(len(_lengths())))


def test_padded_batches_are_full_where_equal_length_ones_are_not():
    seqs = _lengths()
    equal = list(ev._equal_length_batches(seqs, 4))
    padded = list(ev._padded_batches(seqs, 4))
    assert len(padded) < len(equal), "padding must reduce the batch count"
    assert len(padded) == 3, "10 prompts, batch 4 -> 4+4+2"


def test_padded_batches_sort_by_length_to_keep_padding_small():
    seqs = _lengths()
    for batch in ev._padded_batches(seqs, 4):
        widths = [len(seqs[i]) for i in batch]
        assert widths == sorted(widths)
    # And the spread inside a batch stays narrow: sorting is the whole point.
    spans = [max(len(seqs[i]) for i in b) - min(len(seqs[i]) for i in b)
             for b in ev._padded_batches(seqs, 4)]
    assert max(spans) <= 4


def test_padding_is_refused_on_an_arm_with_virtual_tokens():
    cfg = ev.GenerationConfig(allow_padding=True)
    with pytest.raises(ValueError, match="virtual tokens"):
        ev.generate_responses(_Model(n_virtual=100), object(), ["a"], None, config=cfg)


def test_config_records_whether_padding_was_used():
    """The summary has to say how a number was produced."""
    assert ev.GenerationConfig(allow_padding=True).as_dict()["allow_padding"] is True
    assert ev.GenerationConfig().as_dict()["allow_padding"] is False


def test_empty_prompt_list_still_raises_before_the_padding_check():
    with pytest.raises(ValueError, match="no prompts"):
        ev.generate_responses(_Model(), object(), [], None,
                              config=ev.GenerationConfig(allow_padding=True))


class _OOMOnce:
    """Fails the first generate with OOM, then succeeds - a busy neighbour."""

    def __init__(self, fail_above: int = 1):
        self.fail_above = fail_above
        self.widths: list[int] = []
        self.device = "cpu"
        self.calls = 0

    def generate(self, *, input_ids, attention_mask, max_new_tokens, do_sample,
                 pad_token_id):
        self.calls += 1
        if input_ids.shape[0] > self.fail_above:
            raise torch.OutOfMemoryError("CUDA out of memory")
        self.widths.append(int(input_ids.shape[0]))
        grown = torch.zeros((input_ids.shape[0], input_ids.shape[1] + 1),
                            dtype=torch.long)
        grown[:, :input_ids.shape[1]] = input_ids
        return grown


class _Tok:
    pad_token_id = 0
    eos_token_id = 0
    # render_chat refuses a tokenizer without one, and rightly: a hand-rolled
    # prompt format would not match what the arms were trained on.
    chat_template = "{{ messages[-1]['content'] }}"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1] * (len(text) % 5 + 1)}

    def apply_chat_template(self, messages, tokenize=False, **kw):
        return messages[-1]["content"]

    def batch_decode(self, rows, skip_special_tokens=False):
        return ["NONE"] * len(rows)


def test_a_batch_that_runs_out_of_memory_is_split_and_retried(monkeypatch):
    """An hour of finished work must not be lost to one unlucky batch."""
    model = _OOMOnce(fail_above=2)
    prompts = [f"comment {i}" for i in range(8)]
    cfg = ev.GenerationConfig(batch_size=8, allow_padding=True, max_new_tokens=1)
    out, raw = ev.generate_responses(model, _Tok(), prompts, None, config=cfg)
    assert len(out) == len(prompts), "every prompt still gets an answer"
    assert max(model.widths) <= 2, "the retry must use smaller batches"
    assert model.calls > 1, "the first attempt failed and was retried"


def test_a_single_row_that_cannot_fit_is_a_real_shortage():
    """Splitting stops at one: below that it is not contention, it is capacity."""
    model = _OOMOnce(fail_above=0)
    cfg = ev.GenerationConfig(batch_size=2, allow_padding=True, max_new_tokens=1)
    with pytest.raises(torch.OutOfMemoryError):
        ev.generate_responses(model, _Tok(), ["a", "b"], None, config=cfg)
