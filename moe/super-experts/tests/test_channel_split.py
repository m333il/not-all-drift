"""A reasoning model's trace must not reach the label parser.

gpt-oss writes its reasoning and its answer into separate channels of one token
stream. Decoding with ``skip_special_tokens`` drops the channel headers but keeps
the trace's text, so a scorer that parses that string sees prose in front of the
labels and scores every reasoning answer zero -- silently, as an unparsable rate
rather than as an error.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from score_arms import split_channels


class FakeTokenizer:
    """Decodes a list of string pieces; ``skip_special_tokens`` drops the markers."""

    MARKERS = ("<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|return|>")

    def decode(self, ids, skip_special_tokens=False):
        text = "".join(ids)
        if skip_special_tokens:
            for marker in self.MARKERS:
                text = text.replace(marker, "")
        return text


HARMONY = ["<|channel|>", "analysis", "<|message|>",
           "The comment calls someone a liar, so toxicity and insult apply.", "<|end|>",
           "<|start|>", "assistant", "<|channel|>", "final", "<|message|>",
           "toxicity, insult", "<|return|>"]


def test_harmony_answer_excludes_the_trace():
    answer, analysis = split_channels(FakeTokenizer(), HARMONY)
    assert answer == "toxicity, insult"
    assert analysis.startswith("The comment calls someone a liar")


def test_the_plain_decode_would_have_fed_the_trace_to_the_parser():
    # The bug this guards against, stated as a fact about the old path.
    plain = FakeTokenizer().decode(HARMONY, skip_special_tokens=True).strip()
    assert plain != "toxicity, insult"
    assert "The comment calls someone a liar" in plain


def test_a_model_without_channels_is_unchanged():
    answer, analysis = split_channels(FakeTokenizer(), ["toxicity", ", ", "insult"])
    assert answer == "toxicity, insult"
    assert analysis == ""


def test_the_qwen_leg_keeps_the_vendored_renderer():
    # Pins are opt-in: with none, load_contract must hand back the dense/discrete
    # function, so the leg that already passes the contract gate is untouched.
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from interpretability_gepa.prompts import apply_chat_template
    from se_gepa.arms import load_contract

    assert load_contract()[3] is apply_chat_template
    assert load_contract({"date": "2026-09-07"})[3] is not apply_chat_template


def test_pins_reach_the_template():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from se_gepa.arms import load_contract

    seen = {}

    class Recorder:
        def apply_chat_template(self, messages, **options):
            seen.update(options)
            return "rendered"

    class Rendered:
        messages = [{"role": "user", "content": "x"}]

    _labels, _seeds, _render, apply = load_contract(
        {"reasoning_effort": "low", "date": "2026-09-07"})
    assert apply(Recorder(), Rendered(), non_thinking=True) == "rendered"
    assert seen["reasoning_effort"] == "low"
    assert seen["strftime_now"]("%Y-%m-%d") == "2026-09-07"
