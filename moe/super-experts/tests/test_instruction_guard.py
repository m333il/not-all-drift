"""An instruction must not already contain the harness's output contract.

A GEPA run writes two artifacts: the component it optimised, and the rendered
template with ``{text}`` left as a placeholder for reading. Passing the second as
an instruction renders the contract twice and leaves a literal ``{text}`` in the
prompt -- a blank task in front of the real one. That happened, and nothing in
the pipeline noticed: Qwen shrugged it off at a 0.999 parse rate while GPT-OSS
fell to 0.461, so it corrupted one model's numbers and left the other's looking
fine.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from se_gepa.arms import CONTRACT_MARKER, check_instructions

CIVIL = ("toxicity", "obscene", "threat", "insult", "identity_attack")


class Rendered:
    def __init__(self, content):
        self.messages = ({"role": "user", "content": content},)


def render_messages(instruction, labels, text):
    return Rendered(f"{instruction}\n"
                    f"Labels (use exact names): {', '.join(labels)}\n"
                    f"Text:\n{text}\n"
                    "Return every applicable label in the order listed, separated by commas. "
                    "If no label applies, return exactly NONE. Return no other text.\n"
                    "Answer:")


CONTRACT = (CIVIL, {}, render_messages, None)


def test_a_plain_instruction_passes():
    arms = [{"name": "gepa", "kind": "text", "instruction": "Classify the text carefully."}]
    check_instructions(None, arms, CONTRACT)


def test_the_rendered_template_is_refused():
    template = render_messages("Classify the text carefully.", CIVIL, "{text}").messages[0]["content"]
    arms = [{"name": "gepa", "kind": "text", "instruction": template}]
    with pytest.raises(ValueError, match="states the output contract 2 times"):
        check_instructions(None, arms, CONTRACT)


def test_a_leftover_placeholder_is_refused():
    arms = [{"name": "gepa", "kind": "text", "instruction": "Classify {text} carefully."}]
    with pytest.raises(ValueError, match=r"still contains \{text\}"):
        check_instructions(None, arms, CONTRACT)


def test_peft_arms_are_skipped():
    check_instructions(None, [{"name": "prompt-m100", "kind": "peft", "cell": "x"}], CONTRACT)


def test_the_shipped_gepa_instructions_pass():
    arms = []
    for path in sorted(ROOT.glob("prompts/*-instructions.txt")):
        arms.append({"name": path.stem, "kind": "text", "instruction": path.read_text().strip()})
    assert arms, "no instruction artifacts to check"
    check_instructions(None, arms, CONTRACT)
    assert all(CONTRACT_MARKER not in arm["instruction"] for arm in arms)
