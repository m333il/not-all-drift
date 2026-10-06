"""Arm handling shared by the profiling and the scoring entry points.

An *arm* here is one way of adapting the frozen backbone that the v2 user-only
contract can render: a text arm changes the instruction inside the single user
message, a PEFT arm installs an adapter and leaves the text at the seed
instruction. Both the Super-Expert profile and the task score have to build
exactly the same sequences from the same specification, so that logic lives here
rather than in either script.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

SEED_KEY = "civil_comments"


def load_contract(pins=None):
    """The rendering contract, optionally with the chat template's pins.

    The vendored ``apply_chat_template`` passes no ``reasoning_effort`` and no
    ``strftime_now``. Qwen's template reads neither, so for that leg it renders
    exactly what the adapters were trained under. GPT-OSS's harmony preamble
    carries both a current date and a reasoning effort, so an unpinned render
    produces *today's* date -- the same token count as the archived one and
    different ids, which is what the contract gate caught. ``pins`` restates the
    values ``src/mrd/chat.py`` froze, and is absent for the Qwen leg so its path
    is unchanged.
    """
    from interpretability_gepa.datasets import CIVIL_LABELS
    from interpretability_gepa.prompts import SEED_INSTRUCTIONS, apply_chat_template, render_messages

    if not pins:
        return CIVIL_LABELS, SEED_INSTRUCTIONS, render_messages, apply_chat_template

    def pinned(tokenizer, rendered, *, non_thinking):
        options = {"tokenize": False, "add_generation_prompt": True}
        if non_thinking:
            options["enable_thinking"] = False
        if "reasoning_effort" in pins:
            options["reasoning_effort"] = pins["reasoning_effort"]
        if "date" in pins:
            options["strftime_now"] = lambda _format: pins["date"]
        return str(tokenizer.apply_chat_template(list(rendered.messages), **options))

    return CIVIL_LABELS, SEED_INSTRUCTIONS, render_messages, pinned


def resolve_instructions(arms, tokenizer, seeds):
    """Fill in every text arm's instruction, including the length-matched control.

    ``pad_to`` names another arm's instruction file and yields the seed
    instruction padded with task-inert filler to that file's token count. The
    project already knows that a prompt matched to GEPA's length reproduces most
    of its routing drift, so an arm comparison without this control cannot tell
    a Super-Expert effect from a prompt-length effect.
    """
    from interpretability_gepa.prompts import neutral_length_padding

    for arm in arms:
        if arm["kind"] != "text" or "instruction" in arm:
            continue
        if "instruction_key" in arm:
            arm["instruction"] = seeds[arm["instruction_key"]]
        elif "pad_to" in arm:
            arm["instruction"] = neutral_length_padding(
                tokenizer, seeds[SEED_KEY], Path(arm["pad_to"]).read_text().strip())
        else:
            arm["instruction"] = Path(arm["instruction_file"]).read_text().strip()


CONTRACT_MARKER = "Labels (use exact names)"


def check_instructions(tokenizer, arms, contract):
    """Refuse an instruction that already carries the harness's own contract.

    ``render_messages`` appends the label list, the text and the output contract
    after whatever instruction it is given. A GEPA run writes two artifacts: the
    component it optimised, and the *rendered* template with ``{text}`` left as a
    placeholder for reading. Passing the second one as an instruction renders the
    contract twice and leaves a literal ``{text}`` in the prompt -- a blank task
    in front of the real one. Qwen shrugged that off at a 0.999 parse rate while
    GPT-OSS fell to 0.461, so it cost one model's numbers and not the other's, and
    nothing in the pipeline noticed. This is the check that would have.
    """
    probe = "MRD_PROBE_TEXT"
    for arm in arms:
        if arm["kind"] != "text":
            continue
        rendered = contract[2](arm["instruction"], contract[0], probe).messages[0]["content"]
        count = rendered.count(CONTRACT_MARKER)
        if count != 1:
            raise ValueError(
                f"{arm['name']}: the rendered prompt states the output contract {count} times; "
                "the instruction already contains it, so this is the rendered template rather "
                "than the instruction component")
        for placeholder in ("{text}", "{instructions}"):
            if placeholder in rendered:
                raise ValueError(f"{arm['name']}: rendered prompt still contains {placeholder}")


def render(tokenizer, instruction, text, contract):
    labels, _seeds, render_messages, apply_chat_template = contract
    rendered = apply_chat_template(tokenizer, render_messages(instruction, labels, text), non_thinking=True)
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def check_contract(tokenizer, contract, rows, sample):
    """Re-render archived examples and require identical token ids.

    The arms were trained and evaluated under a specific rendering. If this
    script's rendering differs by even one token, every arm is being profiled on
    a sequence the adapter never saw, and the comparison is meaningless -- so
    this is a gate, not a diagnostic.
    """
    _labels, seeds, _render, _apply = contract
    by_id = {row["id"]: row for row in rows}
    checked = 0
    for entry in sample:
        row = by_id.get(entry["key"])
        if row is None:
            continue
        ids = render(tokenizer, seeds[SEED_KEY], row["text"], contract)
        if ids != entry["input_ids"]:
            raise RuntimeError(
                f"Rendering does not reproduce the archived contract for {entry['key']}: "
                f"{len(ids)} tokens against {len(entry['input_ids'])}"
            )
        checked += 1
    if not checked:
        raise RuntimeError("No archived example could be re-rendered; the contract went unchecked")
    return checked


def build_base(model_dir, device="cuda"):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=device,
        experts_implementation="grouped_mm", attn_implementation="eager",
    ).eval()


def peft_type_of(arm):
    return json.loads((Path(arm["dir"]) / "receipt.json").read_text())["peft_type"]


def group_arms(arms):
    """Text arms, then one group per PEFT type.

    peft 0.20.0 refuses to hold adapters of different types in one ``PeftModel``
    ("Cannot combine adapters with different peft types"), so prompt tuning and
    prefix tuning cannot share a wrapper. Neither type touches the base weights,
    so each group wraps the same already-loaded backbone in turn and the 30B
    weights are read once.
    """
    text = [arm for arm in arms if arm["kind"] == "text"]
    groups: dict[str, list] = {}
    for arm in arms:
        if arm["kind"] == "peft":
            groups.setdefault(peft_type_of(arm), []).append(arm)
    return text, groups


def wrap(base, group):
    from peft import PeftModel

    first, *rest = group
    model = PeftModel.from_pretrained(base, str(Path(first["dir"]) / "adapter"),
                                      adapter_name=first["name"], is_trainable=False)
    for arm in rest:
        model.load_adapter(str(Path(arm["dir"]) / "adapter"), adapter_name=arm["name"])
    return model.eval()


def device_of(model):
    return next(model.parameters()).device


def virtual_offset(arm):
    """Positions the arm inserts before the first text token.

    Prompt tuning prepends embeddings, so the offset is its virtual token count.
    Prefix tuning writes into ``past_key_values`` and inserts nothing into the
    stream the experts see, so its offset is zero even though its activations
    differ from the base arm's everywhere.
    """
    if arm["kind"] != "peft":
        return 0
    receipt = json.loads((Path(arm["dir"]) / "receipt.json").read_text())
    return receipt["num_virtual_tokens"] if receipt["peft_type"] == "PROMPT_TUNING" else 0


