"""civil_comments multilabel task: one prompt renderer, one parser, one metric.

Every past failure of this harness that reached a written report came from this
file's territory, so the invariants are enforced here rather than at call sites:

* ``enable_thinking=False`` - Qwen3's chat template turns reasoning on by
  default. With it on, the model opens ``<think>`` and the 24-token budget is
  gone before any label appears, which collapses F1 to ~0.33 for every arm and
  looks like a routing result rather than a formatting bug (30-08-2026).
* ``reasoning_effort="low"`` - the same failure, on the other model, through a
  different knob. gpt-oss's harmony template does not know ``enable_thinking``
  at all: it reads ``reasoning_effort`` and defaults to ``"medium"``. Our flag
  was therefore silently ignored, the model spent all 24 tokens in the
  ``analysis`` channel, and base came out with 82.5% unparsable responses
  (11-09-2026). ``"low"`` is not a tuning choice - it is the value recorded in
  the published runs' own training manifest, so the arms are evaluated under the
  rendering they were trained on. It only *shortens* the reasoning, so the
  generation prompt additionally opens the ``final`` channel - see
  :func:`render_chat`. The protocol is "answer immediately, 24 tokens" for both
  models, and gpt-oss has to meet it the same way Qwen does.
* One renderer only. When two functions render the same prompt (generation and
  segment labelling), they drift apart and the second silently mislabels every
  example. Anything that needs the prompt calls :func:`render_chat`.
* ``NONE`` mixed with real labels is a policy, not an accident: arms append
  ``", NONE"`` to otherwise correct answers. Voiding those answers zeroed 802 of
  2000 prompt-tuning responses in an earlier run. The policy is explicit and
  reported, never implicit.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

logger = logging.getLogger(__name__)

LABELS: tuple[str, ...] = ("toxicity", "obscene", "threat", "insult", "identity_attack")
NONE_TOKEN = "none"
ANSWER_PREFIX = "answer:"
# How every prompt contract introduces the roster of label names, and how the
# model quotes it back when it echoes instead of answering.
_LABEL_ROSTER_PREFIX = "labels (use exact names):"

NonePolicy = Literal["lenient", "strict"]

FINAL_CHANNEL = "<|channel|>final<|message|>"
# A pruned model damages the channel header before it damages the answer. At
# −75% gpt-oss writes `<|channel|>final message<|message|>toxicity, insult` -
# the channel is still the final one, its name has picked up a word - and an
# exact match on FINAL_CHANNEL reads that as "no final channel" and throws the
# answer away. Measured on prefix-m100 at −75%: 404 rows unparsable, 143 of them
# naming labels the run then scored as silence. The bias is not random, it lands
# on the heaviest pruning levels, which is exactly where the curve is read.
_FINAL_CHANNEL_RE = re.compile(r"<\|channel\|>final[^<]*<\|message\|>")
_CHANNEL_MARKER = "<|channel|>"
_END_MARKERS = ("<|return|>", "<|end|>", "<|start|>")
_WARNED_TEMPLATES: set[str] = set()

SEED_SYSTEM_PROMPT = (
    "You are a careful assistant. Answer the user's request, and follow every "
    "formatting and output constraint stated in the request exactly."
)

USER_TEMPLATE = (
    "Text:\n{comment}\n\n"
    "Labels (use exact names): toxicity, obscene, threat, insult, identity_attack\n"
    "Return every applicable label in the order listed, separated by commas. "
    "If no label applies, return exactly NONE. Return no other text.\n"
    "Answer:"
)

# The v25 ("user-only") series is a different prompt contract, taken verbatim
# from the runs' own `prompt_contract.json` (id `civil_comments_seed_v2`):
# "first and only user message; no system message". Three things differ from
# v24, and all three change what the arm sees:
#
#   * the instruction leads, where v24 puts it in a system turn;
#   * the label list comes *before* the text, where v24 puts the text first;
#   * there is no system instruction at all - only harmony's own identity block.
#
# An arm evaluated under the other contract is being shown a prompt it never
# trained on, which reads as a weak method rather than as a wrong renderer.
V25_USER_TEMPLATE = (
    "{instructions}\n"
    "Labels (use exact names): toxicity, obscene, threat, insult, identity_attack\n"
    "Text:\n{comment}\n"
    "Return every applicable label in the order listed, separated by commas. "
    "If no label applies, return exactly NONE. Return no other text.\n"
    "Answer:"
)

# The seed instruction the PEFT arms of that series trained with; a GEPA arm
# substitutes its optimised instruction here instead.
V25_SEED_INSTRUCTION = (
    "You are a text classifier. Classify the text using the provided labels. "
    "More than one label may apply."
)


@dataclass(frozen=True)
class ParseResult:
    """Parsed prediction plus the audit flags a summary has to report."""

    labels: tuple[str, ...]
    saw_none: bool
    saw_labels: bool
    mixed_none: bool  # NONE together with real labels - the policy-sensitive case
    unparsable: bool  # neither NONE nor any known label anywhere in the response


def render_user_prompt(comment: str) -> str:
    """The user turn, byte-identical to the one the published runs used."""
    return USER_TEMPLATE.format(comment=comment)


def render_user_prompt_v25(comment: str, instructions: str | None = None) -> str:
    """The user turn of the v25 contract, instruction included.

    ``instructions`` defaults to the series' seed instruction, which is what its
    PEFT arms trained against; a GEPA arm passes its optimised instruction.
    """
    return V25_USER_TEMPLATE.format(
        instructions=(instructions or V25_SEED_INSTRUCTION).rstrip("\n"),
        comment=comment,
    )


def render_chat(
    tokenizer,
    user_prompt: str,
    system_prompt: str | None,
    *,
    tokenize: bool = True,
    reasoning: bool = False,
    reasoning_effort: str = "medium",
) -> list[int] | str:
    """Render one chat prompt. The only place a chat template is applied.

    ``system_prompt=None`` omits the system turn entirely, which is what the
    PEFT arms were trained with. Passing an empty string instead would add an
    empty system turn and shift every downstream position by a few tokens.

    ``reasoning=True`` leaves the model free to think: the template is asked for
    reasoning rather than against it, and the generation prompt stops before any
    channel is chosen, so a harmony model opens ``analysis`` on its own. This is
    the model's native mode - gpt-oss has no non-reasoning setting - and it
    needs a far larger token budget than the suppressed default.

    ``reasoning_effort`` says *how much* it may think, and the default is the
    template's own ``medium``. That default turned out to be a measurement
    choice nobody made: the published runs pin ``low``, and at ``medium`` the
    same base model writes 11037 characters of ``analysis`` against their 374
    and scores 0.6998 against their 0.6249. It only matters where the model
    actually reasons - the six PEFT arms answer in 79 characters either way and
    reproduce the training manifest to four decimals - so this is a knob for the
    two text arms, ``base`` and ``gepa``, and a no-op everywhere else.
    """
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "tokenizer has no chat_template; refusing to fall back to a hand-rolled "
            "prompt format, which would not match the trained arms"
        )
    template_kwargs = (
        reasoning_on_kwargs(tokenizer, reasoning_effort)
        if reasoning else reasoning_off_kwargs(tokenizer)
    )
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=True,
        **template_kwargs,
    )
    if reasoning:
        # Stop before the channel marker: choosing it is the model's to make.
        if tokenize:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **template_kwargs,
            )
            return list(
                tokenizer(pin_template_date(text), add_special_tokens=False)["input_ids"]
            )
        return pin_template_date(rendered)
    if not speaks_in_channels(tokenizer):
        return rendered
    # Open the `final` channel ourselves so the model answers immediately.
    # Qwen is measured with reasoning off and a 24-token budget, and gpt-oss has
    # to be measured the same way or the two are not comparable. There is no
    # setting that switches harmony's reasoning off - `reasoning_effort="low"`
    # only shortens it - so the generation prompt is left at the point where the
    # answer begins. Everything else is a workaround for reasoning we did not
    # want: a 512-token budget, and truncated responses scored as zero.
    if tokenize:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **reasoning_off_kwargs(tokenizer),
        )
        return list(tokenizer(pin_template_date(text) + FINAL_CHANNEL,
                              add_special_tokens=False)["input_ids"])
    return pin_template_date(rendered) + FINAL_CHANNEL


def speaks_in_channels(tokenizer) -> bool:
    """Whether this model's chat format uses harmony channels."""
    return _CHANNEL_MARKER in (getattr(tokenizer, "chat_template", "") or "")


# harmony writes today's date into its system block via `strftime_now`, so the
# same evaluation run on two days renders two different prompts. Both published
# series recorded the date they trained under, and pinning it here makes our
# prompt byte-identical to theirs instead of drifting daily.
TEMPLATE_DATE = "2026-09-07"
_DATE_LINE = re.compile(r"^Current date: \d{4}-\d{2}-\d{2}$", re.MULTILINE)


def pin_template_date(text: str, date: str = TEMPLATE_DATE) -> str:
    """Freeze the template's self-reported date. No date line, no change."""
    return _DATE_LINE.sub(f"Current date: {date}", text)


def reasoning_on_kwargs(tokenizer, effort: str = "medium") -> dict[str, object]:
    """Template arguments that let the model reason, per model family.

    The mirror of :func:`reasoning_off_kwargs`, and it reads the template the
    same way rather than matching a model name. A model whose template offers
    neither knob gets an empty dict: Qwen3-Instruct does not reason at all, so
    asking it to is not an error, it simply has no effect.
    """
    template = getattr(tokenizer, "chat_template", "") or ""
    kwargs: dict[str, object] = {}
    if "enable_thinking" in template:
        kwargs["enable_thinking"] = True
    if "reasoning_effort" in template:
        kwargs["reasoning_effort"] = effort
    return kwargs


def reasoning_off_kwargs(tokenizer) -> dict[str, object]:
    """Template arguments that suppress the reasoning channel, per model family.

    The two families spell this differently and each silently ignores the
    other's spelling, which is exactly how a formatting bug gets read as a
    routing result. The template text itself decides, rather than a model-name
    match, so a renamed or re-uploaded checkpoint cannot fall through.

    ``reasoning_effort="low"`` does not switch the analysis channel off the way
    ``enable_thinking=False`` does - gpt-oss can still open one. The generation
    budget has to accommodate that; see ``GenerationConfig.max_new_tokens``.
    """
    template = getattr(tokenizer, "chat_template", "") or ""
    kwargs: dict[str, object] = {}
    if "enable_thinking" in template:
        kwargs["enable_thinking"] = False
    if "reasoning_effort" in template:
        kwargs["reasoning_effort"] = "low"  # the value the published runs trained with
    if not kwargs and template not in _WARNED_TEMPLATES:
        # Once per template, not once per example: this is called for every one
        # of 2000 prompts and would otherwise bury the run log. Qwen3-Instruct
        # legitimately has neither knob - it does not reason - so the warning is
        # a prompt to check, not evidence of a fault.
        _WARNED_TEMPLATES.add(template)
        logger.warning(
            "chat template knows neither enable_thinking nor reasoning_effort; "
            "if this model reasons by default the token budget will be spent on it"
        )
    return kwargs


def answer_text(raw: str, clean: str) -> str:
    """The span of a response that is the model's answer.

    ``raw`` keeps the special tokens, ``clean`` is the same decode with them
    stripped. Ordinary chat models answer in ``clean`` directly. gpt-oss speaks
    the harmony format and emits ``analysis`` before ``final``; stripping the
    specials welds the two together (``"analysisWe need to label…finalNONE"``)
    and the answer can no longer be told from the reasoning.

    That is not merely lossy. The model reasons *about the label names*, in a
    comma-separated list - "no toxicity, obscene, threat, insult,
    identity_attack" - so a parser reading the reasoning harvests three labels
    the model never predicted, and reports them as a confident prediction rather
    than as a parse failure (measured 11-09-2026).

    So when a response speaks in channels, only ``final`` counts. A response
    whose budget ran out mid-analysis has no ``final`` and yields ``""``, which
    scores as unparsable - the honest outcome, and loudly visible in the
    unparsable rate rather than silently wrong.
    """
    if _CHANNEL_MARKER not in raw:
        return clean
    # The last one: a response may open several channels before finishing. The
    # name is matched loosely (`final`, `final message`, …) because a degenerate
    # header still marks the same channel, while the `analysis` channel - the
    # one that must never be read as the answer - cannot match this pattern.
    opens = list(_FINAL_CHANNEL_RE.finditer(raw))
    if not opens:
        return ""
    tail = raw[opens[-1].end():]
    for marker in _END_MARKERS:
        tail = tail.split(marker)[0]
    return tail


def _echoes_the_label_list(line: str) -> bool:
    """Is this line the prompt's own roster reproduced verbatim?

    Only the exact roster counts - the five names, in the order the prompt
    lists them. gpt-oss reproduces it word for word and then goes on to repeat
    the input under ``Text:``; that is an echo and carries no answer. A line
    that borrows the same heading but names its own subset ("Labels (use exact
    names): toxicity, insult") is an answer in the prompt's own dress: 153 such
    rows on gepa-n500, none of them followed by ``Text:``, and their label sets
    track the truth. Reading breadth alone would throw those away.
    """
    low = line.strip().lower()
    if not low.startswith(_LABEL_ROSTER_PREFIX):
        return False
    after = low[len(_LABEL_ROSTER_PREFIX):]
    return [part.strip() for part in after.split(",")] == list(LABELS)


def parse_response(response: str) -> ParseResult:
    """Split the first line on commas and keep known label names.

    Only the first line is read: a model that keeps talking after the answer is
    not punished for verbosity, the same rule the binary baseline used.

    Callers hand this the answer span, not the raw response; see
    :func:`answer_text`.

    A leading ``Answer:`` is stripped. The user turn ends with ``Answer:`` and
    an untuned model often echoes it - 153 of 2000 gpt-oss base responses were
    ``"Answer: NONE"`` or ``"Answer: insult"``, correct answers that the parser
    threw away as unreadable (11-09-2026). Since the echo falls on the base far
    more than on a tuned arm, leaving it in place does not merely add noise, it
    biases the base down by 7.7 points against every arm it is compared with.
    Only this one prefix is stripped, and only at the start: anything looser
    would start inventing answers the model did not give.
    """
    lines = [line for line in response.strip().split("\n") if line.strip()]
    first_line = lines[0] if lines else ""
    if _echoes_the_label_list(first_line):
        # The model reproduced the instruction instead of answering it. That
        # line names every label, so reading it as a prediction awards all five
        # at once: on gepa-n1000 it invented 492 `threat` predictions against 72
        # true ones and dropped precision to 0.48. An echo is a missing answer,
        # not a confident one.
        return ParseResult(
            labels=(), saw_none=False, saw_labels=False,
            mixed_none=False, unparsable=True,
        )
    if first_line.lower().lstrip().startswith(_LABEL_ROSTER_PREFIX):
        # Not the verbatim roster, or the check above would have caught it: the
        # model answered under the prompt's own heading.
        first_line = first_line.lstrip()[len(_LABEL_ROSTER_PREFIX):]
    if first_line.lower().lstrip().startswith(ANSWER_PREFIX):
        first_line = first_line.lstrip()[len(ANSWER_PREFIX):]
        # `Answer:` alone on its line, the answer on the next one. GEPA's
        # optimised prompt teaches exactly this layout, and dropping it threw
        # away 458 of 2000 responses on gepa-n1000 - every one of them a valid
        # answer - which is most of why GEPA read worse than the base.
        if not first_line.strip() and len(lines) > 1:
            first_line = lines[1]
    tokens = [t.strip().lower().strip(".") for t in first_line.split(",")]
    found = [t for t in tokens if t in LABELS]
    saw_none = NONE_TOKEN in tokens
    saw_labels = bool(found)
    ordered = tuple(sorted(set(found), key=LABELS.index))
    return ParseResult(
        labels=ordered,
        saw_none=saw_none,
        saw_labels=saw_labels,
        mixed_none=saw_none and saw_labels,
        unparsable=not saw_none and not saw_labels,
    )


def predicted_labels(parsed: ParseResult, none_policy: NonePolicy) -> tuple[str, ...]:
    """Apply the NONE policy to a parse result.

    ``lenient`` (default): NONE alongside real labels is ignored, the labels win.
    ``strict``: any NONE voids the prediction. This reproduces the harness that
    produced the published ``*_score`` fields and exists only for regression
    checks against them.
    """
    if none_policy == "strict":
        return () if parsed.saw_none else parsed.labels
    if none_policy == "lenient":
        return parsed.labels
    raise ValueError(f"unknown none_policy {none_policy!r}")


def f1_sample(pred: Sequence[str], true: Sequence[str]) -> float:
    """Sample-level F1, empty-aware: two empty sets agree, one empty set does not."""
    p, t = set(pred), set(true)
    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0
    return 2 * len(p & t) / (len(p) + len(t))


def exact_match(pred: Sequence[str], true: Sequence[str]) -> bool:
    return set(pred) == set(true)


@dataclass(frozen=True)
class ScoreSummary:
    """Aggregate scores plus the counters that make a bad run visible."""

    n: int
    f1_mean: float
    exact_mean: float
    empty_pred_rate: float
    unparsable_rate: float
    mixed_none_rate: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "f1_mean": self.f1_mean,
            "exact_mean": self.exact_mean,
            "empty_pred_rate": self.empty_pred_rate,
            "unparsable_rate": self.unparsable_rate,
            "mixed_none_rate": self.mixed_none_rate,
        }


def score_batch(
    responses: Iterable[str],
    golds: Iterable[Sequence[str]],
    *,
    none_policy: NonePolicy = "lenient",
) -> tuple[ScoreSummary, list[dict[str, object]]]:
    """Score responses against gold label lists, returning per-example rows too."""
    rows: list[dict[str, object]] = []
    f1s: list[float] = []
    exacts: list[bool] = []
    n_empty = n_unparsable = n_mixed = 0
    for response, gold in zip(responses, golds):
        parsed = parse_response(response)
        pred = predicted_labels(parsed, none_policy)
        f1 = f1_sample(pred, gold)
        exact = exact_match(pred, gold)
        f1s.append(f1)
        exacts.append(exact)
        n_empty += int(not pred)
        n_unparsable += int(parsed.unparsable)
        n_mixed += int(parsed.mixed_none)
        rows.append({
            "response": response,
            "predicted_labels": list(pred),
            "true_labels": list(gold),
            "f1": f1,
            "exact": exact,
            "mixed_none": parsed.mixed_none,
            "unparsable": parsed.unparsable,
        })
    n = len(f1s)
    if n == 0:
        raise ValueError("scored zero examples - refusing to report a summary")
    summary = ScoreSummary(
        n=n,
        f1_mean=sum(f1s) / n,
        exact_mean=sum(exacts) / n,
        empty_pred_rate=n_empty / n,
        unparsable_rate=n_unparsable / n,
        mixed_none_rate=n_mixed / n,
    )
    if summary.unparsable_rate > 0.5:
        logger.warning(
            "%.1f%% of responses contain neither NONE nor any known label - "
            "check max_new_tokens and enable_thinking before reading these scores",
            100 * summary.unparsable_rate,
        )
    return summary, rows
