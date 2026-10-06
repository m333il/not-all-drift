"""Batched generation and scoring, with the guards that stop a broken run.

The rule this module enforces: a measurement that failed must not be able to
produce a summary file. The transplant run wrote thirteen "finished" configs
whose predictions were empty 89-100% of the time, and a whole day was spent
reading routing conclusions off them.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Sequence

from .task import ScoreSummary, NonePolicy, answer_text, render_chat, score_batch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding settings. Greedy by default: this is a measurement, not a demo."""

    max_new_tokens: int = 24
    batch_size: int = 32
    do_sample: bool = False
    log_every: int = 64
    # Off by default: the arms are scored with reasoning suppressed so the two
    # backbones answer under the same rules. Turning it on is the model's native
    # mode and needs a budget an order of magnitude larger.
    reasoning: bool = False
    # How much it may think, when it thinks at all. The template's own default
    # is "medium"; the published runs pin "low", and the gap is worth 0.075 F1
    # on base. Only the two text arms reason, so this is inert elsewhere.
    reasoning_effort: str = "medium"
    # Equal-length batching is mandatory for a prompt-tuning arm and merely
    # wasteful for everything else: the 2000-example test set has 227 distinct
    # prompt lengths, so groups average nine rows and a batch of 32 is never
    # full - 232 batches where padding would need 63. Left padding on a model
    # with no virtual tokens is the ordinary way to batch generation and costs
    # nothing in correctness, so arms without an adapter may opt in. The guard
    # in `generate_responses` refuses it when virtual tokens are present.
    allow_padding: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "do_sample": self.do_sample,
            "reasoning": self.reasoning,
            "reasoning_effort": self.reasoning_effort if self.reasoning else None,
            "allow_padding": self.allow_padding,
        }


@dataclass(frozen=True)
class QualityGates:
    """Limits above which a result is a bug report, not a data point.

    ``max_unparsable_rate`` catches the reasoning-mode failure: responses that
    contain neither NONE nor a label mean the budget went somewhere else. It is
    enforced only where the model is *intact* (``intervened=False``), because a
    heavily pruned model producing unusable text is the finding, not a defect -
    the first version of this gate threw away exactly those cells and left holes
    where the interesting part of the curve is.

    ``max_empty_pred_rate`` is looser for the same reason and only warns unless
    ``strict_empty`` is set.
    """

    max_unparsable_rate: float = 0.30
    max_empty_pred_rate: float = 0.95
    strict_empty: bool = False


def check_gates(
    summary: ScoreSummary,
    gates: QualityGates,
    *,
    context: str,
    intervened: bool = False,
) -> None:
    """``intervened`` means something was deliberately broken in the model.

    With an intervention in place a degenerate output is data: it is recorded,
    reported loudly, and left for the reader to judge. Without one, the same
    numbers mean the harness is misconfigured and the run must stop.
    """
    if summary.unparsable_rate > gates.max_unparsable_rate:
        if not intervened:
            raise RuntimeError(
                f"{context}: {summary.unparsable_rate:.1%} of responses have neither NONE "
                f"nor a label (limit {gates.max_unparsable_rate:.0%}) on an unmodified "
                "model. Check that the chat template was rendered with "
                "enable_thinking=False and that max_new_tokens is large enough - "
                "this is not a quality signal."
            )
        logger.warning(
            "%s: %.1f%% of responses are unusable text. The model is pruned here, so "
            "this is recorded as degradation rather than treated as a harness fault - "
            "read results.jsonl before quoting the F1.",
            context, 100 * summary.unparsable_rate,
        )
    if summary.empty_pred_rate > gates.max_empty_pred_rate:
        message = (
            f"{context}: {summary.empty_pred_rate:.1%} of predictions are empty "
            f"(limit {gates.max_empty_pred_rate:.0%})"
        )
        if gates.strict_empty:
            raise RuntimeError(message)
        logger.warning("%s - kept, but read the responses before trusting F1", message)
    if summary.mixed_none_rate > 0:
        logger.info(
            "%s: %.1f%% of responses mix NONE with real labels; the NONE policy decides "
            "how those score", context, 100 * summary.mixed_none_rate,
        )


@dataclass(frozen=True)
class EvalResult:
    summary: ScoreSummary
    rows: list[dict[str, object]]
    seconds: float

    def as_dict(self) -> dict[str, object]:
        return {**self.summary.as_dict(), "seconds": self.seconds}


def has_virtual_tokens(model) -> bool:
    """Whether this model prepends PEFT virtual tokens to every sequence.

    Prompt- and prefix-tuning put their tokens at position 0, which is what
    makes padded batches wrong for them. The check reads the adapter's own
    config rather than the arm's name, because the name is a label the caller
    chose and the config is what the forward pass will actually do.
    """
    configs = getattr(model, "peft_config", None) or {}
    for cfg in configs.values():
        if int(getattr(cfg, "num_virtual_tokens", 0) or 0) > 0:
            return True
    return False


def _equal_length_batches(encoded: Sequence[Sequence[int]], batch_size: int):
    """Index batches in which every sequence has the same length."""
    by_length: dict[int, list[int]] = {}
    for index, ids in enumerate(encoded):
        by_length.setdefault(len(ids), []).append(index)
    for length in sorted(by_length):
        group = by_length[length]
        for start in range(0, len(group), batch_size):
            yield group[start:start + batch_size]


def _padded_batches(encoded: Sequence[Sequence[int]], batch_size: int):
    """Index batches of near-equal length, for models that tolerate padding.

    Sorted by length first, so a batch spans the smallest length range it can
    and the padding stays near zero. On the 2000-example test set this turns
    232 ragged batches into 63 full ones - the prompts take 227 distinct
    lengths, so equal-length groups average nine rows and a batch of 32 is
    never full.
    """
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


def generate_responses(
    model,
    tokenizer,
    user_prompts: Sequence[str],
    system_prompt: str | None,
    *,
    config: GenerationConfig,
    golds: Sequence[Sequence[str]] | None = None,
    none_policy: NonePolicy = "lenient",
) -> tuple[list[str], list[str]]:
    """Greedy-decode one response per prompt, batched by equal length.

    Returns the parsed answers and the full decodes beside them. In reasoning
    mode the two differ: the answer is the `final` channel alone, the full
    decode still holds the `analysis` the model wrote first.

    Generation goes through ``model.generate``. An earlier hand-rolled decode
    loop rebuilt the attention mask as ``torch.ones`` and quietly fed pad tokens
    into the context of every short example in a batch; the fix took a day and
    a dedicated regression test. There is no reason to own that code here.

    **Batches hold one length only, so nothing is ever padded.** Left padding is
    the usual way to batch generation and it is wrong for a prompt-tuning arm:
    PEFT prepends the virtual tokens at position 0, ahead of the padding, so the
    sequence becomes ``[virtual][pad…][text]`` and the gap between the arm's
    tokens and the text it conditions differs from row to row. Measured on the
    published run's own inputs, reproducing its recorded outputs: 39/40 one at a
    time, 5/40 in padded batches of 8, 4/40 in padded batches of 32. With
    equal-length batches it is 21/24 - batching is fine, padding is not. The
    base model is barely affected, which is exactly why this hid for so long: it
    depresses the arms and leaves the reference alone.

    Grouping costs little. On the project's 2000-example test set the prompts
    take 227 distinct lengths, so the average group holds ~9 examples and a
    whole pass is 232 batches rather than 2000 single rows.
    """
    import torch

    if not user_prompts:
        raise ValueError("no prompts to generate for")

    padded = config.allow_padding
    if padded and has_virtual_tokens(model):
        raise ValueError(
            "allow_padding was asked for on a model with virtual tokens: PEFT "
            "prepends them at position 0, ahead of the padding, so the gap "
            "between the arm's tokens and its text would differ from row to row"
        )

    rendered = [render_chat(tokenizer, prompt, system_prompt, tokenize=False,
                            reasoning=config.reasoning,
                            reasoning_effort=config.reasoning_effort)
                for prompt in user_prompts]
    encoded = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in rendered]

    batches = list(
        _padded_batches(encoded, config.batch_size) if padded
        else _equal_length_batches(encoded, config.batch_size)
    )
    logger.info("%d prompts in %d %s batches", len(encoded), len(batches),
                "padded" if padded else "equal-length")

    responses: list[str | None] = [None] * len(user_prompts)
    # The full decode is kept beside the parsed answer. Scoring wants the answer
    # alone, but a routing map measured in reasoning mode has to count the
    # reasoning tokens too - and `answer_text` has already thrown them away.
    raw_responses: list[str | None] = [None] * len(user_prompts)
    done = n_logged = 0
    started = time.monotonic()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None and padded:
        # Only padded batches ever write this id; equal-length ones pass it to
        # `generate` where None is harmless. Failing here beats filling the
        # context with whatever `None` becomes downstream.
        raise ValueError("padded batches need a pad or eos token id; tokenizer has neither")

    pending = list(batches)
    while pending:
        rows = pending.pop(0)
        width = max(len(encoded[i]) for i in rows)
        # Left padding, so every row's last token is the one generation
        # continues from and the prompt keeps its own positions.
        ids = [[pad_id] * (width - len(encoded[i])) + list(encoded[i]) for i in rows]
        attn = [[0] * (width - len(encoded[i])) + [1] * len(encoded[i]) for i in rows]
        input_ids = torch.tensor(ids, device=model.device)
        attention_mask = torch.tensor(attn, device=model.device)
        try:
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=config.do_sample,
                    pad_token_id=pad_id,
                )
        except torch.OutOfMemoryError:
            # The card is shared. A neighbour's job can take the memory this
            # batch needs, and an hour of finished work is then thrown away
            # over one batch - measured: prompt-m500 died at example 229 of
            # 2000 with four other tenants on the card. Halving and retrying
            # costs a little throughput and saves the cell. A batch of one that
            # still cannot fit is a real shortage, and that one is raised.
            del input_ids, attention_mask
            torch.cuda.empty_cache()
            if len(rows) == 1:
                raise
            half = len(rows) // 2
            logger.warning(
                "I didn’t have enough memory for the batch %d x %d - split on %d and %d and repeat",
                len(rows), width, half, len(rows) - half,
            )
            pending.insert(0, rows[half:])
            pending.insert(0, rows[:half])
            continue
        # Every row of the batch is `width` wide after padding, generated or
        # not, so one slice serves the whole batch in both modes.
        new_tokens = generated[:, width:]
        # Both decodes: harmony models mark the answer with special tokens,
        # and stripping them welds the reasoning onto the answer. See
        # `answer_text`.
        clean = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        raw = tokenizer.batch_decode(new_tokens, skip_special_tokens=False)
        for i, r, c in zip(rows, raw, clean):
            responses[i] = answer_text(r, c)
            raw_responses[i] = r

        done += len(rows)
        if done - n_logged >= config.log_every or done == len(user_prompts):
            # Threshold, not modulo: with batch_size=32 a modulo check on a
            # 64-example step first fires at 800 and the run looks hung.
            n_logged = done
            rate = (time.monotonic() - started) / max(done, 1)
            logger.info("%d/%d generated | %.2fs/example",
                        done, len(user_prompts), rate)
            # Score what is finished, every time the progress line prints.
            #
            # Without this a cell says nothing about its quality until the last
            # row lands, and the run that prompted it took ten hours to report
            # F1 0.3300 with every answer unparsable - ten hours to learn what
            # the first sixty-four rows already knew. Speed is not a substitute:
            # a degenerate arm that babbles to the ceiling is slow, and one that
            # answers instantly with nothing is fast, and neither is visible in
            # seconds-per-example.
            #
            # The rows are scored in completion order rather than input order,
            # which is a sample of the shortest prompts first under equal-length
            # batching. It is a running indicator, not the cell's number - that
            # one is still computed over all rows at the end.
            if golds is not None:
                ready = [(r, golds[i]) for i, r in enumerate(responses)
                         if r is not None and i < len(golds)]
                if ready:
                    try:
                        partial, _ = score_batch([r for r, _ in ready],
                                                 [g for _, g in ready],
                                                 none_policy=none_policy)
                        logger.info(
                            "  intermediate on %d rows: F1 %.4f | empty %.1f%% "
                            "| unparsable %.1f%%",
                            partial.n, partial.f1_mean,
                            100 * partial.empty_pred_rate,
                            100 * partial.unparsable_rate)
                    except ValueError:
                        pass  # nothing scorable yet; the next mark will have some

    missing = [i for i, r in enumerate(responses) if r is None]
    if missing:
        raise RuntimeError(f"{len(missing)} prompts produced no response")
    return (
        [r for r in responses if r is not None],
        [r for r in raw_responses if r is not None],
    )


def evaluate(
    model,
    tokenizer,
    user_prompts: Sequence[str],
    golds: Sequence[Sequence[str]],
    system_prompt: str | None,
    *,
    config: GenerationConfig,
    gates: QualityGates,
    none_policy: NonePolicy = "lenient",
    context: str = "eval",
    intervened: bool = False,
) -> EvalResult:
    """Generate, score, and refuse to return a summary that fails the gates.

    ``intervened`` is passed through to :func:`check_gates`: a pruned model is
    allowed to produce garbage, an unmodified one is not.
    """
    if len(user_prompts) != len(golds):
        raise ValueError(f"{len(user_prompts)} prompts vs {len(golds)} gold lists")
    started = time.monotonic()
    responses, raw_responses = generate_responses(
        model, tokenizer, user_prompts, system_prompt, config=config,
        golds=golds, none_policy=none_policy,
    )
    summary, rows = score_batch(responses, golds, none_policy=none_policy)
    for row, raw in zip(rows, raw_responses):
        row["raw_response"] = raw
    check_gates(summary, gates, context=context, intervened=intervened)
    elapsed = time.monotonic() - started
    logger.info(
        "%s: F1 %.4f | exact %.4f | empty %.1f%% | n=%d | %.1fs",
        context, summary.f1_mean, summary.exact_mean,
        100 * summary.empty_pred_rate, summary.n, elapsed,
    )
    return EvalResult(summary=summary, rows=rows, seconds=elapsed)
