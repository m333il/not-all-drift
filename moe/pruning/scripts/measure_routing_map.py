#!/usr/bin/env python3
"""Measure where every token is routed, per arm and per prompt stage.

Produces the routing map that pruning decisions are made from: one
``expert_counts.npz`` per arm, in the same layout as the published files, so
existing analysis code reads it unchanged.

Three things this has to get right:

**One shared answer.** Every arm is run over the *same* answer text, produced
once by the base arm. Otherwise the counters mix "routes differently" with
"wrote different text" and the two cannot be separated afterwards.

**Stage labels in capture coordinates.** Prompt-tuning prepends
``num_virtual_tokens`` positions that exist in the routed sequence but not in
``input_ids``; labelling against ``input_ids`` shifts every label by 100
positions. The offset is applied explicitly and the length is asserted.

**Speed.** The counting happens on the GPU. A first version moved each layer's
top-k indices to CPU and used ``np.add.at`` per stage: 48 device syncs and 288
mask rebuilds per example, 30 seconds each, 16 hours per arm. Here every gate
hook does one ``scatter_add_`` into a resident ``[layers, stages, experts]``
tensor, and nothing crosses the PCIe bus until the run ends.

    uv run scripts/measure_routing_map.py --arm base --n-examples 2000 \\
        --data data/test.jsonl --answers routing_maps/reference_answers.jsonl \\
        --out routing_maps/base
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

# A GEPA sequence runs to 6.5k tokens and gpt-oss has no SDPA path, so eager
# attention asks for one ~8 GiB buffer per forward. With the default allocator
# the run then dies holding 19 GiB "reserved but unallocated" - fragmentation,
# not a shortage. Expandable segments hand that back. Set before torch is
# imported, and only when the caller has not chosen their own policy.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
from typing import Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mrd_pruning.arms import ArmSpec, load_arm, provenance  # noqa: E402
from mrd_pruning.masking import discover_gates  # noqa: E402
from mrd_pruning.task import (  # noqa: E402
    FINAL_CHANNEL, SEED_SYSTEM_PROMPT, V25_SEED_INSTRUCTION, render_chat,
    render_user_prompt, render_user_prompt_v25,
)

logger = logging.getLogger("routing_map")

# Fixed order so the stage axis means the same thing in every file.
# "reasoning" holds the analysis channel a harmony model writes before its
# answer; it is empty unless the map was measured with --reasoning.
STAGES = ("system", "virtual", "comment", "question", "template",
          "reasoning", "answer")
STAGE_ID = {name: i for i, name in enumerate(STAGES)}
PAD_ID = -1  # padded positions are counted nowhere


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True,
                   choices=("base", "gepa", "prompt_tuning", "prefix_tuning"))
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--gepa-prompt", type=Path, default=None)
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--revision", default="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--n-examples", type=int, default=2000)
    p.add_argument("--answers", type=Path, required=True,
                   help="jsonl of shared reference answers (field 'answer')")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--oom-retries", type=int, default=4,
                   help="how many times to retry a batch that ran out of "
                        "memory; the card is shared, so a neighbour's job can "
                        "take the room between claiming it and using it")
    p.add_argument("--oom-wait", type=float, default=45.0,
                   help="seconds to wait before retrying an out-of-memory batch")
    p.add_argument("--max-batch-tokens", type=int, default=12288,
                   help="cap a batch by tokens as well as by rows; long arms "
                        "(GEPA runs to 6.5k tokens) OOM on eager attention at "
                        "a batch tuned for short ones. 0 disables the cap.")
    p.add_argument("--top-k", type=int, default=None,
                   help="default: read num_experts_per_tok from the model config, "
                        "which is 8 on Qwen and 4 on gpt-oss")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--allow-unpinned-checkpoint", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prompt-contract", choices=("v24", "v25"), default="v24",
                   help="v24 puts the instruction in a system turn; v25 carries it "
                        "in the user turn and leaves the system turn empty. Must "
                        "match the contract the arm was trained under, or the map "
                        "is measured on a prompt the arm never saw.")
    p.add_argument("--instructions", default=None,
                   help="v25 only: the instruction text; defaults to the seed one")
    p.add_argument("--reasoning", action="store_true",
                   help="let the model reason instead of suppressing it. gpt-oss "
                        "then opens its analysis channel, and the answer tokens "
                        "counted here include that reasoning.")
    p.add_argument("--trace-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=200)
    return p.parse_args(argv)


def build_arm(args: argparse.Namespace) -> ArmSpec:
    if args.arm == "base":
        # v25 carries the instruction in the user turn, so the system turn stays
        # empty; giving it the v24 seed prompt would measure a third contract.
        system = None if args.prompt_contract == "v25" else SEED_SYSTEM_PROMPT
        return ArmSpec(name="base", kind="base", system_prompt_text=system,
                       prompt_in_user_turn=args.prompt_contract == "v25")
    if args.arm == "gepa":
        if args.gepa_prompt is None:
            raise SystemExit("--arm gepa needs --gepa-prompt")
        return ArmSpec(name="gepa", kind="gepa",
                       system_prompt_text=args.gepa_prompt.read_text().strip())
    if args.adapter is None:
        raise SystemExit(f"--arm {args.arm} needs --adapter")
    return ArmSpec(name=args.arm, kind=args.arm, adapter_path=args.adapter,
                   system_prompt_text=None,
                   prompt_in_user_turn=args.prompt_contract == "v25",
                   allow_unpinned_checkpoint=args.allow_unpinned_checkpoint)


def label_prompt_stages(tokenizer, system_prompt: str | None, comment: str,
                        user_prompt: str, ids: Sequence[int],
                        *, reasoning: bool = False,
                        reasoning_effort: str = "medium") -> list[str]:
    """One stage name per prompt token, from character offsets.

    `reasoning_effort` must match what the caller rendered with. The spans come
    from re-rendering the prompt and mapping character offsets onto the given
    ids, so a different effort renders a different prompt - on gpt-oss `low` and
    `medium` differ by one token at the same length, which produces ids that
    mismatch without changing the count and suppresses every stage label.
    """
    text = render_chat(tokenizer, user_prompt, system_prompt, tokenize=False,
                       reasoning=reasoning, reasoning_effort=reasoning_effort)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(enc["input_ids"]) != list(ids):
        logger.warning("re-render mismatch (%d vs %d); stages suppressed for one example",
                       len(enc["input_ids"]), len(ids))
        return ["template"] * len(ids)

    spans: dict[str, tuple[int, int]] = {}
    user_at = text.rfind(user_prompt)
    if user_at < 0:
        return ["template"] * len(ids)
    if system_prompt:
        # Before the user turn: GEPA prompts restate their instructions, and the
        # system block is the earlier occurrence.
        sys_at = text.rfind(system_prompt, 0, user_at)
        if sys_at >= 0:
            spans["system"] = (sys_at, sys_at + len(system_prompt))
    c_at = text.find(comment, user_at)
    if c_at >= 0:
        spans["comment"] = (c_at, c_at + len(comment))
    # Match the marker itself, not a particular amount of whitespace before it.
    # The v24 contract puts a blank line ahead of it and v25 a single newline,
    # so requiring "\n\n" silently produced no question span at all under v25:
    # every instruction token fell into `template`, in the routing maps and in
    # the attention measurement alike, and the segment was reported as zero.
    marker = "Labels (use exact names):"
    q_at = user_prompt.rfind(marker)
    if q_at >= 0:
        # Swallow the whitespace that introduces it, so the newline belongs to
        # the question rather than to whatever precedes it.
        while q_at > 0 and user_prompt[q_at - 1] in "\r\n":
            q_at -= 1
        spans["question"] = (user_at + q_at, user_at + len(user_prompt))

    out = []
    for a, b in enc["offset_mapping"]:
        hit = "template"
        for name, (lo, hi) in spans.items():
            if a < hi and b > lo:
                hit = name
                break
        out.append(hit)
    return out


def iter_equal_length_batches(
    seqs: Sequence[Sequence[int]],
    batch_size: int,
    max_batch_tokens: int | None = None,
):
    """Yield index batches whose sequences all share one length.

    Grouping costs nothing - the counts are summed, so order is irrelevant -
    and it is what makes padding unnecessary.

    ``batch_size`` alone is the wrong knob when arms differ by an order of
    magnitude in length: a PEFT prompt is ~700 tokens while a GEPA prompt plus
    its reasoning runs to 6.5k, and eager attention allocates on the square of
    that. A fixed batch of 8 that fits the first ran out of memory on the
    second (4.87 GiB for one attention buffer). ``max_batch_tokens`` caps the
    batch by tokens instead, so short arms keep the full batch and long ones
    shrink to what the card can hold.
    """
    by_length: dict[int, list[int]] = {}
    for index, seq in enumerate(seqs):
        by_length.setdefault(len(seq), []).append(index)
    for length in sorted(by_length):
        group = by_length[length]
        size = batch_size
        if max_batch_tokens and length > 0:
            size = max(1, min(batch_size, max_batch_tokens // length))
        for start in range(0, len(group), size):
            yield group[start:start + size]


def run_forward(model, ids, attn, counter, stage_flat, *,
                attempts: int = 4, wait: float = 45.0) -> None:
    """One counted forward pass, retried while the card is merely busy.

    The GPU is shared. A worker claims a card with room to spare and another
    user's job can take that room before the first batch runs - the failure
    then looks identical to "this arm does not fit", but waiting a minute
    resolves it. Counting stops for the retry: the hooks accumulate on every
    call, so a partial forward that died mid-way must not leave its counts in
    the tensor. The counter is reset to its pre-batch state before each try.
    """
    import torch

    saved = counter.snapshot()
    for attempt in range(1, attempts + 1):
        try:
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=attn)
            return
        except torch.OutOfMemoryError:
            counter.restore(saved)
            if attempt == attempts:
                raise
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            logger.warning(
                "%d x %d (%d attempt from %d, free) "
                "%.1f GB from %.1f – waiting for %.0f",
                ids.shape[0], ids.shape[1], attempt, attempts,
                free / 2**30, total / 2**30, wait,
            )
            time.sleep(wait)


def split_answer(tokenizer, answer: str) -> tuple[list[int], list[str]]:
    """Tokenise a response and label its reasoning apart from its answer.

    A harmony response is ``<|channel|>analysis…<|channel|>final<|message|>…``.
    Counting all of it as "answer" would hide the thing a reasoning-mode map
    exists to show, so everything up to and including the ``final`` marker is
    labelled ``reasoning`` and the rest ``answer``. A response with no channel
    marker - every suppressed-mode run, and Qwen always - is answer throughout,
    which keeps those maps byte-identical to the ones measured before this split.
    """
    if FINAL_CHANNEL in answer:
        head, _, tail = answer.partition(FINAL_CHANNEL)
        head_ids = list(
            tokenizer(head + FINAL_CHANNEL, add_special_tokens=False)["input_ids"]
        )
        tail_ids = list(tokenizer(tail, add_special_tokens=False)["input_ids"])
        return head_ids + tail_ids, ["reasoning"] * len(head_ids) + ["answer"] * len(tail_ids)
    ids = list(tokenizer(answer, add_special_tokens=False)["input_ids"])
    return ids, ["answer"] * len(ids)


class GpuCounter:
    """Accumulates [layer, stage, expert] counts without leaving the GPU.

    ``set_batch`` installs the per-position stage ids for the batch about to be
    run; the hooks read them. Positions marked ``PAD_ID`` (padding) contribute
    nothing.
    """

    def __init__(self, model, *, top_k: int, n_layers: int, n_experts: int, device) -> None:
        import torch

        self.gates = discover_gates(model)
        self.top_k = top_k
        self.n_experts = n_experts
        self.row = {g.layer_idx: i for i, g in enumerate(self.gates)}
        # float64: counts reach ~1e9 for a long-prompt arm, and while each
        # individual cell stays well under float32's exact-integer limit, any
        # reduction over them does not. Cheap here - one [48,6,128] tensor.
        self.counts = torch.zeros((n_layers, len(STAGES), n_experts),
                                  dtype=torch.float64, device=device)
        self.positions = torch.zeros(len(STAGES), dtype=torch.long, device=device)
        self._stage_flat = None   # [B*seq] stage id per routed position
        self._handles: list = []
        self._calls = 0

    def set_batch(self, stage_flat) -> None:
        self._stage_flat = stage_flat

    def _hook(self, layer_idx: int):
        import torch

        def hook(_m, _i, output):
            stage = self._stage_flat
            if stage is None:
                return output
            # Qwen's gate returns logits; gpt-oss's router returns
            # (scores, indices) with the top-k already taken. Using the indices
            # it actually selected is both cheaper and safer than re-deriving
            # them, since the two models differ in whether top-k runs before or
            # after the softmax.
            picked = None
            if isinstance(output, tuple):
                output, picked = output[0], output[1]
            if output.shape[0] != stage.shape[0]:
                raise RuntimeError(
                    f"layer {layer_idx}: gate saw {output.shape[0]} positions, "
                    f"{stage.shape[0]} stage labels were built - virtual-token "
                    "offset or segmentation is wrong"
                )
            self._calls += 1
            with torch.no_grad():
                keep = stage >= 0
                idx = (picked[keep] if picked is not None
                       else output[keep].float().topk(self.top_k, dim=-1).indices)
                st = stage[keep].unsqueeze(1).expand_as(idx)                # [n, k]
                flat = (st * self.n_experts + idx).reshape(-1)
                self.counts[self.row[layer_idx]].view(-1).scatter_add_(
                    0, flat, torch.ones_like(flat, dtype=torch.float64))
            return None  # leave the module's own output untouched

        return hook

    def add_positions(self, stage_flat) -> None:
        import torch

        keep = stage_flat[stage_flat >= 0]
        self.positions += torch.bincount(keep, minlength=len(STAGES))

    def snapshot(self) -> tuple:
        """The accumulated state, so a failed forward can be rolled back.

        A forward that dies part-way has already fired the hooks of the layers
        it reached. Retrying it without undoing those would count those layers
        twice - a silent corruption, since the totals would still look
        plausible. Position counts are added once per batch, before the pass,
        and so are left alone.
        """
        return self.counts.clone(), self._calls

    def restore(self, saved: tuple) -> None:
        counts, calls = saved
        self.counts.copy_(counts)
        self._calls = calls

    def __enter__(self) -> "GpuCounter":
        self._handles = [g.module.register_forward_hook(self._hook(g.layer_idx))
                         for g in self.gates]
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    @property
    def n_calls(self) -> int:
        return self._calls

    def numpy(self):
        return self.counts.detach().cpu().numpy()


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args = parse_args(argv)
    import torch

    spec = build_arm(args)
    spec.validate()

    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    rows = rows[:args.n_examples]
    if len(rows) < args.n_examples:
        raise SystemExit(f"{args.data} has {len(rows)} rows, {args.n_examples} requested")
    ref = [json.loads(line) for line in args.answers.read_text().splitlines() if line.strip()]
    ref = ref[:args.n_examples]
    if len(ref) != len(rows) or any(a["comment"] != r["comment"] for a, r in zip(ref, rows)):
        raise SystemExit("reference answers do not line up with the data rows")

    model, tokenizer = load_arm(spec, model_id=args.model, revision=args.revision,
                                dtype=args.dtype)
    system_prompt = spec.system_prompt("as_trained")
    n_virtual = 0
    if spec.kind == "prompt_tuning":
        cfg = getattr(model, "peft_config", {}).get("default")
        n_virtual = int(getattr(cfg, "num_virtual_tokens", 0) or 0)
    gates = discover_gates(model)
    n_layers, n_experts = len(gates), int(gates[0].num_experts)
    # Qwen routes 8 experts per token, gpt-oss 4. Taking the model's own value
    # keeps a hardcoded default from silently counting the wrong number of
    # assignments on a backbone it was not written for.
    if args.top_k is None:
        args.top_k = int(getattr(model.config, "num_experts_per_tok", 0) or 8)
        logger.info("top_k from the model config: %d", args.top_k)
    logger.info("arm %s: %d gates x %d experts, %d virtual positions",
                spec.name, n_layers, n_experts, n_virtual)

    # ---- build every sequence and its stage labels once ---------------------
    seqs: list[list[int]] = []
    stages: list[list[int]] = []
    if args.prompt_contract == "v25":
        instructions = args.instructions or V25_SEED_INSTRUCTION
        render_user = lambda comment: render_user_prompt_v25(comment, instructions)  # noqa: E731
    else:
        render_user = render_user_prompt
    for row, answer in zip(rows, ref):
        user_prompt = render_user(row["comment"])
        prompt_ids = list(render_chat(tokenizer, user_prompt, system_prompt,
                                      tokenize=True, reasoning=args.reasoning))
        answer_ids, answer_stages = split_answer(tokenizer, answer["answer"])
        names = label_prompt_stages(tokenizer, system_prompt, row["comment"],
                                    user_prompt, prompt_ids,
                                    reasoning=args.reasoning)
        names = names + answer_stages
        ids = ["virtual"] * n_virtual + names
        seqs.append(prompt_ids + answer_ids)
        stages.append([STAGE_ID[s] for s in ids])
    logger.info("built %d sequences, %d..%d tokens",
                len(seqs), min(map(len, seqs)), max(map(len, seqs)))

    # ---- one pass, batched, counting on device ------------------------------
    device = model.device
    traces: dict[str, np.ndarray] = {}
    started = time.monotonic()
    done = n_logged = n_traced = 0
    with GpuCounter(model, top_k=args.top_k, n_layers=n_layers,
                    n_experts=n_experts, device=device) as counter:
        # Batches hold ONE length, so nothing is ever padded. Left padding is
        # wrong here for the same reason it is wrong in generation: PEFT
        # prepends the virtual tokens at position 0, *ahead* of the padding, so
        # the real sequence is [virtual][pad…][text] while a padded stage row
        # says [pad…][virtual][text]. Every virtual position would then be
        # labelled as padding (and dropped) while padding got counted as
        # virtual. Equal-length batches remove the padding and the ambiguity at
        # once, and skip the wasted compute on pad positions as a bonus.
        for chunk in iter_equal_length_batches(
            seqs, args.batch_size, args.max_batch_tokens
        ):
            ids = torch.tensor([seqs[i] for i in chunk], dtype=torch.long)
            attn = torch.ones_like(ids)
            stage_rows = torch.tensor([stages[i] for i in chunk], dtype=torch.long)
            ids, attn = ids.to(device), attn.to(device)
            stage_flat = stage_rows.reshape(-1).to(device)

            counter.set_batch(stage_flat)
            counter.add_positions(stage_flat)
            run_forward(model, ids, attn, counter, stage_flat,
                        attempts=args.oom_retries, wait=args.oom_wait)
            counter.set_batch(None)

            done += len(chunk)
            if done - n_traced >= args.trace_every:
                n_traced = done
                traces[f"@{done}"] = counter.numpy().copy()
            if done - n_logged >= args.log_every or done == len(seqs):
                rate = (time.monotonic() - started) / done
                n_logged = done
                left = rate * (len(seqs) - done)
                logger.info("%d/%d examples | %.2fs each | ~%.0f min left",
                            done, len(seqs), rate, left / 60)

        if counter.n_calls == 0:
            raise RuntimeError("gate hooks never fired - nothing was counted")
        counts = counter.numpy()
        positions = counter.positions.detach().cpu().numpy()

    # ---- write ---------------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    per_stage = {STAGES[s]: counts[:, s, :] for s in range(len(STAGES))
                 if counts[:, s, :].sum() > 0}
    per_stage["__all__"] = counts.sum(axis=1)
    pos = {STAGES[s]: int(positions[s]) for s in range(len(STAGES)) if positions[s] > 0}
    pos["__all__"] = int(positions.sum())

    meta = {
        "arm": spec.name,
        "model": args.model,
        "revision": args.revision,
        "layer_ids": [g.layer_idx for g in gates],
        "num_experts": n_experts,
        "top_k": args.top_k,
        "n_examples": len(rows),
        "n_virtual": n_virtual,
        "stages_present": sorted(per_stage),
        "entries": {st: {"n_positions": pos.get(st, 0),
                         "total_assignments": float(mat.sum())}
                    for st, mat in per_stage.items()},
        "shared_answers": str(args.answers),
        "data": str(args.data),
        "prompt_contract": args.prompt_contract,
        "reasoning": args.reasoning,
        "arm_provenance": provenance(spec),
    }
    np.savez(args.out / "expert_counts.npz",
             _meta=json.dumps(meta, ensure_ascii=False),
             **{f"{spec.name}|{st}": mat for st, mat in per_stage.items()})
    if traces:
        np.savez(args.out / "expert_counts_trace.npz",
                 _meta=json.dumps({"snapshot_every": args.trace_every,
                                   "stages": list(STAGES)}),
                 **{f"{spec.name}|{k}": v for k, v in traces.items()})
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    elapsed = time.monotonic() - started
    logger.info("wrote %s in %.1f min", args.out / "expert_counts.npz", elapsed / 60)
    for st in sorted(per_stage):
        logger.info("  %-9s %9d positions | %14.0f assignments",
                    st, pos.get(st, 0), per_stage[st].sum())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
