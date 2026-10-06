#!/usr/bin/env python3
"""Per-token attention and massive activations, intact and pruned.

For super-expert detection. The reported chain is: a super expert emits a
massive activation on one token, that token becomes an attention sink, and
removing the expert breaks both at once. This measures the two ends of that
chain on the same forward pass and on the same positions, so the pairing can be
checked rather than assumed.

Run over the same cell at several pruning levels and the comparison answers the
question the frequency maps cannot: whether the experts the mask removed were
the ones holding the sink up.

**Sample size.** 64 examples by default. Attention is quadratic in the prompt,
but only the column sums survive the hook - one vector per layer per example,
about 2 KB - so the cost is the forward passes, not the storage. Sixty-four is
enough for a per-layer mean whose standard error is a few percent, and small
enough to sweep sixteen checkpoints at three levels within an hour on one card.
A sink is not a subtle effect: it either holds tens of percent of the mass or it
is not a sink, and that shows up in a handful of examples.

**Prompts are truncated from the left, and that is a real limitation.** The
read position must be inside the window and it is the last token, so the window
is the prompt's tail. A prompt longer than `--max-length` therefore loses its
opening - including the first token, which is where a sink canonically sits.
The run reports how many prompts this happened to; raise the window rather than
reading a first-token result that the window never contained.

    python scripts/measure_token_attention.py \\
        --arm prefix_tuning:adapters-final/qwen/prefix-m100-s42 \\
        --counts-npz routing_maps_final/qwen/prefix-m100-s42/expert_counts.npz \\
        --levels 0,50%,75% --n-examples 64 --out attention_tokens/qwen/prefix-m100
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

logger = logging.getLogger("token_attention")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True,
                   help="base | gepa | prompt_tuning:<path> | prefix_tuning:<path>")
    p.add_argument("--gepa-prompt", type=Path)
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--revision", default="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--data", type=Path, default=ROOT / "data/test_canonical_n2000.jsonl")
    p.add_argument("--n-examples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4,
                   help="small on purpose: eager attention materialises "
                        "[batch, heads, q, k] and that is the memory wall here")
    p.add_argument("--max-length", type=int, default=256,
                   help="prompt window; the attention matrix is quadratic in it")
    p.add_argument("--counts-npz", type=Path,
                   help="expert map the mask is cut from; required for any level but 0")
    p.add_argument("--counts-arm", default="own")
    p.add_argument("--counts-stage", default="__all__")
    p.add_argument("--levels", default="0,50%,75%")
    p.add_argument("--selection", default="per_layer")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--prompt-contract", default="v25")
    p.add_argument("--reasoning", action="store_true")
    p.add_argument("--reasoning-effort", default="low",
                   choices=("low", "medium", "high"))
    p.add_argument("--detail-examples", type=int, default=8,
                   help="How many examples to take with each rib: "
                        "It is this dump that answers the question, \"Who looked where?\" "
                        "0 turns it off")
    p.add_argument("--top-edges", type=int, default=16,
                   help="How many strong pairs (request, key) to keep on the head")
    p.add_argument("--edge-min-context", type=int, default=8,
                   help="query lines that see less than this number of positions "
                        "the edges do not go: the first line of this weight 1.0 "
                        "arithmetic, and without cutting off, the dump consists of it.")
    p.add_argument("--edge-min-weight", type=float, default=0.05,
                   help="Threshold for readable dump ribs: each head has "
                        "The strongest couple, and absent-minded it means nothing")
    p.add_argument("--keep-arrays", action="store_true", default=True,
                   help="Preserve raw materials for each example: "
                        "Profiles, line reading, activation, ribs")
    p.add_argument("--no-keep-arrays", dest="keep_arrays", action="store_false")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


SEGMENT_COLUMNS = None       # filled from the routing maps' stage list at run time


def _open_table(path, columns):
    """A csv writer with its header already down, appended to as batches land.

    Written as it goes rather than assembled at the end: a run that dies on the
    last level should still leave the levels it finished, and the full table at
    sixty-four examples by forty-eight layers by thirty-two heads is a hundred
    thousand rows a level - worth streaming.
    """
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    writer.writerow(columns)
    return handle, writer


def _head_columns():
    # `valid` is not decoration. A row whose layer collected no attention still
    # has an argmax - of all zeros - and its offset is whatever index that
    # landed on. Without the column those rows are indistinguishable from a
    # measured position, so they are marked and their positions left blank.
    return (["example", "level", "layer", "head", "valid", "top1_share",
             "top1_over_null", "top4_share", "top1_offset", "top1_from_end",
             "entropy_ratio", "row_entropy", "first_real"]
            + [f"seg_{name}" for name in SEGMENT_COLUMNS])


def _write_head_rows(writer, profile, seg, level: int, *, first_index: int):
    """One row per (example, layer, head) - the table the analysis runs on.

    Everything here is per head on purpose. A layer mean is what hid the sink
    in the first place, and a mean cannot be taken apart afterwards; a table
    with the heads in it can always be averaged.
    """
    import numpy as np

    from mrd_pruning.token_attention import segment_mass, sink_metrics_batch

    first_real = profile.first_real
    for layer in sorted(profile.received_per_head):
        per_head = profile.received_per_head[layer]           # [B, H, K]
        null = profile.null_top1.get(layer)
        m = sink_metrics_batch(per_head, first_real, null=null)
        rows = profile.last_row.get(layer)
        entropy = profile.row_entropy.get(layer)
        n_seg = len(SEGMENT_COLUMNS)
        for b in range(per_head.shape[0]):
            mass = (segment_mass(rows[b], seg[b], n_seg) if rows is not None
                    else np.zeros((per_head.shape[1], n_seg)))
            for h in range(per_head.shape[1]):
                ok = bool(m["valid"][b, h])
                writer.writerow([
                    first_index + b, level, layer, h, int(ok),
                    f"{m['top1_share'][b, h]:.6g}",
                    f"{m['top1_over_null'][b, h]:.6g}",
                    f"{m['top4_share'][b, h]:.6g}",
                    int(m["top1_offset"][b, h]) if ok else "",
                    int(m["top1_from_end"][b, h]) if ok else "",
                    f"{m['entropy_ratio'][b, h]:.6g}",
                    f"{float(entropy[b, h]):.6g}" if entropy is not None else "",
                    int(first_real[b]),
                ] + [f"{mass[h, s]:.6g}" for s in range(n_seg)])


def _layer_columns():
    return ["example", "level", "layer", "valid", "top1_share", "top1_over_null",
            "top4_share", "top1_offset", "entropy_ratio", "top_head",
            "head_top1_share", "activation_ratio", "activation_offset",
            "activation_max", "activation_median", "n_massive", "first_real"]


def _write_layer_rows(writer, profile, level: int, *, first_index: int,
                      n_virtual: int = 0):
    """One row per (example, layer): the head-averaged view and the activations.

    Kept beside the head table rather than derived from it, because the
    activation numbers have no head axis and the head-averaged attention is not
    the mean of the per-head shares.
    """
    from mrd_pruning.token_attention import (hidden_origin, massive_activations,
                                             sink_metrics_batch)

    first_real = profile.first_real
    for layer in sorted(profile.received):
        received = profile.received[layer]
        null = profile.null_top1.get(layer)
        m = sink_metrics_batch(received, first_real, null=null)
        top = sink_metrics_batch(profile.received_top_head[layer], first_real,
                                 null=null)
        hidden = profile.hidden_norm.get(layer)
        for b in range(received.shape[0]):
            act = (massive_activations(hidden[b], first_real=hidden_origin(
                       int(first_real[b]), hidden.shape[-1],
                       profile.n_positions, n_virtual))
                   if hidden is not None else None)
            ok = bool(m["valid"][b])
            live = act if (act and act.get("valid")) else None
            writer.writerow([
                first_index + b, level, layer, int(ok),
                f"{m['top1_share'][b]:.6g}", f"{m['top1_over_null'][b]:.6g}",
                f"{m['top4_share'][b]:.6g}",
                int(m["top1_offset"][b]) if ok else "",
                f"{m['entropy_ratio'][b]:.6g}",
                int(profile.top_head_index[layer][b]),
                f"{top['top1_share'][b]:.6g}",
                f"{live['ratio']:.6g}" if live else "",
                live["argmax_offset"] if live else "",
                f"{live['max']:.6g}" if live else "",
                f"{live['median']:.6g}" if live else "",
                live["n_massive"] if live else "",
                int(first_real[b]),
            ])


def _write_per_example(out_dir, profile, seg, level: int, *, first_index: int):
    """The arrays themselves, one file per example per level.

    The tables answer the questions already asked. These are for the ones that
    come up later: the whole `[heads, keys]` profile, the read position's row,
    the hidden-state magnitudes, and the segment labelling that goes with them.
    """
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    layers = sorted(profile.received_per_head)
    n = next(iter(profile.received.values())).shape[0]
    for b in range(n):
        arrays = {"segment_ids": seg[b], "first_real": np.asarray([profile.first_real[b]])}
        for layer in layers:
            tag = f"layer{layer:02d}"
            arrays[f"received_per_head_{tag}"] = \
                profile.received_per_head[layer][b].astype(np.float16)
            if layer in profile.last_row:
                arrays[f"last_row_{tag}"] = profile.last_row[layer][b].astype(np.float16)
            if layer in profile.row_entropy:
                arrays[f"row_entropy_{tag}"] = profile.row_entropy[layer][b]
            if layer in profile.hidden_norm:
                arrays[f"hidden_norm_{tag}"] = profile.hidden_norm[layer][b]
            if layer in profile.edges:
                arrays[f"edges_{tag}"] = profile.edges[layer][b]
        np.savez_compressed(
            out_dir / f"level{level:03d}_ex{first_index + b:04d}.npz", **arrays)


def _segment_ids(stage_labels, n_virtual: int, width: int, virtual_id: int):
    """Segment per key position, in the axis the model actually used.

    The key axis is `[virtual][pad][prompt]` - PEFT prepends its virtual tokens
    ahead of the padding, not after it, so a labelling built as `[pad][virtual]`
    would put every virtual position one block off and hand the soft prompt's
    attention mass to whatever sits there.

    `-1` marks a position belonging to no segment. Padding is the only such
    position here, and it is dropped rather than folded into a neighbour.
    """
    import numpy as np

    seg = np.full((len(stage_labels), n_virtual + width), -1, dtype=np.int16)
    if n_virtual:
        seg[:, :n_virtual] = virtual_id
    for j, row_labels in enumerate(stage_labels):
        head = n_virtual + width - len(row_labels)
        seg[j, head:] = row_labels
    return seg


def _detail_row(profile, row_i: int, ids, tokenizer, *, index: int,
                n_virtual: int = 0) -> dict:
    """Everything known about one example, keyed by layer and head.

    The decoded tokens travel with the numbers. Without them the dump is a grid
    of indices, and the question being asked - *which token* holds the sink - is
    exactly the one indices cannot answer.

    The token list is built on the **key axis**, virtual block included. The
    tokenizer only knows the text, which is `n_virtual` positions narrower than
    the axis the attention lives on; taking its length while taking `first_real`
    from the key axis put every offset in the virtual block and made the
    readable edge dump index the wrong token. The virtual positions have no
    text, so they are named for what they are.
    """
    import numpy as np

    origin = int(profile.first_real[row_i])
    tokens = ([f"<vt{i}>" for i in range(n_virtual)]
              + tokenizer.convert_ids_to_tokens(list(ids)))
    layers = sorted(profile.received)
    return {
        "example": index,
        "first_real": origin,
        "tokens": tokens,
        # Offsets from the first real token, so positions line up with the
        # summary and with the other examples in the dump.
        "offsets": [i - origin for i in range(len(tokens))],
        "n_virtual": int(n_virtual),
        "per_head": {
            str(l): np.asarray(profile.received_per_head[l][row_i])
            for l in layers if l in profile.received_per_head
        },
        "edges": {
            str(l): np.asarray(profile.edges[l][row_i])
            for l in layers if l in profile.edges
        },
        "hidden_norm": {
            str(l): np.asarray(profile.hidden_norm[l][row_i])
            for l in layers if l in profile.hidden_norm
        },
    }


def _write_detail(out_dir, level: int, rows, min_weight: float) -> None:
    """Two files per level: the arrays, and the same thing as readable lines."""
    import numpy as np

    from mrd_pruning.token_attention import describe_edges

    if not rows:
        return
    arrays = {}
    for row in rows:
        i = row["example"]
        for layer, arr in row["per_head"].items():
            arrays[f"ex{i:03d}_per_head_layer{int(layer):02d}"] = arr
        for layer, arr in row["edges"].items():
            arrays[f"ex{i:03d}_edges_layer{int(layer):02d}"] = arr
        for layer, arr in row["hidden_norm"].items():
            arrays[f"ex{i:03d}_hidden_layer{int(layer):02d}"] = arr
        arrays[f"ex{i:03d}_first_real"] = np.asarray([row["first_real"]])
    np.savez_compressed(out_dir / f"detail_level{level:03d}.npz", **arrays)

    (out_dir / f"tokens_level{level:03d}.json").write_text(json.dumps(
        [{"example": r["example"], "first_real": r["first_real"],
          "tokens": r["tokens"], "offsets": r["offsets"]} for r in rows],
        indent=2, ensure_ascii=False))

    lines = []
    for row in rows:
        lines.append(f"example {row['example']}")
        for layer in sorted(row["edges"], key=int):
            lines.extend(describe_edges(
                row["edges"][layer], row["tokens"], layer=int(layer),
                min_weight=min_weight, first_real=row["first_real"]))
    (out_dir / f"edges_level{level:03d}.txt").write_text(
        "\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args = parse_args(argv)

    import numpy as np
    import torch

    from measure_routing_map import STAGE_ID, STAGES, label_prompt_stages
    from run_pruning_sweep import build_arm, load_dataset

    from mrd_pruning.arms import load_arm
    from mrd_pruning.frequency import load_counts_npz, resolve_levels, select_pruned
    from mrd_pruning.masking import ExpertMask
    from mrd_pruning.task import render_chat
    from mrd_pruning.token_attention import (TokenAttention,
                                             causal_null_top1_share, force_eager,
                                             hidden_origin, massive_activations,
                                             position_mode, segment_mass,
                                             sink_metrics)

    arm = build_arm(args.arm, args)
    # Under v25 the instruction lives in the user turn, GEPA's optimised one
    # included, and it reaches the prompt only through `instructions=`. Omitting
    # it silently rendered the seed instruction instead: the sixteen-cell
    # delivery shipped two "gepa" directories whose level-0 artefacts are
    # byte-identical to `base`, down to the sha256 of every npz. A GEPA arm that
    # measures the base prompt is not a GEPA arm.
    v25_instructions = None
    if args.prompt_contract == "v25" and args.gepa_prompt is not None:
        v25_instructions = args.gepa_prompt.read_text().strip()
    prompts, _golds = load_dataset(args.data, args.n_examples,
                                   contract=args.prompt_contract,
                                   instructions=v25_instructions)
    if v25_instructions:
        logger.info("GEPA instruction is inserted into the prompt: %d characters",
                    len(v25_instructions))
    # The comment text, for the segment labelling. Read with the same "first n
    # rows" rule as the prompts and then checked against them, because a silent
    # drift here would put the comment's attention mass under the instruction.
    comments = []
    with args.data.open() as handle:
        for line in handle:
            if len(comments) >= args.n_examples:
                break
            comments.append(json.loads(line)["comment"])
    for c, prompt in zip(comments, prompts):
        if c not in prompt:
            raise SystemExit("The comment is not found in your Prompt. "
                             "Segment markup would go silently")
    logger.info("%d examples, %d-token window", len(prompts), args.max_length)

    model, tokenizer = load_arm(arm, model_id=args.model, revision=args.revision,
                                dtype=args.dtype, device_map={"": 0})
    if not force_eager(model):
        logger.warning("could not switch to eager attention; attention weights may be missing")

    # Read from the adapter, never from a flag: every position in the run is
    # shifted by this number, and a CLI value that disagrees with the checkpoint
    # would move the whole map without saying so.
    n_virtual = 0
    cfg = getattr(model, "active_peft_config", None) or getattr(
        model, "peft_config", None)
    if isinstance(cfg, dict):
        cfg = next(iter(cfg.values()), None)
    if cfg is not None:
        n_virtual = int(getattr(cfg, "num_virtual_tokens", 0) or 0)
    logger.info("%d virtual positions", n_virtual)

    counts = None
    if args.counts_npz is not None:
        counts = load_counts_npz(args.counts_npz, arm=args.counts_arm,
                                 stage=args.counts_stage)
    levels = resolve_levels(args.levels.split(","),
                            counts.n_experts if counts is not None else 128)

    system_prompt = arm.system_prompt("as_trained")
    encoded = []
    stage_labels = []
    truncated: list[int] = []
    for text, comment in zip(prompts, comments):
        rendered = render_chat(tokenizer, text, system_prompt, tokenize=False,
                               reasoning=args.reasoning,
                               reasoning_effort=args.reasoning_effort)
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        # Labelled on the whole sequence and cut the same way the ids are: the
        # labels come from character offsets, so labelling a truncated string
        # would relabel every token instead of dropping the ones that went.
        # The labeller re-renders the prompt to find spans by character offset,
        # so it has to render the *same* prompt. Without the effort it used the
        # template's own default: on gpt-oss that is `medium` against a run at
        # `low`, one token either way, so the ids differed at equal length and
        # every one of the 64 examples hit "re-render mismatch; stages
        # suppressed". All eight gpt-oss cells shipped with the segment cut
        # collapsed into a single bucket.
        names = label_prompt_stages(tokenizer, system_prompt, comment, text, ids,
                                    reasoning=args.reasoning,
                                    reasoning_effort=args.reasoning_effort)
        encoded.append(ids[-args.max_length:])
        stage_labels.append([STAGE_ID[n] for n in names][-args.max_length:])
        if len(ids) > args.max_length:
            truncated.append(len(ids))

    global SEGMENT_COLUMNS
    SEGMENT_COLUMNS = list(STAGES)

    # The window is the *tail*: the read position has to be in it, and it is the
    # last token. A prompt longer than the window therefore loses its opening -
    # including the first token, where a sink canonically sits - so the count is
    # reported rather than left for the reader to discover in the token dump.
    if truncated:
        logger.warning(
            "%d of %d prompts are longer than the %d window (up to %d tokens): they lose "
            "the beginning, and a sink on the first token is then not visible in this measurement",
            len(truncated), len(encoded), args.max_length, max(truncated))
    else:
        logger.info("All %d prompts fit into the %d token window",
                    len(encoded), args.max_length)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = []

    for level in levels:
        pruned = {}
        if level > 0:
            if counts is None:
                raise SystemExit("For a level above zero you need -counts-npz")
            pruned = select_pruned(counts, level, top_k=args.top_k,
                                   selection=args.selection)

        head_handle, head_table = _open_table(
            args.out / f"table_heads_level{level:03d}.csv", _head_columns())
        layer_handle, layer_table = _open_table(
            args.out / f"table_layers_level{level:03d}.csv", _layer_columns())

        per_layer_sink: dict[int, list] = {}
        per_layer_seg: dict[int, list] = {}
        per_layer_entropy: dict[int, list] = {}
        per_layer_mass: dict[int, list] = {}
        per_layer_received = {}
        detail_rows: list = []
        detail_left = args.detail_examples
        n_seen = 0
        null_top1: list[float] = []

        with ExpertMask(model, pruned, top_k=args.top_k) as mask:
            for start in range(0, len(encoded), args.batch_size):
                batch = encoded[start:start + args.batch_size]
                width = max(len(x) for x in batch)
                pad = tokenizer.pad_token_id
                if pad is None:
                    pad = tokenizer.eos_token_id or 0
                # Left padding: the read position is then last for every row,
                # which is the convention the rest of this project uses.
                ids = [[pad] * (width - len(x)) + list(x) for x in batch]
                att = [[0] * (width - len(x)) + [1] * len(x) for x in batch]
                input_ids = torch.tensor(ids, device=model.device)
                attention_mask = torch.tensor(att, device=model.device)

                seg = _segment_ids(
                    stage_labels[start:start + args.batch_size],
                    n_virtual, width, STAGE_ID["virtual"])

                # Heads are kept apart on every example, not a sample of them:
                # the per-head rows are the analysis table, and an aggregate
                # cannot be un-aggregated later. The edges are the expensive
                # part and stay on a budget.
                detailed = detail_left > 0
                probe = TokenAttention(
                    model,
                    keep_per_head=True,
                    top_edges=args.top_edges if detailed else 0,
                    edge_min_context=args.edge_min_context)
                probe.set_attention_mask(attention_mask, n_virtual=n_virtual)
                with probe, torch.inference_mode():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                probe.assert_captured()

                first_real = probe.profile.first_real
                for layer, row in probe.profile.last_row.items():
                    # Where the answer position is looking, split by segment.
                    # This is the cut the dense-model side of the project
                    # reports, and the one that separates an arm that leans on
                    # its soft prompt from one that leans on the comment.
                    for row_i in range(row.shape[0]):
                        per_layer_seg.setdefault(layer, []).append(
                            segment_mass(row[row_i], seg[row_i], len(STAGES)))
                    per_layer_entropy.setdefault(layer, []).append(
                        probe.profile.row_entropy[layer])

                _write_head_rows(head_table, probe.profile, seg, level,
                                 first_index=start)
                _write_layer_rows(layer_table, probe.profile, level,
                                  first_index=start, n_virtual=n_virtual)
                if args.keep_arrays:
                    _write_per_example(args.out / "per_example", probe.profile,
                                       seg, level, first_index=start)

                for layer, received in probe.profile.received.items():
                    top_head = probe.profile.received_top_head[layer]
                    for row_i in range(received.shape[0]):
                        # Left padding is per-example, so every index is taken
                        # relative to that example's first real token; absolute
                        # columns are not comparable between examples.
                        origin = int(first_real[row_i])
                        null = probe.profile.null_top1.get(layer)
                        m = sink_metrics(received[row_i], first_real=origin,
                                         null=float(null[row_i]) if null is not None else None)
                        m["head_top1_share"] = sink_metrics(
                            top_head[row_i], first_real=origin)["top1_share"]
                        m["top_head"] = int(
                            probe.profile.top_head_index[layer][row_i])
                        per_layer_sink.setdefault(layer, []).append(m)
                        hid = probe.profile.hidden_norm.get(layer)
                        if hid is not None:
                            # The hidden axis is narrower than the key axis on a
                            # prefix arm; the origin has to be moved onto it.
                            per_layer_mass.setdefault(layer, []).append(
                                massive_activations(hid[row_i], first_real=hidden_origin(
                                    origin, hid.shape[-1],
                                    probe.profile.n_positions, n_virtual)))
                    # One example's full profile per layer, for a picture.
                    if layer not in per_layer_received:
                        per_layer_received[layer] = received[0].tolist()

                if detailed:
                    take = min(detail_left, len(batch))
                    for row_i in range(take):
                        detail_rows.append(_detail_row(
                            probe.profile, row_i, ids[row_i], tokenizer,
                            index=start + row_i, n_virtual=n_virtual))
                    detail_left -= take

                n_seen += len(batch)
                null_top1.extend(float(np.mean(v)) for v in probe.profile.null_top1.values())
                del probe
                torch.cuda.empty_cache()
                logger.info("%d level: %d / %d examples", level, n_seen, len(encoded))

            mask.assert_applied(expect_layers=None if level == 0 else len(pruned))
        head_handle.close()
        layer_handle.close()

        layers = sorted(per_layer_sink)
        record = {
            "level": level,
            "n_examples": n_seen,
            "max_length": args.max_length,
            "layers": [
                {
                    "layer": l,
                    "top1_share_mean": float(np.mean([m["top1_share"] for m in per_layer_sink[l]])),
                    # The share against a model with no sink at all. Under a
                    # causal mask the first position collects H_n/n for free -
                    # 0.024 at a 256-token window, not the 1/n of 0.004 - so a
                    # raw share is unreadable without this beside it.
                    "top1_over_null_mean": float(np.mean(
                        [m["top1_over_null"] for m in per_layer_sink[l]])),
                    # The head-level share beside the head-averaged one. Sinks
                    # live in a minority of heads, so averaging thirty-two of
                    # them is how a real sink comes back reading as ordinary.
                    "head_top1_share_mean": float(np.mean(
                        [m["head_top1_share"] for m in per_layer_sink[l]])),
                    "top4_share_mean": float(np.mean([m["top4_share"] for m in per_layer_sink[l]])),
                    "entropy_ratio_mean": float(np.mean([m["entropy_ratio"] for m in per_layer_sink[l]])),
                    # Which position wins, and how often the same one wins:
                    # a sink is a *stable* position, not merely a peaked one.
                    # Counted from each example's first real token - a mode over
                    # absolute columns mixes examples padded to different widths
                    # and returns a position that belongs to none of them. Both
                    # ends are reported because a sink sits either on the first
                    # token or on the closing delimiter, and the second one is
                    # only stable when counted backwards.
                    # Which head, not just which layer: a sink reported without
                    # its head leaves thirty-two places to look.
                    **position_mode("top_head", per_layer_sink[l]),
                    **position_mode("top1_offset", per_layer_sink[l]),
                    **position_mode("top1_from_end", per_layer_sink[l]),
                    # Mean over examples and heads, per segment: what share of
                    # the read position's attention each part of the prompt got.
                    "segment_mass_mean": {
                        name: float(np.mean([m[..., si].mean()
                                             for m in per_layer_seg.get(l, [])])
                                    ) if per_layer_seg.get(l) else 0.0
                        for si, name in enumerate(STAGES)
                    },
                    "row_entropy_mean": float(np.mean(
                        [e.mean() for e in per_layer_entropy.get(l, [])] or [0.0])),
                    "activation_ratio_mean": float(np.mean(
                        [m["ratio"] for m in per_layer_mass.get(l, [])] or [0.0])),
                    "activation_offset_mode": position_mode(
                        "argmax_offset", per_layer_mass.get(l, []),
                    ).get("argmax_offset_mode", -1),
                    "n_massive_mean": float(np.mean(
                        [m["n_massive"] for m in per_layer_mass.get(l, [])] or [0.0])),
                }
                for l in layers
            ],
        }
        summary.append(record)

        _write_detail(args.out, level, detail_rows, args.edge_min_weight)
        np.savez_compressed(
            args.out / f"received_level{level:03d}.npz",
            **{f"layer{l:02d}": np.asarray(per_layer_received[l]) for l in layers})
        (args.out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False))

        # Ranked by the head-level share: that is where a sink is visible at
        # all, and a layer can hold one while its head-average looks flat.
        peak = max(record["layers"], key=lambda r: r["head_top1_share_mean"])
        # Reported from the measured nulls, not from the window width: the
        # formula assumes a square causal triangle that neither a sliding
        # window nor a prefix arm's always-visible keys satisfy.
        null = float(np.mean(null_top1)) if null_top1 else causal_null_top1_share(args.max_length)
        logger.info("level %d: strongest sink in layer %d (head %d), share %.3f "
                    "(head mean %.3f, %.1fx the no-sink null; "
                    "null for a %d-token window %.3f), "
                    "at offset %+d from the first prompt token "
                    "(mode in %.0f%% of examples), activation peak %.1fx at %+d",
                    level, peak["layer"], peak["top_head_mode"],
                    peak["head_top1_share_mean"],
                    peak["top1_share_mean"], peak["top1_over_null_mean"],
                    args.max_length, null, peak["top1_offset_mode"],
                    100 * peak["top1_offset_mode_frac"],
                    peak["activation_ratio_mean"], peak["activation_offset_mode"])

    logger.info("finished: %s", args.out / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
