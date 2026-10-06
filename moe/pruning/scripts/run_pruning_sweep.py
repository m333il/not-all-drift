#!/usr/bin/env python3
"""Sweep: frequency-prune N experts per layer from a ready arm, measure quality.

One cell of the sweep is (arm, frequency source, pruning level). The model is
loaded once per arm and every level runs against the same weights, because the
pruning is a router mask rather than surgery on the checkpoint.

Each cell writes its own directory as soon as it finishes, so an interrupted
sweep resumes instead of restarting, and a cell that fails a quality gate
leaves an ``error.json`` rather than a summary that reads like a result.

Example:

    uv run scripts/run_pruning_sweep.py \\
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
        --data data/civil_multilabel_test.jsonl --n-examples 500 \\
        --arm base --arm prompt_tuning:checkpoints/epoch_002 \\
        --counts-npz routing/qwen/layers/frozen_base/expert_counts.npz \\
        --counts-arm base --counts-stage comment \\
        --levels 0,8,16,32,48,64 --system-policy as_trained \\
        --out results/pruning_sweep_$(date +%Y%m%d)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mrd_pruning.arms import ArmSpec, load_arm, provenance  # noqa: E402
from mrd_pruning.evaluate import (  # noqa: E402
    GenerationConfig, QualityGates, evaluate, has_virtual_tokens,
)
from mrd_pruning.frequency import (  # noqa: E402
    ExpertCounts, load_counts_npz, parse_layer_spec, pruned_set_stats, resolve_levels,
    select_pruned,
)
from mrd_pruning.masking import (  # noqa: E402
    ExpertMask, RoutingTally, iter_prune_levels,
)
from mrd_pruning.routers import load_router  # noqa: E402
from mrd_pruning.task import (  # noqa: E402
    SEED_SYSTEM_PROMPT, render_user_prompt, render_user_prompt_v25,
)

logger = logging.getLogger("pruning_sweep")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="backbone model id")
    p.add_argument("--revision", default=None, help="pin the model revision")
    p.add_argument("--data", required=True, type=Path, help="jsonl with 'comment' and 'labels'")
    p.add_argument("--n-examples", type=int, default=500)
    p.add_argument("--arm", action="append", required=True,
                   help="NAME or NAME:ADAPTER_PATH; repeatable. NAME in "
                        "{base,gepa,prompt_tuning,prefix_tuning}")
    p.add_argument("--gepa-prompt", type=Path, default=None, help="file with the GEPA system prompt")
    p.add_argument("--system-policy", default="as_trained",
                   choices=["as_trained", "seed_all", "none_all"],
                   help="as_trained keeps each arm's training-time system turn, which is "
                        "asymmetric between base and the PEFT arms by construction")
    p.add_argument("--counts-npz", type=Path, default=None,
                   help="single expert_counts.npz; whose column is used is set by "
                        "--counts-arm. Mutually exclusive with --counts-root")
    p.add_argument("--counts-root", type=Path, default=None,
                   help="directory of per-arm routing maps (<root>/<arm>/expert_counts.npz). "
                        "Each arm is then pruned by its own measured usage")
    p.add_argument("--counts-arm", default="base",
                   help="which arm's frequencies drive pruning: 'base' prunes every arm by the "
                        "same set, 'own' uses each arm's own counts")
    p.add_argument("--counts-stage", default="comment")
    p.add_argument("--levels", default="0,32,64,80,96,104,112,120",
                   help="experts pruned per layer: absolute counts, percentages or "
                        "fractions, mixed freely, e.g. '0,8,25%%,0.5'")
    p.add_argument("--selection", default="per_layer", choices=["per_layer", "global"],
                   help="per_layer removes the level from every eligible layer; global "
                        "spends the same total budget where the load is weakest")
    p.add_argument("--layers", default="all",
                   help="which layers may lose experts: 'all', a range like '0-15' or "
                        "'32-47', or a list like '0,3,7'. Layers outside keep everything")
    p.add_argument("--protect", default="",
                   help="expert ids never pruned in any layer, comma separated")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--reasoning", action="store_true",
                   help="let the model reason instead of suppressing it; needs a "
                        "much larger --max-new-tokens")
    p.add_argument("--reasoning-effort", default="medium",
                   choices=("low", "medium", "high"),
                   help="how much it may reason. 'low' is what the published runs "
                        "pinned and what the arms were evaluated under; 'medium' is "
                        "the template's own default and scores base 0.075 higher "
                        "because it writes thirty times the analysis")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grouped-moe", action="store_true",
                   help="consider Qwen experts as one bmm instead of a cycle; "
                        "order of reduction, therefore NOT for the main set of levels")
    p.add_argument("--reserve-mb", type=int, default=0,
                   help="take so many megabytes on the map before loading the scales, "
                        "So the neighbor doesn't pick them up in the five minutes they load. "
                        "Shards; 0 - do not reserve")
    p.add_argument("--pad-batches", action="store_true",
                   help="batch by near-equal length with left padding instead "
                        "of strictly equal length - ~3.7x fewer batches on the "
                        "2000-example set. Ignored for arms with virtual "
                        "tokens, where padding would shift them.")
    p.add_argument("--none-policy", default="lenient", choices=["lenient", "strict"])
    p.add_argument("--max-unparsable-rate", type=float, default=None,
                   help="override the gate that rejects a cell whose responses "
                        "parse as neither NONE nor a label. Raise it only when "
                        "unusable output is the arm's own property and the run "
                        "is still worth recording - a GEPA arm that echoes its "
                        "instructions, say. The rate lands in the summary either "
                        "way, so nothing is hidden by loosening it.")
    p.add_argument("--router", default=None,
                   help="router.safetensors of a retrained gate, applied after the adapter. "
                        "Use ARM=PATH,ARM=PATH to give each arm its own, or a bare PATH for "
                        "every arm. Omit for the frozen-router half of the 2x2")
    p.add_argument("--allow-unpinned-checkpoint", action="store_true")
    p.add_argument("--record-routing", action="store_true",
                   help="count, per layer, which experts won a slot while the "
                        "mask was on, and write expert_counts_pruned.npz beside "
                        "the summary. The map measured before pruning says "
                        "where the traffic used to go; this says where it went "
                        "instead. Counted over the evaluated generation itself, "
                        "so it costs one topk per layer per forward and no "
                        "extra run.")
    p.add_argument("--allow-map-mismatch", action="store_true",
                   help="prune by a map measured from a different checkpoint "
                        "than the one being scored (normally a mistake)")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--prompt-contract", choices=("v24", "v25"), default="v24",
                   help="v24 puts the instruction in a system turn and the text first; "
                        "v25 ('user-only') has no system instruction and leads with the "
                        "instruction, then the labels, then the text. An arm scored under "
                        "the wrong contract is shown a prompt it never trained on.")
    p.add_argument("--overwrite", action="store_true", help="recompute cells that already exist")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve and print the plan (levels, pruned mass, cells) without "
                        "loading the backbone")
    return p.parse_args(argv)


def build_arm(token: str, args: argparse.Namespace) -> ArmSpec:
    name, _, adapter = token.partition(":")
    # Under the v25 contract nothing goes in a system turn - the instruction is
    # part of the user message, GEPA's optimised one included.
    v25 = args.prompt_contract == "v25"
    if name == "base":
        return ArmSpec(name="base", kind="base",
                       system_prompt_text=None if v25 else SEED_SYSTEM_PROMPT)
    if name == "gepa":
        if args.gepa_prompt is None:
            raise SystemExit("--arm gepa needs --gepa-prompt pointing at the optimised prompt")
        return ArmSpec(name="gepa", kind="gepa",
                       system_prompt_text=None if v25
                       else args.gepa_prompt.read_text().strip(),
                       prompt_in_user_turn=v25)
    if name in ("prompt_tuning", "prefix_tuning"):
        if not adapter:
            raise SystemExit(f"--arm {name} needs an adapter path: {name}:checkpoints/epoch_002")
        return ArmSpec(name=name, kind=name, adapter_path=Path(adapter),
                       system_prompt_text=None,
                       allow_unpinned_checkpoint=args.allow_unpinned_checkpoint)
    raise SystemExit(f"unknown arm {name!r}")


def router_for(arm: ArmSpec, spec: str | None) -> Path | None:
    """Resolve ``--router`` for one arm: a bare path, or ARM=PATH pairs."""
    if not spec:
        return None
    if "=" not in spec:
        return Path(spec)
    mapping = dict(part.split("=", 1) for part in spec.split(",") if part.strip())
    unknown = set(mapping) - {"base", "gepa", "prompt_tuning", "prefix_tuning"}
    if unknown:
        raise SystemExit(f"--router names unknown arms: {sorted(unknown)}")
    path = mapping.get(arm.name)
    return Path(path) if path else None


def load_dataset(path: Path, n: int, contract: str = "v24",
                 instructions: str | None = None) -> tuple[list[str], list[list[str]]]:
    prompts: list[str] = []
    golds: list[list[str]] = []
    render = render_user_prompt if contract == "v24" else (
        lambda comment: render_user_prompt_v25(comment, instructions))
    with path.open() as handle:
        for line in handle:
            if len(prompts) >= n:
                break
            row = json.loads(line)
            prompts.append(render(row["comment"]))
            golds.append(list(row["labels"]))
    if len(prompts) < n:
        raise SystemExit(f"{path} holds {len(prompts)} usable rows, {n} requested")
    return prompts, golds


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def counts_for(arm: ArmSpec, args: argparse.Namespace) -> ExpertCounts:
    """Load the frequencies that decide which experts this arm loses.

    ``--counts-root DIR`` is the routing-map layout: ``DIR/<arm>/expert_counts.npz``
    holds that arm's own measured usage, so every arm drops the experts *it*
    uses least. This is the default question - "can this model afford to lose
    its own tail" - and it needs a map measured with the arm's current prompt,
    not one inherited from an earlier version of it.

    ``--counts-npz FILE`` keeps the older behaviour: one file, and
    ``--counts-arm`` picks whose column inside it is used for every arm.
    """
    if args.counts_root is not None:
        path = Path(args.counts_root) / arm.name / "expert_counts.npz"
        if not path.exists():
            raise SystemExit(
                f"no routing map for arm {arm.name!r} at {path} - measure it first "
                "with scripts/measure_routing_map.py"
            )
        return load_counts_npz(path, arm=arm.name, stage=args.counts_stage)
    key = args.counts_arm
    if key == "own":
        key = npz_arm_key(args.counts_npz, arm.kind)
    return load_counts_npz(args.counts_npz, arm=key, stage=args.counts_stage)


def map_identity(counts_path: Path) -> dict:
    """Which map file this run was masked from, by path and by content.

    `map_matches_arm` compares adapter digests and so says nothing for the two
    cells that carry no adapter - plain `base` and GEPA, where the arm is a
    prompt. For those the only thing that ties the mask to the right checkpoint
    is the map file itself, and a path alone would not survive the file being
    replaced under the same name.
    """
    path = Path(counts_path)
    digest = None
    if path.is_file():
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
    return {"path": str(path), "sha256": digest}


def check_map_matches_arm(arm: ArmSpec, counts_path: Path, *, allow_mismatch: bool) -> dict:
    """Refuse to prune an arm by a map measured from a different checkpoint.

    Two adapter trees can hold the same cell names and different weights -
    `adapters/` and `adapters-v25/` here differ in every file - and nothing in
    the run would say so: the mask would be built from one checkpoint's routing
    and the quality measured on another's. The map records the sha256 of what
    it was measured with, so the mismatch is detectable before the backbone is
    loaded, which is the only place it costs nothing.

    Returns what was compared, so the cell's summary can carry it.
    """
    import numpy as np

    with np.load(counts_path, allow_pickle=True) as bundle:
        meta = json.loads(str(bundle["_meta"])) if "_meta" in bundle.files else {}
    measured = (meta.get("arm_provenance") or {}).get("adapter_sha256") or {}
    if arm.adapter_path is None or not measured:
        return {"checked": False, "reason": "no adapter or no digest in the map"}

    current = provenance(arm).get("adapter_sha256") or {}
    weights = "adapter_model.safetensors"
    same = measured.get(weights) == current.get(weights)
    report = {
        "checked": True,
        "matches": same,
        "map_adapter": (meta.get("arm_provenance") or {}).get("adapter_path"),
        "map_sha256": measured.get(weights),
        "arm_adapter": str(arm.adapter_path),
        "arm_sha256": current.get(weights),
    }
    if same:
        return report
    message = (
        f"{arm.name}: the map was measured from {report['map_adapter']} "
        f"(sha {str(report['map_sha256'])[:16]}…) but this run loads "
        f"{report['arm_adapter']} (sha {str(report['arm_sha256'])[:16]}…). "
        "The pruning mask would come from one checkpoint and the quality from "
        "another. Point --arm at the adapter the map names, or pass "
        "--allow-map-mismatch if the mismatch is the experiment."
    )
    if not allow_mismatch:
        raise SystemExit(message)
    logger.warning("%s (allowed explicitly)", message)
    return report


def npz_arm_key(path: Path, kind: str) -> str:
    """Which arm inside a counts file this arm should read.

    A map measured per cell holds exactly one arm, and its name is whatever
    `measure_routing_map.py` was told - `prefix_tuning`, not the `prefix` this
    used to assume. A single-arm file therefore answers the question itself,
    which is both correct and immune to the naming drifting again. The name
    table is kept only for the older shared files that hold several arms.
    """
    import numpy as np

    with np.load(path, allow_pickle=True) as bundle:
        arms = sorted({k.split("|", 1)[0] for k in bundle.files if k != "_meta"})
    if len(arms) == 1:
        return arms[0]
    table = {"base": "base", "gepa": "gepa",
             "prompt_tuning": "prompt_tuning", "prefix_tuning": "prefix"}
    guess = table[kind]
    if guess not in arms:
        raise SystemExit(
            f"{path.name} holds arms {arms}; none of them is {guess!r} for a "
            f"{kind} arm - pass --counts-arm explicitly"
        )
    return guess


def run_cell(model, tokenizer, arm, level, counts, prompts, golds, args, out_dir: Path,
             router_report=None, map_check: dict | None = None) -> dict:
    layers = parse_layer_spec(args.layers, counts.layer_ids)
    protect = [int(x) for x in args.protect.split(",") if x.strip()]
    pruned = select_pruned(counts, level, top_k=args.top_k, selection=args.selection,
                           protect=protect, layers=layers)
    mass = pruned_set_stats(pruned, counts)
    # Padding is safe only without virtual tokens, and it is worth ~3.7x on the
    # 2000-example test set, where equal-length groups average nine rows. The
    # decision is made from the loaded model, not from the arm's name.
    gen = GenerationConfig(max_new_tokens=args.max_new_tokens,
                           batch_size=args.batch_size, reasoning=args.reasoning,
                           reasoning_effort=args.reasoning_effort,
                           allow_padding=args.pad_batches
                           and not has_virtual_tokens(model))
    gates = (QualityGates() if args.max_unparsable_rate is None
             else QualityGates(max_unparsable_rate=args.max_unparsable_rate))
    system_prompt = arm.system_prompt(args.system_policy)

    tally = (RoutingTally(n_layers=counts.counts.shape[0],
                          n_experts=counts.n_experts)
             if args.record_routing else None)
    with ExpertMask(model, pruned, top_k=args.top_k, tally=tally) as mask:
        result = evaluate(
            model, tokenizer, prompts, golds, system_prompt,
            config=gen, gates=gates, none_policy=args.none_policy,
            context=f"{arm.name}@prune{level}",
            # Level 0 is the control: there the harness must be provably sane,
            # so the gates stay strict. Above it the model is deliberately
            # damaged and degenerate output is the measurement.
            intervened=level > 0,
        )
        mask.assert_applied(expect_layers=None if level == 0 else len(pruned))

    summary = {
        "arm": arm.name,
        "prune_level": level,
        "prune_level_pct": round(100 * level / counts.n_experts, 2),
        "selection": args.selection,
        "layers_eligible": "all" if layers is None else layers,
        "protected_experts": protect,
        "top_k": args.top_k,
        "system_policy": args.system_policy,
        "none_policy": args.none_policy,
        "frequencies": counts.as_metadata(),
        "counts_arm": arm.name if args.counts_root else args.counts_arm,
        "counts_source": "own routing map" if args.counts_root else "shared npz",
        "counts_npz": map_identity(args.counts_npz),
        "map_matches_arm": map_check or {"checked": False},
        "pruned_mass": mass,
        "n_pruned_total": mask.n_pruned_total,
        "mask_audit": {
            "layers_touched": mask.audit.layers_touched(),
            "selected_pruned_experts": mask.audit.max_selected_pruned,
        },
        "generation": gen.as_dict(),
        "model": {"id": args.model, "revision": args.revision, "dtype": args.dtype},
        # Which expert kernel produced this number. The batched form sums the
        # experts in a different order, which moves bf16 activations by ~1e-3 -
        # enough for greedy decoding to pick a different token on a near-tie. A
        # table that mixes the two is subtracting two harnesses again, so the
        # kernel has to travel with the number rather than be remembered.
        "moe_kernel": "grouped_bmm" if args.grouped_moe else "per_expert_loop",
        "arm_provenance": provenance(arm),
        "router": router_report.as_dict() if router_report is not None else "frozen",
        "git_commit": git_commit(),
        **result.as_dict(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (out_dir / "results.jsonl").open("w") as handle:
        for row in result.rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "pruned_experts.json").write_text(
        json.dumps({str(k): v for k, v in pruned.items()}, indent=2)
    )
    if tally is not None:
        # numpy is imported where it is used in this file, not at the top.
        import numpy as np

        routed = tally.as_array().numpy()
        np.savez_compressed(
            out_dir / "expert_counts_pruned.npz",
            **{f"{arm.name}|__all__": routed},
            _meta=json.dumps({
                "what": "Experts who won the slot with the mask on",
                "arm": arm.name,
                "prune_level": level,
                "top_k": args.top_k,
                "data": str(args.data),
                "n_examples": len(prompts),
                "prompt_contract": args.prompt_contract,
                "reasoning": bool(args.reasoning),
                "reasoning_effort": args.reasoning_effort if args.reasoning else None,
                "max_new_tokens": args.max_new_tokens,
                "note": "the quality assessment (test), and the map to "
                        "Pruning - by calibration; you need to compare the shares, not "
                        "absolutes",
            }, ensure_ascii=False),
        )
        logger.info("routing under the mask: %d assignments over %d layers",
                    tally.total, routed.shape[0])
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args = parse_args(argv)
    if (args.counts_npz is None) == (args.counts_root is None):
        raise SystemExit("pass exactly one of --counts-npz or --counts-root")
    level_specs = [x.strip() for x in args.levels.split(",") if x.strip()]
    arms = [build_arm(token, args) for token in args.arm]
    for arm in arms:
        arm.validate()

    # A GEPA arm's optimised text is the instruction of the v25 user turn, not a
    # system prompt: that series has no system message at all.
    v25_instructions = None
    if args.prompt_contract == "v25" and args.gepa_prompt is not None:
        v25_instructions = args.gepa_prompt.read_text().strip()
    prompts, golds = load_dataset(args.data, args.n_examples,
                                  contract=args.prompt_contract,
                                  instructions=v25_instructions)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sweep_config.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items()}, "level_specs": level_specs},
        indent=2, ensure_ascii=False,
    ))

    index: list[dict] = []
    for arm in arms:
        counts = counts_for(arm, args)
        # Resolve and validate everything that depends on the counts before the
        # backbone is loaded: a typo in --layers should cost a second, not a
        # forty-minute load followed by a crash.
        map_check = {"checked": False}
        if args.counts_npz is not None:
            map_check = check_map_matches_arm(
                arm, args.counts_npz, allow_mismatch=args.allow_map_mismatch)
        elif args.counts_root is not None:
            map_check = check_map_matches_arm(
                arm, Path(args.counts_root) / arm.name / "expert_counts.npz",
                allow_mismatch=args.allow_map_mismatch)
        levels = resolve_levels(level_specs, counts.n_experts)
        list(iter_prune_levels(levels, counts.n_experts, args.top_k))
        parse_layer_spec(args.layers, counts.layer_ids)
        logger.info("arm %s: levels %s of %d experts", arm.name, levels, counts.n_experts)

        if args.dry_run:
            plan_layers = parse_layer_spec(args.layers, counts.layer_ids)
            protect = [int(x) for x in args.protect.split(",") if x.strip()]
            for level in levels:
                pruned = select_pruned(counts, level, top_k=args.top_k,
                                       selection=args.selection, protect=protect,
                                       layers=plan_layers)
                mass = pruned_set_stats(pruned, counts)
                index.append({
                    "arm": arm.name, "prune_level": level,
                    "prune_level_pct": round(100 * level / counts.n_experts, 2),
                    "layers_pruned": mass["layers_pruned"],
                    "mass_pruned_mean": round(mass["mass_pruned_mean"], 4),
                    "mass_pruned_max": round(mass["mass_pruned_max"], 4),
                    "system_prompt": "seed" if arm.system_prompt(args.system_policy) else "none",
                })
                logger.info(
                    "  level %3d (%5.1f%%): %2d layers, routed mass removed %.2f%% "
                    "(max layer %.2f%%)", level, 100 * level / counts.n_experts,
                    mass["layers_pruned"], 100 * mass["mass_pruned_mean"],
                    100 * mass["mass_pruned_max"],
                )
            continue
        # One card, named explicitly. A worker already holds exactly one GPU and
        # `CUDA_VISIBLE_DEVICES` makes it device 0, so there is nothing for
        # `device_map="auto"` to plan - and planning is what went wrong: with 75
        # GB free on the card it still put twenty-four of Qwen's layers on the
        # host, because its budget comes from the card's total capacity and the
        # neighbours' share of it, not from what is actually free. Pinning the
        # whole model to device 0 turns that into an out-of-memory error, which
        # is loud, instead of a CPU-bound run, which is silent.
        # Say which card this actually got, in the log, before loading onto it.
        # The queue picks a card by nvidia-smi's numbering and hands it over as
        # CUDA_VISIBLE_DEVICES; when the two numberings disagreed the log said
        # "GPU 4" - 80 GB, 53 free - while the run sat on a 40 GB card and died
        # there. One line makes that visible without reading a traceback.
        import torch  # noqa: PLC0415 - local by module convention

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            logger.info("GPU: %s, %.1f GB total, %.1f GB free (CUDA_VISIBLE_DEVICES=%s)",
                        props.name, props.total_memory / 2**30,
                        torch.cuda.mem_get_info()[0] / 2**30,
                        os.environ.get("CUDA_VISIBLE_DEVICES", " - "))
        model, tokenizer = load_arm(arm, model_id=args.model, revision=args.revision,
                                    dtype=args.dtype, device_map={"": 0},
                                    reserve_mb=args.reserve_mb)
        if args.grouped_moe:
            # Opt-in, and never for the thirty-two-level set: the batched form
            # reduces in a different order, which moves bf16 activations by
            # ~1e-3 and can flip a greedy argmax on a near-tie. Mixing it into a
            # table half-measured with the loop would put us back to subtracting
            # two harnesses. Measured on qwen/prompt-m200: the loop held 59.6 GB
            # at 99.8% CPU with zero SM share for 109 minutes without finishing
            # its first sixty-four rows.
            from mrd_pruning.grouped_moe import group_qwen3_moe  # noqa: PLC0415

            n_blocks = group_qwen3_moe(model)
            if n_blocks == 0:
                raise RuntimeError(
                    "Grouped MoE requested. No Qwen3-MoE block found: "
                    "It is not necessary for gpt-oss, there experts are considered batch.")
        # After the adapter, never before: PEFT rewraps the modules, and the
        # gates have to be the ones the adapter is actually running through.
        router_path = router_for(arm, args.router)
        router_report = load_router(model, router_path) if router_path else None
        try:
            for level in levels:
                suffix = "_router" if args.router else ""
                cell_dir = args.out / f"{arm.name}{suffix}_prune{level:03d}"
                if (cell_dir / "summary.json").exists() and not args.overwrite:
                    logger.info("skip %s (already done)", cell_dir.name)
                    index.append(json.loads((cell_dir / "summary.json").read_text()))
                    continue
                try:
                    index.append(run_cell(model, tokenizer, arm, level, counts,
                                          prompts, golds, args, cell_dir,
                                          router_report, map_check))
                except RuntimeError as exc:
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    (cell_dir / "error.json").write_text(json.dumps(
                        {"arm": arm.name, "prune_level": level, "error": str(exc)}, indent=2,
                        ensure_ascii=False))
                    logger.error("cell %s failed: %s", cell_dir.name, exc)
        finally:
            del model
            _free_memory()

    keys = ("arm", "prune_level", "prune_level_pct", "f1_mean", "exact_mean",
            "empty_pred_rate", "pruned_mass", "mass_pruned_mean", "mass_pruned_max")
    table = [{k: row[k] for k in keys if k in row} for row in index]
    (args.out / "index.json").write_text(json.dumps(table, indent=2, ensure_ascii=False))
    logger.info("wrote %d cells to %s", len(table), args.out)
    return 0


def _free_memory() -> None:
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
