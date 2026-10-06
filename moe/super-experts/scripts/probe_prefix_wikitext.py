#!/usr/bin/env python3
"""Paired WikiText NLL and early-layer mechanisms under prefix interventions."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from se_gepa.arms import build_base, device_of, wrap
from se_gepa.prefix_intervention import PrefixKeyMask, PrefixValueIntervention, localization_designs
from se_gepa.profiler import FusedExpertProfiler
from se_gepa.residual import ResidualNormProbe
from probe_attention_contributions import arm_metadata, sequence_hash
from eval_ppl_arms import windows, WIKITEXT_REVISION


def logits_hash(logits):
    digest = hashlib.sha256()
    for part in logits[0].split(128):
        digest.update(part.float().cpu().numpy().tobytes())
    return digest.hexdigest()


def observe_attention(layer, prefix_tokens, records):
    def hook(_module, _args, output):
        weights = output[1][0]
        if not torch.isfinite(weights).all():
            raise RuntimeError("Non-finite attention weights")
        mass = weights[..., :prefix_tokens].float().sum(-1)
        real = weights[..., prefix_tokens:]
        records.append({"layer": layer, "prefix_mass_per_head": mass.mean(-1).tolist(),
                        "prefix_mass_first3_queries_per_head": mass[..., :3].mean(-1).tolist(),
                        "prefix_mass_after3_queries_per_head": mass[..., 3:].mean(-1).tolist() if mass.shape[-1] > 3 else None,
                        "real_early_mass_per_head": real[..., :3].float().sum(-1).mean(-1).tolist(),
                        "real_first_mass_per_head": real[..., 0].float().mean(-1).tolist(),
                        "normalization_max_error": float((weights.float().sum(-1) - 1).abs().max())})
    return hook


@torch.no_grad()
def measure(model, tokens, layers, tracked, out, designs=None):
    prefix = model.peft_config[model.active_adapter].num_virtual_tokens if hasattr(model, "peft_config") else 0
    base = model.get_base_model() if prefix else model
    conditions = ["intact", "zero_values", "mask_keys"] if prefix else ["intact"]
    designs = designs or [{"name": c, "mode": c, "layers": layers, "scope": "all"} for c in conditions]
    summaries, reference, gates = [], {}, {}
    with (out / "rows.jsonl").open("w") as stream:
        for design in designs:
            condition, mode = design["name"], design["mode"]
            started = time.time()
            rows = []
            for index, ids in enumerate(tokens):
                tensor = ids.unsqueeze(0).to(device_of(model))
                labels = tensor.clone()
                labels[:, 0] = -100
                native_hash = None
                if index == 0:
                    with ExitStack() as stack:
                        for layer in design["layers"] if prefix else []:
                            if mode == "zero_values":
                                stack.enter_context(PrefixValueIntervention(model, layer, mode="zero", scope="all"))
                            elif mode == "mask_keys":
                                stack.enter_context(PrefixKeyMask(model, layer, scope=design["scope"]))
                        native = model(input_ids=tensor, labels=labels, use_cache=False)
                        native_hash = logits_hash(native.logits)
                        del native
                attention = []
                with ExitStack() as stack:
                    interventions = []
                    for layer in design["layers"] if prefix else []:
                        if mode == "zero_values":
                            interventions.append(stack.enter_context(PrefixValueIntervention(model, layer, mode="zero", scope="all")))
                        elif mode == "mask_keys":
                            interventions.append(stack.enter_context(PrefixKeyMask(model, layer, scope=design["scope"])))
                    for layer in layers:
                        handle = base.model.layers[layer].self_attn.register_forward_hook(
                            observe_attention(layer, prefix, attention))
                        stack.callback(handle.remove)
                    residual = stack.enter_context(ResidualNormProbe(model, layers))
                    profiler = stack.enter_context(FusedExpertProfiler(model, track=set(tracked)))
                    residual.begin_example(index)
                    profiler.begin_example(index, ids.tolist())
                    output = model(input_ids=tensor, labels=labels, use_cache=False)
                    if not torch.isfinite(output.loss) or not torch.isfinite(output.logits).all():
                        raise RuntimeError("Non-finite model output")
                    nll = float(output.loss)
                    if native_hash is not None:
                        if logits_hash(output.logits) != native_hash:
                            raise RuntimeError("Measurement hooks changed logits")
                        gates[condition] = {"observation_logits_exact": True, "logits_sha256": native_hash}
                    if any(hook.calls != 1 for hook in interventions):
                        raise RuntimeError("Intervention call count mismatch")
                    if condition == "intact" and index == 0 and prefix:
                        intact_hash = logits_hash(output.logits)
                    del output
                if len(attention) != len(layers) or len(residual.records) != len(layers):
                    raise RuntimeError("Missing observation layers")
                if mode == "mask_keys":
                    field = {"all": "prefix_mass_per_head", "early3": "prefix_mass_first3_queries_per_head",
                             "after3": "prefix_mass_after3_queries_per_head"}[design["scope"]]
                    if any(any(r[field] or []) for r in attention if r["layer"] in design["layers"]):
                        raise RuntimeError("Masked prefix received attention at a targeted query")
                    expected = min(3, len(ids)) if design["scope"] == "early3" else max(0, len(ids) - 3) if design["scope"] == "after3" else len(ids)
                    if any(h.masked_queries != expected for h in interventions):
                        raise RuntimeError("Wrong number of masked real queries")
                if any(r["normalization_max_error"] > .02 for r in attention):
                    raise RuntimeError("Attention normalization failed")
                if condition == "intact":
                    reference[index] = nll
                peaks = []
                for record in residual.records:
                    position = max(range(len(record.per_position)), key=record.per_position.__getitem__)
                    peaks.append({"layer": record.layer, "max_abs": record.per_position[position],
                                  "position": position, "first_three": record.per_position[:3]})
                experts = []
                for layer, expert in tracked:
                    values = profiler.traces[0].values.get((layer, expert), {})
                    position = max(values, key=values.get) if values else None
                    experts.append({"layer": layer, "expert": expert, "routed_tokens": len(values),
                                    "max_abs": values[position] if values else None, "position": position})
                row = {"condition": condition, "window": index, "sequence_sha256": sequence_hash(ids.tolist()),
                       "targets": len(ids) - 1, "nll": nll, "delta_nll": nll - reference[index],
                       "intervention": design,
                       "masked_queries_per_layer": {str(l): h.masked_queries for l, h in zip(design["layers"], interventions)} if mode == "mask_keys" else {},
                       "attention": attention, "residual": peaks, "experts": experts}
                stream.write(json.dumps(row, allow_nan=False) + "\n"); stream.flush()
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("condition", "window", "nll", "delta_nll")}), flush=True)
                if condition == "intact" and index == 0 and prefix:
                    with ExitStack() as stack:
                        for layer in layers:
                            stack.enter_context(PrefixValueIntervention(model, layer, mode="restore", scope="all"))
                        restored = model(input_ids=tensor, labels=labels, use_cache=False)
                        if logits_hash(restored.logits) != intact_hash:
                            raise RuntimeError("Exact restoration changed logits")
                        del restored
                    gates["restore"] = {"logits_exact": True, "window": 0}
            mean_nll = sum(r["nll"] for r in rows) / len(rows)
            summaries.append({"condition": condition, "windows": len(rows), "targets": sum(r["targets"] for r in rows),
                              "nll": mean_nll, "ppl": math.exp(mean_nll),
                              "paired_delta_nll": sum(r["delta_nll"] for r in rows) / len(rows),
                              "elapsed_seconds": time.time() - started})
            (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return summaries, gates


def main():
    import peft
    import transformers
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--arms-spec", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--layers", default="0,1,2,3,4,5")
    parser.add_argument("--localization-panel", choices=["layers", "positions"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1 or args.seqlen < 2:
        raise ValueError("Require positive windows and at least two tokens")
    if transformers.__version__ != "5.16.1" or peft.__version__ != "0.20.0":
        raise RuntimeError("Runtime differs from pinned attention implementation")
    arms = json.loads(args.arms_spec.read_text())
    if len(arms) != 1:
        raise ValueError("One arm per GPU job")
    metadata = arm_metadata(arms[0])
    if arms[0]["kind"] == "peft" and (metadata["receipt"]["peft_type"] != "PREFIX_TUNING" or
            metadata["receipt"]["base_revision"] != args.model_revision):
        raise ValueError("Expected a prefix on the pinned base")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokens = windows(tokenizer, args.seqlen, args.limit)
    if tokens.shape != (args.limit, args.seqlen):
        raise RuntimeError("Requested WikiText windows are unavailable")
    layers = [int(x) for x in args.layers.split(",")]
    model = build_base(args.model_dir, "cuda")
    if not layers or len(set(layers)) != len(layers) or min(layers) < 0 or max(layers) >= model.config.num_hidden_layers:
        raise ValueError("Invalid intervention layers")
    if arms[0]["kind"] == "peft":
        model = wrap(model, arms)
    args.out.mkdir(parents=True, exist_ok=False)
    designs = localization_designs(args.localization_panel) if args.localization_panel else None
    if designs and (arms[0]["kind"] != "peft" or layers != list(range(6))):
        raise ValueError("Localization requires a prefix and observation layers L0-L5")
    summaries, gates = measure(model, tokens, layers, [(1, 68), (2, 92), (3, 82)], args.out, designs)
    manifest = {"status": "PASS", "probe": "prefix_wikitext_keys_values_v1", "arm": metadata,
                "model_revision": args.model_revision, "dataset": "Salesforce/wikitext", "dataset_config": "wikitext-2-raw-v1",
                "dataset_revision": WIKITEXT_REVISION, "split": "test", "rendering": "raw concatenated text; no chat template",
                "selection": "first non-overlapping windows; remainder dropped", "seqlen": args.seqlen,
                "windows": args.limit, "targets_per_window": args.seqlen - 1, "layers": layers,
                "scope": "all prefill queries; teacher forcing; no free generation", "positions": "native positions unchanged by interventions",
                "sequence_hashes": [sequence_hash(ids.tolist()) for ids in tokens], "gates": gates,
                "metric_rows": len(summaries) * args.limit, "conditions": [s["condition"] for s in summaries],
                "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
                "files_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in args.out.iterdir() if f.is_file()}}
    if designs:
        manifest.update(probe="prefix_wikitext_localization_v1", localization_panel=args.localization_panel,
                        intervention_designs=designs, scope="per-condition real query positions; see intervention_designs")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PREFIX_WIKITEXT_ROWS={manifest['metric_rows']}", flush=True)


if __name__ == "__main__":
    main()
