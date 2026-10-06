#!/usr/bin/env python3
"""Summarize hash-verified contribution probes with equal weight per example."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


GROUPS = ("virtual", "real_0", "real_1", "real_2", "real_remaining")


def cosine(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else None


def constancy(vectors, energies=None):
    vectors = np.asarray(vectors, dtype=np.float64)
    mean = vectors.mean(axis=0)
    norms = np.linalg.norm(vectors, axis=1)
    pairs = [float(np.dot(vectors[i], vectors[j]) / (norms[i] * norms[j]))
             for i, j in itertools.combinations(range(len(vectors)), 2)
             if norms[i] > 0 and norms[j] > 0]
    result = {"pooled_mean_norm": float(np.linalg.norm(mean)),
              "example_mean_cosine_avg": float(np.mean(pairs)) if pairs else None,
              "example_mean_cosine_min": min(pairs) if pairs else None}
    if energies is not None:
        energy = float(np.mean(energies))
        other_means = (vectors.sum(axis=0) - vectors) / (len(vectors) - 1)
        mse = np.asarray(energies) - 2 * np.sum(other_means * vectors, axis=1) + np.sum(other_means**2, axis=1)
        result.update(rms_l2=float(np.sqrt(energy)),
                      constant_energy_fraction=float(mean @ mean / energy) if energy else None,
                      loo_constant_energy_fraction=float(1 - mse.mean() / energy) if energy else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs, manifests = {}, {}
    for path in sorted(args.runs_root.glob("*/contributions/manifest.json")):
        manifest = json.loads(path.read_text())
        data = path.with_name("metrics.jsonl").read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest["metrics_sha256"]:
            raise ValueError(f"Metric hash mismatch: {path}")
        rows = [json.loads(line) for line in data.splitlines()]
        assert manifest["status"] == "PASS" and len(rows) == manifest["metrics_rows"]
        arm = manifest["arms"][0]["name"]
        assert all(row["arm"] == arm for row in rows)
        assert {(row["example"], row["layer"]) for row in rows} == set(itertools.product(range(manifest["limit"]), manifest["layers"]))
        assert len(rows) == manifest["limit"] * len(manifest["layers"])
        assert manifest["audit_vector_dims"] == 2048 and manifest["limit"] > 1
        for row in rows:
            assert row["sequence_sha256"] == manifest["sequence_hashes"][arm][row["example"]]["sha256"]
            mass = np.array([row["per_head_attention_mass"][g] for g in GROUPS])
            assert np.max(np.abs(mass.sum(axis=0) - 1)) < 0.01
        runs[arm], manifests[arm] = rows, manifest
    assert "base" in runs and len(runs) == 5
    base = manifests["base"]
    for arm, manifest in manifests.items():
        assert manifest["sequence_hashes"][arm] == base["sequence_hashes"]["base"]
        assert manifest["model_revision"] == base["model_revision"]
        assert manifest["layers"] == base["layers"]
    summaries, layers, mean_totals, group_vectors = [], [], {}, {}
    for arm, rows in runs.items():
        for layer in base["layers"]:
            selected = sorted((r for r in rows if r["layer"] == layer), key=lambda r: r["example"])
            totals = sum(np.array([r["projected_contribution"][g]["mean_vector"] for r in selected]) for g in GROUPS)
            mean_totals[arm, layer] = totals
            layer_row = {"arm": arm, "layer": layer, "examples": len(selected),
                         "key_group_cancellation": float(np.mean([r["between_key_group_cancellation"]["mean_fraction"] for r in selected])),
                         "total_mean_constancy": constancy(totals)}
            head_mass = np.mean([r["per_head_attention_mass"]["virtual"] for r in selected], axis=0)
            layer_row["virtual_attention_mass_by_head"] = head_mass.tolist()
            if selected[0]["within_virtual"] is not None:
                layer_row["within_virtual"] = {}
                for key in ("conditional_effective_count", "conditional_top1_mass", "conditional_top5_mass"):
                    values = np.array([[np.nan if v is None else v for v in r["within_virtual"][key]] for r in selected])
                    layer_row["within_virtual"][key] = {"mean": float(np.nanmean(values)),
                                                          "min": float(np.nanmin(values)),
                                                          "max": float(np.nanmax(values)),
                                                          "by_head": np.nanmean(values, axis=0).tolist()}
            layers.append(layer_row)
            for group in GROUPS:
                contributions = [r["projected_contribution"][group] for r in selected]
                group_vectors[arm, layer, group] = np.array([v["mean_vector"] for v in contributions])
                stats = constancy([v["mean_vector"] for v in contributions], [v["rms_l2"]**2 for v in contributions])
                mass = np.mean([r["per_head_attention_mass"][group] for r in selected], axis=0)
                summaries.append({"arm": arm, "layer": layer, "group": group,
                                  "attention_mass": float(mass.mean()), "head_mass_min": float(mass.min()),
                                  "head_mass_max": float(mass.max()), **stats})
    for row in layers:
        arm, layer = row["arm"], row["layer"]
        if arm != "base":
            row["delta_example_mean_constancy"] = constancy(mean_totals[arm, layer] - mean_totals["base", layer])
    directions = []
    for row in layers:
        arm, layer = row["arm"], row["layer"]
        if arm == "base":
            continue
        virtual = group_vectors[arm, layer, "virtual"].mean(axis=0)
        base_sink = group_vectors["base", layer, "real_1"].mean(axis=0)
        delta = (mean_totals[arm, layer] - mean_totals["base", layer]).mean(axis=0)
        record = {"arm": arm, "layer": layer,
                  "virtual_vs_base_real1_cosine": cosine(virtual, base_sink),
                  "virtual_vs_total_mean_delta_cosine": cosine(virtual, delta),
                  "virtual_mean_norm": float(np.linalg.norm(virtual)),
                  "base_real1_mean_norm": float(np.linalg.norm(base_sink)),
                  "total_mean_delta_norm": float(np.linalg.norm(delta))}
        if arm in {"prefix-m500-s42", "prefix-m500-best"}:
            initial = group_vectors["prefix-m500-init", layer, "virtual"]
            record["virtual_change_from_available_init_constancy"] = constancy(group_vectors[arm, layer, "virtual"] - initial)
            record["trained_virtual_vs_init_virtual_cosine"] = cosine(virtual, initial.mean(axis=0))
        directions.append(record)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "group-summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (args.out / "layer-summary.json").write_text(json.dumps(layers, indent=2, allow_nan=False) + "\n")
    (args.out / "direction-comparisons.json").write_text(json.dumps(directions, indent=2, allow_nan=False) + "\n")
    (args.out / "verification.json").write_text(json.dumps({
        "arms": {arm: {"sha256": m["metrics_sha256"], "rows": m["metrics_rows"]} for arm, m in manifests.items()},
        "weighting": "equal examples; query means and second moments within each example; equal heads for attention mass",
        "constant_energy_fraction": "||mean_example mean_query contribution||^2 / mean_example mean_query ||contribution||^2; uncentered, not R squared",
        "loo_constant_energy_fraction": "1 - held-out-example mean squared prediction error / uncentered energy; predictor trained on other examples",
        "delta_limitation": "Only query-mean deltas are available; no per-query cross-model covariance, so no delta RMS or per-query delta energy fraction",
    }, indent=2) + "\n")
    print(f"Verified {len(runs)} arms, {sum(map(len, runs.values()))} metric records")


if __name__ == "__main__":
    main()
