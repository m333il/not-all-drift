from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


def claim_gate_summary(metrics: dict[str, float | bool]) -> dict[str, bool]:
    """Apply the preregistered minimum criteria without hiding negative results."""
    optimizer = (
        float(metrics.get("optimizer_gain", 0.0)) >= 0.02
        and float(metrics.get("optimizer_gain_ci_low", 0.0)) > 0
        and float(metrics.get("parse_gain_fraction", 1.0)) <= 0.5
    )
    probe = (
        float(metrics.get("probe_selectivity", 0.0)) > 0
        and float(metrics.get("probe_permutation_p", 1.0)) < 0.05
    )
    steering = (
        float(metrics.get("steering_recovery", 0.0)) >= 0.3
        and bool(metrics.get("steering_beats_controls", False))
        and float(metrics.get("parse_failure_rate", 1.0)) < 0.1
    )
    patch = (
        float(metrics.get("patch_recovery", 0.0)) >= 0.5
        and bool(metrics.get("patch_beats_mismatch", False))
    )
    necessity = (
        float(metrics.get("necessity_fraction", 0.0)) >= 0.3
        and bool(metrics.get("necessity_beats_random", False))
        and bool(metrics.get("lm_quality_matched", False))
    )
    sae = (
        bool(metrics.get("sae_reconstruction_gate", False))
        and bool(metrics.get("sae_feature_beats_random", False))
    )
    return {
        "optimizer_gain": optimizer,
        "probe_decodability": probe,
        "global_steering": steering,
        "paired_patching": patch,
        "causal_necessity": necessity,
        "sae_feature_claim": sae,
    }


def build_report(rows: Iterable[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    frame.to_parquet(output_dir / "results.parquet", index=False)
    frame.to_csv(output_dir / "results.csv", index=False)
    summary = {
        "rows": len(frame),
        "conditions": sorted(frame["condition"].dropna().unique().tolist())
        if "condition" in frame
        else [],
        "metrics": sorted(frame["metric"].dropna().unique().tolist()) if "metric" in frame else [],
    }
    target = output_dir / "report.json"
    target.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return target
