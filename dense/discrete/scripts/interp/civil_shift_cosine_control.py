"""Per-example cosine between adapted and seed states, raw and centred.

Centring subtracts the per-layer mean over examples, which removes the component shared
by all inputs. The length-matched padding condition ``C_seed_pad`` is included as a control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.maximum(den, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-condition", default="C_seed")
    parser.add_argument(
        "--conditions",
        default=(
            "C_adapt_s42_n1000,C_adapt_s43_n1000,C_adapt_s44_n1000,"
            "C_prompt_vt500_n1000_s42,C_prompt_vt500_n1000_s43,C_prompt_vt500_n1000_s44,"
            "C_prefix_vt500_n1000_s42,C_prefix_vt500_n1000_s43,C_prefix_vt500_n1000_s44,"
            "C_seed_pad_s42,C_seed_pad_s43,C_seed_pad_s44,C_bland"
        ),
    )
    args = parser.parse_args()

    from e3_rank_analysis import load

    seed = load(args.activations, args.seed_condition)
    s = seed.last_prompt.astype(np.float64)
    n_layers = s.shape[1]
    s_centred = s - s.mean(axis=0, keepdims=True)

    results: dict[str, dict] = {}
    for name in args.conditions.split(","):
        acts = load(args.activations, name)
        if acts.example_ids != seed.example_ids:
            raise SystemExit(f"{name}: example ids not aligned with the seed condition")
        t = acts.last_prompt.astype(np.float64)
        t_centred = t - t.mean(axis=0, keepdims=True)
        raw, centred, norms = [], [], []
        for layer in range(n_layers):
            raw.append(float(cosine_rows(t[:, layer], s[:, layer]).mean()))
            centred.append(float(cosine_rows(t_centred[:, layer], s_centred[:, layer]).mean()))
            ratio = np.linalg.norm(t[:, layer], axis=1) / np.maximum(
                np.linalg.norm(s[:, layer], axis=1), 1e-12
            )
            norms.append(float(ratio.mean()))
        results[name] = {"cos_raw": raw, "cos_centred": centred, "norm_ratio": norms}
        print(
            f"{name:28s} L18 raw={raw[18]:.3f} centred={centred[18]:.3f} norm={norms[18]:.3f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
