"""Mean-shift energy and effective rank of the seed-relative shift, per layer.

With per-example shifts d_i and their mean v,

    (1/N) sum_i ||d_i||^2 = ||v||^2 + (1/N) ||D - 1 v^T||_F^2,

so eta = ||v||^2 / ((1/N) sum_i ||d_i||^2) is the share carried by the mean
(eta = 1 for identical shifts, about 1/N for isotropic ones). The effective rank of the
centred remainder D - 1 v^T is reported alongside.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ALPHAS = (0.5, 1.0, 2.0, 4.0)
SWEEP_LAYERS = (6, 10, 14, 18)
TOPK = 10


def store_path(root: Path, condition: str) -> Path:
    """The store is the directory holding acts.zarr, nested under the run directory."""
    base = root / condition
    if (base / "acts.zarr").exists():
        return base
    stores = {p.parent for p in base.glob("*/*/acts.zarr")}
    stores |= {p.parent for p in base.glob("*/acts.zarr")}
    if len(stores) != 1:
        raise SystemExit(f"{condition}: expected exactly one store, found {len(stores)}")
    return stores.pop()


def load(root: Path, condition: str):
    from interpretability_gepa.activations import load_activation_store

    return load_activation_store(store_path(root, condition))


def effective_rank(sv: np.ndarray) -> float:
    """Roy-Vetterli entropy-based effective rank."""
    total = sv.sum()
    if total <= 0:
        return 0.0
    p = sv / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def analyse(seed_acts: np.ndarray, target_acts: np.ndarray) -> list[dict]:
    n_layers = seed_acts.shape[1]
    rows = []
    for layer in range(n_layers):
        s = seed_acts[:, layer, :].astype(np.float64)
        t = target_acts[:, layer, :].astype(np.float64)
        d = t - s
        n = d.shape[0]

        v = d.mean(axis=0)
        v_norm = float(np.linalg.norm(v))
        per_ex_sq = (d * d).sum(axis=1)
        mean_sq = float(per_ex_sq.mean())
        mean_norm = float(np.sqrt(per_ex_sq).mean())
        eta = v_norm**2 / mean_sq if mean_sq > 0 else 0.0
        state_norm = float(np.linalg.norm(s, axis=1).mean())

        centred = d - v
        sv_c = np.linalg.svd(centred, compute_uv=False)
        sv_u, u1 = None, None
        # top right singular vector of the uncentred matrix via the Gram trick
        gram = d @ d.T
        evals, evecs = np.linalg.eigh(gram)
        top_left = evecs[:, -1]
        u1 = d.T @ top_left
        u1_norm = np.linalg.norm(u1)
        u1 = u1 / u1_norm if u1_norm > 0 else u1
        sv_u = np.sqrt(np.clip(evals[::-1], 0, None))

        energy_c = sv_c**2
        cum = np.cumsum(energy_c) / energy_c.sum() if energy_c.sum() > 0 else energy_c
        k50 = int(np.searchsorted(cum, 0.50) + 1)
        k90 = int(np.searchsorted(cum, 0.90) + 1)

        cos_v_u1 = float(abs(v @ u1) / v_norm) if v_norm > 0 else 0.0

        row = {
            "layer": layer,
            "v_norm": v_norm,
            "mean_shift_norm": mean_norm,
            "coherence_rho": v_norm / mean_norm if mean_norm > 0 else 0.0,
            "eta": eta,
            "eta_isotropic_floor": 1.0 / n,
            "state_norm": state_norm,
            "v_rel_to_state": v_norm / state_norm if state_norm > 0 else 0.0,
            "erank_centred": effective_rank(sv_c),
            "srank_centred": float((sv_c**2).sum() / sv_c[0] ** 2) if sv_c[0] > 0 else 0.0,
            "erank_uncentred": effective_rank(sv_u),
            "k50": k50,
            "k90": k90,
            "cos_v_top_sv": cos_v_u1,
        }
        if layer in SWEEP_LAYERS:
            row["applied_rel_perturbation"] = {
                str(a): a * v_norm / state_norm if state_norm > 0 else 0.0 for a in ALPHAS
            }
        rows.append(row)
    return rows


def subspaces(seed_acts: np.ndarray, target_acts: np.ndarray, layers) -> dict:
    out = {}
    for layer in layers:
        d = target_acts[:, layer, :].astype(np.float64) - seed_acts[:, layer, :].astype(np.float64)
        d = d - d.mean(axis=0)
        gram = d @ d.T
        evals, evecs = np.linalg.eigh(gram)
        idx = np.argsort(evals)[::-1][:TOPK]
        basis = d.T @ evecs[:, idx]
        basis /= np.maximum(np.linalg.norm(basis, axis=0, keepdims=True), 1e-12)
        out[layer] = basis  # [hidden, k]
    return out


def principal_angles(a: np.ndarray, b: np.ndarray) -> list[float]:
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    sv = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return [float(np.degrees(np.arccos(np.clip(x, -1, 1)))) for x in sv]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, required=True, help="activation store root")
    parser.add_argument("--output", type=Path, required=True, help="JSON to write")
    parser.add_argument("--seed-condition", default="C_seed")
    args = parser.parse_args()
    root = args.activations

    targets = [
        "C_adapt_s42_n1000",
        "C_adapt_s43_n1000",
        "C_adapt_s44_n1000",
        "C_prefix_vt500_n1000_s42",
        "C_prefix_vt500_n1000_s43",
        "C_prefix_vt500_n1000_s44",
        "C_prefix_vt100_n1000_s42",
        "C_prompt_vt500_n1000_s42",
        "C_prompt_vt500_n1000_s43",
        "C_prompt_vt500_n1000_s44",
        "C_seed_pad_s42",
        "C_seed_pad_s43",
        "C_seed_pad_s44",
        "C_bland",
    ]

    seed = load(root, args.seed_condition)
    seed_acts = seed.last_prompt
    print(f"C_seed last_prompt shape {seed_acts.shape}", flush=True)

    results = {"seed_shape": list(seed_acts.shape), "conditions": {}}
    bases = {}
    for name in targets:
        tgt = load(root, name)
        if tgt.example_ids != seed.example_ids:
            raise SystemExit(f"{name}: example ids not aligned with C_seed")
        rows = analyse(seed_acts, tgt.last_prompt)
        results["conditions"][name] = rows
        bases[name] = subspaces(seed_acts, tgt.last_prompt, SWEEP_LAYERS)
        sel = {r["layer"]: r for r in rows}
        print(
            f"{name:28s} "
            + " ".join(
                f"L{layer}: eta={sel[layer]['eta']:.4f} erank={sel[layer]['erank_centred']:6.1f}"
                for layer in SWEEP_LAYERS
            ),
            flush=True,
        )

    # cross-branch geometry at the swept layers: GEPA vs prefix, same seed
    cross = {}
    for s in (42, 43, 44):
        g, p = f"C_adapt_s{s}_n1000", f"C_prefix_vt500_n1000_s{s}"
        for layer in SWEEP_LAYERS:
            angles = principal_angles(bases[g][layer], bases[p][layer])
            cross[f"s{s}_L{layer}"] = {
                "principal_angles_deg": angles,
                "mean_angle_deg": float(np.mean(angles)),
            }
    results["cross_branch"] = cross

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
