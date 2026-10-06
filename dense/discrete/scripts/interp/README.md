# scripts/interp

Standalone analysis scripts. Run them with the package installed (`uv sync --extra models`)
from this directory, since `civil_shift_cosine_control.py` imports `e3_rank_analysis`.
Every script documents its arguments in `--help`.

Condition names follow the activation stores: `C_seed`, `C_seed_pad_s<seed>` (length-matched
padding), `C_adapt_s<seed>_n<size>` (GEPA), `C_prompt_vt<m>_n<size>_s<seed>`,
`C_prefix_vt<m>_n<size>_s<seed>`. "E3" is the mean-shift steering experiment.

| Script | Paper item | What it does |
|---|---|---|
| `civil_strict_vs_set_table.py` | Quality figure, quality table | Re-parses GEPA and prompt/prefix predictions on the test split with the strict and the set parser |
| `export_prompt_bundle.py` | GEPA prompt examples | Exports the frozen prompt registry as a bundle with checksums |
| `civil_e3_steer_sweep.py` | Steering figure and table | Adds the mean seed-relative shift at the last prompt token, sweeps layers and relative strengths, runs norm-matched random controls |
| `e3_bootstrap_recovery.py` | Steering CIs | Paired bootstrap over examples for the recovered fraction, pooled over seeds |
| `e3_rank_analysis.py` | Shift structure figure | Mean-shift energy eta and effective rank of the centred remainder per layer |
| `civil_shift_cosine_control.py` | Movement vs. utility figure | Raw and centred cosine to the seed state per layer, with the padding control |

Typical steering run for one target condition (a store is the `activations/` directory
written by `interp activations extract`):

```bash
python civil_e3_steer_sweep.py --config ../../configs/experiments/core.yaml \
  --split-file intervention_val.jsonl \
  --seed-store <C_seed run>/activations --target-store <prefix run>/activations \
  --target-label C_prefix_vt500_n1000_s42 --target-adapter /path/to/adapter \
  --layers 6,10,14,18 --relative-norms 0.05,0.1,0.2 --output sweeps/prefix_s42

python e3_bootstrap_recovery.py --sweeps sweeps --output bootstrap.json
```

For a GEPA target pass `--target-instruction optimized_instructions.txt` instead of
`--target-adapter`.
