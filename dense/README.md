# Dense-model experiments

Section 3 of the paper and its appendix. The backbone is Gemma 2 2B-IT, the
task is five-label Civil Comments, and the arms are the seed prompt, GEPA, prompt
tuning and prefix tuning, with seeds 42, 43 and 44.

| Folder | Scope | Python |
|---|---|---|
| [`discrete/`](discrete/README.md) | Civil Comments v2 splits, GEPA, strict and set-level evaluation, the condition registry shared by GEPA, prompt and prefix runs, probes, tuned lens, mean-shift steering, shift geometry with the padding control, figures | 3.12 |
| [`continuous/`](continuous/README.md) | Prompt and prefix tuning, attention extraction and masking, residual geometry, Gemma Scope SAE analyses and transfer, learned residual corrections, early exits | 3.11 |

The two folders are separate packages with separate environments. Their model
and analysis dependencies are pinned differently, and they exchange only files:
prompts, adapters, activation caches and manifests passed on the command line.

```bash
cd dense/discrete
uv sync --group dev
uv run pytest

cd ../continuous
uv venv --python 3.11
uv pip install -e '.[dev]'
pytest
```

## Paper to code

| Paper | Code |
|---|---|
| Strict versus set-level quality | `discrete/scripts/interp/civil_strict_vs_set_table.py` |
| Probes on seed and adapted states | `interp activations extract`, `interp probes ...` in `discrete/` |
| Tuned lens onsets | `interp tuned-lens train` and `score` in `discrete/` |
| Attention to the adapted context and its masking | `continuous/scripts/attention/` |
| Padding placebo and shift geometry | `discrete/src/interpretability_gepa/conditions.py`, `discrete/scripts/interp/civil_shift_cosine_control.py`, `e3_rank_analysis.py` |
| Mean-shift steering with random controls | `discrete/scripts/interp/civil_e3_steer_sweep.py`, `e3_bootstrap_recovery.py` |
| SAE fidelity, similarity and causal transfer | `continuous/scripts/sae/` |
| Learned corrections | `continuous/scripts/predictors/` |
| Early exits | `continuous/scripts/early_exit/` |
| Figures | `discrete/scripts/paper/export_appendix_data.py`, then `make_figures.py` |

## Comparing methods across the two folders

GEPA is a frozen text template with exactly one `{text}` placeholder. Pass the
complete renderable template, not only the optimized instruction fragment.
Prompt and prefix conditions load PEFT adapters. The baseline is the same frozen
model under the seed template.

For three-way figures, use one implementation for all three conditions. The
discrete pipeline is the reference for probes and the tuned lens, the
continuous one for SAE similarity and SAE transfer. The SAE scripts take a GEPA
template through `--frozen-prompt-file` or `--gepa-prompt-file`, for example:

```bash
cd dense/continuous
python -m scripts.sae.extract_civil_comments_sae_anchor_states \
  --reference-run <matched reference run> --condition gepa \
  --frozen-prompt-file <gepa template> --sample-manifest <sample manifest> \
  --layers 0 6 13 20 25 --output-dir <gepa anchor states>

python -m scripts.sae.analyze_civil_comments_sae_all_layer_similarity \
  --manual-dir <manual> --run prompt:42=<prompt run> --run prefix:42=<prefix run> \
  --run gepa:42=<gepa run> --sae-manifest <sae manifest> --sae-snapshot <sae snapshot> \
  --allow-train-sample-mismatch --allow-virtual-token-mismatch --output-dir <out>
```

Repeat `--run` for every seed. The analyzer checks example order, labels,
model revision, layers and activation shapes before computing any similarity.

Some outputs share a name but not a definition, and should not be averaged:
the continuous logit lens reads the next-token distribution at the prompt
anchor, while the discrete one scores the complete label serialization under
teacher forcing. The two probing pipelines also consume different activation
layouts.
