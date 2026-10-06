# Residual-shift predictors

This directory contains the cache, training, and autoregressive evaluation code for bias-only, full-rank affine, rank-constrained affine, and GELU-MLP corrections. The teacher cache first generates adapted-native continuations, then captures baseline and adapted final-block states on the same token prefixes. This avoids aligning independently generated trajectories.

`train_residual_predictor.py` supports three causal contracts:

- `prefill_once`: predict one correction at the prompt anchor;
- `fixed_recurrent`: reuse the anchor correction at every decoding step;
- `repredict_recurrent`: recompute the correction from each current baseline state.

The supported objectives are `dense`, normalized `dense_plus_kl`, and normalized `dense_sae_kl`. The last option reproduces the four-component DENSE+ENC+DEC+KL objective and requires compatible SAE weights through `--sae-npz`. Every active component is normalized by its train-split mean-shift baseline. The selected checkpoint minimizes the same normalized objective on the validation trajectories.

Example:

```bash
python -m scripts.predictors.train_residual_predictor \
  --config configs/predictors/mlp_sae_composite.json \
  --cache-dir /path/to/teacher_cache \
  --architecture mlp \
  --mode repredict_recurrent \
  --objective dense_sae_kl \
  --sae-npz /path/to/sae_params.npz \
  --output-dir /path/to/predictor
```

`evaluate_residual_predictor.py` reads the saved mode, runs the corresponding intervention during baseline generation, and reports standard multilabel scores plus sample F1 with invalid outputs forced to zero.

The rank ablation uses `--architecture low_rank --rank R`. Its canonical grid is `R in {1, 2, 4, 8}`, seed 42, block 25, DENSE objective, with separately trained fixed and repredict predictors; see `configs/predictors/low_rank.json`.

The input-dependence control uses `--architecture bias_only`. It learns only a
single shared vector `b` while preserving the same trajectory cache, objective,
validation selection, and autoregressive intervention contract as the affine
predictor. For a DENSE+KL run use `configs/predictors/bias_only.json` together
with `--objective dense_plus_kl --mode repredict_recurrent`.

`analyze_predictor_geometry.py` reproduces the affine-map comparisons. Supply
every fit as `--run METHOD:OBJECTIVE:SEED=FIT_DIR`; the script averages
same-seed cross-method matrix cosines and within-method cross-seed cosines.
Optional aligned caches (`--cache METHOD=CACHE_DIR`) add sample-level
prediction-to-truth cosine, mean-direction energy, first-step pairwise shift
cosines, and the separate norms of `W h` and `b`.

`evaluate_recurrent_mean_shift.py` implements the constant-shift control. For
each supplied block it computes the mean adapted-minus-baseline vector on the
train cache, adds the scaled vector at prefill and every decoded position, and
selects block/scale on the supplied validation examples. Its summary labels the
result as validation-only: do not treat this selection score as an untouched
test result. The historical control used Prefix Tuning, 500 intervention-val
examples, and a block/scale grid chosen on those same examples.
