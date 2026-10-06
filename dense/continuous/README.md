# Dense Prompt Interpretability

This repository contains reusable training and analysis code for Prompt Tuning and projected Prefix Tuning on a frozen causal language model. It covers the full path from adapter training to layerwise readouts, residual geometry, sparse-autoencoder interventions, learned residual corrections, and real-layer-skipping early exits.

The repository contains source code and path-free hyperparameter configs only. Datasets, model weights, adapters, activation caches, predictions, plots, and logs are external artifacts passed explicitly on the command line.

## Contents

- `scripts/adapters/`: Prompt Tuning and Prefix Tuning training on fixed Civil Comments splits.
- `scripts/probing/`: hidden-state extraction and regularized multilabel probes.
- `scripts/lenses/`: direct Logit Lens plus calibrated Tuned Lens training and evaluation.
- `scripts/attention/`: attention extraction and causal masking of text or virtual context.
- `scripts/geometry/`: per-example shift angles, norms, mean energy, and effective-rank analyses.
- `scripts/sae/`: Gemma Scope fidelity, feature similarity, dense/SAE causal replacement, and feature-subset controls.
- `scripts/predictors/`: full-rank, low-rank, and GELU-MLP residual-shift prediction with teacher-forced caches and autoregressive evaluation.
- `scripts/early_exit/`: normalized DENSE+KL early exits that physically omit late decoder blocks.
- `src/prompt_optimization/`: shared implementations.
- `configs/`: hyperparameters and experiment contracts without filesystem paths.
- `tests/`: CPU tests for core math, data contracts, hooks, and model components.

## Installation

Python 3.11 is recommended. Install a CUDA-compatible PyTorch build for the target machine, then install this package:

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -e '.[dev]'
```

Run commands from the repository root. Each GPU command expects exactly one device to be exposed:

```bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
```

The code never selects, waits for, or reserves a GPU. Resource scheduling is intentionally outside this repository.

## External inputs

The fixed split directory must contain `manifest.json` plus the JSONL files named by the dataset contract. Adapter paths must be PEFT directories. Text conditions use a UTF-8 template containing exactly one `{text}` placeholder. SAE analyses expect compatible Gemma Scope weights or caches supplied by CLI arguments.

All generated artifacts should be written outside the source tree. Retain the emitted configs, hashes, seed, model revision, and raw metrics together with each run.

## Reproduction order

### 1. Train continuous adapters

The complete grids are recorded in `configs/adapters/`. A single Prompt Tuning run is:

```bash
python -m scripts.adapters.train_civil_comments_peft \
  --method prompt_tuning \
  --split-root /path/to/fixed_splits \
  --train-samples 20000 \
  --split-seed 42 \
  --training-seed 42 \
  --num-virtual-tokens 20 \
  --learning-rate 0.03 \
  --lr-scheduler constant \
  --max-epochs 4 \
  --disable-early-stopping \
  --evaluate-both-checkpoints \
  --output-dir /path/to/output/prompt_s42
```

For Prefix Tuning, use `--method prefix_tuning --prefix-projection --learning-rate 0.0001 --lr-scheduler linear_decay`.

### 2. Extract states and fit probes

```bash
python -m scripts.probing.extract_civil_comments_activations \
  --reference-run /path/to/adapter_run \
  --condition adapted \
  --output-dir /path/to/activations

python -m scripts.probing.run_civil_comments_probes \
  --activation-dir /path/to/activations \
  --split-root /path/to/fixed_splits \
  --probe-seed 0 \
  --output-dir /path/to/probes
```

The extracted tensor has shape `[samples, hidden_states, hidden_size]` at the final non-padding prompt token. The probe command fits one-vs-rest logistic classifiers at every hidden state and evaluates all train-condition/test-condition combinations.

### 3. Evaluate the Logit Lens and Tuned Lens

Use `scripts/lenses/evaluate_civil_comments_logit_lens.py` for the direct frozen readout. Use `train_civil_comments_tuned_lens.py` on training activations and `evaluate_civil_comments_tuned_lens.py` on aligned evaluation activations for the calibrated translator. All commands expose their complete argument contracts through `--help`.

### 4. Analyze attention and residual geometry

```bash
python -m scripts.attention.extract_attention \
  --prompt-template /path/to/prompt.txt \
  --adapter-path /path/to/best_adapter \
  --input-file /path/to/test.jsonl \
  --output-dir /path/to/attention

python -m scripts.attention.evaluate_context_masking \
  --prompt-template /path/to/prompt.txt \
  --adapter-path /path/to/best_adapter \
  --input-file /path/to/test.jsonl \
  --labels toxicity obscene threat insult identity_attack \
  --output-dir /path/to/masking
```

Geometry entry points consume aligned activation banks and write per-example plus aggregate metrics. Their required cache layouts are documented by `--help` and validated before computation.

### 5. Run SAE analyses

First run reconstruction fidelity. Only then run shift coverage, all-layer feature similarity, dense replacement, SAE replacement, and feature-subset replacement. The default layer and metric grid is in `configs/sae/default.json`.

Frozen text prompts can be supplied as external conditions. The SAE anchor extractor,
all-layer similarity, shift-coverage, and causal-transfer entry points accept a frozen
GEPA template, while prompt search and optimization remain in the adjacent discrete
project.

### 6. Train residual-shift predictors

The predictor pipeline is:

1. cache aligned baseline/adapted states and adapted-native trajectories;
2. fit a full affine, rank-constrained affine, or GELU-MLP correction with DENSE, DENSE+KL, or DENSE+ENC+DEC+KL;
3. evaluate `prefill_once`, `fixed_recurrent`, or `repredict_recurrent` intervention during free generation.

The entry points are in `scripts/predictors/`; canonical hyperparameters are in `configs/predictors/`.

### 7. Train real-layer-skipping early exits

The DENSE+KL pipeline predicts an aligned Prompt Tuning or Prefix Tuning final-block state from a baseline source-block state using normalized state MSE plus output KL. `prepare_dense_kl_cache.py` accepts multiple named target caches so both methods can be evaluated under the same source-state contract.

`evaluate_dense_kl.py` replaces the decoder block list by blocks `0..source_block`, injects a recurrent predicted residual after the source block, and uses the original final normalization and LM head. The omitted late blocks are not executed.

Use `prepare_dense_kl_cache`, `train_dense_kl`, and `evaluate_dense_kl`; the canonical layer and optimization settings are in `configs/early_exit/dense_kl.json`.

## Verification

```bash
python -m compileall -q src scripts tests
pytest -q
```

The test suite is CPU-only. End-to-end model runs require the external datasets, pretrained weights, adapters, and SAE checkpoints described above.
