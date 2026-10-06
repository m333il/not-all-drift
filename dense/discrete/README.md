# GEPA and analysis code for the dense-model section

Code behind the GEPA side of the dense-model experiments (Gemma 2 2B-IT, five-label
Civil Comments): split generation, GEPA prompt optimization, strict/set evaluation,
activation extraction, probes, tuned lens, mean-shift steering, shift geometry, and
the figures built from them. Prompt tuning, prefix tuning, SAE analyses and the shift
predictors live in the continuous-methods code base and are only read here as inputs.

No datasets, model weights, activations or results are included.

## Layout

| Path | Contents |
|---|---|
| `src/interpretability_gepa/` | Package and the `interp` CLI |
| `src/interpretability_gepa/civil_splits_v2/` | Civil Comments v2 split generator |
| `scripts/interp/` | Standalone analysis scripts (steering, geometry, strict/set table, prompt export) |
| `scripts/paper/` | Figure data export and figure rendering |
| `configs/` | Experiment config and fixed analysis thresholds |
| `tests/` | Offline CPU tests |

Every directory has a short README.

## Setup

Python 3.12 and `uv`:

```bash
uv sync --group dev                                   # core + tests
uv sync --extra models --extra gepa --extra data      # GPU / GEPA / data work
uv sync --extra serving                               # vLLM for the task model
uv sync --extra plot                                  # figures
```

The GEPA reflector is called through OpenRouter; set `OPENROUTER_API_KEY`.
`interp preflight --config configs/experiments/core.yaml` checks the config and
environment without network access.

Every `interp` command writes to a fresh directory
`<output.root>/<UTC timestamp>_<command>_<config hash>` together with the resolved
config and a provenance manifest.

## Reproduction order

1. **Splits.** Build the v2 multilabel splits from the Civil Comments parquet files:

   ```bash
   python -m interpretability_gepa.civil_splits_v2.cli \
     --train-parquet /path/to/train.parquet --test-parquet /path/to/test.parquet \
     --output-root /path/to/splits --setup multilabel
   ```

2. **GEPA.** Serve `google/gemma-2-2b-it` with vLLM at the endpoint listed in
   `configs/experiments/core.yaml`, then run one cell per (seed, train size):

   ```bash
   interp gepa optimize --config configs/experiments/core.yaml \
     --splits-dir /path/to/splits/civil_comments_splits_v2_multilabel \
     --model-key gemma2_2b --reflector api --seed 42 \
     --set dataset.train_size=1000 --set dataset.optimizer_val_size=200
   ```

   The paper uses seeds 42, 43, 44 and train sizes 200, 500, 1000 with at most
   10^4 metric calls.

3. **Test evaluation.** `interp evaluate --split-file test.jsonl --instruction-file
   <run>/optimized_instructions.txt ...` stores raw responses with both the strict and
   the set parse. `scripts/interp/civil_strict_vs_set_table.py` builds the strict/set
   table for GEPA and the continuous methods.

4. **Prompt registry.** `interp probes build-registry --run-dir <gepa runs>` freezes the
   selected prompts; `scripts/interp/export_prompt_bundle.py` exports them as a bundle.

5. **Activations and probes.** `interp activations extract` (per condition, with
   `--prompt-registry/--prompt-id` for GEPA or `--adapter` for prompt/prefix tuning),
   then `interp probes assemble`, `fit-frozen`, `evaluate-frozen`, `select-layer`.

6. **Tuned lens.** `interp tuned-lens train` once per model, then `interp tuned-lens
   score` per condition.

7. **Steering and geometry.** See `scripts/interp/README.md`: mean-shift steering with
   random controls and bootstrap CIs, mean-shift energy and effective rank, cosine
   geometry with the padding control.

8. **Figures.** `scripts/paper/export_appendix_data.py`, then
   `scripts/paper/make_figures.py`.

## Tests

```bash
uv run pytest
uv run ruff check .
```

The tests run on CPU and do not download data or weights.
