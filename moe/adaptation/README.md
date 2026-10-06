# Adapting the MoE backbones

`mrd` trains the prompt- and prefix-tuning arms on Qwen3-30B-A3B-Instruct-2507
and gpt-oss-20b, runs GEPA against the same backbones, and selects every
checkpoint on validation before anything is measured on test. Both backbones
are loaded through `mrd.models` at the revisions pinned in
`mrd/models/registry.py` (`--model-spec qwen3-2507` or `gpt-oss-20b`).

## Install and test

```bash
uv venv --python 3.12
uv pip install -e '.[dev]'
pytest
```

The tests run on CPU with small randomly initialised models. Tests marked `gpu`
need the real checkpoints and run with `pytest -m gpu`.

## Targets

The data are the v2 multilabel splits of `dense/discrete`. For one seed:

```bash
python scripts/prepare_civil_targets.py --splits-dir <splits>/civil_comments_splits_v2_multilabel \
    --resolved-dataset-revision <dataset commit> --out-dir data/civil_s42
```

This writes `train.jsonl`, `train_targets.jsonl` and `validation.jsonl` from the
1000-row training split and the 200-row validation split, with the target text
in the same output format the GEPA harness asks for.

## Prompt and prefix tuning

```bash
python scripts/train_peft.py --model-spec qwen3-2507 --model-dir <local snapshot> \
    --targets data/civil_s42/train_targets.jsonl --method prompt --virtual-tokens 500 \
    --lr <lr> --out-dir runs/qwen/prompt-m500-s42
```

`--method prompt` learns virtual input embeddings. `--method prefix-projected`
learns per-layer key and value prefixes through PEFT's MLP reparameterization,
which is the prefix tuning of the paper. Training uses AdamW, linear decay and a
checkpoint per epoch under `step_NNNNNN/adapter`. The run writes a
`manifest.json` with the full training contract and the hashes of its inputs.

## Selecting a checkpoint

```bash
python scripts/evaluate_arms.py --model-spec qwen3-2507 --model-dir <local snapshot> \
    --examples data/civil_s42/validation.jsonl --arms arms.json --out-dir runs/qwen/prompt-m500-s42/evaluation
python scripts/score_civil.py --examples data/civil_s42/validation.jsonl \
    --generations runs/qwen/prompt-m500-s42/evaluation/generations.jsonl --output scores.json
python scripts/select_checkpoints.py --evaluation-dir runs/qwen/prompt-m500-s42/evaluation \
    --validation data/civil_s42/validation.jsonl --scores scores.json --output selection.json
```

`arms.json` lists the unadapted model and every saved step, for example
`[{"name": "baseline"}, {"name": "step_000750", "adapter": "runs/.../step_000750/adapter"}]`.
The selected step is the one with the best validation score, the earliest on a
tie. The untrained initialization (step 0) is evaluated as a control and never
selected. `moe/super-experts/scripts/prepare_arm.py` turns the selection into
the arm directory the super-expert probes load.

## GEPA

GEPA needs the task model behind an OpenAI-compatible endpoint and a reflection
model behind an API. The paper uses `gpt-5.6-luna-pro` through OpenRouter.

```bash
python scripts/serve_task_model.py --model-spec qwen3-2507 --model-dir <local snapshot> \
    --port 8010 --out-dir runs/qwen/server
export OPENROUTER_API_KEY=...
python scripts/optimize_gepa.py --train data/civil_s42/train.jsonl \
    --validation data/civil_s42/validation.jsonl --model qwen3-2507 \
    --base-url http://127.0.0.1:8010/v1 --seed 42 --out-dir runs/qwen/gepa-n1000-s42
```

The server checks that it holds the pinned revision, and the optimizer refuses
a server that reports a different one. `--reflection-proxy` routes only the
reflection calls through a proxy.
