# interpretability_gepa

Package behind the `interp` CLI (`cli.py`).

## Data and prompts

| File | Contents |
|---|---|
| `datasets.py` | Label schemas, JSONL split loading, split path helpers, dataset registry |
| `civil_splits_v2/` | Civil Comments v2 split generator (see its README) |
| `schemas.py` | `Example` and `Prediction` records |
| `prompts.py` | Seed instructions, prompt rendering, strict and set label parsers, length-matched padding |
| `conditions.py` | Condition ladder: null, bland, seed, adapted, and their shuffled / random controls |
| `prompt_registry.py` | Frozen registry of GEPA prompts with hashes |

## GEPA and evaluation

| File | Contents |
|---|---|
| `gepa_runner.py` | GEPA adapter: scoring, reflection feedback, reflection prompt with the output contract |
| `providers.py` | vLLM / OpenAI-compatible / Anthropic clients with usage tracking and failure limits |
| `evaluation.py` | Endpoint and HF generation, strict + set parsing of every response |
| `metrics.py` | Empty-aware samples-F1 and the other multilabel metrics |
| `statistics.py` | Bootstrap, Holm, McNemar |

## Model internals

| File | Contents |
|---|---|
| `modeling.py` | Model loading, readout with Gemma 2 logit softcapping |
| `activations.py` | Residual-stream extraction (last prompt token, mean over text), Zarr stores |
| `probes.py`, `probe_data.py`, `probe_workflow.py`, `probe_stage_cli.py` | Layerwise multilabel probes on frozen subsets, transfer across conditions, layer selection |
| `logit_lens.py`, `logit_lens_cli.py` | Teacher-forced logit lens over full label serializations |
| `tuned_lens.py`, `tuned_lens_cli.py` | Tuned lens: training of per-layer translators and scoring |
| `causal.py` | Residual edits during generation, norm-matched random directions, LEACE |
| `geometry.py` | Cosine, linear CKA, Procrustes distance |

## Infrastructure

| File | Contents |
|---|---|
| `config.py` | Pydantic config schema and `--set key=value` overrides |
| `artifacts.py` | Run directories, hashing, provenance manifests |
| `orchestration.py`, `pipeline_cli.py` | Job matrix expansion and job state |
| `phase0_cli.py` | Hook and HF/vLLM parity checks |
| `preregistration.py`, `decisions.py`, `reporting.py` | Fixed thresholds, claim gates, result tables |
| `errors.py` | Exception types |
