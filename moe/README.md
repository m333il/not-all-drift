# MoE experiments

Section 4 of the paper and its appendix. The backbones are
Qwen3-30B-A3B-Instruct-2507 (48 MoE layers, 128 experts, top-8) and gpt-oss-20b
(24 MoE layers, 32 experts, top-4), both at pinned revisions. The task is the
five-label Civil Comments setup of the dense experiments, scored by set-level
$F_1$. The arms are the frozen base, GEPA, and prompt and prefix tuning with 100,
200 and 500 virtual tokens, all with seed 42.

| Package | Python package | What it covers |
|---|---|---|
| [`adaptation/`](adaptation/README.md) | `mrd` | Training targets, prompt and prefix tuning, GEPA, checkpoint selection on validation |
| [`pruning/`](pruning/README.md) | `mrd_pruning` | Routing maps, load statistics, frequency pruning, token-level attention |
| [`super-experts/`](super-experts/README.md) | `se_gepa` | Super-expert identification and pruning, attention sinks, router retraining, expert-parallel timing |
| [`figures/`](figures/README.md) | | Displacement and attention figures |

## Environments

Two environments. The pruning code was written against Transformers 4.x, the
rest against Transformers 5.16.1 and PEFT 0.20.0. The super-expert code reuses
the prompt rendering of `dense/discrete`, which needs Python 3.12.

```bash
# pruning
uv venv --python 3.11 .venv-pruning
uv pip install --python .venv-pruning/bin/python -e 'moe/pruning[run,plot,dev]'

# adaptation and super experts
uv venv --python 3.12 .venv-moe
uv pip install --python .venv-moe/bin/python -e dense/discrete -e 'moe/adaptation[dev]' -e 'moe/super-experts[dev]'
```

Install a Torch build that matches your CUDA driver first if the default wheel
does not. Every package runs its CPU tests with `pytest` from its own directory.
GPU runs need one device per process, except the expert-parallel benchmark.

## From data to the paper

1. **Splits.** Build the v2 multilabel splits with
   `dense/discrete/src/interpretability_gepa/civil_splits_v2` (see
   [`dense/discrete`](../dense/discrete/README.md)). The arms train on the
   1000-row training split of seed 42 and select checkpoints on its 200-row
   validation split.
2. **Arms.** `adaptation/scripts/prepare_civil_targets.py` turns the splits into
   targets, `train_peft.py` trains prompt (`--method prompt`) and prefix
   (`--method prefix-projected`) adapters, and `evaluate_arms.py`,
   `score_civil.py` and `select_checkpoints.py` pick the step on validation.
   GEPA runs through `optimize_gepa.py` against a task model served by
   `serve_task_model.py`, with `gpt-5.6-luna-pro` as the reflector.
3. **Routing maps.** `pruning/scripts/build_prune_splits.py` draws 500
   calibration rows from validation and the 2000-row test set. Maps are measured
   with `pruning/scripts/measure_routing_map.py --prompt-contract v25`, which
   puts the instruction in the user turn as in training. The GEPA maps of the
   paper are re-measured with `super-experts/scripts/comment_map.py`.
4. **Pruning.** `pruning/scripts/run_pruning_sweep.py` masks the least-used
   experts of every layer, ranked on the arm's own calibration map.
5. **Super experts.** See [`super-experts/`](super-experts/README.md).

| Paper | Code |
|---|---|
| Fig. displacement, depth profile of the displacement | `figures/make_moe_figures.py` on the routing maps |
| Tab. expert load | `pruning/scripts/mk_tables.py`, `pruning/scripts/analyze_maps.py` |
| Reasoning switched off on gpt-oss | `pruning/scripts/probe_reasoning.py` |
| Expert-parallel timing | `super-experts/scripts/ep_bench.py` |
| Tab. frequency pruning, complete pruning results | `pruning/scripts/run_pruning_sweep.py`, `collect_pruning_grid.py`, `mk_prune_table.py` |
| Tab. super experts removed by the masks | `pruning/scripts/plan_pruned_experts.py` |
| Super-expert identification | `super-experts/scripts/profile_super_experts.py` |
| Tab. super-expert magnitudes | `super-experts/scripts/profile_arms.py` |
| Tab. super experts pruned | `super-experts/scripts/score_arms.py`, `row_stats.py`, `analyze_score.py` |
| Attention figures | `super-experts/scripts/probe_mechanism.py`, `probe_sink_budget.py`, `probe_expert_channel.py`, then `figures/make_attention_figures.py` |
| Prefix damage outside the task | `super-experts/scripts/eval_ppl_arms.py`, `probe_prefix_wikitext.py`, `probe_prefix_civil.py` |
| Tab. router retraining | `super-experts/scripts/train_router.py` |
| Measurement caveats | `pruning/scripts/compare_scorers.py`, `check_published_rows.py`, `simulate_ceiling.py`, `rescore_levels.py` |
