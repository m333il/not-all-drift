# Routing maps and frequency pruning

`mrd_pruning` measures how often each expert is selected under each arm and
removes the least-used experts of every layer by masking their router logits.
A masked expert can never win a top-k slot again, which for quality is the same
as deleting it. The FLOP saving is not measured here, only the damage.

## Install and test

```bash
uv venv --python 3.11
uv pip install -e '.[run,plot,dev]'
pytest
```

The tests need no GPU and no model download. `pytest -m artifacts` additionally
checks finished runs under `results/` and `routing_maps_final/` if they exist.

## Data

```bash
python scripts/build_canonical_test.py --source <v2 test.jsonl> --out data/test_canonical_n2000.jsonl
python scripts/build_prune_splits.py --validation <v2 validation.jsonl> --test data/test_canonical_n2000.jsonl \
    --out-dir data
```

Calibration rows come from validation and never overlap the evaluation rows,
so a mask is never chosen on the data it is scored on. The script writes
`calib_val_n500.jsonl`, the evaluation split and a provenance file. Input files
are JSONL with `comment` and `labels`.

## Routing maps

```bash
python scripts/measure_routing_map.py --arm prompt_tuning --adapter <selected step>/adapter \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 --data data/calib_val_n500.jsonl --n-examples 500 \
    --answers <reference answers> --prompt-contract v25 --out routing_maps_final/qwen/prompt-m500-s42
```

`--arm` is one of `base`, `gepa` (with `--gepa-prompt`), `prompt_tuning` and
`prefix_tuning`. `--prompt-contract v25` renders the instruction in the user
turn with an empty system turn, which is how the arms were trained. Counts are
kept per layer, per expert and per stage of the sequence (`template`,
`comment`, `virtual`, `reasoning`, `answer`, and `__all__` for their sum) in
`expert_counts.npz`. `make_reference_answers.py` builds the `--answers` file from
the `results.jsonl` of an unpruned sweep cell, so every arm is measured on the
same answer text.

`analyze_maps.py`, `mk_tables.py` and `export_per_layer_tables.py` turn the maps
into the load statistics of the paper: Gini over all experts of a layer, the
busiest expert's share and the number of dead experts, as medians or means over
layers. `summarize_routing_grid.py` and `read_maps.py` are lighter readers.

## Pruning

The paper's runs prune every arm by its own calibration map:

```bash
python scripts/run_pruning_sweep.py \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 --revision 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe \
    --data data/test_canonical_n2000.jsonl --n-examples 2000 \
    --arm prompt_tuning:<selected step>/adapter \
    --counts-root routing_maps_final/qwen --counts-arm own --counts-stage __all__ \
    --levels 0,50%,75% --selection per_layer --system-policy as_trained --none-policy lenient \
    --max-new-tokens 512 --prompt-contract v25 --out results/prune/qwen
```

For gpt-oss use `--model openai/gpt-oss-20b --revision 6cee5e81ee83917806bbde320786a8fb61efebee
--reasoning --reasoning-effort low --max-new-tokens 1024`. The budgets were set
from measured truncation (`simulate_ceiling.py`): pruned gpt-oss arms start
writing reasoning again and need the room, while a healthy Qwen answer is under
forty tokens and a row that fills 512 is degenerate.

`--levels` takes counts, percentages and fractions in one list. Fractions
resolve against the real number of experts, so `50%` is 64 experts on Qwen and
16 on gpt-oss. `--layers` restricts which layers may lose experts, `--protect`
pins expert ids that are never removed, and `--selection global` spends the same
budget wherever the routed load is weakest. Every level keeps at least `top_k`
experts per layer, and a level that would break this fails before the model
loads. `--dry-run` prints the routed mass each level would remove without a GPU.

Each cell writes `summary.json`, `results.jsonl` and `pruned_experts.json`. The
summary records $F_1$, exact match, the rates of empty, unparsable and mixed
answers, the mask audit, the sha256 of the adapter files and `pruned_mass`, the
share of routed load the removed experts carried. Pruning 32 of 128 experts that
carry 2% of the load is a different experiment from pruning 32 that carry 20%,
so levels are compared by mass as well as by count.

`collect_pruning_grid.py` and `mk_prune_table.py` collect the cells into the
pruning tables, and `plan_pruned_experts.py` lists which experts each mask
removes, including the super experts.

## Safeguards

| Safeguard | What it prevents |
|---|---|
| `enable_thinking=False` in the single chat renderer | Qwen3 opening `<think>` and spending the budget, which reads as a routing result |
| One renderer for every arm | Two prompt paths that drift apart and mislabel examples |
| Mask value from `finfo(logits.dtype).min` of the live tensor | A float32 constant in a bf16 tensor that silently breaks every example |
| `ExpertMask.assert_applied()` | Hooks that never fire, or a masked expert that still wins a slot |
| `max_unparsable_rate` gate | Formatting bugs reported as quality |
| Level check before the model loads | A level below `top_k` failing after a long load |
| Adapter path check | Loading the last epoch instead of the checkpoint selected on validation |
| `require_on_gpu` | `device_map="auto"` spilling the model to host memory, which looks like a hang |
| Explicit `--system-policy` and `--none-policy` | Hidden differences in the system turn, and voiding answers that end in `", NONE"` |

`tests/data/published_responses.jsonl` holds 77 real responses. The parser and
metric must reproduce their recorded scores exactly under `--none-policy strict`.

## Token-level attention

`measure_token_attention.py` records, on the same pass, the attention each
position receives, the read position's attention by segment and the largest
hidden-state coordinate per position, intact and under a mask. It locates the
attention sink and the massive activation that creates it.
`verify_grouped_exact.py` checks the batched MoE kernel (`--grouped-moe`)
against float64.

## Caveats checked by scripts

The unpruned scores in the paper come from the run that selected each
checkpoint, the pruned ones from this harness. `compare_scorers.py` compares the
two parsers on the same answers, `check_published_rows.py` re-scores the
training run's saved test answers, and `rescore_levels.py` re-scores finished
levels from stored responses without a GPU. `probe_reasoning.py` checks whether
an arm still reasons on gpt-oss and how the reasoning effort changes the score.
