# Super experts under prompt adaptation

Super experts are the few routed experts whose down-projection writes the
massive activations that create the attention sink (Su et al., arXiv:2507.23279). They are
rarely selected, so frequency pruning removes them, and a model that still
depends on them collapses. `se_gepa` finds them, follows them across adapted
arms, prunes them, and measures where the attention goes without them.

## Install and test

From the repository root, in the environment shared with `moe/adaptation`:

```bash
uv venv --python 3.12 .venv-moe
uv pip install --python .venv-moe/bin/python -e dense/discrete -e 'moe/adaptation[dev]' -e 'moe/super-experts[dev]'
cd moe/super-experts && pytest
```

The prompt rendering comes from `dense/discrete`, so the MoE arms see exactly
the prompts the dense experiments use. The tests run on CPU with tiny models.
A few skip unless the research splits they check are present locally.

## Arms

An arm is a frozen backbone with either an instruction or a PEFT adapter.
`configs/arms_qwen.json` and `configs/arms_gpt_oss.json` list the arms of the
paper. A text arm names its instruction (`instruction_key` for the seed prompt,
`instruction_file` for a GEPA prompt, `pad_to` for the seed padded to the
length of another prompt). A PEFT arm points to a directory made by
`scripts/prepare_arm.py` from a run of `moe/adaptation`:

```bash
python scripts/prepare_arm.py --selection runs/qwen/prompt-m500-s42/selection.json \
    --generations runs/qwen/prompt-m500-s42/evaluation/generations.jsonl \
    --out arms/qwen3-2507/prompt-m500
```

`--step 0` gives the untrained initialization. The probes check that their
prompt rendering reproduces the token ids recorded in `contract_sample.json`
before measuring anything. For gpt-oss, pass `--chat-pins` with the reasoning
effort and date of `configs/super_experts.json`.

## Experiments

The super-expert sets are in `configs/super_experts.json`. Every script takes
them explicitly, since the expert indices of one model mean nothing on another.

| Paper | Command |
|---|---|
| Identification | `profile_super_experts.py --model <id> --revision <rev> --out ...` on 32 WikiText-2 segments of 2048 tokens. On Qwen3-30B-A3B it must return experts 68, 92 and 82 of layers 1 to 3 before anything else is trusted. |
| Magnitudes per arm | `profile_arms.py --model ... --arms configs/arms_qwen.json --track 1:68,2:92,3:82 --out ...` |
| Task score with the super experts pruned | `score_arms.py --model ... --arms ... --ablate 1:68,2:92,3:82 --out ...`, then `row_stats.py` and `analyze_score.py` |
| Attention with and without them | `probe_mechanism.py --conditions intact,ablated --ablate ...` (Qwen and gpt-oss), `probe_sink_budget.py` (learned sink of gpt-oss) |
| Channel of expert 68 and the random control | `probe_expert_channel.py` |
| WikiText-2 perplexity | `eval_ppl_arms.py --ablate ...` |
| Prefix removed from the first layers | `probe_prefix_wikitext.py`, `probe_prefix_civil.py` |
| Router retraining | `train_router.py`, one arm per run, learning rate and epoch chosen on validation |
| Expert-parallel timing | `make_ep_loads.py --maps <routing maps> --out loads.npz`, then `torchrun --nproc_per_node N ep_bench.py --loads loads.npz --out ...` |
| GEPA routing maps of the paper | `comment_map.py` |

Pruning a super expert zeroes its down-projection (`se_gepa.ablation.ExpertAblation`).
Masking it in the router (`RouterMask`) gives the same attention profile on
gpt-oss. Model runs need one GPU, except `ep_bench.py`, which uses one rank per
GPU and random expert weights of the real shapes.
