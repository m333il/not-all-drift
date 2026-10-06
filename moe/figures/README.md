# MoE figures

Plotting only. Every number comes from files written by the measurement
scripts, and nothing is recomputed or filled in here.

## Routing displacement

```bash
python make_moe_figures.py --maps <routing maps> --out figures
```

Writes `moe_displacement.pdf`, the total variation distance between each arm's
expert shares and the frozen base's, layer by layer, on the comment tokens
(`--stage comment`). `<routing maps>` is laid out as
`<model>/<cell>/expert_counts.npz`, as written by
`moe/pruning/scripts/measure_routing_map.py`.

## Attention shares

```bash
python make_attention_figures.py --logs <mechanism logs> --panels panels.json --out figures
```

| Output | Input |
|---|---|
| `moe_attention_grid_qwen.pdf`, `moe_attention_grid_gptoss.pdf` | `attn-q-<arm>.txt` and `attn-g-<arm>.txt`, the stdout of `probe_mechanism.py --conditions intact,ablated` for `base`, `gepa`, `prompt-m500` and `prefix-m500` |
| `moe_attention_prefix_lr.pdf` | the same logs for `prefix-m500-init`, `prefix-m500` and `prefix-m500-lr1e-4` on Qwen |
| `moe_attention_gptoss_arms.pdf` | gpt-oss logs of `base`, `prompt-m500`, `prefix-m100`, `prefix-m200`, `prefix-m500` and `prefix-m500-init`, layers 0 to 3 |
| `moe_attention_qwen_civil.pdf`, `moe_attention_qwen_wiki.pdf` | `panels.json` from `probe_expert_channel.py` |

The logs are read from the `ATTN` lines, which hold the share of attention that
queries from position 3 on give to the virtual tokens or prefix keys, to real
positions 0, 1 and 2, to the learned sink (gpt-oss only) and to the rest of the
sequence. `panels.json` maps `<arm>-<corpus>` (`base`, `init` or `trained`;
`civil` or `wiki`) to the conditions `intact`, `remove` and `control`, each a
list of five rows for layers 1 to 5 with the keys `prefix`, `pos0`, `pos1`,
`pos2` and `rest`. The drawing code checks that shares are finite, nonnegative
and sum to at most one.
