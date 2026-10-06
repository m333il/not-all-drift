# civil_splits_v2

Generator of the Civil Comments splits used in the paper. Labels (score >= 0.5):
`toxicity, obscene, threat, insult, identity_attack`.

| File | Contents |
|---|---|
| `contract.py` | Split sizes, nested train ladders, empty fraction, label enrichment, mask floors (multilabel and binary setups) |
| `corpus.py` | Loading and deduplicating the source parquet files, enriched mask distribution |
| `allocation.py` | Row allocation under exact-mask floors and soft distributional targets |
| `generator.py` | Writes all splits, `manifest.json` and pairwise label audits |
| `cli.py` | Command line entry point |

```bash
python -m interpretability_gepa.civil_splits_v2.cli \
  --train-parquet train.parquet --test-parquet test.parquet \
  --output-root /path/to/splits --setup multilabel
```

Optimizer train splits are nested per seed (`optimizer_train_seed42_n200` is contained
in `_n500` and `_n1000`), so GEPA and the continuous methods see the same rows at a
given size. Probe, intervention and test partitions are disjoint from them. The output
directory must not exist.
