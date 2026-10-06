# tests

CPU-only tests with synthetic data and fake providers. No downloads, no GPU.

```bash
uv run pytest
```

| File | Covers |
|---|---|
| `test_civil_splits_v2.py`, `test_datasets.py` | Split generation, nesting, leakage, label schema |
| `test_prompts_metrics.py` | Rendering, strict and set parsing, empty-aware metrics |
| `test_gepa_runner.py`, `test_providers.py` | GEPA adapter, reflection feedback, provider retries and failure limits |
| `test_activations.py`, `test_logit_lens.py`, `test_tuned_lens.py` | Residual extraction, lens scoring and translators |
| `test_probing_pipeline.py` | Prompt registry, probe assembly, fitting, layer selection |
| `test_analysis.py`, `test_conditions.py` | Causal edits, geometry, statistics, condition ladder |
| `test_config_artifacts.py`, `test_study_contracts.py`, `test_reporting_cli.py` | Config, run directories, CLI smoke test |
