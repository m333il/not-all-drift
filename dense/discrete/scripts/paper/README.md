# scripts/paper

| Script | What it does |
|---|---|
| `export_appendix_data.py` | Reads the paired activation stores and the outputs of `e3_rank_analysis.py` and `civil_shift_cosine_control.py`, writes `dense_appendix_e3.json` (cosine distributions, eta / effective rank curves, movement at layer 18) |
| `make_figures.py` | Renders the dense-section figures as PDF and PNG |

```bash
python export_appendix_data.py --e3-root /path/to/e3 --output figdata/dense_appendix_e3.json
python make_figures.py --figdata figdata --attention /path/to/attention --output figures_dense
```

`make_figures.py` expects in `--figdata`: `lens_curves.csv` (tuned-lens margins per
condition and layer), `selected_layer.json` (probe layer selection) and
`dense_appendix_e3.json`. The attention and masking panels read the JSON files produced
by the attention analysis of the continuous-methods code. Numbers of the quality and
steering tables are written inline at the top of the script.
