# configs

| File | Contents |
|---|---|
| `experiments/core.yaml` | Dataset catalog, pinned model revisions and endpoints, GEPA budget and reflectors, probe / intervention settings, output root |
| `preregistration.yaml` | Primary metric, probe grid, intervention strengths and decision thresholds, fixed before looking at test data |

Values can be overridden on the command line, e.g.
`--set dataset.train_size=500 --set gepa.max_metric_calls=1000`.
