# Probing

`extract_civil_comments_activations.py` stores the residual state at the final non-padding prompt token for every hidden state. `run_civil_comments_probes.py` fits regularized one-vs-rest logistic probes per layer and class, including cross-condition evaluation. Fixed probe subsets are read from the split manifest.
