# Sparse-autoencoder analysis

Run reconstruction fidelity before interpreting SAE coordinates. The remaining entry
points measure recovered shift energy, recovered output KL, signed feature similarity,
dense versus decoded causal replacement, and restricted feature-subset transfer.

Text conditions are external prompt templates. Use
`extract_civil_comments_sae_anchor_states.py --condition gepa
--frozen-prompt-file ...` to place a frozen GEPA prompt in the same aligned activation
contract as Prompt and Prefix. `analyze_civil_comments_sae_all_layer_similarity.py`
then accepts `--run gepa:SEED=DIR`, and the three `evaluate_cross_method_*` entry points
include GEPA when `--gepa-prompt-file` is supplied. The prompt-optimization procedure
itself is not implemented in this project.

For sparse causal transfer, first run
`select_cross_method_feature_sets.py` on a disjoint calibration bank (32
examples in the canonical experiment). `direct_topK` ranks features by the
magnitude of the mean pairwise SAE activation displacement multiplied by the
corresponding decoder-row norm. The manifest records calibration IDs, and
`evaluate_cross_method_sae_feature_subset_gate.py` refuses to evaluate those
IDs. The canonical sparse comparison uses `K=128` at the generation anchor of
block 20 and contrasts exact dense replacement, the full decoded SAE
replacement, and the selected feature subset.

`plot_feature_subset_transfer.py` combines aggregate evaluator CSVs across
seeds and writes PDF/PNG figures plus the exact plotted points. Prompt, Prefix,
and GEPA targets use orange, blue, and green respectively; the script does not
perform feature selection or statistical testing.
