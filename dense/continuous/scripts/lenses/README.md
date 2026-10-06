# Logit Lens and Tuned Lens

`evaluate_civil_comments_logit_lens.py` applies the frozen final RMSNorm and unembedding directly to each intermediate residual state. It has no learned translator. It records KL and top-1 agreement with the model's final next-token distribution plus per-example first-token probabilities for each label and `NONE`. The stored terminal hidden state is already normalized and is not normalized twice.

`train_civil_comments_tuned_lens.py` and `evaluate_civil_comments_tuned_lens.py` instead fit and evaluate an affine translator per hidden state. The translator objective is KL divergence to the frozen model's final distribution.
