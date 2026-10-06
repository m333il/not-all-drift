# Real-layer-skipping early exit

`prepare_dense_kl_cache.py` → `train_dense_kl.py` → `evaluate_dense_kl.py` predicts an aligned adapted final state from an earlier baseline state using normalized DENSE+KL. Pass multiple `--target-cache NAME=PATH` arguments to prepare aligned Prompt Tuning and Prefix Tuning targets under the same source-state contract; train and evaluate each named condition separately.

During evaluation, decoder blocks after `source_block` are removed from the executed module list. The learned correction is applied recurrently at the source block, followed directly by the original final normalization and LM head. The omitted late blocks are not executed.

`benchmark_latency.py` measures the corresponding real inference shortcut and
full-depth native conditions. Loading, tokenisation, warm-up, and decoding are
outside the synchronized timed region. The canonical latency protocol uses
`configs/early_exit/latency.json`: 100 fixed test examples, batch size 16, one
warm-up batch, two repeats, and exactly 16 generated tokens. Paths in the
example config are relative to the config file and must be replaced with the
released split, adapters, and fitted predictors.
