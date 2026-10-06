# Attention and context masking

`extract_attention.py` records head-level attention at the generation anchor and during decoding, grouped into BOS, virtual context, instruction, and example text. `evaluate_context_masking.py` masks access to the adapted context while preserving baseline position IDs. Prompt Tuning masks prepended embeddings; Prefix Tuning masks prefix key/value columns; text conditions mask instruction-token columns.
