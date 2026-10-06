# Adapter training

`train_civil_comments_peft.py` trains one Prompt Tuning or projected Prefix Tuning adapter on an immutable split. It saves the initial adapter, the best validation-score checkpoint, the best validation-loss checkpoint, epoch history, predictions, hashes, and environment metadata. Filesystem paths and the concrete grid are supplied externally; reference hyperparameters are in `configs/adapters/`.
