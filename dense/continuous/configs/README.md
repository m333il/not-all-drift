# Experiment configurations

Configuration files contain reproducible hyperparameters but intentionally no
machine-specific absolute paths. Replace relative `data/`, `prompts/`, and
`artifacts/` entries with released assets while preserving the recorded model
revision, split IDs, seeds, and protocol parameters.

These JSON files record the adapter grid and the default hyperparameters used by each analysis family. They intentionally contain no dataset, checkpoint, cache, or output paths; those are supplied through the corresponding command-line interfaces.
