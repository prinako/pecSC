# KD article experiment outputs

This directory is the fixed layout for reproducible publication artifacts.

- `manifests/`: immutable dataset, run, and evaluation manifests.
- `training/`: training logs and selected-checkpoint references.
- `evaluation/`: raw per-sample predictions and grouped metrics.
- `figures/`: figures derived from recorded evaluation tables.

Final test evaluation must not be run until the protocol and selected
checkpoints are frozen. Raw prediction tables use the columns
`sample_id, model, channel, snr_db, trial_seed, reference, hypothesis,
checkpoint`.
