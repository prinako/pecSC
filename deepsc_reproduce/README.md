# DeepSC reproducibility suite

Repeat the channel probe, automated repository tests, training variants, per-batch
diagnostic summaries, selected-checkpoint comparisons and paired validation used in
this project. Run from your **local pecSC repository root**. This bundle adds a
separate folder; it does not overwrite source code, checkpoints or earlier outputs.

Verified source snapshot: `prinako/pecSC:kd-exp`, commit
`670f7fb5a5b8b060aa6647aa6307ae1cee647989`.
You still need your existing dataset and Teacher/Student checkpoint files for GPU
evaluation. Checkpoints and data are not in this download. Reading saved training
CSVs/manifests needs Python only. Training/evaluation uses your existing uv project.

## 1. Install

Download and extract `DeepSC_reproduce_suite.zip` **into pecSC**:

```bash
unzip ~/Downloads/DeepSC_reproduce_suite.zip
python deepsc_reproduce/reproduce.py --help
python -m unittest discover -s deepsc_reproduce/tests -v
```

This package's tests include independent aggregation/selection/preflight checks and
a numerical paired-input test that requires PyTorch. On a machine without Torch,
that one test is explicitly skipped. Run with `uv run python` instead of `python`
to use your project environment and execute the numerical test.

Run the repository's original publication, diagnostic and paired-evaluator tests:

```bash
python deepsc_reproduce/reproduce.py tests
```

The command uses `uv run pytest` in your repository. If pytest is not yet installed,
use your existing development dependency setup (`uv sync --group dev`).
No dependency installation is performed by the suite itself.

## 2. Reproduce existing comparison tables (no retraining, no GPU)

For the saved files currently tracked in your repository:

```bash
python deepsc_reproduce/reproduce.py compare \
  --run-root results/article_kd/diagnostics \
  --paired-dir results/article_kd/paired_validation/ce-vs-feature-20261026-192316 \
  --output-dir results/article_kd/reports/replay-$(date +%Y%m%d-%H%M%S)
```

Outputs:

- `runs.csv`: best epoch/validation CE, final train CE, weights, run completeness,
  mean epoch time, for every available CSV/manifest pair including original wd runs.
- `epoch_metrics.csv`: train/validation CE and loss components at every epoch.
- `matched_comparisons.csv`: M2, feature-only and logit-only versus the same-seed M1.
- `comparison_summary.json`: mean, sample SD, paired differences and explicit seed
  coverage. Original short or incomplete runs appear in `runs.csv`; they are not
  silently treated as matched ten-epoch comparisons.
- `paired_differences.csv`, `per_seed_summary.csv`, `paired_summary.json`: independently
  regenerated from `trial_metrics.csv`, using the recorded evaluation manifest.

The source CSV minima are verified against each manifest's selected checkpoint.
Loss weights must agree with the CSV; matched comparisons require matching data,
Teacher, architecture, optimizer and other controls. Binary checkpoints are not
required for this **saved-record analysis**; this mode does not independently verify
the checkpoint bytes. Actual reevaluation below does verify their hashes.

Missing expected runs are listed in the JSON rather than silently filled with
invented values. Comparisons use only actually matched seeds. An invalid or stale
manifest/CSV pair stops the analysis with the relevant path.

Expected results for the recorded October 2026 runs:

| Comparison | Seeds | Candidate mean best val CE | Matched M1 mean | Relative change |
|---|---|---:|---:|---:|
| Combined M2 | 42–45 | 2.3498793916 | 2.0981242156 | +11.9991% |
| Feature-only | 42–45 | 2.0902713862 | 2.0981242156 | -0.3743% |
| Logit-only | 42 | 2.3013937938 | 2.0931811207 | +9.9472% |

Paired reevaluation: token-weighted CE difference = **-0.006156192781584592**;
sample SD across training seeds = **0.01113798782533854**; features favored in
**3 of 4** training seeds. These are validation findings, not test results or
statistical-significance claims. Negative differences favor the candidate.

## 3. Repeat raw diagnostic analysis

The large per-batch CSV files were excluded from Git. Use your retained local
copies; this mode streams them and does not retain all batches in memory:

```bash
python deepsc_reproduce/reproduce.py diagnostics \
  --input results/article_kd/diagnostics \
  --output-dir results/article_kd/reports/diagnostics-$(date +%Y%m%d-%H%M%S)
```

`--input` can also be one CSV file. The output `diagnostics_by_epoch.csv` includes
batch count, CE, mean/max gradient norm, clipping percentage at threshold 1,
parameter norms before optimizer steps, nonfinite flags and CE for fades |h|<0.1
versus other fades. Counts expose partial epochs; the tool does not infer that a
partial diagnostic is a complete training epoch. Nonfinite numeric scalars produce
an explicit failure rather than a misleading average.

## 4. Repeat the scalar Rayleigh check

```bash
python deepsc_reproduce/reproduce.py channel-check
```

This uses Torch, seed 42 and 100,000 complex coefficients, reporting magnitudes,
power-amplification p99/max and equalized scalar noise powers at 0, 8 and 20 dB.
It reproduces the earlier mathematical probe; it does not measure Student losses
or prove that fading caused deterioration. Floating-point/library versions can
affect exact values.

## 5. Rerun paired validation with the saved eight Students

First verify artifacts and then perform a two-batch smoke check:

```bash
python deepsc_reproduce/reproduce.py paired \
  --run-root results/article_kd/diagnostics \
  --output-dir results/article_kd/paired_validation/preflight-unused \
  --preflight-only

python deepsc_reproduce/reproduce.py paired \
  --run-root results/article_kd/diagnostics \
  --output-dir results/article_kd/paired_validation/smoke-$(date +%Y%m%d-%H%M%S) \
  --trial-seeds 1001 --max-batches 2
```

Preflight does not create the output directory. Smoke output is marked partial.
For a new complete paired evaluation in the background:

```bash
mkdir -p results/article_kd/logs
log_file="results/article_kd/logs/paired-repeat-$(date +%Y%m%d-%H%M%S).log"
nohup python -u deepsc_reproduce/reproduce.py paired \
  --run-root results/article_kd/diagnostics \
  --output-dir results/article_kd/paired_validation/repeat-$(date +%Y%m%d-%H%M%S) \
  </dev/null >"$log_file" 2>&1 &
printf 'PID: %s\nLog: %s\n' "$!" "$log_file"
```

Defaults are training seeds 42–45, channel trial seeds 1001–1005, Rayleigh 8 dB,
batch size 32 and teacher forcing. Each batch's received tensor is generated once
and reused by all receivers. Both smoothed CE and unsmoothed token NLL are reported.
The inference unit is the training seed after averaging channel trials within it.
Twenty seed/trial combinations are not twenty independent training replicates.
No optimizer is created. Only the validation pickle is loaded; train/test payloads
are not opened. Runtime, code and input artifact identities are recorded.

## 6. Repeat fresh training experiments (optional, expensive)

All training modes use the **existing frozen Teacher epoch 38**, publication
dataset, 32k BPE, maximum length 67, batch 32, fresh seeded Student, four layers /
eight heads, width 128 / FFN 512, dropout 0.1, constant LR 1e-4, Adam betas
(0.9,0.98), epsilon 1e-8, clip 1, Rayleigh train 0–20 dB, validation 8 dB,
label smoothing 0.1 and diagnostic logging. Teacher architecture is eight layers /
sixteen heads. There is no Teacher retraining in this suite.

| Variant | alpha | beta | gamma | Adam weight decay |
|---|---:|---:|---:|---:|
| reference | 1 | 0 | 0 | 0.0005 |
| m1 | 1 | 0 | 0 | 0 |
| m2 | 0.6 | 0.3 | 0.1 | 0 |
| feature | 1 | 0 | 0.1 | 0 |
| logit | 1 | 0.3 | 0 | 0 |

Always use a **new run root**. Preview the exact commands without executing:

```bash
python deepsc_reproduce/reproduce.py train \
  --run-root results/article_kd/reproductions/repeat-01 \
  --variants m1 m2 feature logit --seeds 42 43 44 45 --epochs 10
```

That plan contains **16 runs / 160 epochs**. Add `reference` explicitly if you also
want the original nonzero-decay arm (20 runs / 200 epochs total). Do not rerun all
arms just to regenerate tables. To confirm feature-only on additional matched
training seeds with the configuration fixed, preview instead:

```bash
python deepsc_reproduce/reproduce.py train \
  --run-root results/article_kd/reproductions/new-seeds-46-49 \
  --variants m1 feature --seeds 46 47 48 49 --epochs 10
```

After reviewing a plan, add `--execute` to run it. Example background execution:

```bash
mkdir -p results/article_kd/logs
log_file="results/article_kd/logs/new-seeds-$(date +%Y%m%d-%H%M%S).log"
nohup python -u deepsc_reproduce/reproduce.py train \
  --run-root results/article_kd/reproductions/new-seeds-46-49 \
  --variants m1 feature --seeds 46 47 48 49 --epochs 10 --execute \
  </dev/null >"$log_file" 2>&1 &
printf 'PID: %s\nLog: %s\n' "$!" "$log_file"
```

Jobs run sequentially. Any existing destination causes failure before the first job
starts. Any training failure stops the sequence. No automatic resume, deletion or
skip is performed. All raw training stdout/stderr is streamed into `.log.gz` files
inside the new run root; the outer log records starts/completions. Inspect a child
log with `gzip -cd PATH.log.gz | tail -n 30`. This avoids giant uncompressed logs.
Per-batch diagnostic CSVs are still large and must be retained locally.

For a fresh root, regenerate comparison tables using `compare --run-root NEW_ROOT`
and the corresponding `--seeds`. Reevaluate its M1/feature pairs with `paired
--run-root NEW_ROOT --seeds ...`; paired folder templates assume ten epochs.
Selected best epochs may change across runs or runtimes. Same seeds do not promise
bit-identical GPU results across software/hardware versions.

Recommended ignore entries for new reproductions (add yourself if needed):

```gitignore
/results/article_kd/reproductions/**/diagnostics.csv
/results/article_kd/reproductions/**/*.log.gz
```

Keep manifests, training CSVs, code and compact comparison/evaluation outputs under
version control. Do not change the model/candidate weights based on held-out test
results. This suite does not implement autoregressive BLEU, semantic similarity,
latency, a new test evaluation protocol or an optimizer-resume mechanism.

## Verification performed for this deliverable

The suite was run against the saved CSV/manifests at the pinned GitHub commit and
reproduced the comparison means and full paired summary above. Eleven independent
standard-library checks passed; one numerical PyTorch test was skipped because
Torch is unavailable in the assistant environment. No GPU training or real
checkpoint reevaluation was run here. Run the project-environment tests and smoke
check on your machine before starting a full GPU evaluation.
