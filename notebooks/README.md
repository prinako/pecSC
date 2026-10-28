# DeepSC evaluation notebooks

Open `DeepSC_Evaluation_Comparisons.ipynb` using the Python environment for this
repository (`kd-exp`). For example, from the pecSC root:

```bash
uv run --with jupyterlab --with ipykernel jupyter lab notebooks/DeepSC_Evaluation_Comparisons.ipynb
```

The first code cell prints the selected kernel/interpreter and locates the repo.
Run the notebook top to bottom. Default operation reads the committed training
CSVs, manifests and complete paired-validation results; no GPU or local checkpoint
files are required for that analysis. Pandas, NumPy and Matplotlib are already
project dependencies. JupyterLab/ipykernel in the command above are temporary
additional runtime dependencies, not changes to the project's lockfile.

The configuration cell controls training seeds, epoch budget, run root, an explicit
paired-evaluation directory, raw diagnostic paths and export behavior. Main plots:

1. Training/validation CE curves by model and training seed.
2. Validation curves with selected checkpoints for each training seed.
3. Same-seed relative changes from M1 at the selected checkpoints.
4. Recorded epoch time (training + validation + diagnostics, **not inference**).
5. Raw logit KD and feature-loss curves.
6. Repeated paired CE/NLL differences by training seed and channel trial.
7. A paired CE difference heatmap.

The logit-only experiment has one recorded seed, whereas the other Student groups
have four; tables and legends expose sample counts. Selected-checkpoint statistics
and repeated paired-validation statistics are kept separate. Each model is paired
with the same-seed M1, and compatible experiment controls are verified from run
manifests. The repeated summary is recomputed from raw trial metrics, and evaluated
checkpoint hashes must agree with the selected run manifests. Smoke/partial
evaluations are not treated as full results. No significance or final-test claim
is generated automatically.

With `SAVE_EXPORTS=True`, CSV tables, PNG/PDF figures and an analysis provenance
manifest are saved into a new directory under:

```text
results/article_kd/notebook_reports/<UTC timestamp>/
```

Optional `RUN_PAIRED_EVALUATION=True` runs the existing paired evaluator against
local validation data and checkpoint files. Leave it disabled unless you intend
to spend GPU time. It creates a fresh evaluation directory and never trains or
updates model weights. `RUN_DIAGNOSTICS=True` additionally reads explicitly listed
local raw diagnostic CSVs and plots gradient/clipping/parameter trends; those large
CSVs are intentionally excluded from Git.

This notebook is scoped to **teacher-forced validation**. It does not access the
held-out test payload, calculate autoregressive BLEU/semantic similarity, retrain
the Teacher/Students or benchmark receiver inference latency.

Before committing a locally executed notebook, clear its outputs and execution
counts. The committed notebook is deliberately unexecuted and contains no embedded
plots, datasets, checkpoint bytes or machine-specific generated outputs.
