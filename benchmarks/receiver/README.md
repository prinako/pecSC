# M0–M3 receiver cost benchmark

Compare the full Teacher receiver (M0), CE-only Student (M1), combined KD
Student (M2), and Liu receiver adaptation (M3) on the same device and
fixed-length autoregressive workload. The benchmark loads each selected receiver
checkpoint, verifies its SHA-256 and size, and checks that the manifests identify
the same Teacher, data, and vocabulary. Students must have the same architecture
and training seed. Manifest file paths may be relocated; place each selected
receiver checkpoint beside its own manifest, retaining the recorded filename.

Run from the pecSC repository root. Point the four variables at the **actual**
matched run manifests from your machine and use their corresponding vocabulary:

```bash
uv run python -m benchmarks.receiver \
  --m0-manifest "${M0_MANIFEST:?Set the Teacher manifest path}" \
  --m1-manifest "${M1_MANIFEST:?Set the CE Student manifest path}" \
  --m2-manifest "${M2_MANIFEST:?Set the combined KD manifest path}" \
  --m3-manifest "${M3_MANIFEST:?Set the Liu Student manifest path}" \
  --vocab-file "${BPE_VOCAB:?Set the matching vocab_bpe.json path}" \
  --output-dir results/article_kd/receiver_benchmark/seed42-b1-src32-steps24 \
  --device cuda --batch-size 1 --source-length 32 --decode-steps 24 \
  --warmup 5 --repeats 30 --input-seed 1001
```

Use `--device cpu` for CPU measurements, or `--device auto` to select CUDA
when available. Use a fresh output directory for each run. Batch size one reports
latency per example; other batch sizes report batch latency. Repeat at additional
declared lengths if the article needs a range of decoding workloads.

The input is one seeded synthetic float32 tensor of shape
`[batch_size, source_length, 16]`, shared by every receiver. This controls the
workload without opening validation or test data or loading the transmitter. Each
timed call runs the channel decoder once, then performs exactly the declared
number of greedy steps from `[START, EN]`. It projects the last decoder position
at each step, does not stop on EOS, and does not use a KV cache. Preparation of
inputs/masks, checkpoint loading, transmitter processing, channel simulation,
and text/metric processing are outside the timing. GPU synchronization brackets
each measurement; warmups are discarded and model order rotates across repeats.

- **summary.csv**: parameter counts (no buffers), parameter reduction versus M0,
  exact receiver checkpoint bytes (includes serialization overhead and metadata),
  median and nearest-rank p95 latency, and throughput at median batch latency.
- **latency_samples.csv**: every timed repeat.
- **benchmark_manifest.json**: workload, device/software, code and artifact
  identities, selected epochs and metrics, and summary.

This measures receiver compute for a controlled workload. It does not estimate
natural sentence latency with EOS stopping or reconstruction quality. M1–M3 have
the same architecture, so parameter counts should match. Report any timing
variation without attributing it to the training objective alone. Checkpoint
selection and training normalization are recorded as context; a separate
held-out quality comparison must use a common experiment protocol.

Synthetic verification:

```bash
uv run pytest -q benchmarks/receiver/test_benchmark.py
```

## Plot the report in Jupyter

After running the benchmark, open the [plot notebook](Receiver_Benchmark_Plots.ipynb):

```bash
uv run --with jupyterlab --with ipykernel jupyter lab benchmarks/receiver/Receiver_Benchmark_Plots.ipynb
```

Set `REPORT_DIR` to your run directory, or leave it as `None` to load the newest
complete report under `results/article_kd/receiver_benchmark/`. Run all cells to
see receiver parameters, checkpoint sizes, median/p95 latency, and the raw timing
trace. Optionally set `EXPORT_DIR` to a new directory for PNG and PDF figures.
The notebook uses saved records; plotting requires no GPU or model checkpoints.

## Measure reconstruction quality

The quality evaluator uses the **same four selected manifests** as the cost
benchmark. It loads the verified Teacher encoder, generates one received tensor
per batch, then decodes it with M0–M3 using the existing BPE greedy decoder.
Every model starts with `[START, EN]`, has the same decode budget, and stops at EOS.
The cost benchmark remains a separate fixed-work measurement.

Start with a small **validation smoke run**. Set the manifest and vocabulary
variables as above, and `DATA_DIR` to the matching publication dataset directory:

```bash
uv run python -m benchmarks.receiver.quality \
  --m0-manifest "${M0_MANIFEST:?Set the Teacher manifest path}" \
  --m1-manifest "${M1_MANIFEST:?Set the CE Student manifest path}" \
  --m2-manifest "${M2_MANIFEST:?Set the combined KD manifest path}" \
  --m3-manifest "${M3_MANIFEST:?Set the Liu Student manifest path}" \
  --data-dir "${DATA_DIR:?Set the matching publication dataset directory}" \
  --vocab-file "${BPE_VOCAB:?Set the matching vocab_bpe.json path}" \
  --output-dir results/article_kd/receiver_quality/val-smoke-seed42 \
  --split val --max-examples 64 --channel Rayleigh \
  --snr-db 4 8 12 --trial-seeds 1001 1002 \
  --batch-size 8 --device cuda
```

Each selected receiver checkpoint must be beside its manifest, and the Teacher
encoder must also be beside the Teacher manifest. The dataset manifest, split ID
file, tokenizer, and chosen split are hash-checked before loading. The command
uses the current batch unit-RMS transmitter normalization, and verifies the
recorded normalization and other training controls. A validation run can report
exploratory comparisons with mismatched controls; its manifest records those
warnings. M1 must still be CE-only, M2 must include both KD terms, and M3 must
identify `liu_rx_v1`.

The semantic encoder defaults to the project's multilingual MPNet model. It
loads through Sentence Transformers and may need to download model weights on
the first run. Use `--semantic-device cuda` if there is enough GPU memory, or leave
the default CPU setting. Record a fixed model commit with `--semantic-revision`
for reproducible runs. The evaluator does not enable remote model code.

| Metric | Definition |
|---|---|
| `bleu` | SacreBLEU corpus BLEU, 13a tokenization and exponential smoothing, 0–100 |
| `chrf` | SacreBLEU corpus chrF++ with word order 2, 0–100 |
| `semantic_similarity` | Mean cosine between normalized sentence embeddings |
| `rouge_l` | Mean stemmed sentence ROUGE-L F1, 0–1 |
| `token_accuracy` | Mean position-wise accuracy after re-encoding both decoded strings with the same BPE model; denominator is the longer token count |

Text metrics use NFKC, lowercase, and collapsed whitespace. A trial score covers
the entire evaluated slice, and the summary reports mean and sample SD across
channel trials for one fixed training seed. The SD is not a confidence interval
or a measure of variation across training seeds.

The output directory contains:

- `predictions.csv`: reference and hypothesis, split row index, model, SNR, trial,
  and receiver checkpoint hash.
- `trial_metrics.csv`: the five metrics for each model/SNR/trial.
- `quality_summary.csv`: means and SDs for plotting.
- `paired_differences.csv`: differences against M1 on each shared trial.
- `quality_manifest.json`: workload, model/dataset/code identities, metric
  signatures, environment, and training-control warnings. The manifest is
  written after a successful run.

Open [Receiver_Quality_Plots.ipynb](Receiver_Quality_Plots.ipynb) to plot the saved
report. It shows the five SNR curves, their trial spread, and paired differences:

```bash
uv run --with jupyterlab --with ipykernel \
  jupyter lab benchmarks/receiver/Receiver_Quality_Plots.ipynb
```

Use `REPORT_DIR` to select a report, or leave it unset to load the newest complete
report under `results/article_kd/receiver_quality/`. Set `EXPORT_DIR` for PNG/PDF.
The notebook reads saved scores and does not load models.

For the article, freeze the checkpoint set, channel/SNR grid, trial seeds,
decoding budget, and semantic model revision using validation first. Then run
with `--split test`, a pinned 40-character `--semantic-revision`, no
`--max-examples`, and a new output directory. Test evaluation rejects unmatched
Student controls and KD checkpoints selected by composite loss. Keep each Student
`training.csv` beside its manifest so the evaluator can verify the selected epoch
and value against the minimum recorded validation CE. M1's recorded
objective loss is equivalent to CE when its weights are 1/0/0. The evaluator does
not reselect checkpoints or compare training seeds as independent channel trials.

Synthetic verification (no real model download or held-out data access):

```bash
uv run pytest -q benchmarks/receiver/test_benchmark.py benchmarks/receiver/test_quality.py
```
