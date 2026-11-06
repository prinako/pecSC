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
