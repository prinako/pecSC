# Liu-based receiver adaptation — implementation plan

**Repository:** `prinako/pecSC`, branch `kd-exp`.
**Status:** receiver-side mapping and `liu_rx_v1` implementation added; CPU synthetic verification only. Matched real-data training and final evaluation remain pending.
**Imported:** 2026-10-30 from the approved [Prince_SC paper plan](https://github.com/prinako/Prince_SC/blob/9e9743f5b67f6da9465f24940291fe2fd0970b08/docs/PAPER_PLAN.md).
**Scope:** English receiver-only compression for `Prince_SC`; separate from the shared-TX multi-user paper.

This document makes the approved M3 comparison plan available in the implementation repository. The original imported plan below is followed by the source audit and implementation specification. No real-data M3 training result is established. Recheck the latest run manifests and M1 stability before starting new training; the original prerequisites below are not a claim about the current local run status.

## External published-method baseline — 2026-10-23

**Planned baseline M3:** Liu et al., *Knowledge Distillation-Based Semantic Communications for Multiple Users*, IEEE TWC 2024, DOI [10.1109/TWC.2023.3336941](https://doi.org/10.1109/TWC.2023.3336941), bibliography key `liu2024kdsemcom`. Primary author manuscript: [arXiv:2311.13789](https://arxiv.org/pdf/2311.13789).

The planned comparison is a **receiver-side adaptation**, labelled **“Liu et al. — receiver-side adaptation”**. The original end-to-end Student/multi-user setting differs from our fixed-transmitter receiver-replacement constraint. Document which original transfer signals and training steps remain applicable, every omitted or changed component, and the resulting limitation. This plan is not a claim of an exact reproduction or a completed experiment.

| Model | Role |
|---|---|
| M0 — Teacher | Original receiver reference |
| M1 — CE-only Student | Compact receiver without distillation |
| M2 — our KD Student | Compact receiver with the implemented three-term objective |
| M3 — Liu-based receiver adaptation | External method adapted to the same deployment constraint |

### Implementation and training sequence

1. Finish M1 stability diagnostics and verify Teacher/data identities.
2. Audit Liu's equations, algorithms, compression configuration and any author code. Produce a source-to-implementation mapping with equation/algorithm references, masks, transfer locations, KL direction/normalization, loss weights/temperature, initialization and training schedule. Do not assume its method equals generic CE plus output KL.
3. Decide and document whether a faithful, meaningful receiver-side adaptation is feasible. If essential method elements cannot be retained, report that limitation and revisit the baseline choice rather than silently assigning Liu's name to an unrelated loss.
4. Implement the mapped adaptation in `pecSC:kd-exp` with a distinct method identifier and focused checks of the implemented loss, frozen transmitter and shared channel observation. Record the exact source revision.
5. Train M2 and M3 only after the stable matched protocol is fixed. Use separate immutable run IDs/manifests and select checkpoints using the common validation-CE rule.
6. Freeze all settings and selected checkpoints before held-out evaluation. Evaluate M0–M3 using identical test samples, channel realizations, decoding and metric definitions. Retain raw predictions and uncertainty across the declared trials/seeds.

### Fairness controls

M1–M3 share the 32k training-only tokenizer and splits, verified frozen Teacher transmitter, four-layer/eight-head receiver architecture, paired initial Student weights/seeds, non-loss optimizer settings, training budget, validation protocol, checkpoint-selection rule and decoding budget. Keep channel normalization, SNR convention, batching and data ordering controlled. Any loss-specific changes required by the published method must be declared explicitly.

Tune method-specific loss weights/temperature only on validation, with a declared comparable tuning budget; do not automatically impose our 0.6/0.3/0.1 weights or T=2 on M3. Record deviations from the original method and how they affect interpretation.

### Article reporting

Section V-A identifies M3, its adaptation and all controls. Section V-B adds M3 to corpus-BLEU/semantic-similarity-versus-SNR plots and the quality/efficiency table after validated runs exist. Identical receiver architectures imply the same parameter count; measured quality and inference differences still require evidence.

A CE-plus-logit-KD variant obtained simply by removing our feature term is a **separate ablation**, not automatically a reproduction/adaptation of Liu's complete method. Published metric values must not be mixed with locally measured results as a controlled numerical comparison.

An end-to-end reproduction of Liu is outside this planned receiver-only baseline; it would require a separately specified system-level experiment. The imported plan alone established no implementation or result; the audited receiver-side implementation is specified below, with real-data training and quantitative claims still pending.


## Local completion checklist

- [ ] Verify current Teacher/data identities and M1 stability from local manifests/logs.
- [x] Add a source-to-code mapping with precise source equation/algorithm references and an explicit receiver-side feasibility decision (see audit below).
- [ ] Specify all adaptation differences and method-specific validation tuning before implementation/training.
- [x] Implement a distinct M3 method/config identifier, `liu_rx_v1`; preserve the existing KD objective.
- [x] Check loss mathematics/masking, Student-only gradients, frozen Teacher TX/RX, and shared noisy channel tensor in CPU synthetic tests (real-run artifact checks remain pending).
- [ ] Train matched M1/M2/M3 runs with immutable IDs, configs, source revisions and selected checkpoint hashes; preserve validation/test isolation.
- [ ] Evaluate M0–M3 under one frozen protocol and retain raw predictions, corpus metrics and uncertainty.
- [ ] Report quality and measured deployment cost with adaptation limitations in the paper.

Use [KD_EXPERIMENT_PLAN.md](KD_EXPERIMENT_PLAN.md) for the overall execution sequence. The audit and a command template below specify the implemented adaptation; hyperparameters and a matched real-data training protocol still require a declared validation-only selection procedure.

## Primary-source audit and receiver-side specification — 2026-10-31

### Source identity and limits

Reviewed the published IEEE TWC version, vol. 23, no. 7, pp. 7000–7012,
DOI `10.1109/TWC.2023.3336941`. The supplied 13-page PDF has SHA-256
`951db1bb6490beee689e6fce1f57212b5630261aed801a9c8fefdcdb567a85e5`.
Equations and Table I were checked visually, in addition to text extraction.
The licensed PDF is not committed to this repository.

No verified official author implementation was located in the title/author/arXiv-ID
searches performed for this audit. This is a search limitation, not a claim that
author code does not exist. The inspected paper does not supply a complete numeric
recipe for temperature, individual distillation weights, LR, batch size, training
epochs and transfer-set size. Do not invent these as recovered author settings.

### What the published method specifies

- Eqs. (25)–(27), printed p. 7005: temperature-softmax and KL over a selected
  subset of semantic-encoder `p`, channel-encoder `x`, channel-decoder `p_hat`,
  semantic-decoder `t_hat`, and prediction `s_hat` outputs. The printed KL order
  is **Student || Teacher**, scaled by `T^2`.
- Eq. (28), p. 7005: hard-label CE coefficient is `1 - sum(eta)`; add each
  `eta * T^2 * KL(Student || Teacher)`. This is not our current forward-KL plus
  decoder cosine objective.
- Table I, p. 7008: Teacher/Student 1 have four semantic encoder/decoder layers,
  eight heads and width 128. Student 2 reduces both semantic sides to two layers.
  Students 1–2 transfer `x`, `t_hat`, and `s_hat`. Students 3–4 additionally
  transfer `p` and `p_hat`, with further channel-layer reduction. Only Student 4
  uses post-training dynamic quantization.
- Algorithms 1–4, pp. 7006–7007: train the full Teacher, train a full Student
  transceiver on a constrained transfer set, and optionally quantize. Teacher
  training includes MU interference; Student transfer training omits it.
  Our receiver-only experiment does not reproduce that robustness setting.

There are two source ambiguities. The prose below Table I says semantic-encoder
outputs where the table lists semantic-decoder outputs for Students 1–2. This
adaptation explicitly follows **Table I**. Also, Eq. (9) defines prediction
`s_hat` after softmax, whereas the discussion of soft targets refers to logits.
We explicitly use **pre-softmax prediction logits** as input to Eq. (25), rather
than silently applying a second softmax to probabilities. This interpretation
is an adaptation choice; it is not established by verified author code.

### Feasibility decision and signal mapping

**Feasible as a partial receiver-side adaptation inspired by Table-I Student 2;
not an exact reproduction.** Our label `M3` is an experiment label, not the
paper's `Student 3`. Keep our common four-layer/eight-head receiver, width 128,
FFN 512 and 32k BPE. The source's end-to-end four-to-two layer compression is
not reproduced.

| Source component | pecSC tensor/location | Decision |
|---|---|---|
| `p`, semantic-encoder output | `Transmitter.encoder` output | Not in the selected Table-I signal subset; shared frozen TX has no trainable Student encoder |
| `x`, channel-encoder output | Shared `Transmitter.channel_encoder` output | Omit: both receivers share the exact TX; this term supplies no receiver gradient |
| `p_hat`, channel-decoder output | Second return from `Student` / `Receiver` | Exclude from this variant; adding it would be a separately identified extension |
| `t_hat`, semantic-decoder output | Third return from `Student` / `Receiver` | Retain; temperature-softmax over the final feature axis |
| `s_hat`, prediction transfer | First return, pre-softmax vocabulary logits | Retain; explicitly declared logit interpretation of the prediction signal |

Omitting `x` and retaining two receiver terms changes the weighting problem. No
omitted source weight is implicitly redistributed; select the two retained weights
on validation and compute the CE coefficient from their sum.

### Exact implemented objective: `liu_rx_v1`

For `Q_T(u) = softmax(u/T, dim=-1)`, define

```text
D(uS,uT) = T^2 * mean_valid_tokens sum_final_axis
           Q_T(uS) * (log Q_T(uS) - log Q_T(stop_gradient(uT)))

L = (1 - eta_decoder - eta_logit) * CE_smoothed_0.1
    + eta_decoder * D(student_decoder, teacher_decoder)
    + eta_logit   * D(student_logits, teacher_logits)
```

`utils/liu_kd.py:student_teacher_kl` implements the stable log-softmax calculation.
`train_liu_rx.py` owns M3 arguments and calls the common training runner with
`liu_distillation_terms`. The `receiver_kd` trainer retains the existing
Teacher||Student KL and normalized decoder cosine-squared feature loss.

- Both retained eta values must be explicit, finite and positive; their sum must
  be below one. The M3 entry point rejects legacy alpha/beta/gamma flags rather than
  ignoring them. Temperature must be finite and positive. Its CLI default of 2
  is our configuration default, not an author-recovered value.
- Exclude PAD rows before softmax; sum along features/vocabulary and average
  across valid shifted target positions. An all-PAD tensor gives differentiable
  zero. START is input-only; EN and EOS targets contribute as in our other arms.
- Use final decoder outputs, not layer-by-layer matching. No feature cosine loss
  or additional channel-decoder term is used in M3.
- Preserve smoothed causal teacher-forced CE (`0.1`) for matched local controls.
  This differs from an exact reproduction of the source's hard-label/masked-word
  description. Explicit axis choice and PAD/token normalization are local
  implementation specifications not fully resolved by the source.
- Teacher TX and RX stay frozen and in eval mode. Both receivers process the same
  received tensor. Only the Student receiver is optimized. No Teacher retraining,
  MU-interference generator, quantization or new decoding procedure is added.
- M3 selects by minimum batch-mean smoothed validation CE. The existing
  trainer currently selects by validation objective loss and records that metric.
  Align checkpoint selection across arms before a matched publication comparison.
  Unsmoothed token-weighted NLL and `exp(NLL)` are additional reporting metrics.

Manifests and checkpoint metadata contain the method ID, effective coefficients,
KL direction and alignment definition. Numeric CSV fields retain the shared schema:
`kd` is raw output KL; `tf` is raw decoder KL for M3 (cosine loss for the legacy
method); alpha/beta/gamma are the effective CE/output/decoder weights. Compare CE
across methods; raw auxiliary terms have different definitions.

### Normalization, tuning and real-run gates

The implementation target is current `batch_unit_rms_v1`: batch-wide RMS scaling
using `clamp_min(1e-12)` before the square root; padding participates. This is
recorded in every new Student manifest, with SHA-256 identities of relevant code.
The change already present in commit `12bc092` differs from historical conditional
RMS capping. It does not update or validate historical results retroactively.

Before launching real training, freeze one normalization convention for **all**
matched arms and the selected Teacher deployment. Do not compare newly trained
M3 under unit-RMS against historical M1/M2 under the cap as a loss-only ablation.
The Teacher weights were trained under the historical convention; changing its
runtime TX normalization must be declared and evaluated consistently. Historical
reproduction uses its original code. No normalization choice is silently patched
by this implementation and no real training is launched here.

Keep identical data/Teacher identities, initialization policy/seeds, architecture,
optimizer, batch size, order, SNR schedule and epoch budget across candidate arms.
Declare a validation-only tuning grid and equal trial budget before selecting
eta/temperature. Keep test unopened. Artifact verification and a real-data M1
stability check remain local gates, even though historical CSV evidence exists.

### Command template — requires explicitly chosen validation candidate

Use the dedicated M3 script for new commands. Existing
`train_multi_vocab_one_student.py --method liu_rx_v1` commands forward to it.

From the pecSC root, set `ETA_DECODER`, `ETA_LOGIT`, `KD_TEMPERATURE` and a new
`RUN_ID` as environment variables. No production values are supplied by this plan.
This command uses the recorded data/Teacher locations and matched training settings;
it is a recipe, not evidence that a run occurred or that normalization is frozen.

```bash
mkdir ./results/article_kd/diagnostics/m3-liu-dec010-log020-t2-seed42 && \
  nohup uv run python train_liu_rx.py \
  --data-dir data/train/europarl_bpe/2026-publication-v2 \
  --vocab-file vocab_bpe.json \
  --teacher-encoder results/article_kd/training/teacher-v2-bpe32k-seed42/encoder_37.pth \
  --teacher-decoder results/article_kd/training/teacher-v2-bpe32k-seed42/decoder_37.pth \
  --save-dir results/article_kd/diagnostics \
  --experiment-id m3-liu-dec010-log020-t2-seed42 \
  --seed 42 \
  --student_num_layers 4 --student_num_heads 8 \
  --num-layers 8 --num-heads 16 \
  --max-len 99 --d-model 128 --dff 512 \
  --channel Rayleigh --snr-mode range \
  --snr-db-low 2 --snr-db-high 18 --val-snr-db 8 \
  --batch-size 32 --epochs 10 --lr 1e-4 \
  --weight-decay 0 --dropout 0.1 --grad-clip 1 \
  --liu-decoder-weight 0.10 --liu-logit-weight 0.20 \
  --temperature 1 --diagnostic-mode \
  > ./results/article_kd/diagnostics/m3-liu-dec010-log020-t2-seed42/logo.txt 2>&1 &
```

### Verification and remaining work

Focused tests in `tests/test_liu_kd.py` cover independent KL arithmetic/direction,
Teacher detach, PAD invariance, valid-token averaging, invalid weights/temperature,
shape checks, shared received tensors, frozen Teacher parameters and Student updates.
The publication pipeline test also runs one tiny synthetic CLI epoch with no test
pickle, then verifies selected-checkpoint identity, method metadata and NLL/perplexity.
These are implementation checks, not publication experiments.

Run the checks with the project environment:

```bash
uv run pytest -q tests/test_liu_kd.py tests/test_publication_pipeline.py \
  tests/test_student_diagnostics.py tests/test_paired_validation.py
```

Remaining: decide/freeze normalization and validation tuning budget; verify local
data/Teacher artifacts; train matched arms; extend the current M1-vs-feature paired
evaluator to include M3 under the frozen protocol; finalize autoregressive corpus
evaluation and deployment benchmarks. Current notebooks and paired scripts do not
automatically establish an M3 comparison. No quality, significance or latency claim
is made by adding this implementation.


Verification recorded for this implementation: 56 focused tests passed on CPU
with PyTorch 2.15.0+cpu, including the tiny one-epoch CLI run. This inspection
runtime differs from the historical 2.13.0+cu130 training runtime; no GPU or
full-corpus run was performed.
