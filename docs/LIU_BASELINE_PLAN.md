# Liu-based receiver adaptation — implementation plan

**Repository:** `prinako/pecSC`, branch `kd-exp`.
**Status:** source mapping and receiver-only loss implemented; matched training and final evaluation pending.
**Imported:** 2026-10-30 from the approved [Prince_SC paper plan](https://github.com/prinako/Prince_SC/blob/9e9743f5b67f6da9465f24940291fe2fd0970b08/docs/PAPER_PLAN.md).
**Scope:** English receiver-only compression for `Prince_SC`; separate from the shared-TX multi-user paper.

This document specifies the M3 adaptation implemented in `train_multi_vocab_one_student.py` and `utils/liu_kd.py`. It establishes no new training result. Recheck the latest run manifests and M1 stability before starting new training; the original prerequisites below are not a claim about the current local run status.

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

An end-to-end reproduction of Liu is outside this planned receiver-only baseline; it would require a separately specified system-level experiment. The implementation below retains two receiver-side signals. No matched training or quantitative result is established by this plan.


## Local completion checklist

- [ ] Verify current Teacher/data identities and M1 stability from local manifests/logs.
- [x] Add a source-to-code mapping with precise source equation/algorithm references and an explicit receiver-side feasibility decision.
- [x] Specify adaptation differences and a proposed validation tuning procedure; freeze the actual trial list before training.
- [x] Implement a distinct M3 method/config identifier; do not silently repurpose the existing KD objective.
- [x] Check loss mathematics/masking, Student-only gradients, frozen Teacher TX/RX, and reuse of the same noisy channel tensor (synthetic CPU checks).
- [ ] Train matched M1/M2/M3 runs with immutable IDs, configs, source revisions and selected checkpoint hashes; preserve validation/test isolation.
- [ ] Evaluate M0–M3 under one frozen protocol and retain raw predictions, corpus metrics and uncertainty.
- [ ] Report quality and measured deployment cost with adaptation limitations in the paper.

Use [KD_EXPERIMENT_PLAN.md](KD_EXPERIMENT_PLAN.md) for the overall execution sequence. The specification and command below are implementation instructions, not measured experimental evidence.

## Audited source-to-code specification — 2026-10-31

Primary source inspected: the supplied published IEEE TWC PDF, *Knowledge
Distillation-Based Semantic Communications for Multiple Users*, pp. 7000–7012.
This implementation follows the equations in the published version, not a
verified official author implementation; no official code was used to establish
unspecified settings.

**Feasibility decision:** a meaningful receiver-side adaptation is feasible,
retaining semantic-decoder and final-prediction knowledge from Table I's
Student 2. It is labelled **“Liu et al. — receiver-side adaptation”**, method ID
`M3_liu_receiver_v1`, and must not be described as an exact reproduction.

| Published element | Implemented mapping / decision |
|---|---|
| Eq. (25), temperature softmax | `masked_student_teacher_kl`: softmax over last axis; D model features for final semantic-decoder outputs, vocabulary classes for logits |
| Eqs. (26)–(27), transfer signals | Retain final semantic-decoder output before dense projection and final logits; no layerwise matching or learned projection |
| Eqs. (26), (28), KL direction | Explicit `sum(p_S * (log p_S - log p_T))`, i.e. Student‖Teacher, with detached Teacher and T² scaling |
| Eq. (28), complementary CE coefficient | `liu_weights`: alpha = 1 − eta_decoder − eta_output, beta = eta_output, gamma = eta_decoder |
| Table I, Student 2 channel-encoder output | Omitted: our Student has no transmitter; shared frozen output supplies no receiver gradient. Its weight is removed and CE is recomputed from retained weights, not silently inherited |
| Table I / nearby simulation prose | Table I lists semantic-decoder transfer for Students 1–2; nearby prose refers to semantic-encoder transfer. We explicitly follow Table I and document the ambiguity |
| Algorithms 1–3, original training pipeline | Original end-to-end transmitter/receiver and MU-interference procedure replaced by existing fixed Teacher TX and shared received tensor; Teacher RX stays frozen/eval/no-grad |
| Initialization / compression | Fresh seeded compact receiver, paired with M1/M2; same four layers/eight heads and feature width. No Teacher layer copying, channel compression or quantization |

For target tokens `y = trg[:, 1:]`, let `m = (y != PAD)` and
`N = max(sum(m), 1)`. The decoder is teacher-forced with `trg[:, :-1]`,
using the existing causal and source-padding attention masks. EOS and language
tokens remain valid. Both transfer terms sum their distribution axis, then
average over the same N valid target positions:

```text
K(a_S, a_T) = T² / N * sum_i m_i sum_j
             softmax(a_S[i]/T)[j] *
             (log_softmax(a_S[i]/T)[j] - log_softmax(a_T[i]/T)[j])
L_M3 = (1 - eta_decoder - eta_output) * CE
       + eta_decoder * K(decoder_S, decoder_T)
       + eta_output  * K(logits_S, logits_T)
```

The Teacher tensors are detached inside K as well as generated under no-grad.
Inputs require identical `[B,T,D]` shapes; mismatched lengths or widths fail
instead of silently cropping. All-PAD finite inputs return zero and zero
Student gradients. Half precision inputs are promoted to float32 for KL.
Training and validation use one common loss dispatch. Both receivers consume
exactly the same `z_noisy` object from one transmitter/channel call per batch.
The Student channel decoder receives gradients through retained receiver losses
but has no separate channel-decoder distillation term in this configuration.

### Explicit local adaptation choices and limits

- **Hard-label loss:** retain the existing M1/M2 CE with label smoothing 0.1,
  rather than infer smoothing from the paper. This is an explicit deviation
  from ordinary unsmoothed hard CE and keeps the local CE control common.
  Unsmoothed token-weighted NLL and corpus perplexity remain separate metrics.
- **Masking/reduction:** feature-axis softmax, shifted-target PAD masking and
  valid-token normalization are declared local choices; these details are not
  sufficiently specified for exact reproduction by the inspected paper.
- **Channel:** keep current unconditional batch RMS `PowerNormalize`. This
  change predates M3 and is now recorded in new manifests; historical
  conditional-power runs cannot be pooled into matched results. Freeze one
  common code revision for new M1–M3 runs. No channel implementation is changed
  by adding this loss.
- **Data and training:** retain the verified 70/15/15 English data, train-only
  32k BPE target, common Teacher, seeded receiver, optimizer and budget. The
  original Teacher-with-interference / Students-without-interference setup,
  different architectures and training procedure are not reproduced.
- **Claims:** receiver quality under the declared local channels may be compared.
  No reproduction of the paper's MU-interference generalization, quantization,
  training-time or quality measurements is established here.
- **Checkpoint rule:** existing minimum `val_ce` is retained for all methods;
  it is smoothed CE averaged over batches. Do not call it token-weighted NLL
  or select by composite `val_loss`. Any later selection-rule correction must
  apply to all matched runs and be recorded as a protocol revision.

### Configuration and validation-only tuning

Use `--method liu-receiver`. Both eta weights and `--temperature` must be
explicit finite numbers; weights are nonnegative, their sum is at most one,
at least one is positive, and T is positive. A zero individual weight is an
explicit M3 ablation. The full adapted Student 2 comparison retains both terms.
`--alpha/--beta/--gamma` are rejected with M3; existing runs keep their defaults
and original Teacher‖Student KL / cosine feature objective.

The inspected source does not establish numerical temperature, retained loss
weights, optimizer, batch size or epoch budget sufficient for exact reproduction.
The following is a **proposed local validation search**, not paper values or
an approved final experiment: T in {1,2,4}, eta_decoder in {0.1,0.3},
eta_output in {0.1,0.3}, giving 12 full two-signal candidates. Before execution,
freeze the trial list, common seed set, per-trial training budget, comparable
M2 tuning budget, and tie rule (lowest mean selected validation CE across seeds;
exact ties use the declared trial order). Choose without opening test data,
then record the winning configuration and immutable checkpoint identities.
No tuning or final hyperparameter selection has been performed by this change.

Example **candidate only**, with user-supplied verified paths and a new run ID:

```bash
python train_multi_vocab_one_student.py \
  --data-dir "$DATA_DIR" --vocab-file vocab_bpe.json \
  --teacher-encoder "$TEACHER_ENCODER" --teacher-decoder "$TEACHER_DECODER" \
  --save-dir "$RUN_ROOT" --experiment-id m3-liu-candidate-s42 \
  --method liu-receiver --liu-eta-decoder 0.1 --liu-eta-output 0.3 \
  --temperature 2 --student_num_layers 4 --student_num_heads 8 \
  --seed 42 --lr 1e-4 --weight-decay 0 --batch-size 32 --epochs 10 \
  --channel Rayleigh --snr-mode range --snr-db-low 0 --snr-db-high 20 \
  --val-snr-db 8
```

All run settings above must match the comparison controls; they are illustrative,
not selected measurements. Manifest and checkpoint metadata include method ID,
weights, temperature, KL direction, feature definition and normalization.
Existing CSV `kd`/`tf` columns denote output KL / decoder-distribution KL for M3;
interpret them using the manifest, not as M2's cosine term.

### Verification and remaining work

Focused tests cover the explicit reverse-KL formula, asymmetry, T², masking,
all-PAD behavior, detached Teacher, valid gradients, input/weight validation,
CLI separation and manifest identity. Integration checks use synthetic tiny
models in active training and validation to verify shared channel observations,
frozen Teacher TX/RX and Student updates. These checks establish implementation
behavior, not trained quality. No GPU experiment or held-out test evaluation is
part of this change. Teacher/data identity, M1 stability, protocol freeze,
matched tuning/training and final M0–M3 decoded evaluation remain pending.


Validation commands (CPU, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`):

```bash
python -m pytest tests -q
python -m pytest deepsc_reproduce/tests -q
```

The suites are invoked separately because they contain two modules named
`test_paired_validation.py`. The M3 CLI smoke test creates synthetic train/val
artifacts and a tiny fresh Teacher, runs one epoch, and checks the selected
checkpoint hash and loss metadata. It does not use publication data or establish
model performance. Implementation base: `e23c3b753f9d35c7aecd6b44f83c56184d9097de`;
use the committed change revision (and a clean working tree) for actual runs.
