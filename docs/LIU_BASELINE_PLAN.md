# Liu-based receiver adaptation — implementation plan

**Repository:** `prinako/pecSC`, branch `kd-exp`.
**Status:** planned; source mapping and feasibility review precede implementation.
**Imported:** 2026-10-30 from the approved [Prince_SC paper plan](https://github.com/prinako/Prince_SC/blob/9e9743f5b67f6da9465f24940291fe2fd0970b08/docs/PAPER_PLAN.md).
**Scope:** English receiver-only compression for `Prince_SC`; separate from the shared-TX multi-user paper.

This document makes the approved M3 comparison plan available in the implementation repository. It adds no executable baseline and establishes no new training result. Recheck the latest run manifests and M1 stability before starting new training; the original prerequisites below are not a claim about the current local run status.

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

An end-to-end reproduction of Liu is outside this planned receiver-only baseline; it would require a separately specified system-level experiment. No baseline implementation, training or quantitative result is established by this plan.


## Local completion checklist

- [ ] Verify current Teacher/data identities and M1 stability from local manifests/logs.
- [ ] Add a source-to-code mapping with precise source equation/algorithm references and an explicit receiver-side feasibility decision.
- [ ] Specify all adaptation differences and method-specific validation tuning before implementation/training.
- [ ] Implement a distinct M3 method/config identifier; do not silently repurpose the existing KD objective.
- [ ] Check loss mathematics/masking, Student-only gradients, frozen Teacher TX/RX, and reuse of the same noisy channel tensor.
- [ ] Train matched M1/M2/M3 runs with immutable IDs, configs, source revisions and selected checkpoint hashes; preserve validation/test isolation.
- [ ] Evaluate M0–M3 under one frozen protocol and retain raw predictions, corpus metrics and uncertainty.
- [ ] Report quality and measured deployment cost with adaptation limitations in the paper.

Use [KD_EXPERIMENT_PLAN.md](KD_EXPERIMENT_PLAN.md) for the overall execution sequence. The source-to-code mapping and M3 command are deliberately not invented here; add them after the primary-method audit and implementation are complete.
