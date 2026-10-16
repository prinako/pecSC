# KD Receiver-Compression Publication — Experiment Preparation

**Repository:** `prinako/pecSC`, branch `kd-exp`, created from `BPE`.
**Manuscript:** `prinako/Prince_SC` (not the shared-TX multi-user paper).

## Objective
Provide reproducible final evidence for the IEEE paper comparing:
- **M0:** freshly trained full DeepSC Teacher.
- **M1:** four-layer/eight-head compact receiver trained with CE only; frozen Teacher TX.
- **M2:** same compact receiver trained with receiver-only KD; frozen Teacher TX.

An external published-method baseline (e.g., a carefully adapted KD-SemCom configuration) may be added after the three-model core is validated. Label adaptations accurately.

## Stage 1 — Blocking preparation
- [ ] Validate all split ratios and normalized-content overlap across partitions.
- [ ] Create a fresh, train-only 70/15/15 BPE dataset with manifest and SHA-256 hashes. Confirm vocabulary/special tokens/max length and retained counts.
- [ ] Add explicit `--data-dir` and related vocabulary/tokenizer paths across Teacher, Student, and evaluation entry points.
- [ ] Eliminate unnecessary test-set access during Student training.
- [ ] Replace hardcoded Teacher checkpoint filenames with explicit CLI paths; verify checkpoint model and positional-buffer compatibility.
- [ ] Standardize independent run directories and complete run metadata, including git commit, CLI args, library versions, and RNG seed.

## Stage 2 — Training
- [ ] Fresh Teacher training; select checkpoint on validation CE.
- [ ] Define one **comparable validation selection rule** for CE-only and KD Students.
- [ ] Train CE-only (alpha=1, beta=0, gamma=0) and KD (candidate 0.6/0.3/0.1, T=2) with matched architecture/initialization/optimizer/budget/channel.
- [ ] Collect training and validation loss breakdown, checkpoint identity and logs.

## Stage 3 — Evaluation
- [ ] Build deterministic EN->EN BPE greedy decoder and optional matched beam decoder.
- [ ] Fix SNR/channel grid, independently seeded noise trials, evaluation batch sizes, metric signatures and semantic embedding version **before testing**.
- [ ] Produce raw per-sentence predictions with `sample_id,model,channel,snr_db,seed,reference,hypothesis,checkpoint`.
- [ ] Aggregate **corpus BLEU** per model/channel/SNR/trial; semantic similarity, optional chrF/ROUGE/CER/WER and uncertainty across trials.
- [ ] Measure receiver/full-system parameter counts, serialized checkpoint size, and synchronized warmed-up inference latency on documented hardware.
- [ ] Compare matched CE-only vs KD and run loss ablations if resources permit.

## Stage 4 — Article handoff
- [ ] Produce validated tables/figures for `Prince_SC` Experimental Methodology and Results and Discussion.
- [ ] Record source revision, dataset/tokenizer/checkpoint hashes and exact reproduction commands.
- [ ] Do not import old 90/10 checkpoint results, or claim model compression from decoder layer counts alone.

## Existing implementation requiring audit
- `main_multi_vocab.py` — Teacher trainer; verify dataset root and run identity.
- `train_multi_vocab_one_student.py` — authoritative receiver-only KD; hardcoded Teacher paths and test-set loader need fixing.
- `data_perprocess/preprocess_text_mu_bpe.py` — split ratio/duplicates/dated outputs.
- `dataset_multilingual.py` — dataset root currently defaults to an undated path.
- `metrics/metrics_bpe.py` — corpus-level aggregation needs correct grouped handling.

**Status:** Branch/setup only. No training, metrics or benchmark runs performed by this preparation step.
