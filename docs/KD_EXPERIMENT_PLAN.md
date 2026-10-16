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
- [x] Validate all split ratios and normalized-content overlap across partitions. Duplicate normalized English sources are retained but grouped into one split.
- [ ] Create a fresh, train-only 70/15/15 BPE dataset with manifest and SHA-256 hashes. Confirm vocabulary/special tokens/max length and retained counts.
- [x] Add explicit `--data-dir` and consistent absolute-or-data-relative `--vocab-file` handling to Teacher and Student training.
- [x] Eliminate test-set access during Teacher and Student training.
- [x] Replace hardcoded Teacher filenames with required `--teacher-encoder` and `--teacher-decoder`; strict loading validates tensor/vocabulary/positional compatibility.
- [x] Standardize non-overwriting run directories and run manifests, including source/data/checkpoint hashes, git identity, CLI/configuration, runtime, hardware, seed, optimizer, SNR, selection rule and selected checkpoint.
- [x] Add the validation-only evaluation schema/layout and grouped corpus-BLEU aggregation.
- [x] Make validation cross-entropy the common CE-only/KD Student checkpoint-selection rule.

The remaining unchecked dataset item requires running preprocessing on the actual
Europarl JSON source and reviewing the resulting counts and hashes. No full
training or held-out test evaluation was performed during Stage 1 implementation.

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

## Reproduction command templates

Choose new immutable identifiers and paths. Do not point these commands at the
historical 90/10 artifacts.

```bash
# 1. Fresh 70/15/15 English reconstruction data and train-only BPE.
python data_perprocess/preprocess_text_mu_bpe.py \
  --data-dir /data/europarl/json \
  --en-en-file en_en.json \
  --output-dir data/train/europarl_bpe/2026-publication-v1 \
  --train-ratio 0.70 --val-ratio 0.15 --seed 48 \
  --vocab-size 96000 --force-retrain-tokenizer

# 2. Fresh Teacher; selected by validation cross-entropy.
python main_multi_vocab.py \
  --data-dir data/train/europarl_bpe/2026-publication-v1 \
  --vocab-file vocab_bpe.json \
  --checkpoint-path results/article_kd/training \
  --experiment-id teacher-seed42 --seed 42

# 3. Matched CE-only Student with a fresh seeded initialization.
python train_multi_vocab_one_student.py \
  --data-dir data/train/europarl_bpe/2026-publication-v1 \
  --vocab-file vocab_bpe.json \
  --teacher-encoder results/article_kd/training/teacher-seed42/encoder_NN.pth \
  --teacher-decoder results/article_kd/training/teacher-seed42/decoder_NN.pth \
  --save-dir results/article_kd/training --experiment-id student-ce-seed42 \
  --seed 42 --alpha 1.0 --beta 0.0 --gamma 0.0 --temperature 2.0

# 4. Matched KD Student. Keep every non-loss argument identical to CE-only.
python train_multi_vocab_one_student.py \
  --data-dir data/train/europarl_bpe/2026-publication-v1 \
  --vocab-file vocab_bpe.json \
  --teacher-encoder results/article_kd/training/teacher-seed42/encoder_NN.pth \
  --teacher-decoder results/article_kd/training/teacher-seed42/decoder_NN.pth \
  --save-dir results/article_kd/training --experiment-id student-kd-seed42 \
  --seed 42 --alpha 0.6 --beta 0.3 --gamma 0.1 --temperature 2.0
```

Replace `NN` with the epoch selected in the Teacher manifest; never rename an
older checkpoint to fit the command. Freeze the protocol and selected manifests
before implementing or running final held-out test evaluation.
