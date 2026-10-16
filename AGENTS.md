# pecSC — Experimental branch instructions

## Purpose
Branch `kd-exp` supports the **receiver-compression knowledge-distillation IEEE article** in `prinako/Prince_SC`. Do not confuse this work with the separate shared-transmitter multi-user manuscript in `prinako/Prince_SC_shared-TX`.

## Scope
English-to-English text reconstruction only. Evaluate a full DeepSC Teacher, a compact receiver trained with CE only, and the **same** compact receiver trained with KD. Teacher semantic/channel transmitter is frozen for both Students. Do not introduce LoRA, multilingual claims, or multi-user scaling experiments.

## Source of truth
Read `prinako/Prince_SC/main/AGENTS.md` and `docs/NEXT_TASK.md`, `docs/EXPERIMENTS.md`, `docs/IMPLEMENTATION_AUDIT.md` in that paper repo before scientific changes. For active BPE receiver-only KD, inspect `train_multi_vocab_one_student.py`; do not substitute `R_tr_kd.py` or `train_student.py` without explicit comparison. Verify exact implementation behavior rather than accepting paper descriptions uncritically.

## Experiment requirements
1. Publication dataset: new leakage-checked 70/15/15 Europarl partition; train-only fresh SentencePiece BPE, explicit data paths, manifest/hashes. Old 90/10 files/checkpoints cannot be reused.
2. Retrain Teacher from scratch; select on validation. Load it explicitly into both Student runs.
3. Student controls: identical four-layer/eight-head receiver architecture, initialisation policy, training data, budget, optimizer, transmitter, channel, and validation checkpoint selection. CE-only weights alpha/beta/gamma = 1/0/0; KD candidate 0.6/0.3/0.1 with temperature 2; verify actual active loss implementation.
4. Test remains held out until experiment protocol and model selection are frozen. Evaluation must match decoder strategy and channel/noise trials; corpus BLEU must be calculated over grouped hypotheses/references, not averaged sentence BLEU.
5. Track checkpoint/config/data/tokenizer/code hashes, seed, hardware, dependencies, SNR/decoder settings, counts, model sizes and benchmark protocol. Never invent measurements.

## Working discipline
Implement small, reviewable changes with tests. Prefer branch `kd-exp` only for the original KD publication experiments. Do not modify `BPE`, `a1-*` branches or either article repository unless the user asks. No trained GPU experiments are claimed merely because code was committed.

See `docs/KD_EXPERIMENT_PLAN.md` for ordered steps.
