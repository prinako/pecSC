#!/usr/bin/env bash
set -euo pipefail
# Run from pecSC root after copying the evaluator here.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
uv run python -u evaluate_paired_validation.py \
  --data-dir data/train/europarl_bpe/2026-publication-v1 \
  --teacher-encoder results/article_kd/training/teacher-bpe32k-seed42/encoder_38.pth \
  --teacher-decoder results/article_kd/training/teacher-bpe32k-seed42/decoder_38.pth \
  --run-root results/article_kd/diagnostics \
  --output-dir results/article_kd/paired_validation/ce-vs-feature-$(date +%Y%m%d-%H%M%S) \
  --training-seeds 42 43 44 45 \
  --trial-seeds 1001 1002 1003 1004 1005 \
  "$@"
