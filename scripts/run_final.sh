#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${PYTHONPATH:-}:src"

bash scripts/check_inputs.sh

python -m alpha_scoring.cache \
  --train-data data/train_data.parquet \
  --test-data data/test_data.parquet \
  --train-target data/train_target.csv \
  --output-dir sequence_cache \
  --force

python -m alpha_scoring.train train \
  --cache-dir sequence_cache \
  --output-dir runs/full100_alpha_gru_payment \
  --epochs 9 \
  --batch-size 1024 \
  --workers 6 \
  --learning-rate 0.0015 \
  --sample-weight-strength 1.0

python -m alpha_scoring.train predict \
  --cache-dir sequence_cache \
  --checkpoint runs/full100_alpha_gru_payment/epoch_9.pt \
  --output runs/full100_alpha_gru_payment/submission_epoch_9.csv \
  --batch-size 2048 \
  --workers 6

python -m alpha_scoring.blend \
  --base data/submission_full100_checkpoint_average_idprior.csv \
  --alpha runs/full100_alpha_gru_payment/submission_epoch_9.csv \
  --sample data/sample_submission.csv \
  --alpha-weight 0.161 \
  --output submissions/submission_alpha_gru161.csv
