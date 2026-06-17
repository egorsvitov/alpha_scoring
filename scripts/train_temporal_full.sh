#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${PYTHONPATH:-}:src"

COMMON_ARGS=(
  --architecture pooling
  --hidden-dim 128
  --embedding-dim 8
  --layers 2
  --heads 4
  --dropout 0.1
  --batch-size 1024
  --workers 6
  --epochs 10
  --learning-rate 0.001
  --weight-decay 0.0001
  --pos-weight 5.0
  --grad-clip 1.0
  --temporal-features
  --temporal-bins 50
  --temporal-smoothing 2000
  --full-train
  --save-epoch-checkpoints
)

mkdir -p sequence_runs/temporal_full_seed271 sequence_runs/temporal_full_seed42

.venv/bin/python -m alpha_scoring.models.sequence.train_sequence_model "${COMMON_ARGS[@]}" \
  --seed 271 \
  --output-dir sequence_runs/temporal_full_seed271 \
  > sequence_runs/temporal_full_seed271/train.log 2>&1

.venv/bin/python -m alpha_scoring.models.sequence.train_sequence_model "${COMMON_ARGS[@]}" \
  --seed 42 \
  --output-dir sequence_runs/temporal_full_seed42 \
  > sequence_runs/temporal_full_seed42/train.log 2>&1
