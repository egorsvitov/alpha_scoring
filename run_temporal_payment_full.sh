#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PY=.venv/bin/python
RUN=sequence_runs/full100_temporal_payment
mkdir -p "$RUN"

if [[ ! -f "$RUN/epoch_10.pt" ]]; then
  "$PY" train_sequence_model.py \
    --architecture pooling \
    --payment-encoder transformer \
    --payment-hidden-dim 32 \
    --payment-layers 2 \
    --payment-heads 4 \
    --hidden-dim 128 \
    --embedding-dim 8 \
    --batch-size 1024 \
    --workers 6 \
    --epochs 10 \
    --patience 20 \
    --temporal-features \
    --temporal-bins 50 \
    --temporal-smoothing 2000 \
    --full-train \
    --save-epoch-checkpoints \
    --seed 42 \
    --output-dir "$RUN" \
    > "$RUN/train.log" 2>&1
fi

for epoch in 7 8 9 10; do
  "$PY" train_sequence_model.py \
    --predict-test \
    --checkpoint "$RUN/epoch_${epoch}.pt" \
    --batch-size 2048 \
    --workers 6 \
    --test-output "$RUN/submission_epoch_${epoch}.csv" \
    >> "$RUN/predict.log" 2>&1
done

"$PY" average_sequence_predictions.py "$RUN"/submission_epoch_{7,8,9,10}.csv \
  --output "$RUN/submission.csv"

"$PY" blend_prediction_files.py \
  0.887:sequence_runs/blends/submission_full100_hierarchical_temporal.csv \
  0.113:"$RUN/submission.csv" \
  --output sequence_runs/blends/submission_full100_with_temporal_payment.csv
