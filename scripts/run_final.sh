#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${PYTHONPATH:-}:src"
PY="${PY:-.venv/bin/python}"
RUNS=sequence_runs
STATUS="$RUNS/self_contained_pipeline"
mkdir -p "$STATUS" "$RUNS/blends" submissions

log_step() {
  echo "$(date -Is) $*" | tee -a "$STATUS/status.log"
}

link_inputs() {
  ln -sf data/train_data.parquet train_data.parquet
  ln -sf data/test_data.parquet test_data.parquet
  ln -sf data/train_target.csv train_target.csv
  ln -sf data/sample_submission.csv sample_submission.csv
}

train_sequence() {
  local output=$1
  shift
  mkdir -p "$output"
  if [[ ! -f "$output/last.pt" ]]; then
    "$PY" train_sequence_model.py "$@" \
      --full-train \
      --save-epoch-checkpoints \
      --output-dir "$output" \
      > "$output/train.log" 2>&1
  fi
}

predict_epoch_average() {
  local output=$1
  local final_name=$2
  shift 2
  local inputs=()
  for epoch in "$@"; do
    local prediction="$output/submission_epoch_${epoch}.csv"
    if [[ ! -f "$prediction" ]]; then
      "$PY" train_sequence_model.py \
        --predict-test \
        --checkpoint "$output/epoch_${epoch}.pt" \
        --batch-size 2048 \
        --workers 6 \
        --test-output "$prediction" \
        >> "$output/predict.log" 2>&1
    fi
    inputs+=("$prediction")
  done
  "$PY" average_sequence_predictions.py "${inputs[@]}" --output "$output/$final_name"
}

bash scripts/check_inputs.sh
link_inputs

log_step "build_tabular_features_start"
"$PY" build_tabular_features.py > "$STATUS/build_tabular_features.log" 2>&1
log_step "build_tabular_features_done"

log_step "train_tabular_models_start"
if [[ ! -f submission_catboost_v4.csv ]]; then
  "$PY" train_v4_submission.py > "$STATUS/train_v4.log" 2>&1
fi
if [[ ! -f submission_catboost_v5.csv ]]; then
  "$PY" train_v5_submission.py > "$STATUS/train_v5.log" 2>&1
fi
if [[ ! -f submission_lightgbm_v5.csv ]]; then
  "$PY" train_lightgbm_v5_submission.py > "$STATUS/train_lightgbm_v5.log" 2>&1
fi
log_step "train_tabular_models_done"

log_step "build_sequence_cache_start"
if [[ ! -f sequence_cache/manifest.json ]]; then
  "$PY" build_sequence_cache.py --force > "$STATUS/build_sequence_cache.log" 2>&1
fi
log_step "build_sequence_cache_done"

log_step "train_temporal_pooling_start"
if [[ ! -f "$RUNS/temporal_full_seed271/epoch_10.pt" || ! -f "$RUNS/temporal_full_seed42/epoch_10.pt" ]]; then
  bash train_temporal_full.sh
fi
predict_epoch_average "$RUNS/temporal_full_seed271" submission_epoch_average.csv 7 8 9 10
predict_epoch_average "$RUNS/temporal_full_seed42" submission_epoch_average.csv 7 8 9 10
"$PY" average_sequence_predictions.py \
  "$RUNS/temporal_full_seed271/submission_epoch_average.csv" \
  "$RUNS/temporal_full_seed42/submission_epoch_average.csv" \
  --output "$RUNS/blends/submission_temporal_full_two_seed.csv"
log_step "train_temporal_pooling_done"

log_step "build_tabular_cache_start"
if [[ ! -f tabular_cache/metadata.json ]]; then
  "$PY" build_tabular_cache.py --force > "$STATUS/build_tabular_cache.log" 2>&1
fi
log_step "build_tabular_cache_done"

COMMON=(--hidden-dim 128 --embedding-dim 8 --batch-size 1024 --workers 6 --epochs 10 --patience 20)

log_step "train_sequence_base_start"
train_sequence "$RUNS/full100_pooling_seed42" --architecture pooling "${COMMON[@]}" --seed 42
predict_epoch_average "$RUNS/full100_pooling_seed42" submission_epoch_average.csv 5 6 7

train_sequence "$RUNS/full100_pooling_seed137" --architecture pooling "${COMMON[@]}" --seed 137
predict_epoch_average "$RUNS/full100_pooling_seed137" submission_epoch_average.csv 4 5 6

train_sequence "$RUNS/full100_product_transformer" --architecture transformer --layers 2 --heads 4 "${COMMON[@]}" --seed 42
predict_epoch_average "$RUNS/full100_product_transformer" submission_epoch_average.csv 5 6 7

train_sequence "$RUNS/full100_dual_pooling" --architecture pooling --tabular-cache-dir tabular_cache "${COMMON[@]}" --seed 42
predict_epoch_average "$RUNS/full100_dual_pooling" submission_epoch_average.csv 2 3 4

train_sequence "$RUNS/full100_late_fusion" \
  --architecture pooling \
  --tabular-cache-dir tabular_cache \
  --fusion-mode late \
  --fusion-init 0.1 \
  --pretrained-sequence "$RUNS/full100_pooling_seed42/epoch_6.pt" \
  --freeze-sequence \
  --dropout 0.2 \
  --weight-decay 0.001 \
  --hidden-dim 128 \
  --embedding-dim 8 \
  --batch-size 1024 \
  --workers 6 \
  --epochs 12 \
  --patience 20 \
  --seed 42
predict_epoch_average "$RUNS/full100_late_fusion" submission_epoch_average.csv 2 3 4

train_sequence "$RUNS/full100_payment_transformer" \
  --architecture pooling \
  --payment-encoder transformer \
  --payment-hidden-dim 32 \
  --payment-layers 2 \
  --payment-heads 4 \
  "${COMMON[@]}" \
  --seed 42
predict_epoch_average "$RUNS/full100_payment_transformer" submission_epoch_average.csv 6 7 8

train_sequence "$RUNS/full100_hierarchical_transformer" \
  --architecture transformer \
  --payment-encoder transformer \
  --payment-hidden-dim 32 \
  --payment-layers 2 \
  --payment-heads 4 \
  --layers 2 \
  --heads 4 \
  "${COMMON[@]}" \
  --seed 42
predict_epoch_average "$RUNS/full100_hierarchical_transformer" submission_epoch_average.csv 5 6 7
log_step "train_sequence_base_done"

log_step "assemble_base_with_idprior_start"
"$PY" assemble_checkpoint_average_submission.py \
  --output "$RUNS/blends/submission_full100_checkpoint_average.csv" \
  --prior-output "$RUNS/blends/submission_full100_checkpoint_average_idprior.csv" \
  > "$STATUS/assemble_base.log" 2>&1
log_step "assemble_base_with_idprior_done"

log_step "train_alpha_gru_start"
train_sequence "$RUNS/full100_alpha_gru_payment" \
  --architecture alpha_gru \
  --payment-encoder transformer \
  --hidden-dim 128 \
  --embedding-dim 8 \
  --layers 2 \
  --payment-hidden-dim 32 \
  --payment-layers 2 \
  --payment-heads 4 \
  --batch-size 1024 \
  --workers 6 \
  --epochs 9 \
  --scheduler onecycle \
  --learning-rate 0.0015 \
  --sample-weight-mode linear \
  --sample-weight-strength 1.0 \
  --seed 42
"$PY" train_sequence_model.py \
  --predict-test \
  --checkpoint "$RUNS/full100_alpha_gru_payment/epoch_9.pt" \
  --batch-size 2048 \
  --workers 6 \
  --test-output "$RUNS/full100_alpha_gru_payment/submission_epoch_9.csv" \
  > "$RUNS/full100_alpha_gru_payment/predict.log" 2>&1
log_step "train_alpha_gru_done"

log_step "final_blend_start"
"$PY" blend_prediction_files.py \
  0.839:"$RUNS/blends/submission_full100_checkpoint_average_idprior.csv" \
  0.161:"$RUNS/full100_alpha_gru_payment/submission_epoch_9.csv" \
  --output submissions/submission_alpha_gru161.csv \
  > "$STATUS/final_blend.log" 2>&1
log_step "complete"
