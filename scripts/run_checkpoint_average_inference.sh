#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${PYTHONPATH:-}:src"
PY=.venv/bin/python
RUNS=sequence_runs

predict_average() {
  local directory=$1
  shift
  local inputs=()
  for epoch in "$@"; do
    local output="$RUNS/$directory/submission_epoch_${epoch}.csv"
    "$PY" -m alpha_scoring.models.sequence.train_sequence_model --predict-test \
      --checkpoint "$RUNS/$directory/epoch_${epoch}.pt" \
      --batch-size 2048 --workers 6 --test-output "$output"
    inputs+=("$output")
  done
  "$PY" -m alpha_scoring.ensembling.average_sequence_predictions "${inputs[@]}" \
    --output "$RUNS/$directory/submission_epoch_average.csv"
}

predict_average full100_pooling_seed42 5 6 7
predict_average full100_pooling_seed137 4 5 6
predict_average full100_product_transformer 5 6 7
predict_average full100_dual_pooling 2 3 4
predict_average full100_late_fusion 2 3 4
predict_average full100_payment_transformer 6 7 8
predict_average full100_hierarchical_transformer 5 6 7

"$PY" -m alpha_scoring.ensembling.assemble_checkpoint_average_submission \
  --output "$RUNS/blends/submission_full100_checkpoint_average.csv" \
  --prior-output "$RUNS/blends/submission_full100_checkpoint_average_idprior.csv"
