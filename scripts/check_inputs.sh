#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

required=(
  data/train_data.parquet
  data/test_data.parquet
  data/train_target.csv
  data/sample_submission.csv
  data/submission_full100_checkpoint_average_idprior.csv
)

missing=0
for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "missing: $path" >&2
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo "Put all required inputs into data/ before running the pipeline." >&2
  exit 1
fi

echo "All required inputs are present."
