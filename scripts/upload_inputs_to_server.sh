#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=server_common.sh
source "${SCRIPT_DIR}/server_common.sh"

bash "${ROOT_DIR}/scripts/check_inputs.sh"

run_ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}/data"
rsync -avP \
  -e "${SSH_COMMAND}" \
  "${ROOT_DIR}/data/train_data.parquet" \
  "${ROOT_DIR}/data/test_data.parquet" \
  "${ROOT_DIR}/data/train_target.csv" \
  "${ROOT_DIR}/data/sample_submission.csv" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/data/"
