#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=server_common.sh
source "${SCRIPT_DIR}/server_common.sh"

mkdir -p "${ROOT_DIR}/submissions"
rsync -avP \
  -e "${SSH_COMMAND}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/submissions/submission_alpha_gru161.csv" \
  "${ROOT_DIR}/submissions/"
