#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=server_common.sh
source "${SCRIPT_DIR}/server_common.sh"

rsync -avP \
  --exclude .git \
  --exclude .venv \
  --exclude data \
  --exclude artifacts \
  --exclude sequence_cache \
  --exclude sequence_runs \
  --exclude tabular_cache \
  --exclude features \
  --exclude runs \
  --exclude submissions \
  --exclude __pycache__ \
  --exclude .pytest_cache \
  --exclude server.env \
  -e "${SSH_COMMAND}" \
  "${ROOT_DIR}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
