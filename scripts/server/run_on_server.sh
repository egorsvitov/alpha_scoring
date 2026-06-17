#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=server_common.sh
source "${SCRIPT_DIR}/server_common.sh"

COMMAND="${*:-bash scripts/run_final.sh}"

run_ssh "${REMOTE_USER}@${REMOTE_HOST}" \
  "cd ${REMOTE_DIR} && bash scripts/server/ensure_swap.sh && source .venv/bin/activate && which python && PYTHONUNBUFFERED=1 ${COMMAND}"
