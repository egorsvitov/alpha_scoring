#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SERVER_CONFIG="${SERVER_CONFIG:-${ROOT_DIR}/server.env}"

if [[ ! -f "${SERVER_CONFIG}" ]]; then
  echo "Server config not found: ${SERVER_CONFIG}" >&2
  echo "Create it from server.env.example and edit host/user/port." >&2
  exit 1
fi

set -a
# shellcheck source=/dev/null
source "${SERVER_CONFIG}"
set +a

: "${REMOTE_HOST:?REMOTE_HOST is required}"
: "${REMOTE_PORT:?REMOTE_PORT is required}"
: "${REMOTE_USER:?REMOTE_USER is required}"
REMOTE_DIR="${REMOTE_DIR:-~/alpha_scoring}"

SSH_OPTIONS=(
  -F /dev/null
  -p "${REMOTE_PORT}"
  -o StrictHostKeyChecking=accept-new
)

if [[ -n "${REMOTE_PASSWORD:-}" ]]; then
  export DISPLAY="${DISPLAY:-codex-ssh}"
  export SSH_ASKPASS="${ROOT_DIR}/scripts/server_askpass.sh"
  export SSH_ASKPASS_REQUIRE=force
  SSH_OPTIONS+=(
    -o PreferredAuthentications=password
    -o PubkeyAuthentication=no
  )
fi

printf -v SSH_COMMAND 'ssh'
for option in "${SSH_OPTIONS[@]}"; do
  printf -v SSH_COMMAND '%s %q' "${SSH_COMMAND}" "${option}"
done

run_ssh() {
  if [[ -n "${REMOTE_PASSWORD:-}" ]]; then
    setsid -w ssh "${SSH_OPTIONS[@]}" "$@"
  else
    ssh "${SSH_OPTIONS[@]}" "$@"
  fi
}

