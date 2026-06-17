#!/usr/bin/env bash
set -euo pipefail

SWAP_SIZE_GB="${SWAP_SIZE_GB:-64}"
MIN_SWAP_GB="${MIN_SWAP_GB:-60}"
SWAP_FILE="${SWAP_FILE:-/swapfile_alpha_scoring}"
SWAP_PERSIST="${SWAP_PERSIST:-1}"

total_swap_kb() {
  awk 'NR > 1 { total += $3 } END { print total + 0 }' /proc/swaps
}

as_root() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  else
    sudo -n "$@"
  fi
}

can_run_root() {
  [[ "${EUID}" -eq 0 ]] || sudo -n true 2>/dev/null
}

min_swap_kb=$((MIN_SWAP_GB * 1024 * 1024))
current_swap_kb="$(total_swap_kb)"

if (( current_swap_kb >= min_swap_kb )); then
  echo "swap_ok: current=$((current_swap_kb / 1024 / 1024))GB min=${MIN_SWAP_GB}GB"
  exit 0
fi

if ! can_run_root; then
  cat >&2 <<EOF
Not enough swap: current=$((current_swap_kb / 1024 / 1024))GB, required=${MIN_SWAP_GB}GB.
Creating/enabling swap requires root privileges.

Run manually on the server:

  sudo SWAP_SIZE_GB=${SWAP_SIZE_GB} MIN_SWAP_GB=${MIN_SWAP_GB} SWAP_FILE=${SWAP_FILE} bash scripts/ensure_swap.sh

Then restart the training command.
EOF
  exit 1
fi

echo "creating_swap: file=${SWAP_FILE} size=${SWAP_SIZE_GB}GB"

if [[ ! -f "${SWAP_FILE}" ]]; then
  if command -v fallocate >/dev/null 2>&1; then
    as_root fallocate -l "${SWAP_SIZE_GB}G" "${SWAP_FILE}"
  else
    as_root dd if=/dev/zero of="${SWAP_FILE}" bs=1G count="${SWAP_SIZE_GB}" status=progress
  fi
fi

as_root chmod 600 "${SWAP_FILE}"
if ! grep -qF "${SWAP_FILE}" /proc/swaps; then
  as_root mkswap "${SWAP_FILE}"
  as_root swapon "${SWAP_FILE}"
fi

if [[ "${SWAP_PERSIST}" == "1" ]] && ! grep -qF "${SWAP_FILE}" /etc/fstab; then
  echo "${SWAP_FILE} none swap sw 0 0" | as_root tee -a /etc/fstab >/dev/null
fi

current_swap_kb="$(total_swap_kb)"
echo "swap_ready: current=$((current_swap_kb / 1024 / 1024))GB file=${SWAP_FILE}"
