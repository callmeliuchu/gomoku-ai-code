#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export DEVICE="${DEVICE:-cuda}"
export SELFPLAY_WORKERS="${SELFPLAY_WORKERS:-4}"
export SELFPLAY_DEVICE="${SELFPLAY_DEVICE:-same}"

# Keep the rest of the training recipe unchanged unless the caller overrides it.
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-$ROOT_DIR/gomoku_az_15x15_5_last.pt}"
export OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-$ROOT_DIR/gomoku_az_15x15_5_mp.pt}"

printf 'Running multiprocess 15x15 training\n'
printf 'INIT_CHECKPOINT=%s\n' "$INIT_CHECKPOINT"
printf 'OUTPUT_CHECKPOINT=%s\n' "$OUTPUT_CHECKPOINT"
printf 'SELFPLAY_WORKERS=%s\n' "$SELFPLAY_WORKERS"
printf 'SELFPLAY_DEVICE=%s\n' "$SELFPLAY_DEVICE"

exec "$ROOT_DIR/train_az_15x15_5.sh"
