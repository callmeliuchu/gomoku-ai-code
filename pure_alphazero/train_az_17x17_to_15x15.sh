#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

INIT_CHECKPOINT="${INIT_CHECKPOINT:-$ROOT_DIR/gomoku_az_17x17_5.pt}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-$ROOT_DIR/gomoku_az_17x17_to_15x15.pt}"

export INIT_CHECKPOINT
export OUTPUT_CHECKPOINT

printf 'Running 17x17 -> 15x15 finetune\n'
printf 'INIT_CHECKPOINT=%s\n' "$INIT_CHECKPOINT"
printf 'OUTPUT_CHECKPOINT=%s\n' "$OUTPUT_CHECKPOINT"

exec "$ROOT_DIR/train_az_15x15_5.sh"
