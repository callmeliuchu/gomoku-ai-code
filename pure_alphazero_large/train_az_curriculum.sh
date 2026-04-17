#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

CKPT_5X5="${CKPT_5X5:-$ROOT_DIR/gomoku_az_large_5x5_4.pt}"
CKPT_7X7="${CKPT_7X7:-$ROOT_DIR/gomoku_az_large_7x7_5.pt}"
CKPT_9X9="${CKPT_9X9:-$ROOT_DIR/gomoku_az_large_9x9_5.pt}"
CKPT_11X11="${CKPT_11X11:-$ROOT_DIR/gomoku_az_large_11x11_5.pt}"
CKPT_13X13="${CKPT_13X13:-$ROOT_DIR/gomoku_az_large_13x13_5.pt}"
CKPT_15X15="${CKPT_15X15:-$ROOT_DIR/gomoku_az_large_15x15_5.pt}"

run_stage() {
  local stage_name="$1"
  shift
  echo
  echo "=== ${stage_name} ==="
  "$@"
}

run_stage "Stage 1: AlphaZero Large 5x5 connect4" \
  env OUTPUT_CHECKPOINT="$CKPT_5X5" \
  bash "$ROOT_DIR/train_az_5x5_4.sh"

run_stage "Stage 2: AlphaZero Large 7x7 connect5" \
  env INIT_CHECKPOINT="$CKPT_5X5" OUTPUT_CHECKPOINT="$CKPT_7X7" \
  bash "$ROOT_DIR/train_az_7x7_5.sh"

run_stage "Stage 3: AlphaZero Large 9x9 connect5" \
  env INIT_CHECKPOINT="$CKPT_7X7" OUTPUT_CHECKPOINT="$CKPT_9X9" \
  bash "$ROOT_DIR/train_az_9x9_5.sh"

run_stage "Stage 4: AlphaZero Large 11x11 connect5" \
  env INIT_CHECKPOINT="$CKPT_9X9" OUTPUT_CHECKPOINT="$CKPT_11X11" \
  bash "$ROOT_DIR/train_az_11x11_5.sh"

run_stage "Stage 5: AlphaZero Large 13x13 connect5" \
  env INIT_CHECKPOINT="$CKPT_11X11" OUTPUT_CHECKPOINT="$CKPT_13X13" \
  bash "$ROOT_DIR/train_az_13x13_5.sh"

run_stage "Stage 6: AlphaZero Large 15x15 connect5" \
  env INIT_CHECKPOINT="$CKPT_13X13" OUTPUT_CHECKPOINT="$CKPT_15X15" \
  bash "$ROOT_DIR/train_az_15x15_5.sh"
