#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

CKPT_5X5="${CKPT_5X5:-$ROOT_DIR/gomoku_mcts_5x5_4.pt}"
CKPT_7X7="${CKPT_7X7:-$ROOT_DIR/gomoku_mcts_7x7_5.pt}"
CKPT_9X9="${CKPT_9X9:-$ROOT_DIR/gomoku_mcts_9x9_5.pt}"
CKPT_15X15="${CKPT_15X15:-$ROOT_DIR/gomoku_mcts_15x15_5.pt}"

run_stage() {
  local stage_name="$1"
  shift
  echo
  echo "=== ${stage_name} ==="
  "$@"
}

run_stage "Stage 1: 5x5 connect4" \
  env OUTPUT_CHECKPOINT="$CKPT_5X5" \
  bash "$ROOT_DIR/train_mcts_5x5_4.sh"

run_stage "Stage 2: 7x7 connect5" \
  env INIT_CHECKPOINT="$CKPT_5X5" OUTPUT_CHECKPOINT="$CKPT_7X7" \
  bash "$ROOT_DIR/train_mcts_7x7_5.sh"

run_stage "Stage 3: 9x9 connect5" \
  env INIT_CHECKPOINT="$CKPT_7X7" OUTPUT_CHECKPOINT="$CKPT_9X9" \
  bash "$ROOT_DIR/train_mcts_9x9_5.sh"

run_stage "Stage 4: 15x15 connect5" \
  env INIT_CHECKPOINT="$CKPT_9X9" OUTPUT_CHECKPOINT="$CKPT_15X15" \
  bash "$ROOT_DIR/train_mcts_15x15_5.sh"
