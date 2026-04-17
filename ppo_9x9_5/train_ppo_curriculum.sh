#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

CKPT_5X5="${CKPT_5X5:-$ROOT_DIR/gomoku_ppo_5x5_4.pt}"
CKPT_7X7="${CKPT_7X7:-$ROOT_DIR/gomoku_ppo_7x7_5.pt}"
CKPT_9X9="${CKPT_9X9:-$ROOT_DIR/gomoku_ppo_9x9_5.pt}"

run_stage() {
  local stage_name="$1"
  shift
  echo
  echo "=== ${stage_name} ==="
  "$@"
}

run_stage "Stage 1: PPO 5x5 connect4" \
  env OUTPUT_CHECKPOINT="$CKPT_5X5" \
  bash "$ROOT_DIR/train_ppo_5x5_4.sh"

run_stage "Stage 2: PPO 7x7 connect5" \
  env INIT_CHECKPOINT="$CKPT_5X5" OUTPUT_CHECKPOINT="$CKPT_7X7" \
  bash "$ROOT_DIR/train_ppo_7x7_5.sh"

run_stage "Stage 3: PPO 9x9 connect5" \
  env INIT_CHECKPOINT="$CKPT_7X7" OUTPUT_CHECKPOINT="$CKPT_9X9" \
  bash "$ROOT_DIR/train_ppo_9x9_5.sh"
