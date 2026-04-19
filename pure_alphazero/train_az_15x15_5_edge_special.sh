#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-$ROOT_DIR/gomoku_az_15x15_5_last.pt}"
export OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-$ROOT_DIR/gomoku_az_15x15_5_edge_special.pt}"

# Force self-play to start from edge buildup prefixes instead of generic openings.
export RANDOM_OPENING_MOVES="${RANDOM_OPENING_MOVES:-0}"
export OPENING_SAMPLER="${OPENING_SAMPLER:-uniform}"
export EDGE_BUILDUP_OPENING_PROB="${EDGE_BUILDUP_OPENING_PROB:-1.0}"
export THREAT_OPENING_PROB="${THREAT_OPENING_PROB:-0.0}"
export THREAT_CENTER_WEIGHT="${THREAT_CENTER_WEIGHT:-0.0}"
export THREAT_NEAR_EDGE_WEIGHT="${THREAT_NEAR_EDGE_WEIGHT:-0.0}"
export THREAT_EDGE_WEIGHT="${THREAT_EDGE_WEIGHT:-1.0}"

# Use deeper search so self-play can turn edge mistakes into clearer training signal.
export MCTS_SIMS="${MCTS_SIMS:-768}"
export EVAL_MCTS_SIMS="${EVAL_MCTS_SIMS:-1536}"
export EVAL_THREAT_CASES="${EVAL_THREAT_CASES:-128}"
export EARLY_STOP_MIN_ITERATIONS="${EARLY_STOP_MIN_ITERATIONS:-140}"
export EARLY_STOP_THREAT_BLOCK_RATE="${EARLY_STOP_THREAT_BLOCK_RATE:-0.99}"
export EARLY_STOP_THREAT_PATIENCE="${EARLY_STOP_THREAT_PATIENCE:-8}"

printf 'Running edge-specialized 15x15 training\n'
printf 'INIT_CHECKPOINT=%s\n' "$INIT_CHECKPOINT"
printf 'OUTPUT_CHECKPOINT=%s\n' "$OUTPUT_CHECKPOINT"

exec "$ROOT_DIR/train_az_15x15_5.sh"
