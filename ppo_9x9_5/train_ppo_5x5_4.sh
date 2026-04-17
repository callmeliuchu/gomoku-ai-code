#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

CHANNELS="${CHANNELS:-128}"
CONV_LAYERS="${CONV_LAYERS:-8}"
ITERATIONS="${ITERATIONS:-800}"
GAMES_PER_ITER="${GAMES_PER_ITER:-48}"
PPO_EPOCHS="${PPO_EPOCHS:-8}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-256}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GAMMA="${GAMMA:-0.99}"
CLIP_EPS="${CLIP_EPS:-0.2}"
ENTROPY_COEF="${ENTROPY_COEF:-0.01}"
VALUE_COEF="${VALUE_COEF:-1.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
RANDOM_OPENING_MOVES="${RANDOM_OPENING_MOVES:-0}"
EVAL_EVERY="${EVAL_EVERY:-5}"
EVAL_GAMES="${EVAL_GAMES:-20}"
EVAL_HEURISTIC_GAMES="${EVAL_HEURISTIC_GAMES:-8}"
EVAL_TRACE_GAMES="${EVAL_TRACE_GAMES:-1}"
EVAL_HEURISTIC_TRACE_GAMES="${EVAL_HEURISTIC_TRACE_GAMES:-1}"
EVAL_TRACE_MAX_MOVES="${EVAL_TRACE_MAX_MOVES:-20}"
LOG_EVERY_GAMES="${LOG_EVERY_GAMES:-12}"
SAVE_EVERY="${SAVE_EVERY:-10}"
EARLY_STOP_LOSS="${EARLY_STOP_LOSS:-0.20}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-12}"
EARLY_STOP_MIN_ITERATIONS="${EARLY_STOP_MIN_ITERATIONS:-120}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"

OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-$ROOT_DIR/gomoku_ppo_5x5_4.pt}"

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/gomoku_ppo.py" train
  --board-size 5
  --win-length 4
  --channels "$CHANNELS"
  --conv-layers "$CONV_LAYERS"
  --iterations "$ITERATIONS"
  --games-per-iter "$GAMES_PER_ITER"
  --ppo-epochs "$PPO_EPOCHS"
  --minibatch-size "$MINIBATCH_SIZE"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --gamma "$GAMMA"
  --clip-eps "$CLIP_EPS"
  --entropy-coef "$ENTROPY_COEF"
  --value-coef "$VALUE_COEF"
  --max-grad-norm "$MAX_GRAD_NORM"
  --random-opening-moves "$RANDOM_OPENING_MOVES"
  --eval-every "$EVAL_EVERY"
  --eval-games "$EVAL_GAMES"
  --eval-heuristic-games "$EVAL_HEURISTIC_GAMES"
  --eval-trace-games "$EVAL_TRACE_GAMES"
  --eval-heuristic-trace-games "$EVAL_HEURISTIC_TRACE_GAMES"
  --eval-trace-max-moves "$EVAL_TRACE_MAX_MOVES"
  --log-every-games "$LOG_EVERY_GAMES"
  --save-every "$SAVE_EVERY"
  --early-stop-loss "$EARLY_STOP_LOSS"
  --early-stop-patience "$EARLY_STOP_PATIENCE"
  --early-stop-min-iterations "$EARLY_STOP_MIN_ITERATIONS"
  --seed "$SEED"
  --device "$DEVICE"
  --checkpoint "$OUTPUT_CHECKPOINT"
)

if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  CMD+=(--init-checkpoint "$INIT_CHECKPOINT")
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
exec "${CMD[@]}"
