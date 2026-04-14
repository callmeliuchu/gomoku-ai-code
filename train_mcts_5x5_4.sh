#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-$ROOT_DIR/gomoku_mcts_5x5_4.pt}"

BOARD_SIZE="${BOARD_SIZE:-5}"
WIN_LENGTH="${WIN_LENGTH:-4}"
CHANNELS="${CHANNELS:-64}"

ITERATIONS="${ITERATIONS:-1200}"
GAMES_PER_ITER="${GAMES_PER_ITER:-32}"
TRAIN_STEPS="${TRAIN_STEPS:-64}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BUFFER_SIZE="${BUFFER_SIZE:-30000}"

MCTS_SIMS="${MCTS_SIMS:-96}"
EVAL_MCTS_SIMS="${EVAL_MCTS_SIMS:-192}"
EVAL_EVERY="${EVAL_EVERY:-10}"
EVAL_GAMES="${EVAL_GAMES:-20}"
EVAL_HEURISTIC_GAMES="${EVAL_HEURISTIC_GAMES:-8}"
EVAL_TRACE_GAMES="${EVAL_TRACE_GAMES:-1}"
EVAL_TRACE_MAX_MOVES="${EVAL_TRACE_MAX_MOVES:-20}"
LOG_EVERY_GAMES="${LOG_EVERY_GAMES:-4}"
LOG_EVERY_TRAIN_STEPS="${LOG_EVERY_TRAIN_STEPS:-16}"
SAVE_EVERY="${SAVE_EVERY:-10}"

LR="${LR:-7e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
VALUE_COEF="${VALUE_COEF:-1.0}"
CPUCT="${CPUCT:-1.5}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TEMPERATURE_DROP_MOVES="${TEMPERATURE_DROP_MOVES:-4}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.20}"
NOISE_EPS="${NOISE_EPS:-0.15}"
RANDOM_OPENING_MOVES="${RANDOM_OPENING_MOVES:-0}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"

CMD=(
  python "$ROOT_DIR/gomoku_mcts.py" train
  --board-size "$BOARD_SIZE"
  --win-length "$WIN_LENGTH"
  --channels "$CHANNELS"
  --iterations "$ITERATIONS"
  --games-per-iter "$GAMES_PER_ITER"
  --train-steps "$TRAIN_STEPS"
  --batch-size "$BATCH_SIZE"
  --buffer-size "$BUFFER_SIZE"
  --mcts-sims "$MCTS_SIMS"
  --eval-mcts-sims "$EVAL_MCTS_SIMS"
  --eval-every "$EVAL_EVERY"
  --eval-games "$EVAL_GAMES"
  --eval-heuristic-games "$EVAL_HEURISTIC_GAMES"
  --eval-trace-games "$EVAL_TRACE_GAMES"
  --eval-trace-max-moves "$EVAL_TRACE_MAX_MOVES"
  --log-every-games "$LOG_EVERY_GAMES"
  --log-every-train-steps "$LOG_EVERY_TRAIN_STEPS"
  --save-every "$SAVE_EVERY"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --value-coef "$VALUE_COEF"
  --c-puct "$CPUCT"
  --temperature "$TEMPERATURE"
  --temperature-drop-moves "$TEMPERATURE_DROP_MOVES"
  --dirichlet-alpha "$DIRICHLET_ALPHA"
  --noise-eps "$NOISE_EPS"
  --random-opening-moves "$RANDOM_OPENING_MOVES"
  --seed "$SEED"
  --device "$DEVICE"
  --checkpoint "$OUTPUT_CHECKPOINT"
)

if [[ -n "${INIT_CHECKPOINT:-}" && -f "$INIT_CHECKPOINT" ]]; then
  CMD+=(--init-checkpoint "$INIT_CHECKPOINT")
elif [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  echo "init checkpoint not found, training from scratch: $INIT_CHECKPOINT"
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
exec "${CMD[@]}"
