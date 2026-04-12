#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-lerobot}"

INIT_CHECKPOINT="${INIT_CHECKPOINT:-$ROOT_DIR/gomoku_7x7_5.pt}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-$ROOT_DIR/gomoku_mcts_15x15_5.pt}"

BOARD_SIZE="${BOARD_SIZE:-15}"
WIN_LENGTH="${WIN_LENGTH:-5}"
CHANNELS="${CHANNELS:-64}"

ITERATIONS="${ITERATIONS:-3000}"
GAMES_PER_ITER="${GAMES_PER_ITER:-12}"
TRAIN_STEPS="${TRAIN_STEPS:-64}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BUFFER_SIZE="${BUFFER_SIZE:-50000}"

MCTS_SIMS="${MCTS_SIMS:-64}"
EVAL_MCTS_SIMS="${EVAL_MCTS_SIMS:-160}"
EVAL_EVERY="${EVAL_EVERY:-10}"
EVAL_GAMES="${EVAL_GAMES:-20}"
SAVE_EVERY="${SAVE_EVERY:-10}"

LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
VALUE_COEF="${VALUE_COEF:-1.0}"
CPUCT="${CPUCT:-1.5}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TEMPERATURE_DROP_MOVES="${TEMPERATURE_DROP_MOVES:-10}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.3}"
NOISE_EPS="${NOISE_EPS:-0.25}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"

CMD=(
  "$CONDA_BIN" run -n "$ENV_NAME" python "$ROOT_DIR/gomoku_mcts.py" train
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
  --save-every "$SAVE_EVERY"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --value-coef "$VALUE_COEF"
  --c-puct "$CPUCT"
  --temperature "$TEMPERATURE"
  --temperature-drop-moves "$TEMPERATURE_DROP_MOVES"
  --dirichlet-alpha "$DIRICHLET_ALPHA"
  --noise-eps "$NOISE_EPS"
  --seed "$SEED"
  --device "$DEVICE"
  --checkpoint "$OUTPUT_CHECKPOINT"
)

if [[ -f "$INIT_CHECKPOINT" ]]; then
  CMD+=(--init-checkpoint "$INIT_CHECKPOINT")
else
  echo "init checkpoint not found, training from scratch: $INIT_CHECKPOINT"
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
exec "${CMD[@]}"
