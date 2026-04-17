#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:-$ROOT_DIR/gomoku_az_9x9_5.pt}"
DEVICE="${DEVICE:-auto}"
AGENT="${AGENT:-mcts}"
MCTS_SIMS="${MCTS_SIMS:-128}"
C_PUCT="${C_PUCT:-1.5}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8009}"

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/web_app.py"
  --checkpoint "$CHECKPOINT"
  --device "$DEVICE"
  --agent "$AGENT"
  --mcts-sims "$MCTS_SIMS"
  --c-puct "$C_PUCT"
  --host "$HOST"
  --port "$PORT"
)

printf 'Running command:\n%s\n' "${CMD[*]}"
exec "${CMD[@]}"
