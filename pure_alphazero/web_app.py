#!/usr/bin/env python3
"""Minimal web UI for playing Gomoku against a pure AlphaZero model."""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from gomoku_alphazero import (
    GomokuEnv,
    action_to_coords,
    board_to_string,
    choose_ai_action,
    choose_device,
    coords_to_action,
    count_parameters,
    parameter_size_mib,
    resolve_game_config,
    winner_to_text,
)


INDEX_HTML_PATH = Path(__file__).with_name("web").joinpath("index.html")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Gomoku against a pure AlphaZero model in the browser")
    parser.add_argument("--checkpoint", type=Path, default=Path("gomoku_az_9x9_5.pt"))
    parser.add_argument("--board-size", type=int, default=None)
    parser.add_argument("--win-length", type=int, default=None)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--conv-layers", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--agent", choices=["policy", "mcts"], default="mcts")
    parser.add_argument("--mcts-sims", type=int, default=128)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8009)
    return parser.parse_args()


@dataclass
class GameSession:
    env: GomokuEnv
    human_player: int = 1
    history: list[dict[str, object]] = field(default_factory=list)
    status: str = "Click 'New Game' to start."
    last_error: str = ""

    @property
    def ai_player(self) -> int:
        return -self.human_player


class WebGameController:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = choose_device(args.device)
        self.policy, self.board_size, self.win_length = resolve_game_config(
            checkpoint_path=args.checkpoint,
            arg_board_size=args.board_size,
            arg_win_length=args.win_length,
            arg_channels=args.channels,
            arg_conv_layers=args.conv_layers,
            device=self.device,
        )
        self.params = count_parameters(self.policy)
        self.lock = threading.Lock()
        self.session = GameSession(env=GomokuEnv(self.board_size, self.win_length))
        self.reset_game(human_first=True)

    def _record_move(self, player: int, action: int) -> None:
        row, col = action_to_coords(action, self.board_size)
        self.session.history.append(
            {
                "player": "X" if player == 1 else "O",
                "row": row,
                "col": col,
            }
        )

    def _status_text(self) -> str:
        env = self.session.env
        if env.done:
            if env.winner == 0:
                return "Game over: draw."
            if env.winner == self.session.human_player:
                return "Game over: you win."
            return "Game over: model wins."
        if env.current_player == self.session.human_player:
            return "Your turn."
        return "Model is thinking..."

    def _maybe_ai_move(self) -> None:
        env = self.session.env
        if env.done or env.current_player != self.session.ai_player:
            self.session.status = self._status_text()
            return
        action, _ = choose_ai_action(
            policy=self.policy,
            board=env.board,
            current_player=env.current_player,
            win_length=self.win_length,
            device=self.device,
            agent=self.args.agent,
            mcts_sims=self.args.mcts_sims,
            c_puct=self.args.c_puct,
        )
        player = env.current_player
        env.step(action)
        self._record_move(player, action)
        self.session.status = self._status_text()

    def reset_game(self, human_first: bool) -> dict[str, object]:
        with self.lock:
            self.session = GameSession(
                env=GomokuEnv(self.board_size, self.win_length),
                human_player=1 if human_first else -1,
            )
            self.session.env.reset()
            self.session.status = self._status_text()
            if not human_first:
                self._maybe_ai_move()
            return self.serialize_state()

    def apply_human_move(self, row: int, col: int) -> dict[str, object]:
        with self.lock:
            env = self.session.env
            self.session.last_error = ""
            if env.done:
                self.session.last_error = "Game is already over."
                return self.serialize_state()
            if env.current_player != self.session.human_player:
                self.session.last_error = "It is not your turn."
                return self.serialize_state()
            if not (0 <= row < self.board_size and 0 <= col < self.board_size):
                self.session.last_error = "Move is out of range."
                return self.serialize_state()
            if env.board[row, col] != 0:
                self.session.last_error = "Cell is already occupied."
                return self.serialize_state()
            action = coords_to_action(row, col, self.board_size)
            player = env.current_player
            env.step(action)
            self._record_move(player, action)
            self.session.status = self._status_text()
            self._maybe_ai_move()
            return self.serialize_state()

    def serialize_state(self) -> dict[str, object]:
        env = self.session.env
        return {
            "board_size": self.board_size,
            "win_length": self.win_length,
            "board": env.board.astype(int).tolist(),
            "human_player": self.session.human_player,
            "ai_player": self.session.ai_player,
            "current_player": int(env.current_player),
            "done": bool(env.done),
            "winner": winner_to_text(int(env.winner)),
            "status": self.session.status,
            "history": self.session.history,
            "history_text": " ".join(
                f"{idx + 1}:{move['player']}({int(move['row']) + 1},{int(move['col']) + 1})"
                for idx, move in enumerate(self.session.history)
            ),
            "board_text": board_to_string(env.board),
            "last_error": self.session.last_error,
        }

    def config(self) -> dict[str, object]:
        return {
            "board_size": self.board_size,
            "win_length": self.win_length,
            "device": str(self.device),
            "agent": self.args.agent,
            "mcts_sims": self.args.mcts_sims,
            "checkpoint": str(self.args.checkpoint),
            "params": self.params,
            "approx_mib": round(parameter_size_mib(self.params), 2),
        }


class WebHandler(BaseHTTPRequestHandler):
    controller: WebGameController
    index_html: bytes

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(self.index_html)))
            self.end_headers()
            self.wfile.write(self.index_html)
            return
        if self.path == "/api/config":
            self._send_json(self.controller.config())
            return
        if self.path == "/api/state":
            self._send_json(self.controller.serialize_state())
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/new_game":
            payload = self._read_json()
            human_first = bool(payload.get("human_first", True))
            self._send_json(self.controller.reset_game(human_first=human_first))
            return
        if self.path == "/api/move":
            payload = self._read_json()
            row = int(payload.get("row", -1))
            col = int(payload.get("col", -1))
            self._send_json(self.controller.apply_human_move(row=row, col=col))
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    controller = WebGameController(args)
    index_html = INDEX_HTML_PATH.read_bytes()
    WebHandler.controller = controller
    WebHandler.index_html = index_html
    server = ThreadingHTTPServer((args.host, args.port), WebHandler)
    print(
        f"serving pure_alphazero web app on http://{args.host}:{args.port} "
        f"device={controller.device} board={controller.board_size} win={controller.win_length} "
        f"agent={args.agent} params={controller.params} approx_mib={parameter_size_mib(controller.params):.2f}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
