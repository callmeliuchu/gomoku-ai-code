#!/usr/bin/env python3
"""Heuristic bootstrap + MCTS finetune for Gomoku."""

from __future__ import annotations

import argparse
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gomoku_mcts import (
    GomokuEnv,
    apply_state_symmetry,
    apply_policy_symmetry,
    choose_device,
    choose_heuristic_action,
    evaluate_self_play_trace,
    evaluate_vs_opponent,
    format_trace_moves,
    heuristic_candidate_actions,
    immediate_winning_actions,
    last_checkpoint_path,
    score_heuristic_move,
    self_play_game,
    set_seed,
    train_batch,
    winner_to_text,
    encode_state,
)


class BootstrapPolicyValueNet(nn.Module):
    """~1M parameter fully-convolutional policy/value net."""

    def __init__(self, channels: int = 128, conv_layers: int = 8):
        super().__init__()
        if conv_layers < 2:
            raise ValueError("conv_layers must be >= 2")

        trunk_layers: list[nn.Module] = [
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        for _ in range(conv_layers - 1):
            trunk_layers.extend(
                [
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                ]
            )
        self.trunk = nn.Sequential(*trunk_layers)
        self.policy_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        policy_logits = self.policy_head(features).flatten(start_dim=1)
        value = self.value_head(features).squeeze(-1)
        return policy_logits, value


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def save_bootstrap_checkpoint(
    path: Path,
    policy: BootstrapPolicyValueNet,
    board_size: int,
    win_length: int,
    channels: int,
    conv_layers: int,
) -> None:
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "board_size": board_size,
            "win_length": win_length,
            "channels": channels,
            "conv_layers": conv_layers,
        },
        path,
    )


def load_bootstrap_checkpoint(path: Path, map_location: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint
    raise RuntimeError(f"{path} is not a compatible gomoku_bootstrap checkpoint")


def build_policy_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[BootstrapPolicyValueNet, int, int, int, int]:
    checkpoint = load_bootstrap_checkpoint(checkpoint_path, map_location=device)
    board_size = int(checkpoint.get("board_size") or 15)
    win_length = int(checkpoint.get("win_length") or 5)
    channels = int(checkpoint.get("channels") or 128)
    conv_layers = int(checkpoint.get("conv_layers") or 8)
    policy = BootstrapPolicyValueNet(channels=channels, conv_layers=conv_layers).to(device)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy, board_size, win_length, channels, conv_layers


def heuristic_policy_distribution(
    board: np.ndarray,
    current_player: int,
    win_length: int,
    smoothing: float,
) -> np.ndarray:
    board_size = board.shape[0]
    legal_actions = np.flatnonzero((board == 0).reshape(-1))
    if len(legal_actions) == 0:
        raise ValueError("no legal actions available")

    probs = np.zeros(board_size * board_size, dtype=np.float32)
    own_wins = immediate_winning_actions(board, current_player, win_length)
    if own_wins:
        probs[own_wins] = 1.0 / len(own_wins)
    else:
        candidate_actions = heuristic_candidate_actions(board)
        opp_wins = set(immediate_winning_actions(board, -current_player, win_length))
        scores: list[float] = []
        for action in candidate_actions:
            own_score = score_heuristic_move(board, action, current_player, win_length)
            opp_score = score_heuristic_move(board, action, -current_player, win_length)
            block_bonus = 500_000.0 if action in opp_wins else 0.0
            scores.append(own_score + 1.1 * opp_score + block_bonus)

        shifted = np.asarray(scores, dtype=np.float64)
        shifted = shifted - float(np.min(shifted))
        logits = np.log1p(shifted)
        logits = logits - float(np.max(logits))
        candidate_probs = np.exp(logits)
        candidate_probs /= np.sum(candidate_probs)
        for action, value in zip(candidate_actions, candidate_probs, strict=False):
            probs[action] = float(value)

    if smoothing > 0.0:
        uniform = np.zeros_like(probs)
        uniform[legal_actions] = 1.0 / len(legal_actions)
        probs = (1.0 - smoothing) * probs + smoothing * uniform

    total = float(np.sum(probs))
    if total <= 0.0:
        probs[legal_actions] = 1.0 / len(legal_actions)
    else:
        probs /= total
    return probs.astype(np.float32)


def generate_heuristic_dataset(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    env = GomokuEnv(board_size=args.board_size, win_length=args.win_length)

    states: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    values: list[float] = []
    winners: list[int] = []
    lengths: list[int] = []

    for game_idx in range(1, args.games + 1):
        env.reset()
        history: list[tuple[np.ndarray, np.ndarray, int]] = []

        opening_moves = random.randint(0, max(args.random_opening_moves, 0))
        for _ in range(opening_moves):
            if env.done:
                break
            env.step(int(np.random.choice(env.valid_moves())))

        move_count = 0
        while not env.done:
            state = encode_state(env.board, env.current_player).numpy()
            policy = heuristic_policy_distribution(
                board=env.board,
                current_player=env.current_player,
                win_length=args.win_length,
                smoothing=args.policy_smoothing,
            )
            history.append((state, policy, env.current_player))
            action = choose_heuristic_action(
                board=env.board,
                current_player=env.current_player,
                win_length=args.win_length,
            )
            env.step(action)
            move_count += 1

        for state, policy, player in history:
            if env.winner == 0:
                value = 0.0
            else:
                value = 1.0 if player == env.winner else -1.0
            states.append(state.astype(np.float32))
            policies.append(policy.astype(np.float32))
            values.append(value)

        winners.append(env.winner)
        lengths.append(move_count)
        if args.log_every_games > 0 and game_idx % args.log_every_games == 0:
            avg_len = float(np.mean(lengths)) if lengths else 0.0
            print(
                f"generate game={game_idx:5d}/{args.games:5d} "
                f"avg_len={avg_len:6.2f} samples={len(states):7d}"
            )

    np.savez_compressed(
        args.output,
        states=np.asarray(states, dtype=np.float32),
        policies=np.asarray(policies, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32),
        board_size=np.int32(args.board_size),
        win_length=np.int32(args.win_length),
    )
    p1_wins = sum(1 for winner in winners if winner == 1)
    p2_wins = sum(1 for winner in winners if winner == -1)
    draws = sum(1 for winner in winners if winner == 0)
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    print(
        f"saved dataset to {args.output} samples={len(states)} "
        f"p1={p1_wins} p2={p2_wins} draw={draws} avg_len={avg_len:.2f}"
    )


def load_dataset(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    all_states: list[np.ndarray] = []
    all_policies: list[np.ndarray] = []
    all_values: list[np.ndarray] = []
    board_size: int | None = None
    win_length: int | None = None

    for path in paths:
        with np.load(path) as data:
            dataset_board = int(data["board_size"])
            dataset_win = int(data["win_length"])
            if board_size is None:
                board_size = dataset_board
                win_length = dataset_win
            elif board_size != dataset_board or win_length != dataset_win:
                raise ValueError("all datasets must share board_size and win_length")

            all_states.append(data["states"].astype(np.float32))
            all_policies.append(data["policies"].astype(np.float32))
            all_values.append(data["values"].astype(np.float32))

    if board_size is None or win_length is None:
        raise ValueError("no dataset provided")
    return (
        np.concatenate(all_states, axis=0),
        np.concatenate(all_policies, axis=0),
        np.concatenate(all_values, axis=0),
        board_size,
        win_length,
    )


def sample_supervised_batch(
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    batch_size: int,
) -> list[tuple[torch.Tensor, np.ndarray, float]]:
    count = len(states)
    replace = count < batch_size
    indices = np.random.choice(count, size=batch_size, replace=replace)
    batch: list[tuple[torch.Tensor, np.ndarray, float]] = []
    for idx in indices:
        batch.append(
            (
                torch.from_numpy(states[int(idx)]),
                policies[int(idx)],
                float(values[int(idx)]),
            )
        )
    return batch


def run_pretrain(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = choose_device(args.device)
    states, policies, values, board_size, win_length = load_dataset(args.dataset)

    policy = BootstrapPolicyValueNet(channels=args.channels, conv_layers=args.conv_layers).to(device)
    if args.init_checkpoint is not None:
        checkpoint = load_bootstrap_checkpoint(args.init_checkpoint, map_location=device)
        policy.load_state_dict(checkpoint["state_dict"])

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(
        f"device={device} dataset_samples={len(states)} board={board_size} win={win_length} "
        f"params={count_parameters(policy):,}"
    )

    losses: list[tuple[float, float, float]] = []
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        batch = sample_supervised_batch(states, policies, values, args.batch_size)
        loss_tuple = train_batch(
            policy=policy,
            optimizer=optimizer,
            batch=batch,
            device=device,
            value_coef=args.value_coef,
        )
        losses.append(loss_tuple)

        if args.log_every_steps > 0 and step % args.log_every_steps == 0:
            elapsed = time.perf_counter() - start
            print(
                f"step={step:5d}/{args.steps:5d} loss={loss_tuple[0]:7.4f} "
                f"policy={loss_tuple[1]:7.4f} value={loss_tuple[2]:7.4f} elapsed={elapsed:7.1f}s"
            )

        if args.eval_every > 0 and step % args.eval_every == 0:
            policy.eval()
            random_win_rate, random_wins, random_draws, random_losses = evaluate_vs_opponent(
                policy=policy,
                board_size=board_size,
                win_length=win_length,
                device=device,
                games=args.eval_games,
                opponent="random",
                agent=args.eval_agent,
                mcts_sims=args.eval_mcts_sims,
                c_puct=args.c_puct,
            )
            heuristic_win_rate, heuristic_wins, heuristic_draws, heuristic_losses = evaluate_vs_opponent(
                policy=policy,
                board_size=board_size,
                win_length=win_length,
                device=device,
                games=args.eval_heuristic_games,
                opponent="heuristic",
                agent=args.eval_agent,
                mcts_sims=args.eval_mcts_sims,
                c_puct=args.c_puct,
            )
            print(
                f"eval step={step:5d} random_win_rate={random_win_rate:.3f} "
                f"({random_wins}/{random_draws}/{random_losses}) "
                f"heuristic_win_rate={heuristic_win_rate:.3f} "
                f"({heuristic_wins}/{heuristic_draws}/{heuristic_losses})"
            )
            policy.train()

        if args.save_every > 0 and step % args.save_every == 0:
            path = last_checkpoint_path(args.checkpoint)
            save_bootstrap_checkpoint(
                path=path,
                policy=policy,
                board_size=board_size,
                win_length=win_length,
                channels=args.channels,
                conv_layers=args.conv_layers,
            )
            print(f"saved checkpoint to {path}")

    save_bootstrap_checkpoint(
        path=args.checkpoint,
        policy=policy,
        board_size=board_size,
        win_length=win_length,
        channels=args.channels,
        conv_layers=args.conv_layers,
    )
    if losses:
        avg_loss = float(np.mean([item[0] for item in losses[-min(len(losses), 100) :]]))
        avg_policy = float(np.mean([item[1] for item in losses[-min(len(losses), 100) :]]))
        avg_value = float(np.mean([item[2] for item in losses[-min(len(losses), 100) :]]))
        print(f"final_avg loss={avg_loss:.4f} policy={avg_policy:.4f} value={avg_value:.4f}")
    print(f"saved checkpoint to {args.checkpoint}")


def run_finetune(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = choose_device(args.device)
    policy = BootstrapPolicyValueNet(channels=args.channels, conv_layers=args.conv_layers).to(device)
    if args.init_checkpoint is not None and args.init_checkpoint.exists():
        checkpoint = load_bootstrap_checkpoint(args.init_checkpoint, map_location=device)
        policy.load_state_dict(checkpoint["state_dict"])

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    replay_buffer: deque[tuple[torch.Tensor, np.ndarray, float]] = deque(maxlen=args.buffer_size)
    print(
        f"device={device} board={args.board_size} win={args.win_length} "
        f"params={count_parameters(policy):,}"
    )

    for iteration in range(1, args.iterations + 1):
        policy.eval()
        winners: list[int] = []
        lengths: list[int] = []
        iteration_start = time.perf_counter()
        for _ in range(args.games_per_iter):
            examples, winner, moves = self_play_game(
                policy=policy,
                board_size=args.board_size,
                win_length=args.win_length,
                device=device,
                mcts_sims=args.mcts_sims,
                c_puct=args.c_puct,
                temperature=args.temperature,
                temperature_drop_moves=args.temperature_drop_moves,
                dirichlet_alpha=args.dirichlet_alpha,
                noise_eps=args.noise_eps,
                random_opening_moves=args.random_opening_moves,
            )
            replay_buffer.extend(examples)
            winners.append(winner)
            lengths.append(moves)
            games_done = len(winners)
            if args.log_every_games > 0 and games_done % args.log_every_games == 0:
                elapsed = time.perf_counter() - iteration_start
                avg_len_so_far = float(np.mean(lengths)) if lengths else 0.0
                print(
                    f"iter={iteration:5d} selfplay={games_done:3d}/{args.games_per_iter:3d} "
                    f"avg_len={avg_len_so_far:6.2f} buffer={len(replay_buffer):6d} "
                    f"elapsed={elapsed:7.1f}s"
                )

        losses: list[tuple[float, float, float]] = []
        if len(replay_buffer) >= args.batch_size:
            policy.train()
            for step_idx in range(args.train_steps):
                batch = random.sample(replay_buffer, args.batch_size)
                losses.append(
                    train_batch(
                        policy=policy,
                        optimizer=optimizer,
                        batch=batch,
                        device=device,
                        value_coef=args.value_coef,
                    )
                )
                if args.log_every_train_steps > 0 and (step_idx + 1) % args.log_every_train_steps == 0:
                    elapsed = time.perf_counter() - iteration_start
                    recent = losses[-1]
                    print(
                        f"iter={iteration:5d} train={step_idx + 1:3d}/{args.train_steps:3d} "
                        f"loss={recent[0]:7.4f} policy={recent[1]:7.4f} "
                        f"value={recent[2]:7.4f} elapsed={elapsed:7.1f}s"
                    )

        avg_loss = float(np.mean([item[0] for item in losses])) if losses else 0.0
        avg_policy = float(np.mean([item[1] for item in losses])) if losses else 0.0
        avg_value = float(np.mean([item[2] for item in losses])) if losses else 0.0
        p1_wins = sum(1 for winner in winners if winner == 1)
        p2_wins = sum(1 for winner in winners if winner == -1)
        draws = sum(1 for winner in winners if winner == 0)
        avg_len = float(np.mean(lengths)) if lengths else 0.0

        message = (
            f"iter={iteration:5d} loss={avg_loss:7.4f} policy={avg_policy:7.4f} "
            f"value={avg_value:7.4f} p1={p1_wins:3d} p2={p2_wins:3d} draw={draws:3d} "
            f"avg_len={avg_len:6.2f} buffer={len(replay_buffer):6d}"
        )
        if args.eval_every > 0 and iteration % args.eval_every == 0:
            policy.eval()
            random_win_rate, random_wins, random_draws, random_losses = evaluate_vs_opponent(
                policy=policy,
                board_size=args.board_size,
                win_length=args.win_length,
                device=device,
                games=args.eval_games,
                opponent="random",
                agent="mcts",
                mcts_sims=args.eval_mcts_sims,
                c_puct=args.c_puct,
            )
            message += f" random_win_rate={random_win_rate:.3f} ({random_wins}/{random_draws}/{random_losses})"
            if args.eval_heuristic_games > 0:
                heuristic_win_rate, heuristic_wins, heuristic_draws, heuristic_losses = evaluate_vs_opponent(
                    policy=policy,
                    board_size=args.board_size,
                    win_length=args.win_length,
                    device=device,
                    games=args.eval_heuristic_games,
                    opponent="heuristic",
                    agent="mcts",
                    mcts_sims=args.eval_mcts_sims,
                    c_puct=args.c_puct,
                )
                message += (
                    f" heuristic_win_rate={heuristic_win_rate:.3f} "
                    f"({heuristic_wins}/{heuristic_draws}/{heuristic_losses})"
                )
        print(message)

        if args.eval_every > 0 and iteration % args.eval_every == 0 and args.eval_trace_games > 0:
            for trace_idx in range(args.eval_trace_games):
                winner, trace_moves, final_board = evaluate_self_play_trace(
                    policy=policy,
                    board_size=args.board_size,
                    win_length=args.win_length,
                    device=device,
                    mcts_sims=args.eval_mcts_sims,
                    c_puct=args.c_puct,
                )
                print(
                    f"eval_trace game={trace_idx + 1} winner={winner_to_text(winner)} "
                    f"moves={len(trace_moves)}"
                )
                print(f"eval_trace seq {format_trace_moves(trace_moves, args.eval_trace_max_moves)}")
                print("eval_trace board:")
                print(final_board)

        if args.save_every > 0 and iteration % args.save_every == 0:
            path = last_checkpoint_path(args.checkpoint)
            save_bootstrap_checkpoint(
                path=path,
                policy=policy,
                board_size=args.board_size,
                win_length=args.win_length,
                channels=args.channels,
                conv_layers=args.conv_layers,
            )
            print(f"saved checkpoint to {path}")

    save_bootstrap_checkpoint(
        path=args.checkpoint,
        policy=policy,
        board_size=args.board_size,
        win_length=args.win_length,
        channels=args.channels,
        conv_layers=args.conv_layers,
    )
    print(f"saved checkpoint to {args.checkpoint}")


def run_eval(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    policy, board_size, win_length, channels, conv_layers = build_policy_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
    )
    win_rate, wins, draws, losses = evaluate_vs_opponent(
        policy=policy,
        board_size=board_size,
        win_length=win_length,
        device=device,
        games=args.games,
        opponent=args.opponent,
        agent=args.agent,
        mcts_sims=args.mcts_sims,
        c_puct=args.c_puct,
    )
    print(f"device={device}")
    print(
        f"agent={args.agent} opponent={args.opponent} mcts_sims={args.mcts_sims} "
        f"channels={channels} conv_layers={conv_layers}"
    )
    print(f"win_rate={win_rate:.3f} wins={wins} draws={draws} losses={losses}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Heuristic bootstrap + RL Gomoku trainer")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    generate_parser = subparsers.add_parser("generate", help="generate heuristic dataset")
    generate_parser.add_argument("--board-size", type=int, default=9)
    generate_parser.add_argument("--win-length", type=int, default=5)
    generate_parser.add_argument("--games", type=int, default=2000)
    generate_parser.add_argument("--random-opening-moves", type=int, default=2)
    generate_parser.add_argument("--policy-smoothing", type=float, default=0.05)
    generate_parser.add_argument("--log-every-games", type=int, default=100)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--output", type=Path, default=Path("heuristic_dataset_9x9_5.npz"))
    generate_parser.set_defaults(func=generate_heuristic_dataset)

    pretrain_parser = subparsers.add_parser("pretrain", help="supervised pretrain from heuristic data")
    pretrain_parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    pretrain_parser.add_argument("--channels", type=int, default=128)
    pretrain_parser.add_argument("--conv-layers", type=int, default=8)
    pretrain_parser.add_argument("--steps", type=int, default=4000)
    pretrain_parser.add_argument("--batch-size", type=int, default=256)
    pretrain_parser.add_argument("--lr", type=float, default=3e-4)
    pretrain_parser.add_argument("--weight-decay", type=float, default=1e-4)
    pretrain_parser.add_argument("--value-coef", type=float, default=1.0)
    pretrain_parser.add_argument("--eval-every", type=int, default=400)
    pretrain_parser.add_argument("--eval-games", type=int, default=20)
    pretrain_parser.add_argument("--eval-heuristic-games", type=int, default=8)
    pretrain_parser.add_argument("--eval-agent", choices=["policy", "mcts"], default="policy")
    pretrain_parser.add_argument("--eval-mcts-sims", type=int, default=96)
    pretrain_parser.add_argument("--c-puct", type=float, default=1.5)
    pretrain_parser.add_argument("--save-every", type=int, default=1000)
    pretrain_parser.add_argument("--log-every-steps", type=int, default=100)
    pretrain_parser.add_argument("--seed", type=int, default=42)
    pretrain_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    pretrain_parser.add_argument("--init-checkpoint", type=Path, default=None)
    pretrain_parser.add_argument("--checkpoint", type=Path, default=Path("gomoku_bootstrap_pretrain.pt"))
    pretrain_parser.set_defaults(func=run_pretrain)

    finetune_parser = subparsers.add_parser("finetune", help="MCTS self-play finetune")
    finetune_parser.add_argument("--board-size", type=int, default=15)
    finetune_parser.add_argument("--win-length", type=int, default=5)
    finetune_parser.add_argument("--channels", type=int, default=128)
    finetune_parser.add_argument("--conv-layers", type=int, default=8)
    finetune_parser.add_argument("--iterations", type=int, default=1500)
    finetune_parser.add_argument("--games-per-iter", type=int, default=32)
    finetune_parser.add_argument("--train-steps", type=int, default=64)
    finetune_parser.add_argument("--batch-size", type=int, default=128)
    finetune_parser.add_argument("--buffer-size", type=int, default=100000)
    finetune_parser.add_argument("--lr", type=float, default=3e-4)
    finetune_parser.add_argument("--weight-decay", type=float, default=1e-4)
    finetune_parser.add_argument("--value-coef", type=float, default=1.0)
    finetune_parser.add_argument("--mcts-sims", type=int, default=256)
    finetune_parser.add_argument("--eval-mcts-sims", type=int, default=512)
    finetune_parser.add_argument("--c-puct", type=float, default=1.5)
    finetune_parser.add_argument("--temperature", type=float, default=0.6)
    finetune_parser.add_argument("--temperature-drop-moves", type=int, default=3)
    finetune_parser.add_argument("--dirichlet-alpha", type=float, default=0.03)
    finetune_parser.add_argument("--noise-eps", type=float, default=0.10)
    finetune_parser.add_argument("--random-opening-moves", type=int, default=2)
    finetune_parser.add_argument("--eval-every", type=int, default=10)
    finetune_parser.add_argument("--eval-games", type=int, default=20)
    finetune_parser.add_argument("--eval-heuristic-games", type=int, default=8)
    finetune_parser.add_argument("--eval-trace-games", type=int, default=1)
    finetune_parser.add_argument("--eval-trace-max-moves", type=int, default=20)
    finetune_parser.add_argument("--save-every", type=int, default=10)
    finetune_parser.add_argument("--log-every-games", type=int, default=4)
    finetune_parser.add_argument("--log-every-train-steps", type=int, default=16)
    finetune_parser.add_argument("--seed", type=int, default=42)
    finetune_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    finetune_parser.add_argument("--init-checkpoint", type=Path, default=None)
    finetune_parser.add_argument("--checkpoint", type=Path, default=Path("gomoku_bootstrap_finetune.pt"))
    finetune_parser.set_defaults(func=run_finetune)

    eval_parser = subparsers.add_parser("eval", help="evaluate a bootstrap checkpoint")
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--agent", choices=["policy", "mcts"], default="mcts")
    eval_parser.add_argument("--opponent", choices=["random", "heuristic"], default="heuristic")
    eval_parser.add_argument("--games", type=int, default=40)
    eval_parser.add_argument("--mcts-sims", type=int, default=256)
    eval_parser.add_argument("--c-puct", type=float, default=1.5)
    eval_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    eval_parser.set_defaults(func=run_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
