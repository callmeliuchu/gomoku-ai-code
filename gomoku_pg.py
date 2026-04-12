#!/usr/bin/env python3
"""Minimal Gomoku policy gradient example.

Features:
1. Configurable board size and win length, e.g. 5x5 connect-4 or 15x15 connect-5.
2. Shared-policy self-play with REINFORCE.
3. Fully convolutional policy, so the same code works for different board sizes.
4. Optional random-agent evaluation and CLI human play.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GomokuEnv:
    def __init__(self, board_size: int, win_length: int):
        if board_size <= 1:
            raise ValueError("board_size must be > 1")
        if not 1 < win_length <= board_size:
            raise ValueError("win_length must satisfy 1 < win_length <= board_size")
        self.board_size = board_size
        self.win_length = win_length
        self.reset()

    def reset(self) -> np.ndarray:
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = 0
        return self.board

    def legal_mask(self) -> np.ndarray:
        return self.board == 0

    def valid_moves(self) -> np.ndarray:
        return np.flatnonzero(self.legal_mask().reshape(-1))

    def step(self, action: int) -> tuple[bool, int]:
        if self.done:
            raise RuntimeError("game is already finished")

        row, col = divmod(int(action), self.board_size)
        if self.board[row, col] != 0:
            raise ValueError(f"illegal move at ({row}, {col})")

        player = self.current_player
        self.board[row, col] = player

        if self._is_winning_move(row, col, player):
            self.done = True
            self.winner = player
        elif not np.any(self.board == 0):
            self.done = True
            self.winner = 0
        else:
            self.current_player = -player

        return self.done, self.winner

    def _is_winning_move(self, row: int, col: int, player: int) -> bool:
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        for dr, dc in directions:
            count = 1
            count += self._count_one_side(row, col, dr, dc, player)
            count += self._count_one_side(row, col, -dr, -dc, player)
            if count >= self.win_length:
                return True
        return False

    def _count_one_side(self, row: int, col: int, dr: int, dc: int, player: int) -> int:
        total = 0
        r, c = row + dr, col + dc
        while 0 <= r < self.board_size and 0 <= c < self.board_size:
            if self.board[r, c] != player:
                break
            total += 1
            r += dr
            c += dc
        return total

    def render(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        header = "   " + " ".join(f"{i + 1:2d}" for i in range(self.board_size))
        rows = [header]
        for row_idx in range(self.board_size):
            row = " ".join(f"{symbols[int(v)]:>2}" for v in self.board[row_idx])
            rows.append(f"{row_idx + 1:2d} {row}")
        return "\n".join(rows)


def encode_state(board: np.ndarray, current_player: int) -> torch.Tensor:
    current = (board == current_player).astype(np.float32)
    opponent = (board == -current_player).astype(np.float32)
    legal = (board == 0).astype(np.float32)
    stacked = np.stack([current, opponent, legal], axis=0)
    return torch.from_numpy(stacked)


class PolicyValueNet(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
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


def masked_logits(logits: torch.Tensor, legal_mask: np.ndarray) -> torch.Tensor:
    legal = torch.as_tensor(legal_mask.reshape(-1), device=logits.device, dtype=torch.bool)
    return logits.masked_fill(~legal, -1e9)


def transform_board(board: np.ndarray, rotation_k: int, flip: bool) -> np.ndarray:
    transformed = np.rot90(board, k=rotation_k)
    if flip:
        transformed = np.fliplr(transformed)
    return np.ascontiguousarray(transformed)


def action_to_coords(action: int, board_size: int) -> tuple[int, int]:
    return divmod(int(action), board_size)


def coords_to_action(row: int, col: int, board_size: int) -> int:
    return row * board_size + col


def count_one_side(
    board: np.ndarray,
    row: int,
    col: int,
    dr: int,
    dc: int,
    player: int,
) -> int:
    board_size = board.shape[0]
    total = 0
    r, c = row + dr, col + dc
    while 0 <= r < board_size and 0 <= c < board_size:
        if board[r, c] != player:
            break
        total += 1
        r += dr
        c += dc
    return total


def is_winning_move(
    board: np.ndarray,
    row: int,
    col: int,
    player: int,
    win_length: int,
) -> bool:
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    for dr, dc in directions:
        count = 1
        count += count_one_side(board, row, col, dr, dc, player)
        count += count_one_side(board, row, col, -dr, -dc, player)
        if count >= win_length:
            return True
    return False


def apply_action_to_board(
    board: np.ndarray,
    current_player: int,
    action: int,
    win_length: int,
) -> tuple[np.ndarray, int, bool, int]:
    board_size = board.shape[0]
    row, col = action_to_coords(action, board_size)
    if board[row, col] != 0:
        raise ValueError(f"illegal move at ({row}, {col})")

    next_board = board.copy()
    next_board[row, col] = current_player

    if is_winning_move(next_board, row, col, current_player, win_length):
        return next_board, -current_player, True, current_player
    if not np.any(next_board == 0):
        return next_board, -current_player, True, 0
    return next_board, -current_player, False, 0


def forward_transform_coords(
    row: int,
    col: int,
    board_size: int,
    rotation_k: int,
    flip: bool,
) -> tuple[int, int]:
    for _ in range(rotation_k % 4):
        row, col = board_size - 1 - col, row
    if flip:
        col = board_size - 1 - col
    return row, col


def inverse_transform_coords(
    row: int,
    col: int,
    board_size: int,
    rotation_k: int,
    flip: bool,
) -> tuple[int, int]:
    if flip:
        col = board_size - 1 - col
    for _ in range(rotation_k % 4):
        row, col = col, board_size - 1 - row
    return row, col


def sample_action(
    policy: PolicyValueNet,
    board: np.ndarray,
    current_player: int,
    device: torch.device,
    greedy: bool,
    augment: bool,
) -> tuple[int, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    board_size = board.shape[0]
    rotation_k = random.randint(0, 3) if augment else 0
    flip = bool(random.getrandbits(1)) if augment else False
    transformed_board = transform_board(board, rotation_k=rotation_k, flip=flip)

    state = encode_state(transformed_board, current_player).unsqueeze(0).to(device)
    logits, value = policy(state)
    logits = masked_logits(logits.squeeze(0), transformed_board == 0)

    if greedy:
        action = torch.argmax(logits)
        transformed_row, transformed_col = action_to_coords(int(action.item()), board_size)
        row, col = inverse_transform_coords(
            transformed_row,
            transformed_col,
            board_size,
            rotation_k=rotation_k,
            flip=flip,
        )
        return coords_to_action(row, col, board_size), None, None, value.squeeze(0)

    dist = Categorical(logits=logits)
    action = dist.sample()
    transformed_row, transformed_col = action_to_coords(int(action.item()), board_size)
    row, col = inverse_transform_coords(
        transformed_row,
        transformed_col,
        board_size,
        rotation_k=rotation_k,
        flip=flip,
    )
    return (
        coords_to_action(row, col, board_size),
        dist.log_prob(action),
        dist.entropy(),
        value.squeeze(0),
    )


def evaluate_policy_value(
    policy: PolicyValueNet,
    board: np.ndarray,
    current_player: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    state = encode_state(board, current_player).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, value = policy(state)
        logits = masked_logits(logits.squeeze(0), board == 0)
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
    return probs, float(value.item())


@dataclass
class MCTSNode:
    board: np.ndarray
    current_player: int
    win_length: int
    done: bool = False
    winner: int = 0
    priors: dict[int, float] = field(default_factory=dict)
    visit_counts: dict[int, int] = field(default_factory=dict)
    value_sums: dict[int, float] = field(default_factory=dict)
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    expanded: bool = False

    def expand(self, priors: np.ndarray) -> None:
        legal_actions = np.flatnonzero(self.board.reshape(-1) == 0)
        total_prob = float(np.sum(priors[legal_actions]))
        if total_prob <= 0.0:
            uniform = 1.0 / max(len(legal_actions), 1)
            self.priors = {int(action): uniform for action in legal_actions}
        else:
            self.priors = {
                int(action): float(priors[action] / total_prob)
                for action in legal_actions
            }
        self.visit_counts = {action: 0 for action in self.priors}
        self.value_sums = {action: 0.0 for action in self.priors}
        self.expanded = True

    def q_value(self, action: int) -> float:
        visits = self.visit_counts[action]
        if visits == 0:
            return 0.0
        return self.value_sums[action] / visits

    def select_action(self, c_puct: float) -> int:
        total_visits = sum(self.visit_counts.values())
        sqrt_total = math.sqrt(total_visits + 1.0)
        best_action = -1
        best_score = -float("inf")

        for action, prior in self.priors.items():
            visits = self.visit_counts[action]
            q = self.q_value(action)
            u = c_puct * prior * sqrt_total / (1.0 + visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def child_for_action(self, action: int) -> "MCTSNode":
        child = self.children.get(action)
        if child is not None:
            return child

        next_board, next_player, done, winner = apply_action_to_board(
            board=self.board,
            current_player=self.current_player,
            action=action,
            win_length=self.win_length,
        )
        child = MCTSNode(
            board=next_board,
            current_player=next_player,
            win_length=self.win_length,
            done=done,
            winner=winner,
        )
        self.children[action] = child
        return child


def terminal_value(winner: int, current_player: int) -> float:
    if winner == 0:
        return 0.0
    return 1.0 if winner == current_player else -1.0


def choose_mcts_action(
    policy: PolicyValueNet,
    board: np.ndarray,
    current_player: int,
    win_length: int,
    device: torch.device,
    num_simulations: int,
    c_puct: float,
) -> tuple[int, np.ndarray]:
    root = MCTSNode(
        board=board.copy(),
        current_player=current_player,
        win_length=win_length,
    )

    priors, _ = evaluate_policy_value(policy, root.board, root.current_player, device)
    root.expand(priors)

    for _ in range(num_simulations):
        node = root
        path: list[tuple[MCTSNode, int]] = []

        while node.expanded and not node.done:
            action = node.select_action(c_puct)
            path.append((node, action))
            node = node.child_for_action(action)

        if node.done:
            value = terminal_value(node.winner, node.current_player)
        else:
            priors, value = evaluate_policy_value(policy, node.board, node.current_player, device)
            node.expand(priors)

        for parent, action in reversed(path):
            value = -value
            parent.visit_counts[action] += 1
            parent.value_sums[action] += value

    visits = np.zeros(board.size, dtype=np.float32)
    for action, count in root.visit_counts.items():
        visits[action] = float(count)

    if np.all(visits == 0):
        best_action = int(np.argmax(priors))
    else:
        best_action = int(np.argmax(visits))

    return best_action, visits.reshape(board.shape)


def choose_ai_action(
    policy: PolicyValueNet,
    board: np.ndarray,
    current_player: int,
    win_length: int,
    device: torch.device,
    agent: str,
    mcts_sims: int,
    c_puct: float,
) -> tuple[int, np.ndarray | None]:
    if agent == "mcts":
        return choose_mcts_action(
            policy=policy,
            board=board,
            current_player=current_player,
            win_length=win_length,
            device=device,
            num_simulations=mcts_sims,
            c_puct=c_puct,
        )

    action, _, _, _ = sample_action(
        policy=policy,
        board=board,
        current_player=current_player,
        device=device,
        greedy=True,
        augment=False,
    )
    return action, None


def self_play_episode(
    policy: PolicyValueNet,
    env: GomokuEnv,
    device: torch.device,
    gamma: float,
    augment: bool,
) -> tuple[list[torch.Tensor], list[float], list[torch.Tensor], list[torch.Tensor], int, int]:
    env.reset()
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    players: list[int] = []

    while not env.done:
        player = env.current_player
        action, log_prob, entropy, value = sample_action(
            policy=policy,
            board=env.board,
            current_player=player,
            device=device,
            greedy=False,
            augment=augment,
        )
        log_probs.append(log_prob)
        entropies.append(entropy)
        values.append(value)
        players.append(player)
        env.step(action)

    returns: list[float] = []
    total_moves = len(players)
    for move_idx, player in enumerate(players):
        outcome = 0.0
        if env.winner != 0:
            outcome = 1.0 if player == env.winner else -1.0
        discounted = outcome * (gamma ** (total_moves - move_idx - 1))
        returns.append(discounted)

    return log_probs, returns, entropies, values, env.winner, total_moves


def update_policy(
    optimizer: torch.optim.Optimizer,
    batch_log_probs: list[torch.Tensor],
    batch_returns: list[float],
    batch_entropies: list[torch.Tensor],
    batch_values: list[torch.Tensor],
    entropy_coef: float,
    value_coef: float,
    grad_clip: float,
    device: torch.device,
) -> float:
    returns = torch.tensor(batch_returns, dtype=torch.float32, device=device)
    log_probs = torch.stack(batch_log_probs)
    entropies = torch.stack(batch_entropies)
    values = torch.stack(batch_values)
    advantages = returns - values.detach()
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)

    policy_loss = -(log_probs * advantages).mean()
    value_loss = torch.mean((values - returns) ** 2)
    entropy_bonus = entropies.mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], grad_clip)
    optimizer.step()
    return float(loss.item())


def save_checkpoint(
    path: Path,
    policy: PolicyValueNet,
    args: argparse.Namespace,
) -> None:
    payload = {
        "state_dict": policy.state_dict(),
        "channels": args.channels,
        "board_size": args.board_size,
        "win_length": args.win_length,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint
    if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
        raise RuntimeError(
            f"{path} is an old fixed-board checkpoint from the previous implementation. "
            "It is not compatible with the current fully-convolutional actor-critic model. "
            "Please retrain with the current script."
        )
    return {
        "state_dict": checkpoint,
        "channels": 64,
        "board_size": None,
        "win_length": None,
    }


def load_policy(path: Path, channels: int, device: torch.device) -> PolicyValueNet:
    checkpoint = load_checkpoint(path, map_location=device)
    state_dict = checkpoint["state_dict"]
    saved_channels = int(checkpoint.get("channels", channels))

    policy = PolicyValueNet(channels=saved_channels).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def resolve_game_config(
    checkpoint_path: Path,
    arg_board_size: int | None,
    arg_win_length: int | None,
    arg_channels: int,
    device: torch.device,
) -> tuple[PolicyValueNet, int, int]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    board_size = int(checkpoint.get("board_size") or arg_board_size or 15)
    win_length = int(checkpoint.get("win_length") or arg_win_length or 5)
    channels = int(checkpoint.get("channels") or arg_channels)

    policy = PolicyValueNet(channels=channels).to(device)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy, board_size, win_length


def play_vs_random_once(
    policy: PolicyValueNet,
    board_size: int,
    win_length: int,
    device: torch.device,
    policy_player: int,
    agent: str = "policy",
    mcts_sims: int = 100,
    c_puct: float = 1.5,
) -> int:
    env = GomokuEnv(board_size=board_size, win_length=win_length)
    env.reset()

    while not env.done:
        if env.current_player == policy_player:
            action, _ = choose_ai_action(
                policy=policy,
                board=env.board,
                current_player=env.current_player,
                win_length=win_length,
                device=device,
                agent=agent,
                mcts_sims=mcts_sims,
                c_puct=c_puct,
            )
        else:
            action = int(np.random.choice(env.valid_moves()))
        env.step(action)

    return env.winner


def evaluate_vs_random(
    policy: PolicyValueNet,
    board_size: int,
    win_length: int,
    device: torch.device,
    games: int,
    agent: str = "policy",
    mcts_sims: int = 100,
    c_puct: float = 1.5,
) -> tuple[float, int, int, int]:
    wins = 0
    draws = 0
    losses = 0

    for game_idx in range(games):
        policy_player = 1 if game_idx < games // 2 else -1
        winner = play_vs_random_once(
            policy=policy,
            board_size=board_size,
            win_length=win_length,
            device=device,
            policy_player=policy_player,
            agent=agent,
            mcts_sims=mcts_sims,
            c_puct=c_puct,
        )
        if winner == 0:
            draws += 1
        elif winner == policy_player:
            wins += 1
        else:
            losses += 1

    return wins / max(games, 1), wins, draws, losses


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = choose_device(args.device)
    env = GomokuEnv(board_size=args.board_size, win_length=args.win_length)
    policy = PolicyValueNet(channels=args.channels).to(device)
    if args.init_checkpoint is not None and args.init_checkpoint.exists():
        checkpoint = load_checkpoint(args.init_checkpoint, map_location=device)
        policy.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    recent_winners: deque[int] = deque(maxlen=args.print_every)
    recent_lengths: deque[int] = deque(maxlen=args.print_every)
    batch_log_probs: list[torch.Tensor] = []
    batch_returns: list[float] = []
    batch_entropies: list[torch.Tensor] = []
    batch_values: list[torch.Tensor] = []
    last_loss = 0.0

    print(f"device={device} board={args.board_size} win={args.win_length}")

    for episode in range(1, args.episodes + 1):
        log_probs, returns, entropies, values, winner, moves = self_play_episode(
            policy=policy,
            env=env,
            device=device,
            gamma=args.gamma,
            augment=args.symmetry_augment,
        )
        batch_log_probs.extend(log_probs)
        batch_returns.extend(returns)
        batch_entropies.extend(entropies)
        batch_values.extend(values)
        recent_winners.append(winner)
        recent_lengths.append(moves)

        if episode % args.batch_size == 0 or episode == args.episodes:
            policy.train()
            last_loss = update_policy(
                optimizer=optimizer,
                batch_log_probs=batch_log_probs,
                batch_returns=batch_returns,
                batch_entropies=batch_entropies,
                batch_values=batch_values,
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef,
                grad_clip=args.grad_clip,
                device=device,
            )
            batch_log_probs.clear()
            batch_returns.clear()
            batch_entropies.clear()
            batch_values.clear()

        if episode % args.print_every == 0 or episode == args.episodes:
            p1_wins = sum(1 for x in recent_winners if x == 1)
            p2_wins = sum(1 for x in recent_winners if x == -1)
            draws = sum(1 for x in recent_winners if x == 0)
            avg_len = float(np.mean(recent_lengths)) if recent_lengths else 0.0
            message = (
                f"episode={episode:6d} loss={last_loss:8.4f} "
                f"p1={p1_wins:4d} p2={p2_wins:4d} draw={draws:4d} avg_len={avg_len:6.2f}"
            )
            if args.eval_every > 0 and episode % args.eval_every == 0:
                policy.eval()
                win_rate, wins, eval_draws, losses = evaluate_vs_random(
                    policy=policy,
                    board_size=args.board_size,
                    win_length=args.win_length,
                    device=device,
                    games=args.eval_games,
                )
                message += (
                    f" random_win_rate={win_rate:.3f}"
                    f" ({wins}/{eval_draws}/{losses})"
                )
            print(message)

    save_checkpoint(args.checkpoint, policy, args)
    print(f"saved checkpoint to {args.checkpoint}")


def evaluate(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    policy, board_size, win_length = resolve_game_config(
        checkpoint_path=args.checkpoint,
        arg_board_size=args.board_size,
        arg_win_length=args.win_length,
        arg_channels=args.channels,
        device=device,
    )
    win_rate, wins, draws, losses = evaluate_vs_random(
        policy=policy,
        board_size=board_size,
        win_length=win_length,
        device=device,
        games=args.games,
        agent=args.agent,
        mcts_sims=args.mcts_sims,
        c_puct=args.c_puct,
    )
    print(f"device={device}")
    print(f"agent={args.agent} mcts_sims={args.mcts_sims}")
    print(f"win_rate={win_rate:.3f} wins={wins} draws={draws} losses={losses}")


def ask_human_move(env: GomokuEnv) -> int:
    while True:
        text = input("your move (row col): ").strip()
        parts = text.replace(",", " ").split()
        if len(parts) != 2:
            print("please enter: row col")
            continue
        try:
            row, col = (int(parts[0]) - 1, int(parts[1]) - 1)
        except ValueError:
            print("row and col must be integers")
            continue
        if not (0 <= row < env.board_size and 0 <= col < env.board_size):
            print("move out of range")
            continue
        if env.board[row, col] != 0:
            print("that position is occupied")
            continue
        return row * env.board_size + col


def play(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    policy, board_size, win_length = resolve_game_config(
        checkpoint_path=args.checkpoint,
        arg_board_size=args.board_size,
        arg_win_length=args.win_length,
        arg_channels=args.channels,
        device=device,
    )
    env = GomokuEnv(board_size=board_size, win_length=win_length)
    human_player = 1 if args.human_first else -1

    print(f"device={device}")
    print(
        f"human={'X' if human_player == 1 else 'O'} ai={'O' if human_player == 1 else 'X'} "
        f"agent={args.agent} mcts_sims={args.mcts_sims}"
    )

    while not env.done:
        print()
        print(env.render())
        print()

        if env.current_player == human_player:
            action = ask_human_move(env)
        else:
            action, _ = choose_ai_action(
                policy=policy,
                board=env.board,
                current_player=env.current_player,
                win_length=win_length,
                device=device,
                agent=args.agent,
                mcts_sims=args.mcts_sims,
                c_puct=args.c_puct,
            )
            row, col = divmod(action, env.board_size)
            print(f"ai move: {row + 1} {col + 1}")

        env.step(action)

    print()
    print(env.render())
    if env.winner == 0:
        print("draw")
    elif env.winner == human_player:
        print("you win")
    else:
        print("ai wins")


def gui(args: argparse.Namespace) -> None:
    try:
        import pygame
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "pygame is not installed. Install it with: "
            "~/miniconda3/bin/conda run -n lerobot python -m pip install pygame"
        ) from exc

    device = choose_device(args.device)
    policy, board_size, win_length = resolve_game_config(
        checkpoint_path=args.checkpoint,
        arg_board_size=args.board_size,
        arg_win_length=args.win_length,
        arg_channels=args.channels,
        device=device,
    )
    env = GomokuEnv(board_size=board_size, win_length=win_length)
    human_player = 1 if args.human_first else -1
    last_search_visits: np.ndarray | None = None

    pygame.init()
    pygame.display.set_caption("Gomoku Policy Gradient")
    font = pygame.font.SysFont("Arial", 24)
    small_font = pygame.font.SysFont("Arial", 18)

    cell_size = args.cell_size
    padding = 40
    status_height = 80
    board_pixels = board_size * cell_size
    screen = pygame.display.set_mode(
        (board_pixels + padding * 2, board_pixels + padding * 2 + status_height)
    )
    clock = pygame.time.Clock()

    background = (236, 196, 122)
    line_color = (80, 55, 20)
    black_stone = (20, 20, 20)
    white_stone = (245, 245, 245)
    accent = (180, 40, 40)

    def board_to_screen(row: int, col: int) -> tuple[int, int]:
        x = padding + col * cell_size + cell_size // 2
        y = padding + row * cell_size + cell_size // 2
        return x, y

    def mouse_to_action(pos: tuple[int, int]) -> int | None:
        x, y = pos
        left = padding
        top = padding
        if x < left or y < top:
            return None
        col = (x - left) // cell_size
        row = (y - top) // cell_size
        if not (0 <= row < env.board_size and 0 <= col < env.board_size):
            return None
        if env.board[row, col] != 0:
            return None
        return row * env.board_size + col

    def restart() -> None:
        nonlocal last_search_visits
        env.reset()
        last_search_visits = None
        if env.current_player != human_player:
            ai_step()

    def ai_step() -> None:
        nonlocal last_search_visits
        if env.done or env.current_player == human_player:
            return
        action, visits = choose_ai_action(
            policy=policy,
            board=env.board,
            current_player=env.current_player,
            win_length=win_length,
            device=device,
            agent=args.agent,
            mcts_sims=args.mcts_sims,
            c_puct=args.c_puct,
        )
        last_search_visits = visits
        env.step(action)

    def status_text() -> str:
        if env.done:
            if env.winner == 0:
                return "Draw. Press R to restart."
            if env.winner == human_player:
                return "You win. Press R to restart."
            return "AI wins. Press R to restart."
        if env.current_player == human_player:
            return "Your turn. Left click to place."
        return "AI is thinking..."

    if env.current_player != human_player:
        ai_step()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    restart()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if env.done or env.current_player != human_player:
                    continue
                action = mouse_to_action(event.pos)
                if action is None:
                    continue
                env.step(action)
                ai_step()

        screen.fill(background)

        for idx in range(board_size + 1):
            x = padding + idx * cell_size
            pygame.draw.line(screen, line_color, (x, padding), (x, padding + board_pixels), 2)
            y = padding + idx * cell_size
            pygame.draw.line(screen, line_color, (padding, y), (padding + board_pixels, y), 2)

        for row in range(env.board_size):
            for col in range(env.board_size):
                stone = int(env.board[row, col])
                if stone == 0:
                    continue
                x, y = board_to_screen(row, col)
                color = black_stone if stone == 1 else white_stone
                pygame.draw.circle(screen, color, (x, y), cell_size // 2 - 4)
                pygame.draw.circle(screen, line_color, (x, y), cell_size // 2 - 4, 1)

        for idx in range(board_size):
            label = small_font.render(str(idx + 1), True, line_color)
            screen.blit(
                label,
                (padding + idx * cell_size + cell_size // 2 - label.get_width() // 2, 8),
            )
            screen.blit(
                label,
                (8, padding + idx * cell_size + cell_size // 2 - label.get_height() // 2),
            )

        info = (
            f"{board_size}x{board_size}  connect={win_length}  "
            f"device={device}  human={'X' if human_player == 1 else 'O'}  "
            f"agent={args.agent}"
        )
        info_surface = small_font.render(info, True, line_color)
        status_surface = font.render(status_text(), True, accent)
        screen.blit(info_surface, (padding, padding + board_pixels + 16))
        screen.blit(status_surface, (padding, padding + board_pixels + 42))

        if last_search_visits is not None and args.agent == "mcts":
            peak = float(np.max(last_search_visits))
            if peak > 0:
                stats = small_font.render(
                    f"mcts_sims={args.mcts_sims} peak_visits={int(peak)}",
                    True,
                    line_color,
                )
                screen.blit(stats, (padding + 380, padding + board_pixels + 16))

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Gomoku policy gradient example")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common_arguments(subparser: argparse.ArgumentParser, defaults_from_checkpoint: bool = False) -> None:
        board_default = None if defaults_from_checkpoint else 15
        win_default = None if defaults_from_checkpoint else 5
        subparser.add_argument("--board-size", type=int, default=board_default)
        subparser.add_argument("--win-length", type=int, default=win_default)
        subparser.add_argument("--channels", type=int, default=64)
        subparser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
        subparser.add_argument("--checkpoint", type=Path, default=Path("gomoku_policy.pt"))

    def add_inference_arguments(subparser: argparse.ArgumentParser, default_agent: str = "mcts") -> None:
        subparser.add_argument("--agent", choices=["policy", "mcts"], default=default_agent)
        subparser.add_argument("--mcts-sims", type=int, default=120)
        subparser.add_argument("--c-puct", type=float, default=1.5)

    train_parser = subparsers.add_parser("train", help="self-play training")
    add_common_arguments(train_parser)
    train_parser.add_argument("--episodes", type=int, default=5000)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--entropy-coef", type=float, default=0.01)
    train_parser.add_argument("--value-coef", type=float, default=0.5)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--print-every", type=int, default=100)
    train_parser.add_argument("--eval-every", type=int, default=500)
    train_parser.add_argument("--eval-games", type=int, default=40)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--init-checkpoint", type=Path, default=None)
    train_parser.add_argument(
        "--no-symmetry-augment",
        dest="symmetry_augment",
        action="store_false",
        help="disable random rotation/flip augmentation during training",
    )
    train_parser.set_defaults(symmetry_augment=True)
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval", help="evaluate against random agent")
    add_common_arguments(eval_parser)
    eval_parser.add_argument("--games", type=int, default=100)
    add_inference_arguments(eval_parser)
    eval_parser.set_defaults(func=evaluate)

    play_parser = subparsers.add_parser("play", help="play against the trained model")
    add_common_arguments(play_parser, defaults_from_checkpoint=True)
    play_parser.add_argument("--human-first", action="store_true", help="human plays X")
    add_inference_arguments(play_parser)
    play_parser.set_defaults(func=play)

    gui_parser = subparsers.add_parser("gui", help="pygame GUI for testing against the model")
    add_common_arguments(gui_parser, defaults_from_checkpoint=True)
    gui_parser.add_argument("--human-first", action="store_true", help="human plays X")
    gui_parser.add_argument("--cell-size", type=int, default=48)
    gui_parser.add_argument("--fps", type=int, default=30)
    add_inference_arguments(gui_parser)
    gui_parser.set_defaults(func=gui)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
