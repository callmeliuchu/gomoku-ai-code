#!/usr/bin/env python3
"""Minimal Gomoku MCTS example.

This file is intentionally separate from gomoku_pg.py.
It uses the simpler AlphaZero-style recipe:
1. self-play with MCTS
2. policy/value targets from search
3. supervised update on policy + value heads
"""

from __future__ import annotations

import argparse
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn


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

    def valid_moves(self) -> np.ndarray:
        return np.flatnonzero((self.board == 0).reshape(-1))

    def step(self, action: int) -> tuple[bool, int]:
        next_board, next_player, done, winner = apply_action_to_board(
            board=self.board,
            current_player=self.current_player,
            action=action,
            win_length=self.win_length,
        )
        self.board = next_board
        self.current_player = next_player
        self.done = done
        self.winner = winner
        return done, winner

    def render(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        header = "   " + " ".join(f"{i + 1:2d}" for i in range(self.board_size))
        rows = [header]
        for row_idx in range(self.board_size):
            row = " ".join(f"{symbols[int(v)]:>2}" for v in self.board[row_idx])
            rows.append(f"{row_idx + 1:2d} {row}")
        return "\n".join(rows)


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


def count_one_side_with_open_end(
    board: np.ndarray,
    row: int,
    col: int,
    dr: int,
    dc: int,
    player: int,
) -> tuple[int, bool]:
    board_size = board.shape[0]
    total = 0
    r, c = row + dr, col + dc
    while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
        total += 1
        r += dr
        c += dc
    open_end = 0 <= r < board_size and 0 <= c < board_size and board[r, c] == 0
    return total, open_end


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


def encode_state(board: np.ndarray, current_player: int) -> torch.Tensor:
    current = (board == current_player).astype(np.float32)
    opponent = (board == -current_player).astype(np.float32)
    legal = (board == 0).astype(np.float32)
    return torch.from_numpy(np.stack([current, opponent, legal], axis=0))


def apply_state_symmetry(state: torch.Tensor, symmetry: int) -> torch.Tensor:
    transformed = state
    rotations = symmetry % 4
    if rotations:
        transformed = torch.rot90(transformed, k=rotations, dims=(-2, -1))
    if symmetry >= 4:
        transformed = torch.flip(transformed, dims=(-1,))
    return transformed.contiguous()


def apply_policy_symmetry(policy: np.ndarray, board_size: int, symmetry: int) -> np.ndarray:
    transformed = policy.reshape(board_size, board_size)
    rotations = symmetry % 4
    if rotations:
        transformed = np.rot90(transformed, k=rotations)
    if symmetry >= 4:
        transformed = np.flip(transformed, axis=1)
    return np.ascontiguousarray(transformed.reshape(-1), dtype=np.float32)


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


def masked_logits(logits: torch.Tensor, board: np.ndarray) -> torch.Tensor:
    legal = torch.as_tensor((board == 0).reshape(-1), device=logits.device, dtype=torch.bool)
    return logits.masked_fill(~legal, -1e9)


def evaluate_policy_value(
    policy: PolicyValueNet,
    board: np.ndarray,
    current_player: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    state = encode_state(board, current_player).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, value = policy(state)
        logits = masked_logits(logits.squeeze(0), board)
        probs = torch.softmax(logits, dim=0).cpu().numpy()
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

    def expand(
        self,
        priors: np.ndarray,
        add_noise: bool = False,
        dirichlet_alpha: float = 0.3,
        noise_eps: float = 0.25,
    ) -> None:
        legal_actions = np.flatnonzero((self.board == 0).reshape(-1))
        legal_priors = priors[legal_actions]
        total_prob = float(np.sum(legal_priors))
        if total_prob <= 0.0:
            legal_priors = np.full(len(legal_actions), 1.0 / max(len(legal_actions), 1), dtype=np.float32)
        else:
            legal_priors = legal_priors / total_prob

        if add_noise and len(legal_actions) > 0:
            noise = np.random.dirichlet([dirichlet_alpha] * len(legal_actions))
            legal_priors = (1.0 - noise_eps) * legal_priors + noise_eps * noise

        self.priors = {
            int(action): float(prior)
            for action, prior in zip(legal_actions, legal_priors, strict=False)
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
        best_score = -float("inf")
        best_actions: list[int] = []
        for action, prior in self.priors.items():
            q = self.q_value(action)
            u = c_puct * prior * sqrt_total / (1.0 + self.visit_counts[action])
            score = q + u
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif math.isclose(score, best_score, rel_tol=1e-9, abs_tol=1e-12):
                best_actions.append(action)
        return int(random.choice(best_actions))

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


def random_argmax(values: np.ndarray) -> int:
    flat = np.asarray(values).reshape(-1)
    max_value = np.max(flat)
    candidates = np.flatnonzero(np.isclose(flat, max_value))
    return int(np.random.choice(candidates))


def sample_from_visits(visits: np.ndarray, temperature: float) -> tuple[int, np.ndarray]:
    flat = visits.reshape(-1).astype(np.float64)
    if np.all(flat == 0):
        flat = np.ones_like(flat)

    if temperature <= 1e-6:
        probs = np.zeros_like(flat, dtype=np.float64)
        probs[random_argmax(flat)] = 1.0
    else:
        adjusted = np.power(flat, 1.0 / temperature)
        probs = adjusted / np.sum(adjusted)

    action = int(np.random.choice(len(probs), p=probs))
    return action, probs.reshape(visits.shape).astype(np.float32)


def choose_mcts_action(
    policy: PolicyValueNet,
    board: np.ndarray,
    current_player: int,
    win_length: int,
    device: torch.device,
    num_simulations: int,
    c_puct: float,
    temperature: float,
    add_root_noise: bool,
    dirichlet_alpha: float,
    noise_eps: float,
) -> tuple[int, np.ndarray]:
    root = MCTSNode(board=board.copy(), current_player=current_player, win_length=win_length)
    priors, _ = evaluate_policy_value(policy, root.board, root.current_player, device)
    root.expand(
        priors,
        add_noise=add_root_noise,
        dirichlet_alpha=dirichlet_alpha,
        noise_eps=noise_eps,
    )

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

    visits = np.zeros(board.shape, dtype=np.float32)
    for action, count in root.visit_counts.items():
        row, col = action_to_coords(action, board.shape[0])
        visits[row, col] = float(count)

    action, visit_probs = sample_from_visits(visits, temperature=temperature)
    return action, visit_probs


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
    if agent == "policy":
        priors, _ = evaluate_policy_value(policy, board, current_player, device)
        return int(np.argmax(priors)), None
    return choose_mcts_action(
        policy=policy,
        board=board,
        current_player=current_player,
        win_length=win_length,
        device=device,
        num_simulations=mcts_sims,
        c_puct=c_puct,
        temperature=1e-6,
        add_root_noise=False,
        dirichlet_alpha=0.3,
        noise_eps=0.25,
    )


def immediate_winning_actions(board: np.ndarray, player: int, win_length: int) -> list[int]:
    winning_actions: list[int] = []
    for action in np.flatnonzero((board == 0).reshape(-1)):
        row, col = action_to_coords(int(action), board.shape[0])
        next_board = board.copy()
        next_board[row, col] = player
        if is_winning_move(next_board, row, col, player, win_length):
            winning_actions.append(int(action))
    return winning_actions


def heuristic_candidate_actions(board: np.ndarray, radius: int = 2) -> list[int]:
    board_size = board.shape[0]
    occupied = np.argwhere(board != 0)
    if len(occupied) == 0:
        center = board_size // 2
        return [coords_to_action(center, center, board_size)]

    candidates: set[int] = set()
    for row, col in occupied:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = int(row) + dr
                cc = int(col) + dc
                if 0 <= rr < board_size and 0 <= cc < board_size and board[rr, cc] == 0:
                    candidates.add(coords_to_action(rr, cc, board_size))
    if candidates:
        return sorted(candidates)
    return [int(action) for action in np.flatnonzero((board == 0).reshape(-1))]


def heuristic_pattern_score(length: int, open_ends: int, win_length: int) -> float:
    if length >= win_length:
        return 1_000_000.0
    if length == win_length - 1:
        return 100_000.0 if open_ends == 2 else 20_000.0
    if length == win_length - 2:
        return 8_000.0 if open_ends == 2 else 1_500.0
    if length == win_length - 3:
        return 400.0 if open_ends == 2 else 80.0
    return float(length * length)


def score_heuristic_move(
    board: np.ndarray,
    action: int,
    player: int,
    win_length: int,
) -> float:
    board_size = board.shape[0]
    row, col = action_to_coords(action, board_size)
    trial_board = board.copy()
    trial_board[row, col] = player

    score = 0.0
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        forward_count, forward_open = count_one_side_with_open_end(trial_board, row, col, dr, dc, player)
        backward_count, backward_open = count_one_side_with_open_end(
            trial_board, row, col, -dr, -dc, player
        )
        length = 1 + forward_count + backward_count
        open_ends = int(forward_open) + int(backward_open)
        score += heuristic_pattern_score(length, open_ends, win_length)

    center = (board_size - 1) / 2.0
    center_bias = board_size - (abs(row - center) + abs(col - center))
    r0 = max(0, row - 1)
    r1 = min(board_size, row + 2)
    c0 = max(0, col - 1)
    c1 = min(board_size, col + 2)
    neighborhood = board[r0:r1, c0:c1]
    neighbor_bias = float(np.count_nonzero(neighborhood)) * 3.0

    own_next_wins = len(immediate_winning_actions(trial_board, player, win_length))
    opp_next_wins = len(immediate_winning_actions(trial_board, -player, win_length))
    if own_next_wins >= 2:
        score += 700_000.0
    elif own_next_wins == 1:
        score += 90_000.0
    if opp_next_wins >= 2:
        score -= 800_000.0
    elif opp_next_wins == 1:
        score -= 120_000.0

    return score + center_bias + neighbor_bias


def choose_heuristic_action(
    board: np.ndarray,
    current_player: int,
    win_length: int,
) -> int:
    candidate_actions = heuristic_candidate_actions(board)
    if not candidate_actions:
        raise ValueError("no legal actions available")

    own_wins = immediate_winning_actions(board, current_player, win_length)
    if own_wins:
        return own_wins[0]

    opp_wins = set(immediate_winning_actions(board, -current_player, win_length))
    best_action = candidate_actions[0]
    best_score = -float("inf")
    for action in candidate_actions:
        own_score = score_heuristic_move(board, action, current_player, win_length)
        opp_score = score_heuristic_move(board, action, -current_player, win_length)
        block_bonus = 500_000.0 if action in opp_wins else 0.0
        score = own_score + 1.1 * opp_score + block_bonus
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def self_play_game(
    policy: PolicyValueNet,
    board_size: int,
    win_length: int,
    device: torch.device,
    mcts_sims: int,
    c_puct: float,
    temperature: float,
    temperature_drop_moves: int,
    dirichlet_alpha: float,
    noise_eps: float,
    random_opening_moves: int,
) -> tuple[list[tuple[torch.Tensor, np.ndarray, float]], int, int]:
    env = GomokuEnv(board_size=board_size, win_length=win_length)
    env.reset()
    history: list[tuple[torch.Tensor, np.ndarray, int]] = []

    move_idx = 0
    opening_moves = random.randint(0, max(random_opening_moves, 0))
    for _ in range(opening_moves):
        if env.done:
            break
        env.step(int(np.random.choice(env.valid_moves())))
        move_idx += 1

    while not env.done:
        move_temp = temperature if move_idx < temperature_drop_moves else 1e-6
        action, visit_probs = choose_mcts_action(
            policy=policy,
            board=env.board,
            current_player=env.current_player,
            win_length=win_length,
            device=device,
            num_simulations=mcts_sims,
            c_puct=c_puct,
            temperature=move_temp,
            add_root_noise=True,
            dirichlet_alpha=dirichlet_alpha,
            noise_eps=noise_eps,
        )
        history.append((encode_state(env.board, env.current_player), visit_probs.reshape(-1), env.current_player))
        env.step(action)
        move_idx += 1

    examples: list[tuple[torch.Tensor, np.ndarray, float]] = []
    for state, visit_probs, player in history:
        if env.winner == 0:
            outcome = 0.0
        else:
            outcome = 1.0 if player == env.winner else -1.0
        examples.append((state, visit_probs, outcome))
    return examples, env.winner, move_idx


def train_batch(
    policy: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    batch: list[tuple[torch.Tensor, np.ndarray, float]],
    device: torch.device,
    value_coef: float,
) -> tuple[float, float, float]:
    augmented_states: list[torch.Tensor] = []
    augmented_policies: list[np.ndarray] = []
    for state, target_policy, _ in batch:
        symmetry = random.randrange(8)
        board_size = state.shape[-1]
        augmented_states.append(apply_state_symmetry(state, symmetry))
        augmented_policies.append(apply_policy_symmetry(target_policy, board_size, symmetry))

    states = torch.stack(augmented_states).to(device)
    target_policies = torch.tensor(
        np.stack(augmented_policies),
        dtype=torch.float32,
        device=device,
    )
    target_values = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=device)

    logits, values = policy(states)
    log_probs = torch.log_softmax(logits, dim=1)
    policy_loss = -(target_policies * log_probs).sum(dim=1).mean()
    value_loss = torch.mean((values - target_values) ** 2)
    loss = policy_loss + value_coef * value_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return float(loss.item()), float(policy_loss.item()), float(value_loss.item())


def save_checkpoint(path: Path, policy: PolicyValueNet, args: argparse.Namespace) -> None:
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "channels": args.channels,
            "board_size": args.board_size,
            "win_length": args.win_length,
        },
        path,
    )


def last_checkpoint_path(base_path: Path) -> Path:
    return base_path.with_name(f"{base_path.stem}_last{base_path.suffix}")


def load_checkpoint(path: Path, map_location: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint
    raise RuntimeError(f"{path} is not a compatible gomoku_mcts checkpoint")


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


def choose_opponent_action(board: np.ndarray, current_player: int, win_length: int, opponent: str) -> int:
    if opponent == "random":
        return int(np.random.choice(np.flatnonzero((board == 0).reshape(-1))))
    if opponent == "heuristic":
        return choose_heuristic_action(board=board, current_player=current_player, win_length=win_length)
    raise ValueError(f"unsupported opponent: {opponent}")


def play_vs_opponent_once(
    policy: PolicyValueNet,
    board_size: int,
    win_length: int,
    device: torch.device,
    policy_player: int,
    opponent: str,
    agent: str,
    mcts_sims: int,
    c_puct: float,
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
            action = choose_opponent_action(
                board=env.board,
                current_player=env.current_player,
                win_length=win_length,
                opponent=opponent,
            )
        env.step(action)
    return env.winner


def evaluate_vs_opponent(
    policy: PolicyValueNet,
    board_size: int,
    win_length: int,
    device: torch.device,
    games: int,
    opponent: str,
    agent: str,
    mcts_sims: int,
    c_puct: float,
) -> tuple[float, int, int, int]:
    wins = 0
    draws = 0
    losses = 0
    for game_idx in range(games):
        policy_player = 1 if game_idx < games // 2 else -1
        winner = play_vs_opponent_once(
            policy=policy,
            board_size=board_size,
            win_length=win_length,
            device=device,
            policy_player=policy_player,
            opponent=opponent,
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


def evaluate_self_play_trace(
    policy: PolicyValueNet,
    board_size: int,
    win_length: int,
    device: torch.device,
    mcts_sims: int,
    c_puct: float,
) -> tuple[int, list[str], str]:
    env = GomokuEnv(board_size=board_size, win_length=win_length)
    env.reset()
    moves: list[str] = []

    while not env.done:
        player = env.current_player
        action, _ = choose_mcts_action(
            policy=policy,
            board=env.board,
            current_player=env.current_player,
            win_length=win_length,
            device=device,
            num_simulations=mcts_sims,
            c_puct=c_puct,
            temperature=1e-6,
            add_root_noise=False,
            dirichlet_alpha=0.3,
            noise_eps=0.25,
        )
        row, col = action_to_coords(action, env.board_size)
        stone = "X" if player == 1 else "O"
        moves.append(f"{len(moves) + 1}:{stone}({row + 1},{col + 1})")
        env.step(action)

    return env.winner, moves, env.render()


def winner_to_text(winner: int) -> str:
    if winner == 1:
        return "X"
    if winner == -1:
        return "O"
    return "draw"


def format_trace_moves(moves: list[str], max_moves: int) -> str:
    if max_moves <= 0 or len(moves) <= max_moves:
        return " ".join(moves)
    head = " ".join(moves[:max_moves])
    return f"{head} ... total_moves={len(moves)}"


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = choose_device(args.device)
    policy = PolicyValueNet(channels=args.channels).to(device)
    if args.init_checkpoint is not None and args.init_checkpoint.exists():
        checkpoint = load_checkpoint(args.init_checkpoint, map_location=device)
        policy.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    replay_buffer: deque[tuple[torch.Tensor, np.ndarray, float]] = deque(maxlen=args.buffer_size)

    print(f"device={device} board={args.board_size} win={args.win_length}")

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
                    recent_loss, recent_policy_loss, recent_value_loss = losses[-1]
                    print(
                        f"iter={iteration:5d} train={step_idx + 1:3d}/{args.train_steps:3d} "
                        f"loss={recent_loss:7.4f} policy={recent_policy_loss:7.4f} "
                        f"value={recent_value_loss:7.4f} elapsed={elapsed:7.1f}s"
                    )

        avg_loss = float(np.mean([x[0] for x in losses])) if losses else 0.0
        avg_policy_loss = float(np.mean([x[1] for x in losses])) if losses else 0.0
        avg_value_loss = float(np.mean([x[2] for x in losses])) if losses else 0.0
        p1_wins = sum(1 for x in winners if x == 1)
        p2_wins = sum(1 for x in winners if x == -1)
        draws = sum(1 for x in winners if x == 0)
        avg_len = float(np.mean(lengths)) if lengths else 0.0

        message = (
            f"iter={iteration:5d} loss={avg_loss:7.4f} policy={avg_policy_loss:7.4f} "
            f"value={avg_value_loss:7.4f} p1={p1_wins:3d} p2={p2_wins:3d} draw={draws:3d} "
            f"avg_len={avg_len:6.2f} buffer={len(replay_buffer):6d}"
        )
        if args.eval_every > 0 and iteration % args.eval_every == 0:
            policy.eval()
            win_rate, wins, eval_draws, eval_losses = evaluate_vs_opponent(
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
            message += f" random_win_rate={win_rate:.3f} ({wins}/{eval_draws}/{eval_losses})"
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
            checkpoint_path = last_checkpoint_path(args.checkpoint)
            save_checkpoint(checkpoint_path, policy, args)
            print(f"saved checkpoint to {checkpoint_path}")

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
    print(f"agent={args.agent} opponent={args.opponent} mcts_sims={args.mcts_sims}")
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
        return coords_to_action(row, col, env.board_size)


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
            row, col = action_to_coords(action, env.board_size)
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
    pygame.display.set_caption("Gomoku MCTS")
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
        if x < padding or y < padding:
            return None
        col = (x - padding) // cell_size
        row = (y - padding) // cell_size
        if not (0 <= row < env.board_size and 0 <= col < env.board_size):
            return None
        if env.board[row, col] != 0:
            return None
        return coords_to_action(row, col, env.board_size)

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

    def restart() -> None:
        nonlocal last_search_visits
        env.reset()
        last_search_visits = None
        if env.current_player != human_player:
            ai_step()

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
            y = padding + idx * cell_size
            pygame.draw.line(screen, line_color, (x, padding), (x, padding + board_pixels), 2)
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
            screen.blit(label, (padding + idx * cell_size + cell_size // 2 - label.get_width() // 2, 8))
            screen.blit(label, (8, padding + idx * cell_size + cell_size // 2 - label.get_height() // 2))

        info = (
            f"{board_size}x{board_size} connect={win_length} device={device} "
            f"agent={args.agent} sims={args.mcts_sims}"
        )
        screen.blit(small_font.render(info, True, line_color), (padding, padding + board_pixels + 16))
        screen.blit(font.render(status_text(), True, accent), (padding, padding + board_pixels + 42))

        if last_search_visits is not None and args.agent == "mcts":
            peak = int(np.max(last_search_visits))
            screen.blit(
                small_font.render(f"peak_visits={peak}", True, line_color),
                (padding + 420, padding + board_pixels + 16),
            )

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Gomoku MCTS example")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common_arguments(subparser: argparse.ArgumentParser, defaults_from_checkpoint: bool = False) -> None:
        board_default = None if defaults_from_checkpoint else 15
        win_default = None if defaults_from_checkpoint else 5
        subparser.add_argument("--board-size", type=int, default=board_default)
        subparser.add_argument("--win-length", type=int, default=win_default)
        subparser.add_argument("--channels", type=int, default=64)
        subparser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
        subparser.add_argument("--checkpoint", type=Path, default=Path("gomoku_mcts.pt"))

    def add_inference_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--agent", choices=["policy", "mcts"], default="mcts")
        subparser.add_argument("--mcts-sims", type=int, default=120)
        subparser.add_argument("--c-puct", type=float, default=1.5)

    train_parser = subparsers.add_parser("train", help="MCTS self-play training")
    add_common_arguments(train_parser)
    train_parser.add_argument("--iterations", type=int, default=200)
    train_parser.add_argument("--games-per-iter", type=int, default=8)
    train_parser.add_argument("--train-steps", type=int, default=32)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--buffer-size", type=int, default=20000)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--value-coef", type=float, default=1.0)
    train_parser.add_argument("--mcts-sims", type=int, default=64)
    train_parser.add_argument("--eval-mcts-sims", type=int, default=120)
    train_parser.add_argument("--c-puct", type=float, default=1.5)
    train_parser.add_argument("--temperature", type=float, default=1.0)
    train_parser.add_argument("--temperature-drop-moves", type=int, default=8)
    train_parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    train_parser.add_argument("--noise-eps", type=float, default=0.25)
    train_parser.add_argument("--random-opening-moves", type=int, default=0)
    train_parser.add_argument("--eval-every", type=int, default=10)
    train_parser.add_argument("--eval-games", type=int, default=20)
    train_parser.add_argument("--eval-heuristic-games", type=int, default=0)
    train_parser.add_argument("--eval-trace-games", type=int, default=0)
    train_parser.add_argument("--eval-trace-max-moves", type=int, default=20)
    train_parser.add_argument("--log-every-games", type=int, default=0)
    train_parser.add_argument("--log-every-train-steps", type=int, default=0)
    train_parser.add_argument("--save-every", type=int, default=10)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--init-checkpoint", type=Path, default=None)
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval", help="evaluate against a baseline opponent")
    add_common_arguments(eval_parser)
    add_inference_arguments(eval_parser)
    eval_parser.add_argument("--opponent", choices=["random", "heuristic"], default="random")
    eval_parser.add_argument("--games", type=int, default=40)
    eval_parser.set_defaults(func=evaluate)

    play_parser = subparsers.add_parser("play", help="play against the model")
    add_common_arguments(play_parser, defaults_from_checkpoint=True)
    add_inference_arguments(play_parser)
    play_parser.add_argument("--human-first", action="store_true")
    play_parser.set_defaults(func=play)

    gui_parser = subparsers.add_parser("gui", help="pygame GUI")
    add_common_arguments(gui_parser, defaults_from_checkpoint=True)
    add_inference_arguments(gui_parser)
    gui_parser.add_argument("--human-first", action="store_true")
    gui_parser.add_argument("--cell-size", type=int, default=48)
    gui_parser.add_argument("--fps", type=int, default=30)
    gui_parser.set_defaults(func=gui)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
