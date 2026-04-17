#!/usr/bin/env python3
"""Pure AlphaZero-style Gomoku training with curriculum support."""

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


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def parameter_size_mib(param_count: int) -> float:
    return (param_count * 4) / (1024 * 1024)


def load_matching_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    model_state = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            matched[key] = value
    model_state.update(matched)
    model.load_state_dict(model_state)
    skipped = len(model_state) - len(matched)
    return len(matched), skipped


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
        return board_to_string(self.board)


def board_to_string(board: np.ndarray) -> str:
    board_size = board.shape[0]
    symbols = {1: "X", -1: "O", 0: "."}
    header = "   " + " ".join(f"{i + 1:2d}" for i in range(board_size))
    rows = [header]
    for row_idx in range(board_size):
        row = " ".join(f"{symbols[int(v)]:>2}" for v in board[row_idx])
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


def opening_region_sets(board_size: int) -> tuple[set[int], set[int]]:
    center = (board_size - 1) / 2.0
    outer_threshold = max(2.5, board_size * 0.32)
    center_threshold = max(1.5, board_size * 0.18)
    outer_actions: set[int] = set()
    non_center_actions: set[int] = set()
    for row in range(board_size):
        for col in range(board_size):
            dist = abs(row - center) + abs(col - center)
            action = coords_to_action(row, col, board_size)
            if dist >= outer_threshold:
                outer_actions.add(action)
            if dist >= center_threshold:
                non_center_actions.add(action)
    return outer_actions, non_center_actions


def nearby_legal_actions(board: np.ndarray, radius: int) -> list[int]:
    board_size = board.shape[0]
    occupied = np.argwhere(board != 0)
    if len(occupied) == 0:
        return []
    candidates: set[int] = set()
    for row, col in occupied:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = int(row) + dr
                cc = int(col) + dc
                if 0 <= rr < board_size and 0 <= cc < board_size and board[rr, cc] == 0:
                    candidates.add(coords_to_action(rr, cc, board_size))
    return sorted(candidates)


def sample_diverse_opening_action(
    board: np.ndarray,
    outer_ring_prob: float,
    non_center_prob: float,
    neighbor_radius: int,
) -> int:
    legal_actions = [int(action) for action in np.flatnonzero((board == 0).reshape(-1))]
    if not legal_actions:
        raise RuntimeError("no legal opening actions available")

    board_size = board.shape[0]
    outer_actions, non_center_actions = opening_region_sets(board_size)
    nearby_actions = nearby_legal_actions(board, neighbor_radius)

    legal_outer = [action for action in legal_actions if action in outer_actions]
    legal_non_center = [action for action in legal_actions if action in non_center_actions]
    legal_nearby_outer = [action for action in nearby_actions if action in outer_actions]
    legal_nearby_non_center = [action for action in nearby_actions if action in non_center_actions]

    if np.any(board != 0):
        if legal_nearby_outer and random.random() < outer_ring_prob:
            return int(random.choice(legal_nearby_outer))
        if legal_nearby_non_center and random.random() < non_center_prob:
            return int(random.choice(legal_nearby_non_center))
        if nearby_actions:
            return int(random.choice(nearby_actions))

    if legal_outer and random.random() < outer_ring_prob:
        return int(random.choice(legal_outer))
    if legal_non_center and random.random() < non_center_prob:
        return int(random.choice(legal_non_center))
    return int(random.choice(legal_actions))


def encode_state(board: np.ndarray, current_player: int) -> torch.Tensor:
    current = (board == current_player).astype(np.float32)
    opponent = (board == -current_player).astype(np.float32)
    legal = (board == 0).astype(np.float32)
    return torch.from_numpy(np.stack([current, opponent, legal], axis=0))


def apply_state_symmetry(state: np.ndarray, symmetry: int) -> np.ndarray:
    transformed = state
    rotations = symmetry % 4
    if rotations:
        transformed = np.rot90(transformed, k=rotations, axes=(-2, -1))
    if symmetry >= 4:
        transformed = np.flip(transformed, axis=-1)
    return np.ascontiguousarray(transformed, dtype=np.float32)


def apply_policy_symmetry(policy: np.ndarray, board_size: int, symmetry: int) -> np.ndarray:
    transformed = policy.reshape(board_size, board_size)
    rotations = symmetry % 4
    if rotations:
        transformed = np.rot90(transformed, k=rotations)
    if symmetry >= 4:
        transformed = np.flip(transformed, axis=1)
    return np.ascontiguousarray(transformed.reshape(-1), dtype=np.float32)


class PolicyValueNet(nn.Module):
    def __init__(self, channels: int = 64, conv_layers: int = 4):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be > 0")
        if conv_layers < 1:
            raise ValueError("conv_layers must be >= 1")

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
        backward_count, backward_open = count_one_side_with_open_end(trial_board, row, col, -dr, -dc, player)
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


def choose_heuristic_action(board: np.ndarray, current_player: int, win_length: int) -> int:
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
    opening_sampler: str,
    opening_outer_ring_prob: float,
    opening_non_center_prob: float,
    opening_neighbor_radius: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], int, int]:
    env = GomokuEnv(board_size=board_size, win_length=win_length)
    env.reset()
    history: list[tuple[np.ndarray, np.ndarray, int]] = []
    move_idx = 0
    opening_moves = random.randint(0, max(random_opening_moves, 0))
    for _ in range(opening_moves):
        if env.done:
            break
        if opening_sampler == "diverse":
            action = sample_diverse_opening_action(
                env.board,
                outer_ring_prob=opening_outer_ring_prob,
                non_center_prob=opening_non_center_prob,
                neighbor_radius=opening_neighbor_radius,
            )
        else:
            action = int(np.random.choice(env.valid_moves()))
        env.step(action)
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
        history.append(
            (
                encode_state(env.board, env.current_player).numpy().astype(np.float32),
                visit_probs.reshape(-1).astype(np.float32),
                env.current_player,
            )
        )
        env.step(action)
        move_idx += 1

    examples: list[tuple[np.ndarray, np.ndarray, float]] = []
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
    batch: list[tuple[np.ndarray, np.ndarray, float]],
    device: torch.device,
    value_coef: float,
) -> tuple[float, float, float]:
    augmented_states: list[np.ndarray] = []
    augmented_policies: list[np.ndarray] = []
    values: list[float] = []
    for state, target_policy, target_value in batch:
        symmetry = random.randrange(8)
        board_size = state.shape[-1]
        augmented_states.append(apply_state_symmetry(state, symmetry))
        augmented_policies.append(apply_policy_symmetry(target_policy, board_size, symmetry))
        values.append(target_value)

    states = torch.tensor(np.stack(augmented_states), dtype=torch.float32, device=device)
    target_policies = torch.tensor(np.stack(augmented_policies), dtype=torch.float32, device=device)
    target_values = torch.tensor(values, dtype=torch.float32, device=device)

    logits, predicted_values = policy(states)
    log_probs = torch.log_softmax(logits, dim=1)
    policy_loss = -(target_policies * log_probs).sum(dim=1).mean()
    value_loss = torch.mean((predicted_values - target_values) ** 2)
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
            "conv_layers": args.conv_layers,
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
    raise RuntimeError(f"{path} is not a compatible gomoku_alphazero checkpoint")


def build_policy(
    channels: int,
    conv_layers: int,
    device: torch.device,
    checkpoint: dict | None = None,
) -> PolicyValueNet:
    saved_channels = int(checkpoint.get("channels", channels)) if checkpoint is not None else channels
    saved_layers = int(checkpoint.get("conv_layers", conv_layers)) if checkpoint is not None else conv_layers
    policy = PolicyValueNet(channels=saved_channels, conv_layers=saved_layers).to(device)
    if checkpoint is not None:
        policy.load_state_dict(checkpoint["state_dict"])
    return policy


def resolve_game_config(
    checkpoint_path: Path,
    arg_board_size: int | None,
    arg_win_length: int | None,
    arg_channels: int,
    arg_conv_layers: int,
    device: torch.device,
) -> tuple[PolicyValueNet, int, int]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    board_size = int(checkpoint.get("board_size") or arg_board_size or 9)
    win_length = int(checkpoint.get("win_length") or arg_win_length or 5)
    policy = build_policy(arg_channels, arg_conv_layers, device, checkpoint=checkpoint)
    policy.eval()
    return policy, board_size, win_length


def choose_opponent_action(board: np.ndarray, current_player: int, win_length: int, opponent: str) -> int:
    if opponent == "random":
        return int(np.random.choice(np.flatnonzero((board == 0).reshape(-1))))
    if opponent == "heuristic":
        return choose_heuristic_action(board=board, current_player=current_player, win_length=win_length)
    raise ValueError(f"unsupported opponent: {opponent}")


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
            action = choose_opponent_action(env.board, env.current_player, win_length, opponent)
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
    for game_index in range(games):
        policy_player = 1 if game_index < games // 2 else -1
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


def winner_to_text(winner: int) -> str:
    if winner == 1:
        return "X"
    if winner == -1:
        return "O"
    return "draw"


def format_trace_moves(moves: list[str], max_moves: int) -> str:
    if max_moves <= 0 or len(moves) <= max_moves:
        return " ".join(moves)
    return f"{' '.join(moves[:max_moves])} ... total_moves={len(moves)}"


def format_trace_boards(moves: list[str], board_snapshots: list[str], max_moves: int) -> str:
    if len(moves) != len(board_snapshots):
        raise ValueError("moves and board_snapshots must have the same length")
    if max_moves <= 0 or len(moves) <= max_moves:
        limit = len(moves)
        suffix = ""
    else:
        limit = max_moves
        suffix = f"\n... total_moves={len(moves)}"
    parts = []
    for move, board in zip(moves[:limit], board_snapshots[:limit], strict=False):
        parts.append(f"after {move}\n{board}")
    return "\n\n".join(parts) + suffix


def evaluate_self_play_trace(
    policy: PolicyValueNet,
    board_size: int,
    win_length: int,
    device: torch.device,
    mcts_sims: int,
    c_puct: float,
) -> tuple[int, list[str], list[str], str]:
    env = GomokuEnv(board_size=board_size, win_length=win_length)
    env.reset()
    moves: list[str] = []
    board_snapshots: list[str] = []
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
        row, col = action_to_coords(action, board_size)
        stone = "X" if player == 1 else "O"
        moves.append(f"{len(moves) + 1}:{stone}({row + 1},{col + 1})")
        env.step(action)
        board_snapshots.append(env.render())
    return env.winner, moves, board_snapshots, env.render()


def evaluate_heuristic_trace(
    policy: PolicyValueNet,
    board_size: int,
    win_length: int,
    device: torch.device,
    policy_player: int,
    agent: str,
    mcts_sims: int,
    c_puct: float,
) -> tuple[int, list[str], list[str], str]:
    env = GomokuEnv(board_size=board_size, win_length=win_length)
    env.reset()
    moves: list[str] = []
    board_snapshots: list[str] = []
    while not env.done:
        player = env.current_player
        if player == policy_player:
            action, _ = choose_ai_action(
                policy=policy,
                board=env.board,
                current_player=player,
                win_length=win_length,
                device=device,
                agent=agent,
                mcts_sims=mcts_sims,
                c_puct=c_puct,
            )
        else:
            action = choose_heuristic_action(env.board, env.current_player, win_length)
        row, col = action_to_coords(action, board_size)
        stone = "X" if player == 1 else "O"
        side = "policy" if player == policy_player else "heuristic"
        moves.append(f"{len(moves) + 1}:{side}:{stone}({row + 1},{col + 1})")
        env.step(action)
        board_snapshots.append(env.render())
    return env.winner, moves, board_snapshots, env.render()


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = choose_device(args.device)
    checkpoint = None
    if args.init_checkpoint is not None and args.init_checkpoint.exists():
        checkpoint = load_checkpoint(args.init_checkpoint, map_location=device)
    policy = PolicyValueNet(channels=args.channels, conv_layers=args.conv_layers).to(device)
    if checkpoint is not None:
        matched, skipped = load_matching_state_dict(policy, checkpoint["state_dict"])
        print(
            f"loaded init checkpoint from {args.init_checkpoint} "
            f"matched={matched} skipped={skipped}"
        )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    replay_buffer: deque[tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=args.buffer_size)
    early_stop_hits = 0
    heuristic_stop_hits = 0
    params = count_parameters(policy)
    print(
        f"device={device} board={args.board_size} win={args.win_length} "
        f"params={params} approx_mib={parameter_size_mib(params):.2f}"
    )

    for iteration in range(1, args.iterations + 1):
        policy.eval()
        iteration_start = time.perf_counter()
        winners: list[int] = []
        lengths: list[int] = []
        for _ in range(args.games_per_iter):
            game_examples, winner, moves = self_play_game(
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
                opening_sampler=args.opening_sampler,
                opening_outer_ring_prob=args.opening_outer_ring_prob,
                opening_non_center_prob=args.opening_non_center_prob,
                opening_neighbor_radius=args.opening_neighbor_radius,
            )
            replay_buffer.extend(game_examples)
            winners.append(winner)
            lengths.append(moves)
            games_done = len(winners)
            if args.log_every_games > 0 and games_done % args.log_every_games == 0:
                avg_len = float(np.mean(lengths)) if lengths else 0.0
                elapsed = time.perf_counter() - iteration_start
                print(
                    f"iter={iteration:5d} selfplay={games_done:3d}/{args.games_per_iter:3d} "
                    f"avg_len={avg_len:6.2f} buffer={len(replay_buffer):6d} elapsed={elapsed:7.1f}s"
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
                    recent_loss, recent_policy, recent_value = losses[-1]
                    print(
                        f"iter={iteration:5d} train={step_idx + 1:3d}/{args.train_steps:3d} "
                        f"loss={recent_loss:7.4f} policy={recent_policy:7.4f} "
                        f"value={recent_value:7.4f} elapsed={elapsed:7.1f}s"
                    )

        avg_loss = float(np.mean([item[0] for item in losses])) if losses else 0.0
        avg_policy_loss = float(np.mean([item[1] for item in losses])) if losses else 0.0
        avg_value_loss = float(np.mean([item[2] for item in losses])) if losses else 0.0
        p1_wins = sum(1 for item in winners if item == 1)
        p2_wins = sum(1 for item in winners if item == -1)
        draws = sum(1 for item in winners if item == 0)
        avg_len = float(np.mean(lengths)) if lengths else 0.0
        message = (
            f"iter={iteration:5d} loss={avg_loss:7.4f} policy={avg_policy_loss:7.4f} "
            f"value={avg_value_loss:7.4f} p1={p1_wins:3d} p2={p2_wins:3d} draw={draws:3d} "
            f"avg_len={avg_len:6.2f} buffer={len(replay_buffer):6d}"
        )

        if args.eval_every > 0 and iteration % args.eval_every == 0:
            policy.eval()
            random_win_rate, rwins, rdraws, rlosses = evaluate_vs_opponent(
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
            message += f" random_win_rate={random_win_rate:.3f} ({rwins}/{rdraws}/{rlosses})"
            if args.eval_heuristic_games > 0:
                heuristic_win_rate, hwins, hdraws, hlosses = evaluate_vs_opponent(
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
                message += f" heuristic_win_rate={heuristic_win_rate:.3f} ({hwins}/{hdraws}/{hlosses})"
                if (
                    args.early_stop_heuristic_win_rate > 0.0
                    and iteration >= args.early_stop_min_iterations
                    and heuristic_win_rate >= args.early_stop_heuristic_win_rate
                ):
                    heuristic_stop_hits += 1
                    message += (
                        f" heuristic_stop_hits={heuristic_stop_hits}/{args.early_stop_heuristic_patience}"
                    )
                elif args.early_stop_heuristic_win_rate > 0.0:
                    heuristic_stop_hits = 0
        print(message)

        if args.eval_every > 0 and iteration % args.eval_every == 0:
            for trace_index in range(args.eval_trace_games):
                winner, trace_moves, board_snapshots, final_board = evaluate_self_play_trace(
                    policy=policy,
                    board_size=args.board_size,
                    win_length=args.win_length,
                    device=device,
                    mcts_sims=args.eval_mcts_sims,
                    c_puct=args.c_puct,
                )
                print(
                    f"eval_trace game={trace_index + 1} winner={winner_to_text(winner)} "
                    f"moves={len(trace_moves)}"
                )
                print(f"eval_trace seq {format_trace_moves(trace_moves, args.eval_trace_max_moves)}")
                if args.eval_trace_print_boards:
                    print("eval_trace boards:")
                    print(format_trace_boards(trace_moves, board_snapshots, args.eval_trace_max_moves))
                print("eval_trace board:")
                print(final_board)
            for trace_index in range(args.eval_heuristic_trace_games):
                policy_player = 1 if trace_index % 2 == 0 else -1
                winner, trace_moves, board_snapshots, final_board = evaluate_heuristic_trace(
                    policy=policy,
                    board_size=args.board_size,
                    win_length=args.win_length,
                    device=device,
                    policy_player=policy_player,
                    agent="mcts",
                    mcts_sims=args.eval_mcts_sims,
                    c_puct=args.c_puct,
                )
                role = "X" if policy_player == 1 else "O"
                print(
                    f"heuristic_trace game={trace_index + 1} policy_as={role} "
                    f"winner={winner_to_text(winner)} moves={len(trace_moves)}"
                )
                print(f"heuristic_trace seq {format_trace_moves(trace_moves, args.eval_trace_max_moves)}")
                if args.eval_trace_print_boards:
                    print("heuristic_trace boards:")
                    print(format_trace_boards(trace_moves, board_snapshots, args.eval_trace_max_moves))
                print("heuristic_trace board:")
                print(final_board)

        if args.save_every > 0 and iteration % args.save_every == 0:
            checkpoint_path = last_checkpoint_path(args.checkpoint)
            save_checkpoint(checkpoint_path, policy, args)
            print(f"saved checkpoint to {checkpoint_path}")

        if (
            args.early_stop_loss > 0.0
            and iteration >= args.early_stop_min_iterations
            and losses
            and avg_loss <= args.early_stop_loss
        ):
            early_stop_hits += 1
            print(
                f"early_stop progress hits={early_stop_hits}/{args.early_stop_patience} "
                f"threshold={args.early_stop_loss:.4f}"
            )
        else:
            early_stop_hits = 0

        if (
            args.early_stop_loss > 0.0
            and args.early_stop_patience > 0
            and early_stop_hits >= args.early_stop_patience
        ):
            save_checkpoint(args.checkpoint, policy, args)
            print(
                f"early_stop triggered at iter={iteration} "
                f"avg_loss={avg_loss:.4f} threshold={args.early_stop_loss:.4f}"
            )
            print(f"saved checkpoint to {args.checkpoint}")
            return

        if (
            args.early_stop_heuristic_win_rate > 0.0
            and args.early_stop_heuristic_patience > 0
            and heuristic_stop_hits >= args.early_stop_heuristic_patience
        ):
            save_checkpoint(args.checkpoint, policy, args)
            print(
                f"early_stop heuristic triggered at iter={iteration} "
                f"threshold={args.early_stop_heuristic_win_rate:.3f}"
            )
            print(f"saved checkpoint to {args.checkpoint}")
            return

    save_checkpoint(args.checkpoint, policy, args)
    print(f"saved checkpoint to {args.checkpoint}")


def evaluate(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    policy, board_size, win_length = resolve_game_config(
        checkpoint_path=args.checkpoint,
        arg_board_size=args.board_size,
        arg_win_length=args.win_length,
        arg_channels=args.channels,
        arg_conv_layers=args.conv_layers,
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
        f"win_rate={win_rate:.3f} wins={wins} draws={draws} losses={losses}"
    )
    if args.trace_games > 0 and args.opponent == "heuristic":
        for trace_index in range(args.trace_games):
            policy_player = 1 if trace_index % 2 == 0 else -1
            winner, trace_moves, board_snapshots, final_board = evaluate_heuristic_trace(
                policy=policy,
                board_size=board_size,
                win_length=win_length,
                device=device,
                policy_player=policy_player,
                agent=args.agent,
                mcts_sims=args.mcts_sims,
                c_puct=args.c_puct,
            )
            print(
                f"heuristic_trace game={trace_index + 1} policy_as={'X' if policy_player == 1 else 'O'} "
                f"winner={winner_to_text(winner)} moves={len(trace_moves)}"
            )
            print(f"heuristic_trace seq {format_trace_moves(trace_moves, args.trace_max_moves)}")
            if args.trace_print_boards:
                print("heuristic_trace boards:")
                print(format_trace_boards(trace_moves, board_snapshots, args.trace_max_moves))
            print("heuristic_trace board:")
            print(final_board)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pure AlphaZero Gomoku")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common_arguments(subparser: argparse.ArgumentParser, defaults_from_checkpoint: bool = False) -> None:
        board_default = None if defaults_from_checkpoint else 9
        win_default = None if defaults_from_checkpoint else 5
        subparser.add_argument("--board-size", type=int, default=board_default)
        subparser.add_argument("--win-length", type=int, default=win_default)
        subparser.add_argument("--channels", type=int, default=128)
        subparser.add_argument("--conv-layers", type=int, default=8)
        subparser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
        subparser.add_argument("--checkpoint", type=Path, default=Path("gomoku_alphazero.pt"))

    train_parser = subparsers.add_parser("train", help="self-play training")
    add_common_arguments(train_parser)
    train_parser.add_argument("--iterations", type=int, default=2000)
    train_parser.add_argument("--games-per-iter", type=int, default=32)
    train_parser.add_argument("--train-steps", type=int, default=64)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--buffer-size", type=int, default=100000)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--value-coef", type=float, default=1.0)
    train_parser.add_argument("--mcts-sims", type=int, default=160)
    train_parser.add_argument("--eval-mcts-sims", type=int, default=320)
    train_parser.add_argument("--c-puct", type=float, default=1.5)
    train_parser.add_argument("--temperature", type=float, default=0.7)
    train_parser.add_argument("--temperature-drop-moves", type=int, default=4)
    train_parser.add_argument("--dirichlet-alpha", type=float, default=0.08)
    train_parser.add_argument("--noise-eps", type=float, default=0.10)
    train_parser.add_argument("--random-opening-moves", type=int, default=1)
    train_parser.add_argument("--opening-sampler", choices=["uniform", "diverse"], default="uniform")
    train_parser.add_argument("--opening-outer-ring-prob", type=float, default=0.35)
    train_parser.add_argument("--opening-non-center-prob", type=float, default=0.70)
    train_parser.add_argument("--opening-neighbor-radius", type=int, default=2)
    train_parser.add_argument("--eval-every", type=int, default=10)
    train_parser.add_argument("--eval-games", type=int, default=20)
    train_parser.add_argument("--eval-heuristic-games", type=int, default=8)
    train_parser.add_argument("--eval-trace-games", type=int, default=1)
    train_parser.add_argument("--eval-heuristic-trace-games", type=int, default=1)
    train_parser.add_argument("--eval-trace-max-moves", type=int, default=20)
    train_parser.add_argument("--eval-trace-print-boards", action="store_true")
    train_parser.add_argument("--save-every", type=int, default=10)
    train_parser.add_argument("--log-every-games", type=int, default=4)
    train_parser.add_argument("--log-every-train-steps", type=int, default=16)
    train_parser.add_argument("--early-stop-loss", type=float, default=0.0)
    train_parser.add_argument("--early-stop-patience", type=int, default=0)
    train_parser.add_argument("--early-stop-min-iterations", type=int, default=0)
    train_parser.add_argument("--early-stop-heuristic-win-rate", type=float, default=0.0)
    train_parser.add_argument("--early-stop-heuristic-patience", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--init-checkpoint", type=Path, default=None)
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval", help="evaluate against heuristic or random")
    add_common_arguments(eval_parser, defaults_from_checkpoint=True)
    eval_parser.add_argument("--games", type=int, default=40)
    eval_parser.add_argument("--opponent", choices=["random", "heuristic"], default="heuristic")
    eval_parser.add_argument("--agent", choices=["policy", "mcts"], default="mcts")
    eval_parser.add_argument("--mcts-sims", type=int, default=320)
    eval_parser.add_argument("--c-puct", type=float, default=1.5)
    eval_parser.add_argument("--trace-games", type=int, default=1)
    eval_parser.add_argument("--trace-max-moves", type=int, default=20)
    eval_parser.add_argument("--trace-print-boards", action="store_true")
    eval_parser.set_defaults(func=evaluate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
