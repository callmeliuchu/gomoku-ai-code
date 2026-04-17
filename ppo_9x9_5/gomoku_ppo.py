#!/usr/bin/env python3
"""Standalone Gomoku PPO training with curriculum-friendly defaults."""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
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


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def parameter_size_mib(param_count: int) -> float:
    return (param_count * 4) / (1024 * 1024)


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
        if is_winning_move(self.board, row, col, player, self.win_length):
            self.done = True
            self.winner = player
        elif not np.any(self.board == 0):
            self.done = True
            self.winner = 0
        else:
            self.current_player = -player
        return self.done, self.winner

    def render(self) -> str:
        return board_to_string(self.board)


def encode_state(board: np.ndarray, current_player: int) -> torch.Tensor:
    current = (board == current_player).astype(np.float32)
    opponent = (board == -current_player).astype(np.float32)
    legal = (board == 0).astype(np.float32)
    stacked = np.stack([current, opponent, legal], axis=0)
    return torch.from_numpy(stacked)


def transform_board(board: np.ndarray, rotation_k: int, flip: bool) -> np.ndarray:
    transformed = np.rot90(board, k=rotation_k)
    if flip:
        transformed = np.fliplr(transformed)
    return np.ascontiguousarray(transformed)


def action_to_coords(action: int, board_size: int) -> tuple[int, int]:
    return divmod(int(action), board_size)


def coords_to_action(row: int, col: int, board_size: int) -> int:
    return row * board_size + col


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


def random_argmax(values: np.ndarray) -> int:
    flat = np.asarray(values).reshape(-1)
    max_value = np.max(flat)
    candidates = np.flatnonzero(np.isclose(flat, max_value))
    return int(np.random.choice(candidates))


def board_to_string(board: np.ndarray) -> str:
    board_size = board.shape[0]
    symbols = {1: "X", -1: "O", 0: "."}
    header = "   " + " ".join(f"{i + 1:2d}" for i in range(board_size))
    rows = [header]
    for row_idx in range(board_size):
        row = " ".join(f"{symbols[int(v)]:>2}" for v in board[row_idx])
        rows.append(f"{row_idx + 1:2d} {row}")
    return "\n".join(rows)


def format_move_sequence(moves: list[tuple[int, int, int]], max_moves: int) -> str:
    rendered: list[str] = []
    for move_index, (player, row, col) in enumerate(moves[:max_moves], start=1):
        symbol = "X" if player == 1 else "O"
        rendered.append(f"{move_index}:{symbol}({row + 1},{col + 1})")
    suffix = f" ... total_moves={len(moves)}" if len(moves) > max_moves else ""
    return " ".join(rendered) + suffix


def format_policy_vs_opponent_sequence(
    moves: list[tuple[int, int, int]],
    policy_player: int,
    max_moves: int,
) -> str:
    rendered: list[str] = []
    for move_index, (player, row, col) in enumerate(moves[:max_moves], start=1):
        actor = "policy" if player == policy_player else "heuristic"
        symbol = "X" if player == 1 else "O"
        rendered.append(f"{move_index}:{actor}:{symbol}({row + 1},{col + 1})")
    suffix = f" ... total_moves={len(moves)}" if len(moves) > max_moves else ""
    return " ".join(rendered) + suffix


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


def masked_logits(logits: torch.Tensor, legal_mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(legal_mask, np.ndarray):
        legal = torch.as_tensor(legal_mask.reshape(-1), device=logits.device, dtype=torch.bool)
    else:
        legal = legal_mask.reshape(-1).to(device=logits.device, dtype=torch.bool)
    return logits.masked_fill(~legal, -1e9)


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


def score_heuristic_move(board: np.ndarray, action: int, player: int, win_length: int) -> float:
    board_size = board.shape[0]
    row, col = action_to_coords(action, board_size)
    trial_board = board.copy()
    trial_board[row, col] = player

    score = 0.0
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        forward_count, forward_open = count_one_side_with_open_end(trial_board, row, col, dr, dc, player)
        backward_count, backward_open = count_one_side_with_open_end(
            trial_board,
            row,
            col,
            -dr,
            -dc,
            player,
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


def choose_heuristic_action(board: np.ndarray, current_player: int, win_length: int) -> int:
    candidate_actions = heuristic_candidate_actions(board)
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


class PolicyValueNet(nn.Module):
    def __init__(self, channels: int = 320, conv_layers: int = 12):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be > 0")
        if conv_layers < 2:
            raise ValueError("conv_layers must be >= 2")

        layers: list[nn.Module] = [
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        for _ in range(conv_layers - 1):
            layers.extend(
                [
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                ]
            )
        self.trunk = nn.Sequential(*layers)
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


@dataclass
class RolloutSample:
    state: torch.Tensor
    legal_mask: np.ndarray
    action: int
    old_log_prob: float
    old_value: float
    ret: float
    advantage: float


@dataclass
class EpisodeTrace:
    winner: int
    moves: int
    history: list[tuple[int, int, int]]
    final_board: np.ndarray


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


def policy_step(
    policy: PolicyValueNet,
    board: np.ndarray,
    current_player: int,
    device: torch.device,
    greedy: bool,
    augment: bool,
) -> tuple[int, RolloutSample | None]:
    board_size = board.shape[0]
    rotation_k = random.randint(0, 3) if augment else 0
    flip = bool(random.getrandbits(1)) if augment else False
    transformed_board = transform_board(board, rotation_k=rotation_k, flip=flip)
    legal_mask = transformed_board == 0
    state = encode_state(transformed_board, current_player).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, value = policy(state)
        logits = masked_logits(logits.squeeze(0), legal_mask)

    if greedy:
        transformed_action = random_argmax(logits.detach().cpu().numpy())
        row_t, col_t = action_to_coords(transformed_action, board_size)
        row, col = inverse_transform_coords(row_t, col_t, board_size, rotation_k, flip)
        return coords_to_action(row, col, board_size), None

    dist = Categorical(logits=logits)
    action_tensor = dist.sample()
    transformed_action = int(action_tensor.item())
    row_t, col_t = action_to_coords(transformed_action, board_size)
    row, col = inverse_transform_coords(row_t, col_t, board_size, rotation_k, flip)
    sample = RolloutSample(
        state=encode_state(transformed_board, current_player),
        legal_mask=legal_mask.reshape(-1).astype(np.bool_),
        action=transformed_action,
        old_log_prob=float(dist.log_prob(action_tensor).item()),
        old_value=float(value.item()),
        ret=0.0,
        advantage=0.0,
    )
    return coords_to_action(row, col, board_size), sample


def self_play_episode(
    policy: PolicyValueNet,
    board_size: int,
    win_length: int,
    device: torch.device,
    gamma: float,
    symmetry_augment: bool,
    random_opening_moves: int,
) -> tuple[list[RolloutSample], int, int]:
    env = GomokuEnv(board_size=board_size, win_length=win_length)
    env.reset()
    move_index = 0
    opening_moves = random.randint(0, max(random_opening_moves, 0))
    for _ in range(opening_moves):
        if env.done:
            break
        env.step(int(np.random.choice(env.valid_moves())))
        move_index += 1

    episode_samples: list[tuple[int, RolloutSample]] = []
    while not env.done:
        player = env.current_player
        action, sample = policy_step(
            policy=policy,
            board=env.board,
            current_player=player,
            device=device,
            greedy=False,
            augment=symmetry_augment,
        )
        if sample is None:
            raise RuntimeError("policy_step must return a sample during training")
        episode_samples.append((player, sample))
        env.step(action)
        move_index += 1

    completed: list[RolloutSample] = []
    total_moves = len(episode_samples)
    for sample_index, (player, sample) in enumerate(episode_samples):
        if env.winner == 0:
            outcome = 0.0
        else:
            outcome = 1.0 if player == env.winner else -1.0
        sample.ret = outcome * (gamma ** (total_moves - sample_index - 1))
        sample.advantage = sample.ret - sample.old_value
        completed.append(sample)
    return completed, env.winner, move_index


def ppo_update(
    policy: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    rollout: list[RolloutSample],
    device: torch.device,
    ppo_epochs: int,
    minibatch_size: int,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
) -> tuple[float, float, float, float]:
    states = torch.stack([item.state for item in rollout]).to(device)
    legal_masks = torch.as_tensor(
        np.stack([item.legal_mask for item in rollout]),
        dtype=torch.bool,
        device=device,
    )
    actions = torch.as_tensor([item.action for item in rollout], dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor(
        [item.old_log_prob for item in rollout],
        dtype=torch.float32,
        device=device,
    )
    returns = torch.as_tensor([item.ret for item in rollout], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(
        [item.advantage for item in rollout],
        dtype=torch.float32,
        device=device,
    )
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)

    total_loss = 0.0
    total_policy = 0.0
    total_value = 0.0
    total_entropy = 0.0
    updates = 0
    indices = np.arange(len(rollout))

    for _ in range(ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), minibatch_size):
            batch_indices = indices[start:start + minibatch_size]
            batch_tensor = torch.as_tensor(batch_indices, dtype=torch.long, device=device)
            batch_states = states.index_select(0, batch_tensor)
            batch_masks = legal_masks.index_select(0, batch_tensor)
            batch_actions = actions.index_select(0, batch_tensor)
            batch_old_log_probs = old_log_probs.index_select(0, batch_tensor)
            batch_returns = returns.index_select(0, batch_tensor)
            batch_advantages = advantages.index_select(0, batch_tensor)

            logits, values = policy(batch_states)
            logits = torch.stack(
                [masked_logits(logits[i], batch_masks[i]) for i in range(logits.shape[0])],
                dim=0,
            )
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - batch_old_log_probs)
            unclipped = ratios * batch_advantages
            clipped = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * batch_advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = torch.mean((values - batch_returns) ** 2)
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += float(loss.item())
            total_policy += float(policy_loss.item())
            total_value += float(value_loss.item())
            total_entropy += float(entropy.item())
            updates += 1

    scale = max(updates, 1)
    return (
        total_loss / scale,
        total_policy / scale,
        total_value / scale,
        total_entropy / scale,
    )


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
    raise RuntimeError(f"{path} is not a compatible gomoku_ppo checkpoint")


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
    board_size = int(checkpoint.get("board_size") or arg_board_size or 15)
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


def choose_policy_action(
    policy: PolicyValueNet,
    board: np.ndarray,
    current_player: int,
    device: torch.device,
) -> int:
    action, _ = policy_step(
        policy=policy,
        board=board,
        current_player=current_player,
        device=device,
        greedy=True,
        augment=False,
    )
    return action


def play_vs_opponent_once(
    policy: PolicyValueNet,
    board_size: int,
    win_length: int,
    device: torch.device,
    policy_player: int,
    opponent: str,
) -> int:
    env = GomokuEnv(board_size=board_size, win_length=win_length)
    env.reset()
    while not env.done:
        if env.current_player == policy_player:
            action = choose_policy_action(policy, env.board, env.current_player, device)
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
        )
        if winner == 0:
            draws += 1
        elif winner == policy_player:
            wins += 1
        else:
            losses += 1
    return wins / max(games, 1), wins, draws, losses


def play_eval_trace_once(
    policy: PolicyValueNet,
    board_size: int,
    win_length: int,
    device: torch.device,
) -> EpisodeTrace:
    env = GomokuEnv(board_size=board_size, win_length=win_length)
    env.reset()
    history: list[tuple[int, int, int]] = []
    while not env.done:
        player = env.current_player
        action = choose_policy_action(policy, env.board, player, device)
        row, col = action_to_coords(action, board_size)
        history.append((player, row, col))
        env.step(action)
    return EpisodeTrace(
        winner=env.winner,
        moves=len(history),
        history=history,
        final_board=env.board.copy(),
    )


def play_vs_opponent_trace_once(
    policy: PolicyValueNet,
    board_size: int,
    win_length: int,
    device: torch.device,
    policy_player: int,
    opponent: str,
) -> EpisodeTrace:
    env = GomokuEnv(board_size=board_size, win_length=win_length)
    env.reset()
    history: list[tuple[int, int, int]] = []
    while not env.done:
        player = env.current_player
        if player == policy_player:
            action = choose_policy_action(policy, env.board, player, device)
        else:
            action = choose_opponent_action(env.board, player, win_length, opponent)
        row, col = action_to_coords(action, board_size)
        history.append((player, row, col))
        env.step(action)
    return EpisodeTrace(
        winner=env.winner,
        moves=len(history),
        history=history,
        final_board=env.board.copy(),
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = choose_device(args.device)
    checkpoint = None
    if args.init_checkpoint is not None and args.init_checkpoint.exists():
        checkpoint = load_checkpoint(args.init_checkpoint, map_location=device)

    policy = build_policy(args.channels, args.conv_layers, device, checkpoint=checkpoint)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    param_count = count_parameters(policy)
    print(
        f"device={device} board={args.board_size} win={args.win_length} "
        f"params={param_count} approx_mib={parameter_size_mib(param_count):.2f}"
    )

    stable_iterations = 0
    start_time = time.time()
    for iteration in range(1, args.iterations + 1):
        rollout: list[RolloutSample] = []
        winners: list[int] = []
        lengths: list[int] = []

        for game_index in range(1, args.games_per_iter + 1):
            episode_samples, winner, moves = self_play_episode(
                policy=policy,
                board_size=args.board_size,
                win_length=args.win_length,
                device=device,
                gamma=args.gamma,
                symmetry_augment=args.symmetry_augment,
                random_opening_moves=args.random_opening_moves,
            )
            rollout.extend(episode_samples)
            winners.append(winner)
            lengths.append(moves)
            if args.log_every_games > 0 and game_index % args.log_every_games == 0:
                elapsed = time.time() - start_time
                avg_len = float(np.mean(lengths)) if lengths else 0.0
                print(
                    f"iter={iteration:5d} selfplay={game_index:3d}/{args.games_per_iter:3d} "
                    f"avg_len={avg_len:6.2f} samples={len(rollout):6d} elapsed={elapsed:7.1f}s"
                )

        policy.train()
        loss, policy_loss, value_loss, entropy = ppo_update(
            policy=policy,
            optimizer=optimizer,
            rollout=rollout,
            device=device,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            clip_eps=args.clip_eps,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
        )
        elapsed = time.time() - start_time
        p1_wins = sum(1 for item in winners if item == 1)
        p2_wins = sum(1 for item in winners if item == -1)
        draws = sum(1 for item in winners if item == 0)
        avg_len = float(np.mean(lengths)) if lengths else 0.0
        message = (
            f"iter={iteration:5d} loss={loss:7.4f} policy={policy_loss:7.4f} "
            f"value={value_loss:7.4f} entropy={entropy:7.4f} "
            f"p1={p1_wins:3d} p2={p2_wins:3d} draw={draws:3d} "
            f"avg_len={avg_len:6.2f} samples={len(rollout):6d} elapsed={elapsed:7.1f}s"
        )

        policy.eval()
        if args.eval_every > 0 and iteration % args.eval_every == 0:
            random_win_rate, rwins, rdraws, rlosses = evaluate_vs_opponent(
                policy=policy,
                board_size=args.board_size,
                win_length=args.win_length,
                device=device,
                games=args.eval_games,
                opponent="random",
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
                )
                message += f" heuristic_win_rate={heuristic_win_rate:.3f} ({hwins}/{hdraws}/{hlosses})"
            print(message)
            for trace_index in range(args.eval_trace_games):
                trace = play_eval_trace_once(
                    policy=policy,
                    board_size=args.board_size,
                    win_length=args.win_length,
                    device=device,
                )
                winner_symbol = "draw" if trace.winner == 0 else ("X" if trace.winner == 1 else "O")
                print(f"eval_trace game={trace_index + 1} winner={winner_symbol} moves={trace.moves}")
                print(f"eval_trace seq {format_move_sequence(trace.history, args.eval_trace_max_moves)}")
                print("eval_trace board:")
                print(board_to_string(trace.final_board))
            for trace_index in range(args.eval_heuristic_trace_games):
                policy_player = 1 if trace_index % 2 == 0 else -1
                trace = play_vs_opponent_trace_once(
                    policy=policy,
                    board_size=args.board_size,
                    win_length=args.win_length,
                    device=device,
                    policy_player=policy_player,
                    opponent="heuristic",
                )
                if trace.winner == 0:
                    winner_label = "draw"
                elif trace.winner == policy_player:
                    winner_label = "policy"
                else:
                    winner_label = "heuristic"
                policy_symbol = "X" if policy_player == 1 else "O"
                print(
                    f"heuristic_trace game={trace_index + 1} policy_as={policy_symbol} "
                    f"winner={winner_label} moves={trace.moves}"
                )
                print(
                    "heuristic_trace seq "
                    f"{format_policy_vs_opponent_sequence(trace.history, policy_player, args.eval_trace_max_moves)}"
                )
                print("heuristic_trace board:")
                print(board_to_string(trace.final_board))
        else:
            print(message)

        if args.save_every > 0 and iteration % args.save_every == 0:
            last_path = last_checkpoint_path(args.checkpoint)
            save_checkpoint(last_path, policy, args)
            print(f"saved checkpoint to {last_path}")

        if (
            args.early_stop_patience > 0
            and iteration >= args.early_stop_min_iterations
            and loss <= args.early_stop_loss
        ):
            stable_iterations += 1
        else:
            stable_iterations = 0
        if args.early_stop_patience > 0 and stable_iterations >= args.early_stop_patience:
            print(
                f"early_stop triggered after iter={iteration} "
                f"(loss<={args.early_stop_loss} for {stable_iterations} iters)"
            )
            break

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
    )
    print(
        f"opponent={args.opponent} games={args.games} "
        f"win_rate={win_rate:.3f} wins={wins} draws={draws} losses={losses}"
    )


def play(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    policy, board_size, win_length = resolve_game_config(
        checkpoint_path=args.checkpoint,
        arg_board_size=args.board_size,
        arg_win_length=args.win_length,
        arg_channels=args.channels,
        arg_conv_layers=args.conv_layers,
        device=device,
    )
    env = GomokuEnv(board_size=board_size, win_length=win_length)
    env.reset()
    human_player = 1 if args.human_first else -1
    print(env.render())

    while not env.done:
        if env.current_player == human_player:
            raw = input("Your move (row col): ").strip()
            try:
                row_str, col_str = raw.split()
                row = int(row_str) - 1
                col = int(col_str) - 1
                action = coords_to_action(row, col, board_size)
            except Exception as exc:  # noqa: BLE001
                print(f"invalid input: {exc}")
                continue
            if action not in env.valid_moves():
                print("illegal move")
                continue
        else:
            action = choose_policy_action(policy, env.board, env.current_player, device)
            row, col = action_to_coords(action, board_size)
            print(f"AI plays: {row + 1} {col + 1}")
        env.step(action)
        print(env.render())
        print()

    if env.winner == 0:
        print("draw")
    elif env.winner == human_player:
        print("you win")
    else:
        print("AI wins")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Gomoku PPO trainer")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common_arguments(subparser: argparse.ArgumentParser, defaults_from_checkpoint: bool = False) -> None:
        board_default = None if defaults_from_checkpoint else 9
        win_default = None if defaults_from_checkpoint else 5
        subparser.add_argument("--board-size", type=int, default=board_default)
        subparser.add_argument("--win-length", type=int, default=win_default)
        subparser.add_argument("--channels", type=int, default=128)
        subparser.add_argument("--conv-layers", type=int, default=8)
        subparser.add_argument("--device", type=str, default="auto")
        subparser.add_argument("--checkpoint", type=Path, default=Path("gomoku_ppo.pt"))

    train_parser = subparsers.add_parser("train", help="self-play PPO training")
    add_common_arguments(train_parser)
    train_parser.add_argument("--iterations", type=int, default=1500)
    train_parser.add_argument("--games-per-iter", type=int, default=32)
    train_parser.add_argument("--ppo-epochs", type=int, default=8)
    train_parser.add_argument("--minibatch-size", type=int, default=256)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--clip-eps", type=float, default=0.2)
    train_parser.add_argument("--entropy-coef", type=float, default=0.01)
    train_parser.add_argument("--value-coef", type=float, default=1.0)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.add_argument("--eval-every", type=int, default=10)
    train_parser.add_argument("--eval-games", type=int, default=20)
    train_parser.add_argument("--eval-heuristic-games", type=int, default=8)
    train_parser.add_argument("--eval-trace-games", type=int, default=1)
    train_parser.add_argument("--eval-heuristic-trace-games", type=int, default=1)
    train_parser.add_argument("--eval-trace-max-moves", type=int, default=20)
    train_parser.add_argument("--log-every-games", type=int, default=8)
    train_parser.add_argument("--save-every", type=int, default=10)
    train_parser.add_argument("--random-opening-moves", type=int, default=2)
    train_parser.add_argument("--symmetry-augment", action="store_true", default=True)
    train_parser.add_argument("--no-symmetry-augment", dest="symmetry_augment", action="store_false")
    train_parser.add_argument("--early-stop-loss", type=float, default=0.0)
    train_parser.add_argument("--early-stop-patience", type=int, default=0)
    train_parser.add_argument("--early-stop-min-iterations", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--init-checkpoint", type=Path, default=None)
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval", help="evaluate against random or heuristic")
    add_common_arguments(eval_parser, defaults_from_checkpoint=True)
    eval_parser.add_argument("--games", type=int, default=40)
    eval_parser.add_argument("--opponent", choices=["random", "heuristic"], default="heuristic")
    eval_parser.set_defaults(func=evaluate)

    play_parser = subparsers.add_parser("play", help="play against the trained PPO model")
    add_common_arguments(play_parser, defaults_from_checkpoint=True)
    play_parser.add_argument("--human-first", action="store_true", default=False)
    play_parser.set_defaults(func=play)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
