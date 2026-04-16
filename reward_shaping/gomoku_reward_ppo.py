#!/usr/bin/env python3
"""Reward-shaped self-play PPO for Gomoku."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ppo_curriculum.gomoku_ppo import (  # noqa: E402
    PolicyValueNet,
    action_to_coords,
    board_to_string,
    choose_device,
    choose_heuristic_action,
    coords_to_action,
    count_one_side_with_open_end,
    count_parameters,
    encode_state,
    format_move_sequence,
    immediate_winning_actions,
    inverse_transform_coords,
    is_winning_move,
    masked_logits,
    parameter_size_mib,
    random_argmax,
    set_seed,
    transform_board,
)


@dataclass
class RewardConfig:
    win: float
    draw: float
    block_win: float
    open_four: float
    closed_four: float
    open_three: float
    closed_three: float
    open_two: float
    double_threat: float
    allow_opp_win: float
    center_bias: float
    max_abs_step_reward: float


@dataclass
class Transition:
    state: torch.Tensor
    legal_mask: np.ndarray
    action: int
    old_log_prob: float
    old_value: float
    reward: float
    ret: float = 0.0
    advantage: float = 0.0


@dataclass
class TraceResult:
    winner: int
    moves: int
    history: list[tuple[int, int, int]]
    final_board: np.ndarray


class ShapedGomokuEnv:
    def __init__(self, board_size: int, win_length: int, reward_cfg: RewardConfig):
        if board_size <= 1:
            raise ValueError("board_size must be > 1")
        if not 1 < win_length <= board_size:
            raise ValueError("win_length must satisfy 1 < win_length <= board_size")
        self.board_size = board_size
        self.win_length = win_length
        self.reward_cfg = reward_cfg
        self.reset()

    def reset(self) -> np.ndarray:
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = 0
        return self.board

    def valid_moves(self) -> np.ndarray:
        return np.flatnonzero((self.board == 0).reshape(-1))

    def step(self, action: int) -> tuple[bool, int, float]:
        if self.done:
            raise RuntimeError("game is already finished")

        row, col = action_to_coords(int(action), self.board_size)
        if self.board[row, col] != 0:
            raise ValueError(f"illegal move at ({row}, {col})")

        player = self.current_player
        before_board = self.board.copy()
        self.board[row, col] = player

        if is_winning_move(self.board, row, col, player, self.win_length):
            self.done = True
            self.winner = player
            reward = self.reward_cfg.win
        elif not np.any(self.board == 0):
            self.done = True
            self.winner = 0
            reward = self.reward_cfg.draw
        else:
            reward = shaped_move_reward(
                before_board=before_board,
                after_board=self.board,
                row=row,
                col=col,
                player=player,
                win_length=self.win_length,
                reward_cfg=self.reward_cfg,
            )
            self.current_player = -player

        reward = float(np.clip(reward, -self.reward_cfg.max_abs_step_reward, self.reward_cfg.max_abs_step_reward))
        return self.done, self.winner, reward


def line_pattern_counts(
    board: np.ndarray,
    row: int,
    col: int,
    player: int,
    win_length: int,
) -> dict[str, int]:
    counts = {
        "open_four": 0,
        "closed_four": 0,
        "open_three": 0,
        "closed_three": 0,
        "open_two": 0,
    }
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        forward_count, forward_open = count_one_side_with_open_end(board, row, col, dr, dc, player)
        backward_count, backward_open = count_one_side_with_open_end(board, row, col, -dr, -dc, player)
        length = 1 + forward_count + backward_count
        open_ends = int(forward_open) + int(backward_open)
        if length >= win_length:
            counts["open_four"] += 1
        elif length == win_length - 1:
            if open_ends == 2:
                counts["open_four"] += 1
            elif open_ends == 1:
                counts["closed_four"] += 1
        elif length == win_length - 2:
            if open_ends == 2:
                counts["open_three"] += 1
            elif open_ends == 1:
                counts["closed_three"] += 1
        elif length == win_length - 3 and open_ends == 2:
            counts["open_two"] += 1
    return counts


def shaped_move_reward(
    before_board: np.ndarray,
    after_board: np.ndarray,
    row: int,
    col: int,
    player: int,
    win_length: int,
    reward_cfg: RewardConfig,
) -> float:
    reward = 0.0
    action = coords_to_action(row, col, before_board.shape[0])

    opp_immediate_before = set(immediate_winning_actions(before_board, -player, win_length))
    if action in opp_immediate_before:
        reward += reward_cfg.block_win

    patterns = line_pattern_counts(after_board, row, col, player, win_length)
    reward += reward_cfg.open_four * patterns["open_four"]
    reward += reward_cfg.closed_four * patterns["closed_four"]
    reward += reward_cfg.open_three * patterns["open_three"]
    reward += reward_cfg.closed_three * patterns["closed_three"]
    reward += reward_cfg.open_two * patterns["open_two"]

    own_immediate_after = len(immediate_winning_actions(after_board, player, win_length))
    opp_immediate_after = len(immediate_winning_actions(after_board, -player, win_length))
    if own_immediate_after >= 2:
        reward += reward_cfg.double_threat
    elif own_immediate_after == 1:
        reward += reward_cfg.closed_four
    if opp_immediate_after > 0:
        reward -= reward_cfg.allow_opp_win * opp_immediate_after

    center = (before_board.shape[0] - 1) / 2.0
    distance = abs(row - center) + abs(col - center)
    reward += reward_cfg.center_bias * (before_board.shape[0] - distance)
    return reward


def select_action(
    policy: PolicyValueNet,
    board: np.ndarray,
    current_player: int,
    device: torch.device,
    greedy: bool,
    augment: bool,
) -> tuple[int, Transition | None]:
    board_size = board.shape[0]
    rotation_k = np.random.randint(0, 4) if augment else 0
    flip = bool(np.random.randint(0, 2)) if augment else False
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
    sample = Transition(
        state=encode_state(transformed_board, current_player),
        legal_mask=legal_mask.reshape(-1).astype(np.bool_),
        action=transformed_action,
        old_log_prob=float(dist.log_prob(action_tensor).item()),
        old_value=float(value.item()),
        reward=0.0,
    )
    return coords_to_action(row, col, board_size), sample


def finalize_returns(trajectory: list[Transition], gamma: float, return_clip: float) -> None:
    running = 0.0
    for sample in reversed(trajectory):
        running = sample.reward - gamma * running
        if return_clip > 0.0:
            running = float(np.clip(running, -return_clip, return_clip))
        sample.ret = running
        sample.advantage = sample.ret - sample.old_value


def self_play_episode(
    policy: PolicyValueNet,
    board_size: int,
    win_length: int,
    reward_cfg: RewardConfig,
    device: torch.device,
    gamma: float,
    return_clip: float,
    symmetry_augment: bool,
    random_opening_moves: int,
) -> tuple[list[Transition], int, int]:
    env = ShapedGomokuEnv(board_size=board_size, win_length=win_length, reward_cfg=reward_cfg)
    env.reset()
    opening_moves = np.random.randint(0, max(random_opening_moves, 0) + 1)
    for _ in range(opening_moves):
        if env.done:
            break
        env.step(int(np.random.choice(env.valid_moves())))

    trajectory: list[Transition] = []
    move_count = 0
    while not env.done:
        action, sample = select_action(
            policy=policy,
            board=env.board,
            current_player=env.current_player,
            device=device,
            greedy=False,
            augment=symmetry_augment,
        )
        if sample is None:
            raise RuntimeError("sample missing during training")
        done, winner, reward = env.step(action)
        sample.reward = reward
        trajectory.append(sample)
        move_count += 1
        if done and winner == 0:
            break

    finalize_returns(trajectory, gamma=gamma, return_clip=return_clip)
    return trajectory, env.winner, move_count


def ppo_update(
    policy: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    rollout: list[Transition],
    device: torch.device,
    ppo_epochs: int,
    minibatch_size: int,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
) -> tuple[float, float, float, float]:
    states = torch.stack([item.state for item in rollout]).to(device)
    legal_masks = torch.as_tensor(np.stack([item.legal_mask for item in rollout]), dtype=torch.bool, device=device)
    actions = torch.as_tensor([item.action for item in rollout], dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor([item.old_log_prob for item in rollout], dtype=torch.float32, device=device)
    returns = torch.as_tensor([item.ret for item in rollout], dtype=torch.float32, device=device)
    advantages = torch.as_tensor([item.advantage for item in rollout], dtype=torch.float32, device=device)
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
            logits = torch.stack([masked_logits(logits[i], batch_masks[i]) for i in range(logits.shape[0])], dim=0)
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
    return total_loss / scale, total_policy / scale, total_value / scale, total_entropy / scale


def save_checkpoint(path: Path, policy: PolicyValueNet, args: argparse.Namespace) -> None:
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "channels": args.channels,
            "conv_layers": args.conv_layers,
            "board_size": args.board_size,
            "win_length": args.win_length,
            "reward_profile": "shaped_v1",
        },
        path,
    )


def last_checkpoint_path(base_path: Path) -> Path:
    return base_path.with_name(f"{base_path.stem}_last{base_path.suffix}")


def load_checkpoint(path: Path, map_location: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint
    raise RuntimeError(f"{path} is not a compatible gomoku_reward_ppo checkpoint")


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
    channels = int(checkpoint.get("channels") or arg_channels)
    conv_layers = int(checkpoint.get("conv_layers") or arg_conv_layers)
    policy = PolicyValueNet(channels=channels, conv_layers=conv_layers).to(device)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy, board_size, win_length


def choose_policy_action(policy: PolicyValueNet, board: np.ndarray, current_player: int, device: torch.device) -> int:
    action, _ = select_action(policy=policy, board=board, current_player=current_player, device=device, greedy=True, augment=False)
    return action


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
    reward_cfg: RewardConfig,
    device: torch.device,
    policy_player: int,
    opponent: str,
) -> int:
    env = ShapedGomokuEnv(board_size=board_size, win_length=win_length, reward_cfg=reward_cfg)
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
    reward_cfg: RewardConfig,
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
            reward_cfg=reward_cfg,
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
    reward_cfg: RewardConfig,
    device: torch.device,
) -> TraceResult:
    env = ShapedGomokuEnv(board_size=board_size, win_length=win_length, reward_cfg=reward_cfg)
    env.reset()
    history: list[tuple[int, int, int]] = []
    while not env.done:
        player = env.current_player
        action = choose_policy_action(policy, env.board, player, device)
        row, col = action_to_coords(action, board_size)
        history.append((player, row, col))
        env.step(action)
    return TraceResult(winner=env.winner, moves=len(history), history=history, final_board=env.board.copy())


def reward_config_from_args(args: argparse.Namespace) -> RewardConfig:
    return RewardConfig(
        win=args.reward_win,
        draw=args.reward_draw,
        block_win=args.reward_block_win,
        open_four=args.reward_open_four,
        closed_four=args.reward_closed_four,
        open_three=args.reward_open_three,
        closed_three=args.reward_closed_three,
        open_two=args.reward_open_two,
        double_threat=args.reward_double_threat,
        allow_opp_win=args.reward_allow_opp_win,
        center_bias=args.reward_center_bias,
        max_abs_step_reward=args.max_abs_step_reward,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = choose_device(args.device)
    reward_cfg = reward_config_from_args(args)
    policy = PolicyValueNet(channels=args.channels, conv_layers=args.conv_layers).to(device)
    if args.init_checkpoint is not None and args.init_checkpoint.exists():
        checkpoint = load_checkpoint(args.init_checkpoint, map_location=device)
        policy.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    params = count_parameters(policy)
    print(
        f"device={device} board={args.board_size} win={args.win_length} "
        f"params={params} approx_mib={parameter_size_mib(params):.2f}"
    )

    start_time = time.time()
    for iteration in range(1, args.iterations + 1):
        rollout: list[Transition] = []
        winners: list[int] = []
        lengths: list[int] = []
        rewards: list[float] = []

        for game_index in range(1, args.games_per_iter + 1):
            episode_samples, winner, moves = self_play_episode(
                policy=policy,
                board_size=args.board_size,
                win_length=args.win_length,
                reward_cfg=reward_cfg,
                device=device,
                gamma=args.gamma,
                return_clip=args.return_clip,
                symmetry_augment=args.symmetry_augment,
                random_opening_moves=args.random_opening_moves,
            )
            rollout.extend(episode_samples)
            winners.append(winner)
            lengths.append(moves)
            rewards.extend([sample.reward for sample in episode_samples])
            if args.log_every_games > 0 and game_index % args.log_every_games == 0:
                elapsed = time.time() - start_time
                avg_len = float(np.mean(lengths)) if lengths else 0.0
                avg_reward = float(np.mean(rewards)) if rewards else 0.0
                print(
                    f"iter={iteration:5d} selfplay={game_index:3d}/{args.games_per_iter:3d} "
                    f"avg_len={avg_len:6.2f} avg_step_reward={avg_reward:7.4f} "
                    f"samples={len(rollout):6d} elapsed={elapsed:7.1f}s"
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

        p1_wins = sum(1 for item in winners if item == 1)
        p2_wins = sum(1 for item in winners if item == -1)
        draws = sum(1 for item in winners if item == 0)
        avg_len = float(np.mean(lengths)) if lengths else 0.0
        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        elapsed = time.time() - start_time
        message = (
            f"iter={iteration:5d} loss={loss:7.4f} policy={policy_loss:7.4f} "
            f"value={value_loss:7.4f} entropy={entropy:7.4f} "
            f"avg_step_reward={avg_reward:7.4f} p1={p1_wins:3d} p2={p2_wins:3d} draw={draws:3d} "
            f"avg_len={avg_len:6.2f} samples={len(rollout):6d} elapsed={elapsed:7.1f}s"
        )

        policy.eval()
        if args.eval_every > 0 and iteration % args.eval_every == 0:
            random_win_rate, rwins, rdraws, rlosses = evaluate_vs_opponent(
                policy=policy,
                board_size=args.board_size,
                win_length=args.win_length,
                reward_cfg=reward_cfg,
                device=device,
                games=args.eval_games,
                opponent="random",
            )
            heuristic_win_rate, hwins, hdraws, hlosses = evaluate_vs_opponent(
                policy=policy,
                board_size=args.board_size,
                win_length=args.win_length,
                reward_cfg=reward_cfg,
                device=device,
                games=args.eval_heuristic_games,
                opponent="heuristic",
            )
            message += (
                f" random_win_rate={random_win_rate:.3f} ({rwins}/{rdraws}/{rlosses}) "
                f"heuristic_win_rate={heuristic_win_rate:.3f} ({hwins}/{hdraws}/{hlosses})"
            )
            print(message)
            for trace_index in range(args.eval_trace_games):
                trace = play_eval_trace_once(
                    policy=policy,
                    board_size=args.board_size,
                    win_length=args.win_length,
                    reward_cfg=reward_cfg,
                    device=device,
                )
                winner_text = "draw" if trace.winner == 0 else ("X" if trace.winner == 1 else "O")
                print(f"eval_trace game={trace_index + 1} winner={winner_text} moves={trace.moves}")
                print(f"eval_trace seq {format_move_sequence(trace.history, args.eval_trace_max_moves)}")
                print("eval_trace board:")
                print(board_to_string(trace.final_board))
        else:
            print(message)

        if args.save_every > 0 and iteration % args.save_every == 0:
            last_path = last_checkpoint_path(args.checkpoint)
            save_checkpoint(last_path, policy, args)
            print(f"saved checkpoint to {last_path}")

    save_checkpoint(args.checkpoint, policy, args)
    print(f"saved checkpoint to {args.checkpoint}")


def evaluate(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    reward_cfg = reward_config_from_args(args)
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
        reward_cfg=reward_cfg,
        device=device,
        games=args.games,
        opponent=args.opponent,
    )
    print(
        f"opponent={args.opponent} games={args.games} "
        f"win_rate={win_rate:.3f} wins={wins} draws={draws} losses={losses}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reward-shaped Gomoku PPO")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common_arguments(subparser: argparse.ArgumentParser, defaults_from_checkpoint: bool = False) -> None:
        board_default = None if defaults_from_checkpoint else 15
        win_default = None if defaults_from_checkpoint else 5
        subparser.add_argument("--board-size", type=int, default=board_default)
        subparser.add_argument("--win-length", type=int, default=win_default)
        subparser.add_argument("--channels", type=int, default=320)
        subparser.add_argument("--conv-layers", type=int, default=12)
        subparser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
        subparser.add_argument("--checkpoint", type=Path, default=Path("gomoku_reward_ppo.pt"))
        subparser.add_argument("--reward-win", type=float, default=8.0)
        subparser.add_argument("--reward-draw", type=float, default=0.0)
        subparser.add_argument("--reward-block-win", type=float, default=2.0)
        subparser.add_argument("--reward-open-four", type=float, default=1.5)
        subparser.add_argument("--reward-closed-four", type=float, default=0.8)
        subparser.add_argument("--reward-open-three", type=float, default=0.55)
        subparser.add_argument("--reward-closed-three", type=float, default=0.2)
        subparser.add_argument("--reward-open-two", type=float, default=0.05)
        subparser.add_argument("--reward-double-threat", type=float, default=2.5)
        subparser.add_argument("--reward-allow-opp-win", type=float, default=3.0)
        subparser.add_argument("--reward-center-bias", type=float, default=0.01)
        subparser.add_argument("--max-abs-step-reward", type=float, default=10.0)
        subparser.add_argument("--return-clip", type=float, default=10.0)

    train_parser = subparsers.add_parser("train", help="train with shaped self-play PPO")
    add_common_arguments(train_parser)
    train_parser.add_argument("--iterations", type=int, default=3000)
    train_parser.add_argument("--games-per-iter", type=int, default=64)
    train_parser.add_argument("--ppo-epochs", type=int, default=8)
    train_parser.add_argument("--minibatch-size", type=int, default=256)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--clip-eps", type=float, default=0.2)
    train_parser.add_argument("--entropy-coef", type=float, default=0.01)
    train_parser.add_argument("--value-coef", type=float, default=1.0)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.add_argument("--random-opening-moves", type=int, default=2)
    train_parser.add_argument("--eval-every", type=int, default=10)
    train_parser.add_argument("--eval-games", type=int, default=20)
    train_parser.add_argument("--eval-heuristic-games", type=int, default=8)
    train_parser.add_argument("--eval-trace-games", type=int, default=1)
    train_parser.add_argument("--eval-trace-max-moves", type=int, default=20)
    train_parser.add_argument("--log-every-games", type=int, default=8)
    train_parser.add_argument("--save-every", type=int, default=10)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--symmetry-augment", action="store_true", default=True)
    train_parser.add_argument("--no-symmetry-augment", dest="symmetry_augment", action="store_false")
    train_parser.add_argument("--init-checkpoint", type=Path, default=None)
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval", help="evaluate against random or heuristic")
    add_common_arguments(eval_parser, defaults_from_checkpoint=True)
    eval_parser.add_argument("--games", type=int, default=40)
    eval_parser.add_argument("--opponent", choices=["random", "heuristic"], default="heuristic")
    eval_parser.set_defaults(func=evaluate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
