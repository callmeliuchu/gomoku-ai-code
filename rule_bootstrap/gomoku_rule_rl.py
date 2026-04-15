#!/usr/bin/env python3
"""Heuristic imitation + PPO finetune pipeline for Gomoku."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gomoku_bootstrap import generate_heuristic_dataset, run_pretrain
from ppo_curriculum.gomoku_ppo import evaluate as run_ppo_eval
from ppo_curriculum.gomoku_ppo import train as run_ppo_train


def copy_namespace(args: argparse.Namespace, **updates: object) -> argparse.Namespace:
    payload = vars(args).copy()
    payload.update(updates)
    return argparse.Namespace(**payload)


def run_pipeline(args: argparse.Namespace) -> None:
    dataset_args = argparse.Namespace(
        board_size=args.board_size,
        win_length=args.win_length,
        games=args.generate_games,
        random_opening_moves=args.generate_random_opening_moves,
        policy_smoothing=args.policy_smoothing,
        log_every_games=args.generate_log_every_games,
        seed=args.seed,
        output=args.dataset,
    )
    print("=== Stage 1: generate heuristic dataset ===")
    generate_heuristic_dataset(dataset_args)

    pretrain_args = argparse.Namespace(
        dataset=[args.dataset],
        channels=args.channels,
        conv_layers=args.conv_layers,
        steps=args.pretrain_steps,
        batch_size=args.pretrain_batch_size,
        lr=args.pretrain_lr,
        weight_decay=args.pretrain_weight_decay,
        value_coef=args.value_coef,
        eval_every=args.pretrain_eval_every,
        eval_games=args.eval_games,
        eval_heuristic_games=args.eval_heuristic_games,
        eval_agent=args.pretrain_eval_agent,
        eval_mcts_sims=args.pretrain_eval_mcts_sims,
        c_puct=args.c_puct,
        save_every=args.pretrain_save_every,
        log_every_steps=args.pretrain_log_every_steps,
        seed=args.seed,
        device=args.device,
        init_checkpoint=args.pretrain_init_checkpoint,
        checkpoint=args.pretrain_checkpoint,
    )
    print("=== Stage 2: heuristic pretrain ===")
    run_pretrain(pretrain_args)

    finetune_args = argparse.Namespace(
        board_size=args.board_size,
        win_length=args.win_length,
        channels=args.channels,
        conv_layers=args.conv_layers,
        device=args.device,
        checkpoint=args.finetune_checkpoint,
        iterations=args.finetune_iterations,
        games_per_iter=args.games_per_iter,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        lr=args.finetune_lr,
        weight_decay=args.finetune_weight_decay,
        gamma=args.gamma,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        eval_every=args.finetune_eval_every,
        eval_games=args.eval_games,
        eval_heuristic_games=args.eval_heuristic_games,
        eval_trace_games=args.eval_trace_games,
        eval_trace_max_moves=args.eval_trace_max_moves,
        log_every_games=args.finetune_log_every_games,
        save_every=args.finetune_save_every,
        random_opening_moves=args.finetune_random_opening_moves,
        symmetry_augment=not args.no_symmetry_augment,
        early_stop_loss=args.early_stop_loss,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_iterations=args.early_stop_min_iterations,
        seed=args.seed,
        init_checkpoint=args.pretrain_checkpoint,
    )
    print("=== Stage 3: PPO finetune ===")
    run_ppo_train(finetune_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rule imitation + PPO RL pipeline")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_shared(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--board-size", type=int, default=15)
        subparser.add_argument("--win-length", type=int, default=5)
        subparser.add_argument("--channels", type=int, default=320)
        subparser.add_argument("--conv-layers", type=int, default=12)
        subparser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
        subparser.add_argument("--seed", type=int, default=42)
        subparser.add_argument("--value-coef", type=float, default=1.0)
        subparser.add_argument("--eval-games", type=int, default=20)
        subparser.add_argument("--eval-heuristic-games", type=int, default=8)
        subparser.add_argument("--c-puct", type=float, default=1.5)

    pipeline_parser = subparsers.add_parser("pipeline", help="run generate -> pretrain -> PPO finetune")
    add_shared(pipeline_parser)
    pipeline_parser.add_argument("--dataset", type=Path, default=Path("rule_bootstrap/heuristic_dataset_15x15_5.npz"))
    pipeline_parser.add_argument("--generate-games", type=int, default=12000)
    pipeline_parser.add_argument("--generate-random-opening-moves", type=int, default=2)
    pipeline_parser.add_argument("--policy-smoothing", type=float, default=0.05)
    pipeline_parser.add_argument("--generate-log-every-games", type=int, default=200)
    pipeline_parser.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        default=Path("rule_bootstrap/gomoku_rule_pretrain_15x15_5.pt"),
    )
    pipeline_parser.add_argument("--pretrain-init-checkpoint", type=Path, default=None)
    pipeline_parser.add_argument("--pretrain-steps", type=int, default=12000)
    pipeline_parser.add_argument("--pretrain-batch-size", type=int, default=256)
    pipeline_parser.add_argument("--pretrain-lr", type=float, default=3e-4)
    pipeline_parser.add_argument("--pretrain-weight-decay", type=float, default=1e-4)
    pipeline_parser.add_argument("--pretrain-eval-every", type=int, default=500)
    pipeline_parser.add_argument("--pretrain-eval-agent", choices=["policy", "mcts"], default="policy")
    pipeline_parser.add_argument("--pretrain-eval-mcts-sims", type=int, default=96)
    pipeline_parser.add_argument("--pretrain-save-every", type=int, default=1000)
    pipeline_parser.add_argument("--pretrain-log-every-steps", type=int, default=100)
    pipeline_parser.add_argument(
        "--finetune-checkpoint",
        type=Path,
        default=Path("rule_bootstrap/gomoku_rule_rl_15x15_5.pt"),
    )
    pipeline_parser.add_argument("--finetune-iterations", type=int, default=3000)
    pipeline_parser.add_argument("--games-per-iter", type=int, default=64)
    pipeline_parser.add_argument("--ppo-epochs", type=int, default=8)
    pipeline_parser.add_argument("--minibatch-size", type=int, default=256)
    pipeline_parser.add_argument("--finetune-lr", type=float, default=3e-4)
    pipeline_parser.add_argument("--finetune-weight-decay", type=float, default=1e-4)
    pipeline_parser.add_argument("--gamma", type=float, default=0.99)
    pipeline_parser.add_argument("--clip-eps", type=float, default=0.2)
    pipeline_parser.add_argument("--entropy-coef", type=float, default=0.01)
    pipeline_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    pipeline_parser.add_argument("--finetune-eval-every", type=int, default=10)
    pipeline_parser.add_argument("--eval-trace-games", type=int, default=1)
    pipeline_parser.add_argument("--eval-trace-max-moves", type=int, default=20)
    pipeline_parser.add_argument("--finetune-log-every-games", type=int, default=8)
    pipeline_parser.add_argument("--finetune-save-every", type=int, default=10)
    pipeline_parser.add_argument("--finetune-random-opening-moves", type=int, default=2)
    pipeline_parser.add_argument("--no-symmetry-augment", action="store_true", default=False)
    pipeline_parser.add_argument("--early-stop-loss", type=float, default=0.0)
    pipeline_parser.add_argument("--early-stop-patience", type=int, default=0)
    pipeline_parser.add_argument("--early-stop-min-iterations", type=int, default=0)
    pipeline_parser.set_defaults(func=run_pipeline)

    generate_parser = subparsers.add_parser("generate", help="generate heuristic dataset")
    generate_parser.add_argument("--board-size", type=int, default=15)
    generate_parser.add_argument("--win-length", type=int, default=5)
    generate_parser.add_argument("--games", type=int, default=12000)
    generate_parser.add_argument("--random-opening-moves", type=int, default=2)
    generate_parser.add_argument("--policy-smoothing", type=float, default=0.05)
    generate_parser.add_argument("--log-every-games", type=int, default=200)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--output", type=Path, default=Path("rule_bootstrap/heuristic_dataset_15x15_5.npz"))
    generate_parser.set_defaults(func=generate_heuristic_dataset)

    pretrain_parser = subparsers.add_parser("pretrain", help="run heuristic imitation pretrain")
    pretrain_parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    pretrain_parser.add_argument("--channels", type=int, default=320)
    pretrain_parser.add_argument("--conv-layers", type=int, default=12)
    pretrain_parser.add_argument("--steps", type=int, default=12000)
    pretrain_parser.add_argument("--batch-size", type=int, default=256)
    pretrain_parser.add_argument("--lr", type=float, default=3e-4)
    pretrain_parser.add_argument("--weight-decay", type=float, default=1e-4)
    pretrain_parser.add_argument("--value-coef", type=float, default=1.0)
    pretrain_parser.add_argument("--eval-every", type=int, default=500)
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
    pretrain_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("rule_bootstrap/gomoku_rule_pretrain_15x15_5.pt"),
    )
    pretrain_parser.set_defaults(func=run_pretrain)

    finetune_parser = subparsers.add_parser("finetune", help="run PPO finetune from a pretrained checkpoint")
    finetune_parser.add_argument("--board-size", type=int, default=15)
    finetune_parser.add_argument("--win-length", type=int, default=5)
    finetune_parser.add_argument("--channels", type=int, default=320)
    finetune_parser.add_argument("--conv-layers", type=int, default=12)
    finetune_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    finetune_parser.add_argument("--checkpoint", type=Path, default=Path("rule_bootstrap/gomoku_rule_rl_15x15_5.pt"))
    finetune_parser.add_argument("--iterations", type=int, default=3000)
    finetune_parser.add_argument("--games-per-iter", type=int, default=64)
    finetune_parser.add_argument("--ppo-epochs", type=int, default=8)
    finetune_parser.add_argument("--minibatch-size", type=int, default=256)
    finetune_parser.add_argument("--lr", type=float, default=3e-4)
    finetune_parser.add_argument("--weight-decay", type=float, default=1e-4)
    finetune_parser.add_argument("--gamma", type=float, default=0.99)
    finetune_parser.add_argument("--clip-eps", type=float, default=0.2)
    finetune_parser.add_argument("--entropy-coef", type=float, default=0.01)
    finetune_parser.add_argument("--value-coef", type=float, default=1.0)
    finetune_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    finetune_parser.add_argument("--eval-every", type=int, default=10)
    finetune_parser.add_argument("--eval-games", type=int, default=20)
    finetune_parser.add_argument("--eval-heuristic-games", type=int, default=8)
    finetune_parser.add_argument("--eval-trace-games", type=int, default=1)
    finetune_parser.add_argument("--eval-trace-max-moves", type=int, default=20)
    finetune_parser.add_argument("--log-every-games", type=int, default=8)
    finetune_parser.add_argument("--save-every", type=int, default=10)
    finetune_parser.add_argument("--random-opening-moves", type=int, default=2)
    finetune_parser.add_argument("--no-symmetry-augment", action="store_true", default=False)
    finetune_parser.add_argument("--early-stop-loss", type=float, default=0.0)
    finetune_parser.add_argument("--early-stop-patience", type=int, default=0)
    finetune_parser.add_argument("--early-stop-min-iterations", type=int, default=0)
    finetune_parser.add_argument("--seed", type=int, default=42)
    finetune_parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=Path("rule_bootstrap/gomoku_rule_pretrain_15x15_5.pt"),
    )
    finetune_parser.set_defaults(func=run_ppo_train)

    eval_parser = subparsers.add_parser("eval", help="evaluate the PPO finetuned checkpoint")
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--board-size", type=int, default=None)
    eval_parser.add_argument("--win-length", type=int, default=None)
    eval_parser.add_argument("--channels", type=int, default=320)
    eval_parser.add_argument("--conv-layers", type=int, default=12)
    eval_parser.add_argument("--games", type=int, default=40)
    eval_parser.add_argument("--opponent", choices=["random", "heuristic"], default="heuristic")
    eval_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    eval_parser.set_defaults(func=run_ppo_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
