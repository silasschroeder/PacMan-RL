from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from rl.training import TrainingConfig, evaluate_policy, load_training_config, run_training


def _apply_overrides(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    if args.episodes is not None:
        config.episodes = args.episodes
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.buffer_size is not None:
        config.buffer_size = args.buffer_size
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.epsilon_start is not None:
        config.epsilon_start = args.epsilon_start
    if args.epsilon_end is not None:
        config.epsilon_end = args.epsilon_end
    if args.epsilon_decay_steps is not None:
        config.epsilon_decay_steps = args.epsilon_decay_steps
    if args.frame_skip is not None:
        config.frame_skip = args.frame_skip
    if args.gamma is not None:
        config.gamma = args.gamma
    return config


def main(cli_args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train a DQN agent for Pacman")
    parser.add_argument("--config", type=str, help="Path to JSON training config", default=None)
    parser.add_argument("--episodes", type=int, help="Override number of training episodes")
    parser.add_argument("--max-steps", type=int, help="Override max steps per episode")
    parser.add_argument("--buffer-size", type=int, help="Override replay buffer capacity")
    parser.add_argument("--batch-size", type=int, help="Override training batch size")
    parser.add_argument("--learning-rate", type=float, help="Override optimizer learning rate")
    parser.add_argument("--gamma", type=float, help="Override discount factor")
    parser.add_argument("--epsilon-start", type=float, help="Override starting epsilon")
    parser.add_argument("--epsilon-end", type=float, help="Override final epsilon")
    parser.add_argument("--epsilon-decay-steps", type=int, help="Override epsilon decay steps")
    parser.add_argument("--frame-skip", type=int, help="Override frame skip value")
    parser.add_argument("--seed", type=int, help="Override RNG seed")
    parser.add_argument("--device", type=str, help="Torch device to train on (cpu/cuda)")
    parser.add_argument("--output", type=str, default="runs/latest", help="Directory for checkpoints and metrics")
    parser.add_argument("--eval", dest="run_eval", action="store_true", help="Run evaluation after training")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Episodes for post-training evaluation")
    args = parser.parse_args(cli_args)

    config = load_training_config(args.config)
    config = _apply_overrides(config, args)

    result = run_training(config, output_dir=args.output)

    last_metrics = result.metrics[-1] if result.metrics else {}
    print("Training complete")
    if last_metrics:
        print(json.dumps(last_metrics, indent=2))

    if args.run_eval:
        metrics = evaluate_policy(
            result.agent,
            result.config,
            episodes=args.eval_episodes,
        )
        print("Evaluation summary:")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
