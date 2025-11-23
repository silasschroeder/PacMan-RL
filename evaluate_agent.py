from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from rl.training import TrainingConfig, evaluate_policy, load_checkpoint, load_training_config, run_training


def main(cli_args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Pacman DQN agent")
    parser.add_argument("--checkpoint", type=str, help="Path to saved checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=None, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per evaluation episode")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Exploration epsilon during evaluation")
    parser.add_argument("--config", type=str, default=None, help="Optional training config (overrides checkpoint settings)")
    parser.add_argument("--train-if-missing", action="store_true", help="Train from config if checkpoint is missing")
    parser.add_argument("--output", type=str, default=None, help="Optional directory to save metrics JSON")
    args = parser.parse_args(cli_args)

    checkpoint_path = args.checkpoint
    agent: Optional = None
    config: TrainingConfig

    try:
        agent, config = load_checkpoint(checkpoint_path)
    except FileNotFoundError:
        if not args.train_if_missing or args.config is None:
            raise
        config = load_training_config(args.config)
        training_result = run_training(config, output_dir=args.output)
        agent = training_result.agent

    evaluation = evaluate_policy(
        agent,
        config,
        episodes=args.episodes,
        max_steps=args.max_steps,
        epsilon=args.epsilon,
    )

    print("Evaluation summary:")
    print(json.dumps(evaluation, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "evaluation.json", "w", encoding="utf-8") as handle:
            json.dump(evaluation, handle, indent=2)


if __name__ == "__main__":
    main()
