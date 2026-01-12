from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from rl.evolutionary_training import (
    EvolutionaryConfig,
    evaluate_evolutionary_policy,
    load_evolutionary_checkpoint,
    load_evolutionary_config,
    run_evolutionary_training,
)


def main(cli_args: Optional[list[str]] = None) -> None:
    """Evaluate a trained evolutionary agent."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Pacman evolutionary agent"
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to saved checkpoint", required=True
    )
    parser.add_argument(
        "--episodes", type=int, default=None, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Max steps per evaluation episode"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Exploration epsilon during evaluation",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional training config (overrides checkpoint settings)",
    )
    parser.add_argument(
        "--train-if-missing",
        action="store_true",
        help="Train from config if checkpoint is missing",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Optional directory to save metrics JSON"
    )
    args = parser.parse_args(cli_args)

    checkpoint_path = args.checkpoint
    agent: Optional = None
    config: EvolutionaryConfig

    try:
        agent, config = load_evolutionary_checkpoint(checkpoint_path)
    except FileNotFoundError:
        if not args.train_if_missing or args.config is None:
            raise
        config = load_evolutionary_config(args.config)
        training_result = run_evolutionary_training(config, output_dir=args.output)
        agent = training_result.agent

    evaluation = evaluate_evolutionary_policy(
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
