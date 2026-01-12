from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Optional

from rl.evolutionary_training import (
    EvolutionaryConfig,
    evaluate_evolutionary_policy,
    load_evolutionary_config,
    run_evolutionary_training,
)


def _apply_overrides(config: EvolutionaryConfig, args: argparse.Namespace) -> EvolutionaryConfig:
    """Apply command-line argument overrides to config."""
    if args.generations is not None:
        config.generations = args.generations
    if args.population_size is not None:
        config.population_size = args.population_size
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.fitness_episodes is not None:
        config.fitness_episodes = args.fitness_episodes
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.frame_skip is not None:
        config.frame_skip = args.frame_skip
    if args.elite_fraction is not None:
        config.elite_fraction = args.elite_fraction
    if args.tournament_size is not None:
        config.tournament_size = args.tournament_size
    if args.crossover_rate is not None:
        config.crossover_rate = args.crossover_rate
    if args.mutation_rate is not None:
        config.mutation_rate = args.mutation_rate
    if args.mutation_std is not None:
        config.mutation_std = args.mutation_std
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    return config


def main(cli_args: Optional[list[str]] = None) -> None:
    """Main entry point for evolutionary training."""
    parser = argparse.ArgumentParser(
        description="Train a Genetic Algorithm agent for Pacman"
    )
    parser.add_argument(
        "--config", type=str, help="Path to JSON training config", default=None
    )
    parser.add_argument(
        "--generations", type=int, help="Override number of generations"
    )
    parser.add_argument(
        "--population-size", type=int, help="Override population size"
    )
    parser.add_argument(
        "--max-steps", type=int, help="Override max steps per episode"
    )
    parser.add_argument(
        "--fitness-episodes",
        type=int,
        help="Override number of episodes for fitness evaluation",
    )
    parser.add_argument(
        "--elite-fraction", type=float, help="Override elite fraction (0.0-1.0)"
    )
    parser.add_argument(
        "--tournament-size", type=int, help="Override tournament size"
    )
    parser.add_argument(
        "--crossover-rate", type=float, help="Override crossover rate (0.0-1.0)"
    )
    parser.add_argument(
        "--mutation-rate", type=float, help="Override mutation rate (0.0-1.0)"
    )
    parser.add_argument(
        "--mutation-std", type=float, help="Override mutation standard deviation"
    )
    parser.add_argument("--frame-skip", type=int, help="Override frame skip value")
    parser.add_argument("--seed", type=int, help="Override RNG seed")
    parser.add_argument("--device", type=str, help="Torch device to use (cpu/cuda)")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect CPU count)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/evolutionary",
        help="Directory for checkpoints and metrics",
    )
    parser.add_argument(
        "--eval",
        dest="run_eval",
        action="store_true",
        help="Run evaluation after training",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Episodes for post-training evaluation",
    )
    args = parser.parse_args(cli_args)

    # Load and apply config
    config = load_evolutionary_config(args.config)
    config = _apply_overrides(config, args)
    
    # Auto-detect CPU count if num_workers not specified
    if config.num_workers == 1 and args.num_workers is None:
        cpu_count = mp.cpu_count()
        config.num_workers = max(1, cpu_count - 1)  # Leave one core free
        print(f"Auto-detected {cpu_count} CPUs, using {config.num_workers} workers")

    # Run training
    result = run_evolutionary_training(config, output_dir=args.output)

    # Print final metrics
    last_metrics = result.metrics[-1] if result.metrics else {}
    print("\n=== Training Summary ===")
    if last_metrics:
        print(json.dumps(last_metrics, indent=2))

    # Optional post-training evaluation
    if args.run_eval:
        print(f"\n=== Running Evaluation ({args.eval_episodes} episodes) ===")
        eval_results = evaluate_evolutionary_policy(
            result.agent,
            config,
            episodes=args.eval_episodes,
            epsilon=0.0,
        )
        print(json.dumps(eval_results, indent=2))

        # Save evaluation results
        if args.output:
            output_path = Path(args.output)
            eval_path = output_path / "evaluation.json"
            with open(eval_path, "w", encoding="utf-8") as handle:
                json.dump(eval_results, handle, indent=2)
            print(f"Saved evaluation to {eval_path}")


if __name__ == "__main__":
    main()
