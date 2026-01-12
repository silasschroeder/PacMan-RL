"""Utility script to compare DQN and Evolutionary Algorithm results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_metrics(path: Path) -> List[Dict[str, Any]]:
    """Load metrics JSON from a training run."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_dqn_stats(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics from DQN training metrics."""
    rewards = [m["reward"] for m in metrics]
    steps = [m["steps"] for m in metrics]
    
    return {
        "total_episodes": len(metrics),
        "final_reward": rewards[-1] if rewards else 0,
        "max_reward": max(rewards) if rewards else 0,
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0,
        "mean_steps": sum(steps) / len(steps) if steps else 0,
        "final_epsilon": metrics[-1].get("epsilon", 0) if metrics else 0,
    }


def compute_ga_stats(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics from GA training metrics."""
    best_fitnesses = [m["best_fitness"] for m in metrics]
    mean_fitnesses = [m["mean_fitness"] for m in metrics]
    
    return {
        "total_generations": len(metrics),
        "final_best_fitness": best_fitnesses[-1] if best_fitnesses else 0,
        "max_fitness_ever": max(m["best_fitness_ever"] for m in metrics) if metrics else 0,
        "final_mean_fitness": mean_fitnesses[-1] if mean_fitnesses else 0,
        "mean_best_fitness": sum(best_fitnesses) / len(best_fitnesses) if best_fitnesses else 0,
        "final_diversity": metrics[-1].get("std_fitness", 0) if metrics else 0,
    }


def main() -> None:
    """Compare DQN and GA training results."""
    parser = argparse.ArgumentParser(
        description="Compare DQN and Evolutionary Algorithm results"
    )
    parser.add_argument(
        "--dqn-metrics",
        type=Path,
        help="Path to DQN metrics.json",
        required=True,
    )
    parser.add_argument(
        "--ga-metrics",
        type=Path,
        help="Path to GA metrics.json",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save comparison JSON",
        default=None,
    )
    args = parser.parse_args()

    # Load metrics
    dqn_metrics = load_metrics(args.dqn_metrics)
    ga_metrics = load_metrics(args.ga_metrics)

    # Compute statistics
    dqn_stats = compute_dqn_stats(dqn_metrics)
    ga_stats = compute_ga_stats(ga_metrics)

    # Create comparison
    comparison = {
        "dqn": dqn_stats,
        "genetic_algorithm": ga_stats,
        "comparison": {
            "dqn_final_reward": dqn_stats["final_reward"],
            "ga_final_best_fitness": ga_stats["final_best_fitness"],
            "dqn_max_reward": dqn_stats["max_reward"],
            "ga_max_fitness": ga_stats["max_fitness_ever"],
            "winner_by_final": "DQN" if dqn_stats["final_reward"] > ga_stats["final_best_fitness"] else "GA",
            "winner_by_max": "DQN" if dqn_stats["max_reward"] > ga_stats["max_fitness_ever"] else "GA",
        }
    }

    # Print comparison
    print("=" * 60)
    print("TRAINING COMPARISON: DQN vs. Genetic Algorithm")
    print("=" * 60)
    print("\nDQN Statistics:")
    print(f"  Total episodes:     {dqn_stats['total_episodes']}")
    print(f"  Final reward:       {dqn_stats['final_reward']:.2f}")
    print(f"  Max reward:         {dqn_stats['max_reward']:.2f}")
    print(f"  Mean reward:        {dqn_stats['mean_reward']:.2f}")
    print(f"  Mean steps:         {dqn_stats['mean_steps']:.1f}")
    print(f"  Final epsilon:      {dqn_stats['final_epsilon']:.4f}")
    
    print("\nGenetic Algorithm Statistics:")
    print(f"  Total generations:  {ga_stats['total_generations']}")
    print(f"  Final best fitness: {ga_stats['final_best_fitness']:.2f}")
    print(f"  Max fitness ever:   {ga_stats['max_fitness_ever']:.2f}")
    print(f"  Final mean fitness: {ga_stats['final_mean_fitness']:.2f}")
    print(f"  Mean best fitness:  {ga_stats['mean_best_fitness']:.2f}")
    print(f"  Final diversity:    {ga_stats['final_diversity']:.2f}")
    
    print("\nComparison:")
    print(f"  Winner (final):     {comparison['comparison']['winner_by_final']}")
    print(f"    DQN final:        {dqn_stats['final_reward']:.2f}")
    print(f"    GA final:         {ga_stats['final_best_fitness']:.2f}")
    print(f"  Winner (peak):      {comparison['comparison']['winner_by_max']}")
    print(f"    DQN max:          {dqn_stats['max_reward']:.2f}")
    print(f"    GA max:           {ga_stats['max_fitness_ever']:.2f}")
    print("=" * 60)

    # Save comparison if output specified
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to {args.output}")


if __name__ == "__main__":
    main()
