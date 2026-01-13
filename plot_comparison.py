"""Optional visualization script for comparing DQN and GA learning curves.

Requires matplotlib (not in requirements.txt). Install with:
    pip install matplotlib

Usage:
    python plot_comparison.py --dqn runs/dqn_exp/metrics.json --ga runs/ga_exp/metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot_comparison(dqn_metrics_path: Path, ga_metrics_path: Path, output_path: Path | None = None):
    """Plot DQN and GA learning curves for comparison."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Error: matplotlib not installed. Install with: pip install matplotlib")
        return

    # Load metrics
    with open(dqn_metrics_path, "r") as f:
        dqn_metrics = json.load(f)
    
    with open(ga_metrics_path, "r") as f:
        ga_metrics = json.load(f)

    # Extract data
    dqn_episodes = [m["episode"] for m in dqn_metrics]
    dqn_rewards = [m["reward"] for m in dqn_metrics]
    
    ga_generations = [m["generation"] for m in ga_metrics]
    ga_best_fitness = [m["best_fitness"] for m in ga_metrics]
    ga_mean_fitness = [m["mean_fitness"] for m in ga_metrics]

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Learning curves
    ax1.plot(dqn_episodes, dqn_rewards, label="DQN Episode Reward", alpha=0.6, linewidth=1)
    ax1.plot(dqn_episodes, _moving_average(dqn_rewards, 10), 
             label="DQN (MA-10)", linewidth=2, color='blue')
    
    ax1.plot(ga_generations, ga_best_fitness, label="GA Best Fitness", alpha=0.6, linewidth=1)
    ax1.plot(ga_generations, _moving_average(ga_best_fitness, 5), 
             label="GA Best (MA-5)", linewidth=2, color='red')
    
    ax1.set_xlabel("Episode / Generation")
    ax1.set_ylabel("Reward / Fitness")
    ax1.set_title("Learning Curves: DQN vs. Genetic Algorithm")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: GA population statistics
    ax2.plot(ga_generations, ga_best_fitness, label="Best", linewidth=2)
    ax2.plot(ga_generations, ga_mean_fitness, label="Mean", linewidth=2)
    ax2.fill_between(
        ga_generations,
        [m["mean_fitness"] - m["std_fitness"] for m in ga_metrics],
        [m["mean_fitness"] + m["std_fitness"] for m in ga_metrics],
        alpha=0.3,
        label="±1 Std Dev"
    )
    
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Fitness")
    ax2.set_title("GA Population Statistics")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def _moving_average(data: list[float], window: int) -> list[float]:
    """Compute moving average with given window size."""
    if len(data) < window:
        return data
    
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result


def main():
    """Main entry point for plotting script."""
    parser = argparse.ArgumentParser(
        description="Plot DQN vs GA learning curves (requires matplotlib)"
    )
    parser.add_argument(
        "--dqn",
        type=Path,
        required=True,
        help="Path to DQN metrics.json",
    )
    parser.add_argument(
        "--ga",
        type=Path,
        required=True,
        help="Path to GA metrics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (e.g., comparison.png). If not provided, shows plot.",
    )
    args = parser.parse_args()
    
    if not args.dqn.exists():
        print(f"Error: DQN metrics file not found: {args.dqn}")
        return
    
    if not args.ga.exists():
        print(f"Error: GA metrics file not found: {args.ga}")
        return
    
    plot_comparison(args.dqn, args.ga, args.output)


if __name__ == "__main__":
    main()
