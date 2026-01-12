"""Plot episode rewards from a metrics.json file.

Usage:
    python visualize_rewards.py --metrics-path runs/exp14/metrics.json
    python visualize_rewards.py --metrics-path runs/exp14/metrics.json --save-path runs/exp14/reward_curve.png

Requires matplotlib (pip install matplotlib) and supports optional file output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize per-episode rewards from training metrics.")
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("runs/exp14/metrics.json"),
        help="Path to the metrics JSON file exported during training.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Optional path to save the generated plot as an image (e.g., PNG). If omitted, the plot is only shown.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Episode Rewards",
        help="Custom title for the plot.",
    )
    return parser.parse_args(argv)


def load_metrics(path: Path) -> tuple[list[int], list[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    episodes = []
    rewards = []
    for entry in data:
        if "episode" in entry and "reward" in entry:
            episodes.append(int(entry["episode"]))
            rewards.append(float(entry["reward"]))

    if not episodes:
        raise ValueError(f"No episode/reward pairs found in {path}")

    return episodes, rewards


def plot_rewards(episodes: Sequence[int], rewards: Sequence[float], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, color="tab:blue", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    episodes, rewards = load_metrics(args.metrics_path)
    fig = plot_rewards(episodes, rewards, args.title)

    if args.save_path:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_path, dpi=150, bbox_inches="tight")
        print(f"Saved reward plot to {args.save_path}")

    plt.show()


if __name__ == "__main__":
    main()
