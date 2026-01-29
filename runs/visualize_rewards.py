"""Plot episode rewards from one or more metrics.json files.

Usage:
    python visualize_rewards.py --metrics-path runs/exp14/metrics.json
    python visualize_rewards.py --metrics-path runs/exp14/metrics.json --metrics-path runs/exp15/metrics.json
    python visualize_rewards.py --metrics-path runs/exp14/metrics.json --metrics-path runs/exp15/metrics.json \
        --labels exp14 exp15 --save-path runs/comparison.png

Requires matplotlib (pip install matplotlib) and supports optional file output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize per-episode rewards from training metrics.")
    parser.add_argument(
        "--metrics-path",
        dest="metrics_paths",
        type=Path,
        action="append",
        help="Path to a metrics JSON file (repeat for multiple runs). Defaults to runs/exp14/metrics.json if omitted.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Optional path to save the generated plot as an image (e.g., PNG). If omitted, the plot is only shown.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels for each metrics path (must match the number of --metrics-path arguments).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Episode Rewards",
        help="Custom title for the plot.",
    )
    parser.add_argument(
        "--moving-average-window",
        type=int,
        default=50,
        help="Window size for plotting a moving average over rewards (<=1 disables it).",
    )
    args = parser.parse_args(argv)

    if not args.metrics_paths:
        args.metrics_paths = [Path("runs/exp14/metrics.json")]

    if args.labels and len(args.labels) != len(args.metrics_paths):
        parser.error("Number of labels must match number of --metrics-path entries.")

    return args


def load_metrics(path: Path) -> tuple[list[int], list[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    episodes = []
    rewards = []
    for entry in data:
        if "episode" in entry and "reward" in entry or "generation" in entry and "median_fitness" in entry:
            try:
                episodes.append(int(entry["episode"]))
            except:
                episodes.append(int(entry["generation"]))
            try:
                rewards.append(float(entry["reward"]))
            except:
                rewards.append(float(entry["mean_fitness"]))

    if not episodes:
        raise ValueError(f"No episode/reward pairs found in {path}")

    return episodes, rewards


def moving_average(values: Sequence[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)
    if window > len(values):
        return []
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    averages = (cumsum[window:] - cumsum[:-window]) / window
    return averages.tolist()


def plot_rewards(
    series: Sequence[tuple[str, Sequence[int], Sequence[float]]],
    title: str,
    moving_average_window: int,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, episodes, rewards in series:
        ax.plot(episodes, rewards, linewidth=1.5, label=label)
        if moving_average_window > 1:
            ma_values = moving_average(rewards, moving_average_window)
            if ma_values:
                ma_episodes = episodes[moving_average_window - 1 :]
                ax.plot(
                    ma_episodes,
                    ma_values,
                    linewidth=2.0,
                    linestyle="--",
                    label=f"{label} (MA{moving_average_window})",
                )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    if len(series) > 1 or moving_average_window > 1:
        ax.legend()
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    labels = args.labels or []

    series: list[tuple[str, list[int], list[float]]] = []
    for idx, path in enumerate(args.metrics_paths):
        episodes, rewards = load_metrics(path)
        if labels:
            label = labels[idx]
        else:
            label = path.parent.name or path.stem
        series.append((label, episodes, rewards))

    fig = plot_rewards(series, args.title, args.moving_average_window)

    if args.save_path:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_path, dpi=150, bbox_inches="tight")
        print(f"Saved reward plot to {args.save_path}")

    plt.show()


if __name__ == "__main__":
    main()
