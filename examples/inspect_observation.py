"""Utility script to probe PacmanEnv observations without rendering."""

from __future__ import annotations

import argparse

from rl.env import PacmanEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PacmanEnv observations")
    parser.add_argument(
        "--mode",
        choices=["structured", "vector", "pack"],
        default="structured",
        help="Observation mode to request",
    )
    parser.add_argument(
        "--no-board",
        action="store_true",
        help="Exclude board consumables from the observation",
    )
    args = parser.parse_args()

    env = PacmanEnv(
        render_mode="none",
        observation_mode=args.mode,
        include_board_in_observation=not args.no_board,
    )

    try:
        obs, info = env.reset()
        print(f"reset observation type: {type(obs).__name__}")
        summarise(obs)
        print(f"info: {info}")

        obs, reward, terminated, truncated, info = env.step(0)
        print("\nstep(0) ->")
        print(f"  reward={reward} | terminated={terminated} | truncated={truncated}")
        summarise(obs)
    finally:
        env.close()


def summarise(observation) -> None:
    if hasattr(observation, "structured") and hasattr(observation, "vector"):
        print("  structured keys:", list(observation.structured.keys()))
        print("  ghosts:", len(observation.structured.get("ghosts", [])))
        print("  vector shape:", observation.vector.shape)
        return

    if isinstance(observation, dict):
        print("  keys:", list(observation.keys()))
        print("  ghosts:", len(observation.get("ghosts", [])))
        return

    if hasattr(observation, "shape"):
        print("  shape:", observation.shape)
        preview = observation[:6] if len(observation) >= 6 else observation
        print("  preview:", preview)
        return

    try:
        length = len(observation)
    except TypeError:
        length = None
    print("  length:", length)


if __name__ == "__main__":
    main()
