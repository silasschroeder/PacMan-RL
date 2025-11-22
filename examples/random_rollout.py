"""Minimal smoke test for the PacmanEnv wrapper.

Run with:
    python -m examples.random_rollout --episodes 2 --steps 200
"""

from __future__ import annotations

import argparse
import random
from typing import Optional

from rl.env import PacmanEnv, RewardConfig


def run_episode(env: PacmanEnv, max_steps: int, render: bool = False, log_obs: bool = False) -> None:
    observation, info = env.reset()
    total_reward = 0.0

    if render:
        env.render_mode = "human"

    if log_obs:
        _summarise_observation("reset", observation)

    for step in range(max_steps):
        action = random.randint(0, 4)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if log_obs and step == 0:
            _summarise_observation("step", observation)

        if render:
            env.render()

        if terminated or truncated:
            break

    print(
        f"Episode finished after {step + 1} steps | score={info['score']} | "
        f"reward={total_reward:.1f} | game_over={info['powerup_active'] and 'powerup'}"
    )


def main(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run random rollouts in PacmanEnv")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument(
        "--render", action="store_true", help="Render gameplay (slows down execution)"
    )
    parser.add_argument(
        "--obs-mode",
        choices=["structured", "vector", "pack"],
        default="structured",
        help="Select observation return format",
    )
    parser.add_argument(
        "--no-board",
        action="store_true",
        help="Exclude flattened board consumables from observations",
    )
    parser.add_argument(
        "--log-obs",
        action="store_true",
        help="Print summaries of the reset and first-step observations",
    )
    parsed = parser.parse_args(args=args)

    env = PacmanEnv(
        frame_skip=2,
        render_mode="human" if parsed.render else "none",
        reward_config=RewardConfig(pellet_reward=1.0, power_pellet_reward=5.0, ghost_reward=25.0),
        max_episode_steps=parsed.steps,
        observation_mode=parsed.obs_mode,
        include_board_in_observation=not parsed.no_board,
    )

    try:
        for episode in range(parsed.episodes):
            print(f"\n=== Episode {episode + 1}/{parsed.episodes} ===")
            run_episode(env, parsed.steps, render=parsed.render, log_obs=parsed.log_obs)
    finally:
        env.close()


if __name__ == "__main__":
    main()


def _summarise_observation(stage: str, observation) -> None:
    """Print lightweight details about the observation structure."""

    print(f"[{stage}] observation type: {type(observation).__name__}")

    if hasattr(observation, "structured") and hasattr(observation, "vector"):
        struct = observation.structured
        vector = observation.vector
        print(f"  structured keys: {list(struct.keys())}")
        print(f"  ghosts: {len(struct.get('ghosts', []))} entries | board included: {'board_consumables' in struct}")
        print(f"  vector shape: {vector.shape}")
        return

    if isinstance(observation, dict):
        print(f"  keys: {list(observation.keys())}")
        ghosts = observation.get("ghosts", [])
        print(f"  ghosts: {len(ghosts)} | board included: {'board_consumables' in observation}")
        return

    if hasattr(observation, "shape"):
        print(f"  shape: {observation.shape}")
        preview = observation[:5] if len(observation) >= 5 else observation
        print(f"  preview: {preview}")
        return

    try:
        length = len(observation)
    except TypeError:
        length = None
    if length is not None:
        print(f"  length: {length}")
