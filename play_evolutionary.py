from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rl.evolutionary_training import load_evolutionary_checkpoint
from rl.env import PacmanEnv


def _run_episode(
    env: PacmanEnv, agent, epsilon: float, max_steps: int | None
) -> float:
    """Run a single episode with the evolutionary agent."""
    observation, _ = env.reset()
    state = np.asarray(observation, dtype=np.float32)
    total_reward = 0.0

    for step in range(1, (max_steps or env.max_episode_steps) + 1):
        action = agent.select_action(state, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        state = np.asarray(next_obs, dtype=np.float32)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward


def main() -> None:
    """Render an evolved Pacman agent playing episodes."""
    parser = argparse.ArgumentParser(
        description="Render a trained evolutionary agent playing Pacman"
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to saved checkpoint (best.pt or latest.pt)",
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to render"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Exploration epsilon while rendering",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Step cap per episode (default: 10000 for full games, use --max-steps 1500 to match training)",
    )
    parser.add_argument(
        "--no-limit",
        action="store_true",
        help="Remove step limit entirely (game ends only when won/lost)",
    )
    args = parser.parse_args()

    # Load checkpoint
    agent, config = load_evolutionary_checkpoint(str(args.checkpoint))

    # Determine max steps - default to 10000 for watching full games
    if args.no_limit:
        env_max_steps = 1000000  # Effectively unlimited
    elif args.max_steps is not None:
        env_max_steps = args.max_steps
    else:
        env_max_steps = 10000  # High enough for full games

    # Create environment with rendering
    env = PacmanEnv(
        frame_skip=config.frame_skip,
        render_mode="human",
        reward_config=config.reward_config,
        max_episode_steps=env_max_steps,
        observation_mode="vector",
        include_board_in_observation=config.observation_include_board,
        seed=config.seed,
    )

    try:
        for episode in range(1, args.episodes + 1):
            reward = _run_episode(env, agent, epsilon=args.epsilon, max_steps=args.max_steps)
            print(f"Episode {episode}: reward={reward:.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
