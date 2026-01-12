"""Visual demo: Watch a randomly-acting or trained agent play Pacman."""

from __future__ import annotations

import argparse
import numpy as np

from rl.env import PacmanEnv


def visual_demo(episodes: int = 2, random: bool = False):
    """Run a visual demo of Pacman gameplay.
    
    Args:
        episodes: Number of episodes to play
        random: If True, use random actions; if False, try to load trained agent
    """
    print("=" * 70)
    print("PACMAN VISUAL DEMO")
    print("=" * 70)
    print(f"\nMode: {'Random Actions' if random else 'Random Actions (Training not implemented yet)'}")
    print(f"Episodes: {episodes}")
    print("\nClose the game window to exit early.\n")
    
    # Create environment with rendering
    env = PacmanEnv(
        frame_skip=2,
        render_mode="human",  # This enables the pygame window!
        max_episode_steps=1500,
        observation_mode="vector",
        include_board_in_observation=False,
        seed=None,
    )
    
    try:
        for episode in range(1, episodes + 1):
            print(f"\nEpisode {episode}/{episodes}...")
            observation, _ = env.reset()
            state = np.asarray(observation, dtype=np.float32)
            episode_reward = 0.0
            steps = 0
            
            for step in range(env.max_episode_steps):
                # Random action for demo
                action = np.random.randint(0, len(PacmanEnv.ACTION_MEANINGS))
                
                next_obs, reward, terminated, truncated, _ = env.step(action)
                state = np.asarray(next_obs, dtype=np.float32)
                episode_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            print(f"  Reward: {episode_reward:.2f}, Steps: {steps}")
    finally:
        env.close()
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Pacman demo")
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes")
    parser.add_argument("--random", action="store_true", help="Use random actions")
    args = parser.parse_args()
    
    visual_demo(episodes=args.episodes, random=args.random)
