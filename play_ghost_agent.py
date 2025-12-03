"""Quick play script to test ghost agents interactively.

Run a trained ghost agent and watch them play!
"""

import argparse
from pathlib import Path

from train_ghost_agent import load_checkpoint
from rl.ghost_env import GhostEnv, GhostEnvConfig


def play_ghost_agent(checkpoint_path: str, num_games: int = 5):
    """Play with a trained ghost agent."""
    
    print(f"Loading agent from {checkpoint_path}...")
    agent, config = load_checkpoint(checkpoint_path)
    
    env_config = GhostEnvConfig(
        frame_skip=config.frame_skip,
        max_episode_steps=config.max_episode_steps,
        reward_config=config.reward_config,
    )
    
    env = GhostEnv(env_config, render_mode="human")
    
    print("\nPress Ctrl+C to stop at any time\n")
    
    try:
        for game in range(num_games):
            print(f"\n{'='*60}")
            print(f"GAME {game + 1}/{num_games}")
            print(f"{'='*60}\n")
            
            observations, _ = env.reset()
            total_reward = 0.0
            
            for step in range(env_config.max_episode_steps):
                # Use greedy policy (no exploration)
                actions = agent.select_actions(observations, epsilon=0.0)
                observations, rewards, terminated, truncated, info = env.step(actions)
                
                total_reward += rewards.sum()
                
                if terminated or truncated:
                    if terminated and env.game_engine.player.is_eaten():
                        print(f"✓ GHOSTS WIN! Caught Pac-Man at step {step + 1}")
                    else:
                        print(f"✗ Pac-Man escaped at step {step + 1}")
                    print(f"Total reward: {total_reward:.2f}")
                    break
            
            if game < num_games - 1:
                input("\nPress Enter to start next game...")
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        env.close()
    
    print("\nThanks for playing!")


def main():
    parser = argparse.ArgumentParser(description="Play with trained ghost agent")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--games", type=int, default=5, help="Number of games to play")
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    play_ghost_agent(args.checkpoint, args.games)


if __name__ == "__main__":
    main()
