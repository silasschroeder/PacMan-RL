"""Quick play script to test ghost agents interactively.

Run a trained ghost agent and play as Pac-Man against them!
"""

import argparse
from pathlib import Path
import pygame

from train_ghost_agent import load_checkpoint
from rl.ghost_env import GhostEnv, GhostEnvConfig
from model.direction import Direction


def play_ghost_agent(checkpoint_path: str, num_games: int = 5):
    """Play as Pac-Man against a trained ghost agent."""
    
    print(f"Loading agent from {checkpoint_path}...")
    agent, config = load_checkpoint(checkpoint_path)
    
    env_config = GhostEnvConfig(
        frame_skip=config.frame_skip,
        max_episode_steps=None,  # No time limit for human play
        reward_config=config.reward_config,
    )
    
    env = GhostEnv(env_config, render_mode="human")
    
    print("\n" + "="*60)
    print("CONTROLS:")
    print("  Arrow Keys - Move Pac-Man")
    print("  ESC or Close Window - Quit")
    print("="*60 + "\n")
    
    try:
        for game in range(num_games):
            print(f"\nGAME {game + 1}/{num_games}")
            print(f"Use arrow keys to control Pac-Man!")
            
            observations, _ = env.reset()
            total_reward = 0.0
            running = True
            direction_command = Direction.LEFT
            
            # Update initial direction
            env.game_engine.direction_command = direction_command
            
            while running:
                # Handle pygame events for player control
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        game = num_games  # Exit all games
                        break
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            game = num_games  # Exit all games
                            break
                        if event.key == pygame.K_LEFT:
                            direction_command = Direction.LEFT
                        if event.key == pygame.K_RIGHT:
                            direction_command = Direction.RIGHT
                        if event.key == pygame.K_DOWN:
                            direction_command = Direction.DOWN
                        if event.key == pygame.K_UP:
                            direction_command = Direction.UP
                
                if not running:
                    break
                
                # Set player direction
                env.game_engine.direction_command = direction_command
                
                # Get ghost actions from RL agent
                actions = agent.select_actions(observations, epsilon=0.0)
                observations, rewards, terminated, truncated, info = env.step(actions)
                
                total_reward += rewards.sum()
                
                if terminated or truncated:
                    if terminated and env.game_engine.player.is_eaten():
                        print(f"  Game Over! Ghosts caught you!")
                    elif env.game_engine.game_over:
                        print(f"  You lost all lives!")
                    else:
                        print(f"  Time's up!")
                    print(f"  Final Score: {env.game_engine.level.score}")
                    print(f"  Ghost Reward: {total_reward:.2f}")
                    break
            
            if not running:
                break
                
            if game < num_games - 1:
                print("\nPress Enter to start next game (or Ctrl+C to quit)...")
                try:
                    input()
                except KeyboardInterrupt:
                    break
    
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
