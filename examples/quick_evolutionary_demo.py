"""Example: Quick evolutionary algorithm demo with small population."""

from __future__ import annotations

import numpy as np
import torch

from rl.env import PacmanEnv
from rl.evolutionary_training import EvolutionaryConfig, run_evolutionary_training


def quick_evolutionary_demo():
    """Run a quick GA demo with small population and few generations."""
    print("=" * 70)
    print("QUICK EVOLUTIONARY ALGORITHM DEMO")
    print("=" * 70)
    print("\nThis is a minimal example to demonstrate GA training.")
    print("Using small population (20) and few generations (5) for speed.\n")
    
    # Configure for quick demo
    config = EvolutionaryConfig(
        generations=5,
        population_size=20,
        max_steps=500,
        fitness_episodes=2,
        elite_fraction=0.15,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=0.05,
        mutation_std=0.1,
        frame_skip=3,
        observation_include_board=False,
        seed=42,
        device="cpu",
    )
    
    print("Configuration:")
    print(f"  Generations:      {config.generations}")
    print(f"  Population size:  {config.population_size}")
    print(f"  Max steps:        {config.max_steps}")
    print(f"  Fitness episodes: {config.fitness_episodes}")
    print(f"  Elite fraction:   {config.elite_fraction}")
    print(f"  Mutation rate:    {config.mutation_rate}")
    print()
    
    # Run training
    result = run_evolutionary_training(config, output_dir=None)
    
    # Print summary
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    
    if result.metrics:
        print("\nFitness progression:")
        for m in result.metrics:
            print(f"  Gen {m['generation']}: "
                  f"best={m['best_fitness']:.2f}, "
                  f"mean={m['mean_fitness']:.2f}, "
                  f"diversity={m['std_fitness']:.2f}")
        
        final = result.metrics[-1]
        print(f"\nFinal generation statistics:")
        print(f"  Best fitness:      {final['best_fitness']:.2f}")
        print(f"  Mean fitness:      {final['mean_fitness']:.2f}")
        print(f"  Worst fitness:     {final['worst_fitness']:.2f}")
        print(f"  Population spread: {final['std_fitness']:.2f}")
        print(f"  Best ever:         {final['best_fitness_ever']:.2f}")
    
    # Now show a visual demo of the best agent
    print("\n" + "=" * 70)
    print("VISUAL DEMO - Watch the best evolved agent play!")
    print("=" * 70)
    print("\nPlaying 2 episodes with the best agent...")
    print("Close the game window or press ESC to end early.\n")
    
    # Create environment with rendering
    env = PacmanEnv(
        frame_skip=config.frame_skip,
        render_mode="human",
        reward_config=config.reward_config,
        max_episode_steps=config.max_steps,
        observation_mode="vector",
        include_board_in_observation=config.observation_include_board,
        seed=None,  # Random episodes
    )
    
    try:
        for episode in range(1, 3):
            print(f"Episode {episode}/2...")
            observation, _ = env.reset()
            state = np.asarray(observation, dtype=np.float32)
            episode_reward = 0.0
            
            for step in range(config.max_steps):
                action = result.agent.select_action(state, epsilon=0.0)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                state = np.asarray(next_obs, dtype=np.float32)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            print(f"  Episode {episode} reward: {episode_reward:.2f}")
    finally:
        env.close()
    
    print()
    print("=" * 70)
    print("\nTo run a full training session:")
    print("  python train_evolutionary.py --generations 50 --output runs/ga_full")
    print()


if __name__ == "__main__":
    quick_evolutionary_demo()
