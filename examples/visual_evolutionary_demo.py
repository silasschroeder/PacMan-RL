"""Visual Evolutionary Demo: Watch multiple agents from the population play."""

from __future__ import annotations

import numpy as np
import torch

from rl.agents.genetic import GeneticAgent, GeneticConfig, GeneticPopulation
from rl.env import PacmanEnv
from rl.evolutionary_training import EvolutionaryConfig


def visual_evolutionary_demo():
    """Run GA training with visual evaluation of select individuals."""
    print("=" * 70)
    print("VISUAL EVOLUTIONARY ALGORITHM DEMO")
    print("=" * 70)
    print("\nThis demo shows you different agents from the population playing.")
    print("Watch how strategies evolve across generations!\n")
    
    # Configure for visual demo
    config = EvolutionaryConfig(
        generations=3,  # Fewer generations for demo
        population_size=10,  # Smaller population for speed
        max_steps=300,  # Shorter episodes
        fitness_episodes=1,  # Just 1 episode per evaluation for speed
        elite_fraction=0.2,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=0.05,
        mutation_std=0.1,
        frame_skip=2,
        observation_include_board=False,
        seed=42,
        device="cpu",
    )
    
    print("Configuration:")
    print(f"  Generations:      {config.generations}")
    print(f"  Population size:  {config.population_size}")
    print(f"  Max steps:        {config.max_steps}")
    print()
    
    # Initialize agent and population
    env = PacmanEnv(
        frame_skip=config.frame_skip,
        render_mode="none",  # Headless for initialization
        reward_config=config.reward_config,
        max_episode_steps=config.max_steps,
        observation_mode="vector",
        include_board_in_observation=config.observation_include_board,
        seed=config.seed,
    )
    
    observation, _ = env.reset()
    state = np.asarray(observation, dtype=np.float32)
    action_dim = len(PacmanEnv.ACTION_MEANINGS)
    env.close()
    
    # Create agent and population
    genetic_config = GeneticConfig(
        state_dim=state.size,
        action_dim=action_dim,
        hidden_sizes=(256, 256),
        population_size=config.population_size,
        elite_fraction=config.elite_fraction,
        tournament_size=config.tournament_size,
        crossover_rate=config.crossover_rate,
        mutation_rate=config.mutation_rate,
        mutation_std=config.mutation_std,
        device=config.device,
    )
    agent = GeneticAgent(genetic_config)
    population = GeneticPopulation(genetic_config, agent)
    
    print("=" * 70)
    print("Starting evolution with VISUAL evaluation of select individuals")
    print("=" * 70)
    
    # Create rendering environment (reused)
    render_env = PacmanEnv(
        frame_skip=config.frame_skip,
        render_mode="human",
        reward_config=config.reward_config,
        max_episode_steps=config.max_steps,
        observation_mode="vector",
        include_board_in_observation=config.observation_include_board,
        seed=None,
    )
    
    try:
        for generation in range(1, config.generations + 1):
            print(f"\n{'=' * 70}")
            print(f"GENERATION {generation}/{config.generations}")
            print(f"{'=' * 70}")
            
            # Evaluate all individuals (headless for most)
            eval_env = PacmanEnv(
                frame_skip=config.frame_skip,
                render_mode="none",
                reward_config=config.reward_config,
                max_episode_steps=config.max_steps,
                observation_mode="vector",
                include_board_in_observation=config.observation_include_board,
                seed=config.seed,
            )
            
            # Indices to show visually (first, middle, last)
            visual_indices = [0, config.population_size // 2, config.population_size - 1]
            
            for idx in range(config.population_size):
                genome = population.genomes[idx]
                
                # Decide if we show this one visually
                show_visual = idx in visual_indices
                current_env = render_env if show_visual else eval_env
                
                if show_visual:
                    print(f"\nShowing Individual {idx + 1}/{config.population_size} VISUALLY...")
                
                agent.set_weights(genome)
                observation, _ = current_env.reset()
                state = np.asarray(observation, dtype=np.float32)
                episode_reward = 0.0
                
                for step in range(config.max_steps):
                    action = agent.select_action(state, epsilon=0.0)
                    next_obs, reward, terminated, truncated, _ = current_env.step(action)
                    state = np.asarray(next_obs, dtype=np.float32)
                    episode_reward += reward
                    
                    if terminated or truncated:
                        break
                
                population.update_fitness(idx, episode_reward)
                
                if show_visual:
                    print(f"  Individual {idx + 1} fitness: {episode_reward:.2f}")
                elif (idx + 1) % 3 == 0:
                    print(f"  Evaluated {idx + 1}/{config.population_size} individuals (headless)...", end="\r")
            
            eval_env.close()
            
            # Show generation statistics
            stats = population.get_population_stats()
            print(f"\n\nGeneration {generation} Results:")
            print(f"  Best fitness:   {stats['best']:.2f}")
            print(f"  Mean fitness:   {stats['mean']:.2f}")
            print(f"  Worst fitness:  {stats['worst']:.2f}")
            print(f"  Diversity:      {stats['std']:.2f}")
            
            # Evolve to next generation (except last)
            if generation < config.generations:
                print(f"\nEvolving to generation {generation + 1}...")
                population.evolve()
        
    finally:
        render_env.close()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print(f"\nBest fitness achieved: {population.best_fitness:.2f}")
    print("\nYou saw 3 different agents per generation playing visually.")
    print("The rest were evaluated in the background (headless) for speed.")
    print("\nTo run full training without visual interruption:")
    print("  python train_evolutionary.py --generations 50 --output runs/ga_full")
    print()


if __name__ == "__main__":
    visual_evolutionary_demo()
