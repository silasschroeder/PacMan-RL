from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from rl.agents.genetic import GeneticAgent, GeneticConfig, GeneticPopulation
from rl.env import PacmanEnv, RewardConfig


@dataclass
class EvolutionaryConfig:
    """Configuration for evolutionary training."""

    generations: int = 50
    population_size: int = 100
    max_steps: int = 1500
    fitness_episodes: int = 3  # Average fitness over multiple episodes
    elite_fraction: float = 0.1
    tournament_size: int = 5
    crossover_rate: float = 0.8
    mutation_rate: float = 0.05
    mutation_std: float = 0.1
    frame_skip: int = 2
    observation_include_board: bool = False
    seed: Optional[int] = 42
    device: str = "cpu"
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    num_workers: int = 1  # Number of parallel workers (1 = no parallelization)


@dataclass
class EvolutionaryResult:
    """Results from evolutionary training."""

    metrics: List[Dict[str, Any]]
    agent: GeneticAgent
    config: EvolutionaryConfig


def load_evolutionary_config(path: Optional[str]) -> EvolutionaryConfig:
    """Load evolutionary training config from JSON file."""
    if path is None:
        return EvolutionaryConfig()

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    def build_reward_config(data: Dict[str, Any]) -> RewardConfig:
        return RewardConfig(
            score_scale=data.get("score_scale", 1.0),
            step_penalty=data.get("step_penalty", -0.1),
            pellet_reward=data.get("pellet_reward", 0.5),
            power_pellet_reward=data.get("power_pellet_reward", 10.0),
            ghost_reward=data.get("ghost_reward", 30.0),
            life_lost_penalty=data.get("life_lost_penalty", -100.0),
            death_penalty=data.get("death_penalty", -500.0),
        )

    reward_cfg = payload.get("reward_config", {})
    config_kwargs = {k: v for k, v in payload.items() if k != "reward_config"}
    config = EvolutionaryConfig(**config_kwargs)
    config.reward_config = build_reward_config(reward_cfg)
    return config


def _prepare_env(config: EvolutionaryConfig) -> PacmanEnv:
    """Create PacmanEnv from evolutionary config."""
    env = PacmanEnv(
        frame_skip=config.frame_skip,
        render_mode="none",
        reward_config=config.reward_config,
        max_episode_steps=config.max_steps,
        observation_mode="vector",
        include_board_in_observation=config.observation_include_board,
        seed=config.seed,
    )
    return env


def _make_agent(config: EvolutionaryConfig, state_dim: int, action_dim: int) -> GeneticAgent:
    """Create a GeneticAgent from evolutionary config."""
    genetic_config = GeneticConfig(
        state_dim=state_dim,
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
    return GeneticAgent(genetic_config)


def evaluate_individual(
    agent: GeneticAgent,
    genome: np.ndarray,
    config: EvolutionaryConfig,
    num_episodes: int,
) -> float:
    """Evaluate a single genome's fitness over multiple episodes.

    Args:
        agent: GeneticAgent to load genome into
        genome: Weight vector to evaluate
        config: Training configuration
        num_episodes: Number of episodes to average

    Returns:
        Average episode reward (fitness)
    """
    env = _prepare_env(config)
    agent.set_weights(genome)

    total_rewards: List[float] = []
    try:
        for _ in range(num_episodes):
            observation, _ = env.reset()
            state = np.asarray(observation, dtype=np.float32)
            episode_reward = 0.0

            for _ in range(config.max_steps):
                action = agent.select_action(state, epsilon=0.0)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                state = np.asarray(next_obs, dtype=np.float32)
                episode_reward += reward

                if done:
                    break

            total_rewards.append(episode_reward)
    finally:
        env.close()

    return float(np.mean(total_rewards))


def _evaluate_genome_worker(args: tuple) -> float:
    """Worker function for parallel genome evaluation.
    
    Args:
        args: Tuple of (genome, config_dict, state_dim, action_dim, num_episodes)
    
    Returns:
        Fitness value
    """
    genome, config_dict, state_dim, action_dim, num_episodes = args
    
    # Reconstruct config
    reward_dict = config_dict.pop("reward_config", {})
    config = EvolutionaryConfig(**config_dict)
    config.reward_config = RewardConfig(**reward_dict)
    
    # Create agent
    genetic_config = GeneticConfig(
        state_dim=state_dim,
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
    
    # Evaluate
    return evaluate_individual(agent, genome, config, num_episodes)


def evaluate_population_parallel(
    population: GeneticPopulation,
    config: EvolutionaryConfig,
    state_dim: int,
    action_dim: int,
    num_workers: int,
) -> None:
    """Evaluate entire population in parallel.
    
    Args:
        population: Population to evaluate
        config: Training configuration
        state_dim: State dimension
        action_dim: Action dimension
        num_workers: Number of parallel workers
    """
    # Prepare config dict for serialization
    config_dict = asdict(config)
    config_dict.pop("num_workers", None)  # Remove to avoid confusion
    
    # Prepare arguments for each genome
    args_list = [
        (
            population.genomes[idx],
            config_dict.copy(),
            state_dim,
            action_dim,
            config.fitness_episodes,
        )
        for idx in range(config.population_size)
    ]
    
    # Evaluate in parallel
    with mp.Pool(processes=num_workers) as pool:
        fitnesses = pool.map(_evaluate_genome_worker, args_list)
    
    # Update population fitnesses
    for idx, fitness in enumerate(fitnesses):
        population.update_fitness(idx, fitness)


def run_evolutionary_training(
    config: EvolutionaryConfig, output_dir: Optional[str] = None
) -> EvolutionaryResult:
    """Run evolutionary training using genetic algorithm.

    Args:
        config: Training configuration
        output_dir: Optional directory to save checkpoints and metrics

    Returns:
        EvolutionaryResult with metrics, final agent, and config
    """
    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    # Initialize environment to get dimensions
    env = _prepare_env(config)
    observation, _ = env.reset()
    state = np.asarray(observation, dtype=np.float32)
    action_dim = len(PacmanEnv.ACTION_MEANINGS)
    env.close()

    # Create agent and population
    agent = _make_agent(config, state_dim=state.size, action_dim=action_dim)
    population = GeneticPopulation(agent.config, agent)

    metrics: List[Dict[str, Any]] = []
    best_fitness_ever = -float("inf")
    output_path: Optional[Path] = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    print(f"Starting evolutionary training: {config.generations} generations, population {config.population_size}")
    if config.num_workers > 1:
        print(f"Using {config.num_workers} parallel workers for evaluation")

    for generation in range(1, config.generations + 1):
        print(f"\n=== Generation {generation}/{config.generations} ===")

        # Evaluate all individuals in population
        if config.num_workers > 1:
            # Parallel evaluation
            print(f"  Evaluating {config.population_size} individuals in parallel...")
            evaluate_population_parallel(
                population, config, state.size, action_dim, config.num_workers
            )
            print(f"  Evaluated {config.population_size}/{config.population_size} individuals")
        else:
            # Sequential evaluation
            for idx in range(config.population_size):
                genome = population.genomes[idx]
                fitness = evaluate_individual(
                    agent, genome, config, num_episodes=config.fitness_episodes
                )
                population.update_fitness(idx, fitness)

                if (idx + 1) % 10 == 0:
                    print(f"  Evaluated {idx + 1}/{config.population_size} individuals", end="\r")

        # Gather statistics
        stats = population.get_population_stats()
        print(f"\n  Fitness: best={stats['best']:.2f}, mean={stats['mean']:.2f}, "
              f"worst={stats['worst']:.2f}, std={stats['std']:.2f}")

        # Track best individual
        if population.best_fitness > best_fitness_ever:
            best_fitness_ever = population.best_fitness
            if output_path:
                agent.set_weights(population.best_genome)
                agent.generation = generation
                save_evolutionary_checkpoint(agent, config, output_path / "best.pt")
                print(f"  ✓ New best fitness: {best_fitness_ever:.2f}")

        # Record metrics
        generation_metrics = {
            "generation": generation,
            "best_fitness": stats["best"],
            "mean_fitness": stats["mean"],
            "worst_fitness": stats["worst"],
            "std_fitness": stats["std"],
            "median_fitness": stats["median"],
            "best_fitness_ever": best_fitness_ever,
        }
        metrics.append(generation_metrics)

        # Evolve to next generation (except last)
        if generation < config.generations:
            population.evolve()

    # Load best genome into agent
    if population.best_genome is not None:
        agent.set_weights(population.best_genome)

    # Save final checkpoint and metrics
    if output_path:
        save_evolutionary_checkpoint(agent, config, output_path / "latest.pt")
        metrics_path = output_path / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"\nSaved metrics to {metrics_path}")

    print(f"\n=== Training Complete ===")
    print(f"Best fitness achieved: {best_fitness_ever:.2f}")

    return EvolutionaryResult(metrics=metrics, agent=agent, config=config)


def evaluate_evolutionary_policy(
    agent: GeneticAgent,
    config: EvolutionaryConfig,
    episodes: Optional[int] = None,
    max_steps: Optional[int] = None,
    epsilon: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate an evolved agent over multiple episodes.

    Args:
        agent: Trained GeneticAgent
        config: Training configuration
        episodes: Number of evaluation episodes (default: 10)
        max_steps: Max steps per episode (default: config.max_steps)
        epsilon: Exploration epsilon (typically 0.0)

    Returns:
        Dictionary with evaluation statistics
    """
    env = _prepare_env(config)
    eval_episodes = episodes if episodes is not None else 10
    eval_max_steps = max_steps if max_steps is not None else config.max_steps

    rewards: List[float] = []
    lengths: List[int] = []

    try:
        for _ in range(eval_episodes):
            observation, _ = env.reset()
            state = np.asarray(observation, dtype=np.float32)
            total_reward = 0.0
            step = 0

            for step in range(1, eval_max_steps + 1):
                action = agent.select_action(state, epsilon)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = np.asarray(next_obs, dtype=np.float32)
                total_reward += reward
                if done:
                    break

            rewards.append(total_reward)
            lengths.append(step)
    finally:
        env.close()

    return {
        "episodes": len(rewards),
        "average_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "average_length": float(np.mean(lengths)),
    }


def save_evolutionary_checkpoint(
    agent: GeneticAgent, config: EvolutionaryConfig, path: Path
) -> None:
    """Save evolutionary agent checkpoint.

    Args:
        agent: GeneticAgent to save
        config: Training configuration
        path: File path for checkpoint
    """
    payload = {
        "agent": agent.save_state_dict(),
        "config": asdict(config),
    }
    torch.save(payload, path)


def load_evolutionary_checkpoint(path: str) -> tuple[GeneticAgent, EvolutionaryConfig]:
    """Load evolutionary agent from checkpoint.

    Args:
        path: Path to checkpoint file

    Returns:
        Tuple of (GeneticAgent, EvolutionaryConfig)
    """
    checkpoint = torch.load(path, map_location="cpu")
    cfg_dict = checkpoint["config"].copy()
    reward_dict = cfg_dict.pop("reward_config", {})
    config = EvolutionaryConfig(**cfg_dict)
    config.reward_config = RewardConfig(**reward_dict)

    # Get state dimensions
    env = _prepare_env(config)
    observation, _ = env.reset()
    state_dim = int(np.asarray(observation).size)
    env.close()

    agent = _make_agent(config, state_dim=state_dim, action_dim=len(PacmanEnv.ACTION_MEANINGS))
    agent.load_state_dict(checkpoint["agent"])
    return agent, config
