"""Training script for ghost RL agents.

This script trains a shared DQN policy to control all 4 ghosts simultaneously.
Features:
- Parallel environment training for efficiency
- Curriculum learning for personality preservation
- Comprehensive metrics tracking
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from rl.agents.dqn import DQNConfig
from rl.agents.ghost_dqn import GhostDQNAgent, VectorizedGhostReplayBuffer
from rl.ghost_env import GhostEnv, VectorizedGhostEnv, GhostEnvConfig
from rl.ghost_reward import GhostRewardConfig


@dataclass
class GhostTrainingConfig:
    """Configuration for ghost agent training."""
    
    # Environment settings
    num_parallel_envs: int = 50
    max_episode_steps: int = 2000
    frame_skip: int = 1
    
    # Training settings
    total_timesteps: int = 1_000_000
    buffer_size: int = 100_000
    batch_size: int = 256
    warmup_steps: int = 10_000
    train_frequency: int = 4  # Train every N steps
    target_update_interval: int = 1_000
    
    # Agent settings
    gamma: float = 0.99
    learning_rate: float = 3e-4
    tau: float = 1.0  # Hard update by default
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.5  # Fraction of total timesteps to decay over
    
    # Network architecture
    hidden_sizes: tuple[int, ...] = (256, 256, 128)
    
    # Evaluation
    eval_frequency: int = 10_000  # Evaluate every N timesteps
    eval_episodes: int = 5
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: Optional[int] = 42
    
    # Reward configuration
    reward_config: GhostRewardConfig = field(default_factory=GhostRewardConfig)
    
    def get_epsilon_decay(self) -> float:
        """Calculate epsilon decay rate."""
        decay_steps = int(self.total_timesteps * self.epsilon_decay_fraction)
        return (self.epsilon_start - self.epsilon_end) / max(1, decay_steps)


def load_ghost_training_config(path: Optional[str]) -> GhostTrainingConfig:
    """Load training configuration from JSON file."""
    if path is None:
        return GhostTrainingConfig()
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Separate reward config
    reward_data = data.pop("reward_config", {})
    reward_config = GhostRewardConfig(**reward_data)
    
    # Convert hidden_sizes from list to tuple
    if "hidden_sizes" in data:
        data["hidden_sizes"] = tuple(data["hidden_sizes"])
    
    config = GhostTrainingConfig(**data)
    config.reward_config = reward_config
    
    return config


def create_agent(config: GhostTrainingConfig, observation_dim: int, action_dim: int) -> GhostDQNAgent:
    """Create a ghost DQN agent."""
    dqn_config = DQNConfig(
        state_dim=observation_dim,
        action_dim=action_dim,
        hidden_sizes=config.hidden_sizes,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        tau=config.tau,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay=config.get_epsilon_decay(),
        device=config.device,
    )
    
    return GhostDQNAgent(dqn_config)


def evaluate_agent(
    agent: GhostDQNAgent,
    config: GhostTrainingConfig,
    episodes: int = 5,
    render: bool = False,
) -> Dict[str, float]:
    """Evaluate the agent's performance."""
    env_config = GhostEnvConfig(
        frame_skip=config.frame_skip,
        max_episode_steps=config.max_episode_steps,
        reward_config=config.reward_config,
        seed=config.seed + 999999 if config.seed else None,
    )
    
    env = GhostEnv(env_config, render_mode="human" if render else "none")
    
    episode_rewards = []
    episode_lengths = []
    catches = 0
    catch_times = []  # Track time-to-catch for successful catches
    
    try:
        for ep in range(episodes):
            observations, _ = env.reset()
            episode_reward = 0.0
            steps = 0
            
            for step in range(config.max_episode_steps):
                # Use greedy policy (epsilon=0)
                actions = agent.select_actions(observations, epsilon=0.0)
                observations, rewards, terminated, truncated, info = env.step(actions)
                
                episode_reward += rewards.sum()
                steps += 1
                
                if terminated:
                    if env.game_engine.player.is_eaten():
                        catches += 1  # Ghosts caught Pacman
                        catch_times.append(steps)  # Record time-to-catch
                    break
                
                if truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
    
    finally:
        env.close()
    
    results = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "catch_rate": catches / episodes,
        "total_catches": catches,
    }
    
    # Add time-to-catch statistics (only for successful catches)
    if catch_times:
        results["mean_time_to_catch"] = float(np.mean(catch_times))
        results["std_time_to_catch"] = float(np.std(catch_times))
        results["min_time_to_catch"] = float(np.min(catch_times))
        results["max_time_to_catch"] = float(np.max(catch_times))
    else:
        results["mean_time_to_catch"] = None
        results["std_time_to_catch"] = None
        results["min_time_to_catch"] = None
        results["max_time_to_catch"] = None
    
    return results


def train_ghost_agent(
    config: GhostTrainingConfig,
    output_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> GhostDQNAgent:
    """Train a ghost agent with parallel environments.
    
    Args:
        config: Training configuration
        output_dir: Directory to save checkpoints and metrics
        resume_from: Path to checkpoint to resume training from
    """
    
    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
    
    # Create vectorized environments
    env_config = GhostEnvConfig(
        frame_skip=config.frame_skip,
        max_episode_steps=config.max_episode_steps,
        reward_config=config.reward_config,
        seed=config.seed,
    )
    
    envs = VectorizedGhostEnv(
        num_envs=config.num_parallel_envs,
        config=env_config,
        render_mode="none",
    )
    
    # Create agent
    agent = create_agent(config, envs.observation_dim, envs.action_dim)
    
    # Resume from checkpoint if specified
    starting_steps = 0
    if resume_from:
        print(f"Resuming training from {resume_from}...")
        checkpoint = torch.load(resume_from, map_location=config.device)
        agent.load_state_dict(checkpoint["agent"])
        starting_steps = checkpoint.get("total_steps", 0)
        print(f"Resumed from step {starting_steps:,}")
    
    # Create replay buffer
    buffer = VectorizedGhostReplayBuffer(
        capacity=config.buffer_size,
        state_shape=(envs.observation_dim,),
        num_ghosts=envs.num_ghosts,
    )
    
    # Tracking
    metrics: List[Dict[str, Any]] = []
    best_reward = -math.inf
    total_steps = starting_steps
    episode_count = 0
    
    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Training ghost agent with {config.num_parallel_envs} parallel environments")
    print(f"Device: {config.device}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"Observation dim: {envs.observation_dim}, Action dim: {envs.action_dim}")
    
    # Initialize environments
    observations, _ = envs.reset()
    
    try:
        while total_steps < config.total_timesteps:
            # Select actions for all environments and ghosts
            actions = np.zeros((config.num_parallel_envs, envs.num_ghosts), dtype=np.int64)
            for env_idx in range(config.num_parallel_envs):
                actions[env_idx] = agent.select_actions(observations[env_idx])
            
            # Step environments
            next_observations, rewards, terminateds, truncateds, infos = envs.step(actions)
            
            # Store experiences
            buffer.push_batch(
                observations,
                actions,
                rewards,
                next_observations,
                terminateds,
            )
            
            observations = next_observations
            total_steps += config.num_parallel_envs
            
            # Count episodes that finished
            episode_count += int(np.sum(terminateds + truncateds))
            
            # Training
            if (
                total_steps >= config.warmup_steps
                and len(buffer) >= config.batch_size
                and total_steps % config.train_frequency == 0
            ):
                loss = agent.train_step(buffer, config.batch_size, config.gamma)
                agent.decay_epsilon()
                
                # Target network update
                if total_steps % config.target_update_interval == 0:
                    agent.update_target()
            
            # Evaluation and logging
            if total_steps % config.eval_frequency == 0:
                eval_metrics = evaluate_agent(agent, config, config.eval_episodes)
                
                metric_entry = {
                    "timestep": total_steps,
                    "episode": episode_count,
                    "epsilon": agent.epsilon,
                    "buffer_size": len(buffer),
                    **eval_metrics,
                }
                
                metrics.append(metric_entry)
                
                # Format time-to-catch display
                ttc_str = ""
                if eval_metrics.get("mean_time_to_catch") is not None:
                    ttc_str = f" | TTC: {eval_metrics['mean_time_to_catch']:.0f}±{eval_metrics['std_time_to_catch']:.0f}"
                
                print(f"Step {total_steps:,}/{config.total_timesteps:,} | "
                      f"Ep {episode_count} | "
                      f"Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f} | "
                      f"Catch: {eval_metrics['catch_rate']:.1%}{ttc_str} | "
                      f"ε: {agent.epsilon:.3f}")
                
                # Save best model
                if output_path and eval_metrics["mean_reward"] > best_reward:
                    best_reward = eval_metrics["mean_reward"]
                    save_checkpoint(agent, config, output_path / "best.pt")
    
    finally:
        envs.close()
    
    # Save final model and metrics
    if output_path:
        save_checkpoint(agent, config, output_path / "latest.pt")
        
        with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nTraining complete! Saved to {output_path}")
    
    return agent


def save_checkpoint(
    agent: GhostDQNAgent,
    config: GhostTrainingConfig,
    path: Path,
) -> None:
    """Save agent checkpoint."""
    payload = {
        "agent": agent.save_state_dict(),
        "config": asdict(config),
    }
    torch.save(payload, path)


def load_checkpoint(path: str) -> tuple[GhostDQNAgent, GhostTrainingConfig]:
    """Load agent from checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    
    # Reconstruct config
    config_dict = checkpoint["config"].copy()
    reward_dict = config_dict.pop("reward_config", {})
    config_dict["hidden_sizes"] = tuple(config_dict["hidden_sizes"])
    
    config = GhostTrainingConfig(**config_dict)
    config.reward_config = GhostRewardConfig(**reward_dict)
    
    # Create agent
    # We need to know observation dim - create a temporary env
    env_config = GhostEnvConfig(reward_config=config.reward_config)
    temp_env = GhostEnv(env_config)
    observation_dim = temp_env.observation_dim
    action_dim = temp_env.action_dim
    temp_env.close()
    
    agent = create_agent(config, observation_dim, action_dim)
    agent.load_state_dict(checkpoint["agent"])
    
    return agent, config


def main(cli_args: Optional[list[str]] = None) -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ghost RL agent")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--output", type=str, default="runs/ghost_latest", help="Output directory")
    parser.add_argument("--resume", type=str, help="Checkpoint to resume training from")
    parser.add_argument("--num-envs", type=int, help="Number of parallel environments")
    parser.add_argument("--timesteps", type=int, help="Total training timesteps")
    parser.add_argument("--device", type=str, help="Device (cpu/cuda)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to load for evaluation")
    
    args = parser.parse_args(cli_args)
    
    # Load config
    config = load_ghost_training_config(args.config)
    
    # Apply CLI overrides
    if args.num_envs is not None:
        config.num_parallel_envs = args.num_envs
    if args.timesteps is not None:
        config.total_timesteps = args.timesteps
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    
    if args.eval_only:
        if not args.checkpoint:
            print("Error: --checkpoint required for --eval-only")
            return
        
        agent, config = load_checkpoint(args.checkpoint)
        print("Evaluating agent...")
        metrics = evaluate_agent(agent, config, episodes=20, render=True)
        print("\nEvaluation Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    else:
        # Train
        train_ghost_agent(config, output_dir=args.output, resume_from=args.resume)


if __name__ == "__main__":
    main()
