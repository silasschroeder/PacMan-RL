"""Evaluation script for trained ghost agents.

Evaluate ghost agent performance with visualization and detailed metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from train_ghost_agent import load_checkpoint, GhostTrainingConfig
from rl.ghost_env import GhostEnv, GhostEnvConfig


def evaluate_ghost_agent(
    agent,
    config: GhostTrainingConfig,
    num_episodes: int = 10,
    render: bool = True,
    max_steps: Optional[int] = None,
) -> dict:
    """Evaluate a trained ghost agent.
    
    Args:
        agent: Trained ghost DQN agent
        config: Training configuration
        num_episodes: Number of episodes to evaluate
        render: Whether to render the game
        max_steps: Maximum steps per episode (uses config default if None)
    
    Returns:
        Dictionary with evaluation metrics
    """
    env_config = GhostEnvConfig(
        frame_skip=config.frame_skip,
        max_episode_steps=max_steps or config.max_episode_steps,
        reward_config=config.reward_config,
        seed=config.seed + 99999 if config.seed else None,
    )
    
    env = GhostEnv(env_config, render_mode="human" if render else "none")
    
    # Metrics
    episode_rewards = []
    episode_lengths = []
    pacman_catches = 0
    catch_times = []  # Time-to-catch for successful catches
    ghost_deaths = 0
    powerup_activations = 0
    
    # Per-ghost metrics
    ghost_names = ["blinky", "pinky", "inky", "clyde"]
    ghost_rewards = {name: [] for name in ghost_names}
    personality_adherence = {name: [] for name in ghost_names}
    
    try:
        for episode in range(num_episodes):
            observations, info = env.reset()
            episode_reward = 0.0
            steps = 0
            episode_ghost_rewards = {name: 0.0 for name in ghost_names}
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            for step in range(env_config.max_episode_steps):
                # Use greedy policy
                actions = agent.select_actions(observations, epsilon=0.0)
                
                next_observations, rewards, terminated, truncated, info = env.step(actions)
                
                # Track rewards
                episode_reward += rewards.sum()
                for i, name in enumerate(ghost_names):
                    episode_ghost_rewards[name] += rewards[i]
                
                # Track personality adherence from info
                for i, name in enumerate(ghost_names):
                    if f"{name}_reward" in info:
                        reward_breakdown = info[f"{name}_reward"]
                        if "personality_reward" in reward_breakdown:
                            personality_adherence[name].append(
                                reward_breakdown["personality_reward"]
                            )
                
                # Track events
                if "powerup_active" in info and info.get("powerup_active"):
                    powerup_activations += 1
                
                observations = next_observations
                steps += 1
                
                if terminated:
                    # Check who won
                    if env.game_engine.player.is_eaten():
                        pacman_catches += 1
                        catch_times.append(steps)  # Record time-to-catch
                        print(f"  Ghosts won! Caught Pacman at step {steps}")
                    else:
                        print(f"  Pacman won! Ghosts failed at step {steps}")
                    
                    # Count ghost deaths in this episode
                    for ghost in env.game_engine.ghosts:
                        if ghost.is_eaten():
                            ghost_deaths += 1
                    
                    break
                
                if truncated:
                    print(f"  Episode truncated at step {steps}")
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            for name in ghost_names:
                ghost_rewards[name].append(episode_ghost_rewards[name])
            
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Individual rewards: " + 
                  ", ".join([f"{name}: {episode_ghost_rewards[name]:.2f}" 
                            for name in ghost_names]))
    
    finally:
        env.close()
    
    # Compile results
    results = {
        "episodes": num_episodes,
        "mean_total_reward": float(np.mean(episode_rewards)),
        "std_total_reward": float(np.std(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "pacman_catch_rate": pacman_catches / num_episodes,
        "total_catches": pacman_catches,
        "ghost_deaths": ghost_deaths,
        "avg_ghost_deaths_per_episode": ghost_deaths / num_episodes,
        "powerup_activations": powerup_activations,
    }
    
    # Add time-to-catch statistics (only for successful catches)
    if catch_times:
        results["mean_time_to_catch"] = float(np.mean(catch_times))
        results["std_time_to_catch"] = float(np.std(catch_times))
        results["min_time_to_catch"] = float(np.min(catch_times))
        results["max_time_to_catch"] = float(np.max(catch_times))
        results["median_time_to_catch"] = float(np.median(catch_times))
    else:
        results["mean_time_to_catch"] = None
        results["std_time_to_catch"] = None
        results["min_time_to_catch"] = None
        results["max_time_to_catch"] = None
        results["median_time_to_catch"] = None
    
    # Per-ghost statistics
    for name in ghost_names:
        results[f"{name}_mean_reward"] = float(np.mean(ghost_rewards[name]))
        results[f"{name}_std_reward"] = float(np.std(ghost_rewards[name]))
        if personality_adherence[name]:
            results[f"{name}_personality_score"] = float(np.mean(personality_adherence[name]))
    
    return results


def compare_with_baseline(checkpoint_path: str, num_episodes: int = 20):
    """Compare RL agent with original deterministic behavior."""
    
    print("Loading trained agent...")
    agent, config = load_checkpoint(checkpoint_path)
    
    print("\n" + "="*60)
    print("EVALUATING RL AGENT")
    print("="*60)
    
    rl_results = evaluate_ghost_agent(
        agent, config, num_episodes=num_episodes, render=False
    )
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print("\nOverall Performance:")
    print(f"  Mean Reward: {rl_results['mean_total_reward']:.2f} ± {rl_results['std_total_reward']:.2f}")
    print(f"  Mean Episode Length: {rl_results['mean_episode_length']:.1f} steps")
    print(f"  Pacman Catch Rate: {rl_results['pacman_catch_rate']:.1%}")
    print(f"  Total Catches: {rl_results['total_catches']}/{num_episodes}")
    print(f"  Avg Ghost Deaths/Episode: {rl_results['avg_ghost_deaths_per_episode']:.2f}")
    
    # Time-to-catch statistics
    if rl_results['mean_time_to_catch'] is not None:
        print(f"\n  Time-to-Catch Metrics (when successful):")
        print(f"    Mean: {rl_results['mean_time_to_catch']:.1f} steps")
        print(f"    Std:  {rl_results['std_time_to_catch']:.1f} steps")
        print(f"    Min:  {rl_results['min_time_to_catch']:.0f} steps (fastest catch)")
        print(f"    Max:  {rl_results['max_time_to_catch']:.0f} steps (slowest catch)")
        print(f"    Median: {rl_results['median_time_to_catch']:.1f} steps")
    else:
        print(f"\n  Time-to-Catch: N/A (no successful catches)")
    
    print("\nPer-Ghost Performance:")
    ghost_names = ["blinky", "pinky", "inky", "clyde"]
    for name in ghost_names:
        reward_key = f"{name}_mean_reward"
        personality_key = f"{name}_personality_score"
        print(f"  {name.capitalize()}:")
        print(f"    Reward: {rl_results[reward_key]:.2f}")
        if personality_key in rl_results:
            print(f"    Personality Score: {rl_results[personality_key]:.3f}")
    
    return rl_results


def main(cli_args: Optional[list[str]] = None):
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained ghost agent")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--max-steps", type=int, help="Max steps per episode")
    parser.add_argument("--compare-baseline", action="store_true", help="Compare with baseline")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    
    args = parser.parse_args(cli_args)
    
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    if args.compare_baseline:
        results = compare_with_baseline(args.checkpoint, args.episodes)
    else:
        print("Loading agent...")
        agent, config = load_checkpoint(args.checkpoint)
        
        print(f"\nEvaluating agent for {args.episodes} episodes...")
        results = evaluate_ghost_agent(
            agent,
            config,
            num_episodes=args.episodes,
            render=not args.no_render,
            max_steps=args.max_steps,
        )
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(json.dumps(results, indent=2))
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
