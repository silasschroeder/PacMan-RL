# Ghost RL Training

This directory contains the RL infrastructure for training intelligent ghost agents in Pac-Man.

## Overview

The ghost training system uses:

- **Shared Policy**: All 4 ghosts use the same DQN network but receive individual observations
- **Parallel Training**: 50+ environments run simultaneously for efficient data collection
- **Curriculum Learning**: Ghosts are initially rewarded for maintaining their personality, then gradually focus on catching Pac-Man
- **Multi-Agent Coordination**: Rewards encourage ghosts to spread out and corner Pac-Man

## Architecture

### Key Components

1. **`rl/ghost_observation.py`**: Builds observations for each ghost including:

   - Ghost's position, direction, and state
   - Pac-Man's position, direction, and powerup status
   - Other ghosts' positions (for coordination)
   - Nearest power pellet location and distance
   - Local grid (7×7) showing walls and pellets
   - **Observation dimension**: 74 features per ghost

2. **`rl/ghost_reward.py`**: Calculates shaped rewards with:

   - Distance reduction to Pac-Man (+1.0 per normalized unit)
   - Catching Pac-Man (+100.0)
   - Safety when frightened (flee +2.0, approach -3.0)
   - Personality adherence (+0.5, decays over 5000 episodes)
   - Team coordination (+0.3 for optimal spacing)
   - Small step penalty (-0.01 for efficiency)

3. **`rl/ghost_env.py`**: Environment wrapper providing:

   - Individual observations for each ghost
   - Shared action space (5 actions: STAY, LEFT, RIGHT, UP, DOWN)
   - Vectorized environments for parallel training
   - Auto-reset on episode completion

4. **`rl/agents/ghost_dqn.py`**: Specialized DQN agent:

   - Handles multiple ghost observations per step
   - Shared Q-network for all ghosts
   - Replay buffer optimized for multi-agent experiences

5. **`train_ghost_agent.py`**: Main training script with:

   - Parallel environment support (default: 50 envs)
   - Configurable hyperparameters
   - Periodic evaluation and checkpointing
   - Progress logging and metrics tracking

6. **`evaluate_ghost_agent.py`**: Evaluation tools:
   - Visualize trained agents in action
   - Detailed per-ghost metrics
   - Performance comparison utilities

## Quick Start

### Basic Training

Train with default settings (50 parallel environments, 1M timesteps):

```bash
python train_ghost_agent.py --output runs/ghost_exp01
```

### Custom Configuration

Create a config file (e.g., `ghost_config.json`):

```json
{
  "num_parallel_envs": 32,
  "total_timesteps": 2000000,
  "batch_size": 256,
  "learning_rate": 0.0003,
  "hidden_sizes": [256, 256, 128],
  "reward_config": {
    "distance_reduction_reward": 1.0,
    "catch_pacman_reward": 100.0,
    "personality_match_reward": 0.5,
    "personality_weight_start": 1.0,
    "personality_weight_end": 0.1,
    "personality_decay_episodes": 5000,
    "coordination_reward": 0.3
  }
}
```

Train with config:

```bash
python train_ghost_agent.py --config ghost_config.json --output runs/ghost_exp01
```

### Evaluation

Evaluate a trained agent with visualization:

```bash
python evaluate_ghost_agent.py runs/ghost_exp01/best.pt --episodes 10
```

Evaluate without rendering:

```bash
python evaluate_ghost_agent.py runs/ghost_exp01/best.pt --episodes 20 --no-render
```

## Training Parameters

### Environment Settings

- `num_parallel_envs` (50): Number of parallel environments
- `max_episode_steps` (2000): Maximum steps per episode
- `frame_skip` (1): Number of game frames per RL step

### Training Settings

- `total_timesteps` (1M): Total training steps
- `buffer_size` (100K): Replay buffer capacity
- `batch_size` (256): Training batch size
- `warmup_steps` (10K): Random exploration before training
- `train_frequency` (4): Train every N steps
- `target_update_interval` (1000): Update target network every N steps

### Agent Settings

- `gamma` (0.99): Discount factor
- `learning_rate` (3e-4): Adam learning rate
- `epsilon_start` (1.0): Initial exploration rate
- `epsilon_end` (0.05): Final exploration rate
- `epsilon_decay_fraction` (0.5): Fraction of training for epsilon decay
- `hidden_sizes` ([256, 256, 128]): Network architecture

### Reward Configuration

See `rl/ghost_reward.py` for detailed reward component descriptions.

## Understanding Curriculum Learning

The personality adherence reward helps ghosts learn their characteristic behaviors:

- **Blinky**: Directly chases Pac-Man (aggressive)
- **Pinky**: Targets position ahead of Pac-Man (ambusher)
- **Inky**: Uses Blinky's position to flank (strategic)
- **Clyde**: Switches between chase and scatter based on distance (unpredictable)

The curriculum weight decays from 1.0 to 0.1 over 5000 episodes, allowing ghosts to:

1. **Early training**: Learn their personality patterns
2. **Mid training**: Balance personality with performance
3. **Late training**: Focus primarily on catching Pac-Man

## Performance Metrics

Training logs track:

- Mean episode reward (all 4 ghosts combined)
- Pac-Man catch rate (success metric)
- Episode length (efficiency metric)
- Per-ghost reward breakdowns
- Personality adherence scores

Evaluation provides:

- Win rate against Pac-Man
- Average ghost deaths per episode
- Individual ghost performance
- Team coordination metrics

## Tips for Tuning

1. **Slow convergence?**

   - Increase `num_parallel_envs` (more data)
   - Increase `learning_rate`
   - Decrease `batch_size`

2. **Unstable training?**

   - Decrease `learning_rate`
   - Increase `batch_size`
   - Adjust reward scales

3. **Ghosts too aggressive/timid?**

   - Adjust `distance_reduction_reward`
   - Modify `catch_pacman_reward`
   - Tune safety rewards

4. **Want more personality preservation?**

   - Increase `personality_weight_end`
   - Increase `personality_decay_episodes`
   - Boost `personality_match_reward`

5. **Poor coordination?**
   - Increase `coordination_reward`
   - Adjust `optimal_ghost_distance`

## System Requirements

- **RAM**: 8GB+ (for 50 parallel environments)
- **GPU**: Recommended but optional (training works on CPU)
- **Python**: 3.9+
- **Dependencies**: See `requirements.txt`

## File Structure

```
rl/
├── ghost_observation.py    # Observation builder
├── ghost_reward.py         # Reward calculator with curriculum
├── ghost_env.py            # Environment wrapper
└── agents/
    └── ghost_dqn.py        # Ghost DQN agent

train_ghost_agent.py        # Training script
evaluate_ghost_agent.py     # Evaluation script
```

## Integration with Existing Code

The ghost RL system:

- ✅ Does NOT modify Pac-Man agent code
- ✅ Preserves original ghost personalities
- ✅ Can be toggled on/off via `ghost.set_rl_mode(enabled)`
- ✅ Works alongside existing player RL training

## Next Steps

1. Run baseline training to establish performance
2. Experiment with reward weights
3. Try different network architectures
4. Adjust curriculum learning schedule
5. Fine-tune coordination rewards

## Troubleshooting

**Import errors**: Ensure you're running from the project root directory

**Pygame display issues**: Use `--no-render` for headless training

**CUDA errors**: Set `--device cpu` if GPU is unavailable

**Slow training**: Reduce `num_parallel_envs` if memory is limited

## Questions?

See the code comments for detailed implementation notes. Each module has comprehensive docstrings explaining the design decisions.
