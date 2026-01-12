# Pacman DQN Agent Overview

This document summarizes the learning agent implemented in this repository: the inputs it consumes, how environment state is represented, the value function it optimizes, the reward shaping used during training, and the actions it can issue.

## Observation / Input Features

The agent consumes the **vector observation** produced by `PacmanEnv` with `observation_mode="vector"` @rl/env.py#216-224. It is a flattened float32 array built by `ObservationBuilder`, composed of:

- **Global scalars**
  - Normalized frame index (episode progress)
  - Current score (scaled)
  - Boolean game-over flag (as 0/1)
- **Pacman features**
  - Normalized `(x, y)` position in the maze (`player.location / screen_size`)
  - One-hot direction vector `[LEFT, RIGHT, UP, DOWN]`
  - Discrete state encoding (chasing, frightened, eaten, etc.)
  - Remaining lives (clipped to ≥0)
  - Power-up active flag, multiplier, and counter
- **Ghost features (per ghost)**
  - Normalized position `(x, y)`
  - One-hot direction vector
  - Discrete state flags (`CHASE`, `FRIGHTENED`, `EATEN`, `SCATTER`)
  - Whether the ghost is inside the house
- **Board representation (optional)**
  - When `include_board_in_observation=True`, the tile map is flattened to int8 values indicating walls, pellets, power pellets, etc.

These inputs provide sufficient context for the DQN policy to estimate future returns without pixel data.

## Environment State

Internally the environment keeps rich state via the pygame engine:

- `GameEngine` tracks Pacman, ghosts, pellets, timers, and renders frames.
- The observation vector is derived from a `RewardSnapshot` + raw entity attributes @rl/reward.py#17-124, ensuring consistency between reward computation and state features.
- Episode termination occurs when Pacman loses all lives or the step budget (`max_episode_steps`) is reached @rl/env.py#130-152.

## Value Function

The agent approximates the state-action value function **Q(s, a)** using a PyTorch neural network (`DQNAgent.policy_net`) @rl/agents/dqn.py#41-145:

- Input dimension equals the observation vector length.
- Two hidden layers of 256 ReLU units each (configurable via `hidden_sizes`).
- Output dimension is the discrete action space size (`len(PacmanEnv.ACTION_MEANINGS)` = 5).
- Training uses the DQN update with target network, smooth L1 (Huber) loss, and experience replay sampled uniformly from `ReplayBuffer` @rl/agents/replay_buffer.py#1-57.

The target network is synced periodically (hard update every `target_update_interval` steps) @rl/agents/dqn.py#124-145, stabilizing learning.

## Reward Function

Rewards are shaped via `RewardCalculator` using the configurable `RewardConfig` @rl/env.py#21-30:

- **Score-based reward**: difference in game score between frames, scaled by `score_scale` (default 1.0).
- **Step penalty**: constant negative reward to encourage efficiency (default -0.1).
- **Pellet / power pellet bonuses**: optional positive shaping when Pacman eats pellets (`pellet_reward`, `power_pellet_reward`).
- **Ghost reward**: bonus for eating frightened ghosts.
- **Life lost penalty**: applied when Pacman loses a life.
- **Death penalty**: additional penalty when the episode terminates due to game over.

The calculator keeps the previous snapshot and computes deltas each frame, so shaping aligns with observation timing @rl/reward.py#62-118.

## Action Space

The discrete action space is defined in `PacmanEnv.ACTION_MEANINGS` @rl/env.py#38-51:

1. **STAY** – no direction change
2. **LEFT** – request move left
3. **RIGHT** – request move right
4. **UP** – request move up
5. **DOWN** – request move down

Actions are converted to `Direction` commands and injected into the game engine before each frame skip cycle @rl/env.py#119-125. `frame_skip` controls how many engine ticks a single agent action is held.

## Training

The DQN agent is trained with experience replay and target-network stabilization @rl/agents/dqn.py#93-145, orchestrated by the training loop in `rl/training.py` @rl/training.py#102-179.

- **Architecture recap**: The policy network (`QNetwork`) comprises two hidden layers of 256 ReLU units, mapping from the observation dimension to the five Q-values (one per action). A separate target network with identical structure lags behind the policy network to provide stable bootstrap targets.
- **Replay buffer**: Transitions `(state, action, reward, next_state, done)` are stored in a ring buffer. After the warmup phase, each environment step samples a mini-batch of size `batch_size` for updates @rl/agents/replay_buffer.py#13-54.
- **Update rule**:
  1. Compute current Q-values `Q(s, a)` from the policy network.
  2. Estimate target values `r + γ * max_a' Q_target(s', a')` (zeroing out the bootstrapped term when `done=True`).
  3. Minimize the smooth L1 (Huber) loss between current Q-values and targets via Adam @rl/agents/dqn.py#101-123.
- **Update frequency**: A gradient step is executed every environment tick once `warmup_steps` have passed and the buffer holds at least `batch_size` samples @rl/training.py#140-146. This means weights are updated as soon as sufficient data exists, typically after the first thousand transitions (default warmup).
- **Target sync**: Every `target_update_interval` environment steps the target network receives the policy network weights (`agent.update_target()`), ensuring the targets evolve slowly relative to the policy @rl/training.py#147-151.
- **Exploration schedule**: `epsilon` decays linearly from `epsilon_start` to `epsilon_end` across `epsilon_decay_steps`, gradually shifting from exploratory to greedy behavior. During evaluation (`epsilon=0` by default) no noise is injected.

---

This description captures the baseline DQN setup; customizing observation components, network architecture, or reward coefficients allows experimentation with different agent behaviors.
