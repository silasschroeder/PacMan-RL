# Project Overview: Pacman Reinforcement Learning Integration

## Motivation and Objectives
The project extends a classic pygame-based Pacman clone with reinforcement learning (RL) capabilities. The primary goals were to (1) expose a gym-like environment interface for the existing game loop, (2) design structured observations and reward shaping suitable for agent training, (3) implement a baseline Deep Q-Network (DQN) agent with reproducible training and evaluation tooling, and (4) document the system so future research can iterate on algorithms, reward schemes, and curriculum design.

## Environment Abstraction
`PacmanEnv` (implemented in `rl/env.py`) wraps the original game engine, providing `reset()`/`step()` semantics, configurable frame skipping, and optional human rendering. The environment aggregates state via an `ObservationBuilder` that outputs both structured dictionaries and flattened vector features capturing Pacman/ghost kinematics, pellet distribution, episode counters, and optional board grids. Reward shaping is centralized in `RewardCalculator`, enabling experiments with pellet bonuses, survival penalties, and death punishments while retaining access to the raw score delta.

## Agent Architecture
The baseline agent is a DQN that consumes the flattened observation vector emitted by `PacmanEnv`. The input dimensionality is inferred from the observation builder at runtime and recorded in `DQNConfig.state_dim`, ensuring parity between the environment features (Pacman pose, ghost kinematics, pellet counts, timers, etc.) and the neural network. The action head matches the discrete move set exposed by the wrapper (`action_dim`), enabling the model to score one Q-value per legal joystick command.

Both the policy and target networks share an identical multi-layer perceptron defined in `QNetwork`: two fully connected hidden layers with 256 ReLU units each by default, customizable via the `hidden_sizes` tuple. Layers are initialized with Kaiming fan-in scaling and small uniform biases to accelerate early training. Gradients are clipped at a global norm of 10.0 to prevent value spikes when rare high-reward transitions occur, while an Adam optimizer (configurable learning rate) performs parameter updates.

Exploration is governed by an epsilon-greedy scheduler stored on the agent (`epsilon_start`, `epsilon_end`, `epsilon_decay`). During training the epsilon value is decayed every environment step until it reaches the floor, but it can be overridden for evaluation or playback to trade off randomness and exploitation. Experience tuples are buffered in `ReplayBuffer`, allowing uniform sampling of decorrelated batches that mix short-term and long-term outcomes.

Temporal-difference targets rely on the synchronized target network: with `tau=1.0` the implementation performs hard copies at the cadence specified by the training loop; setting `tau<1.0` activates soft Polyak averaging for smoother tracking. Targets use the standard Bellman backup `r + γ max_a' Q_target(s', a')`, which is compared against the policy network’s Q-value for the selected action via mean-squared error. This arrangement, combined with device-aware tensor utilities, encapsulates a minimal-yet-extensible DQN suitable for experimentation with double critics, dueling heads, or prioritized sampling in future work.

## Training Pipeline and Tooling
`rl/training.py` encapsulates training loops, evaluation routines, checkpoint persistence, and configuration loading. CLI entrypoints `train_agent.py` and `evaluate_agent.py` expose hyperparameters such as episode count, learning rate, reward coefficients, and device selection. Metrics (episode reward, mean loss, epsilon) and model snapshots (`best.pt`, `latest.pt`) are written per run, facilitating experiment tracking. The `play_agent.py` helper renders trained checkpoints interactively, reusing stored reward settings during playback.

## Experimental Configuration
The repository includes JSON config files (e.g., `training_configs/config08.json`–`config12.json`) showcasing different reward shaping regimes, epsilon schedules, and episode budgets. These artifacts enable systematic comparisons (e.g., aggressive step penalties vs. dense pellet rewards) and provide starting points for extended studies such as curriculum learning, Double DQN, or prioritized replay.

## Evaluation and Findings
Preliminary runs demonstrate how reward shaping and exploration schedules affect convergence: harsher step penalties discourage aimless wandering, while moderate pellet/ghost bonuses prevent the agent from ignoring edible objectives. Despite longer training horizons (hundreds of episodes), the baseline DQN struggles to achieve consistently positive returns—highlighting opportunities for future work on alternative algorithms, richer observations, or improved credit assignment. The infrastructure nonetheless produces reproducible metrics and visual rollouts to aid qualitative assessment.

## Future Directions
Next steps include adding instrumentation (TensorBoard/CSV logging), video capture for qualitative analysis, and extended documentation or notebooks. Research extensions could explore:
- Double/dueling DQN or distributional critics
- Prioritized replay buffers
- Curriculum strategies (simplified mazes, fewer ghosts)
- Alternative observation modalities (pixel inputs, attention over board state)
- Reward shaping refinements or intrinsic motivation signals

This overview provides the context needed to position the project within reinforcement learning research, detailing the system architecture, methodology, experimental levers, and open challenges suitable for a scientific manuscript.
