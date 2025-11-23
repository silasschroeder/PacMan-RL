# Pacman

Old classic Pacman game written in Python using pygame library.

 <img width="892" alt="Screenshot 2024-06-25 at 12 31 20" src="https://github.com/MathMark/PacMan/assets/13971845/19fa8ce2-a5ed-4cb2-81d6-0130cdf5268e">

# Description

The game contains a single demo-level. To increase difficulty you can play around with variables in settings file changing POWERUP_LIMIT, SCATTER_DISABLE_TRIGGER, SCATTER_ENABLE_TRIGGER, VELOCITY. You can also set 
variable DEBUG to True to see ghosts targets visualization and grid.

## Ghosts behaviour

| Name   | Description  | Behaviour                                                                                                                                                                                                                                                               |
|--------|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Blinky | Red Ghost    | Follows Pac-Man directly during Chase mode, and heads to the upper-right corner during Scatter mode.                                                                                                                                                                    |
| Pinky  | Pink ghost   | Chases towards the spot 2 Pac-Dots in front of Pac-Man. Due to a bug in the original game's coding, if Pac-Man faces upwards, Pinky's target will be 2 Pac-Dots in front of and 2 to the left of Pac-Man. During Scatter mode, she heads towards the upper-left corner. |
| Inky   | Blue ghost   | During Chase mode, his target is a bit complex. His target is relative to both Blinky and Pac-Man, where the distance Blinky is from Pinky's target is doubled to get Inky's target. He heads to the lower-right corner during Scatter mode.                            |
| Clyde  | Yellow ghost | Chases directly after Pac-Man, but tries to head to his Scatter corner when within an 8-Dot radius of Pac-Man. His Scatter Mode corner is the lower-left.                                                                                                               |


## Reinforcement Learning Extensions

The repository now includes an RL-ready wrapper over the pygame game loop to let agents control Pacman programmatically.

### Environment wrapper
- Module: `rl/env.py`
- Class: `PacmanEnv`
- Features:
  - `reset()` / `step()` Gym-style API with configurable frame skip.
  - Reward shaping hooks via `RewardConfig` (score scaling, pellet/ghost bonuses, life penalties).
  - Headless mode by default; enable human rendering with `render_mode="human"`.

### Quick smoke test

Use the random rollout example to verify setup:

```bash
python -m examples.random_rollout --episodes 1 --steps 200
```

Flags:
- `--render` to watch gameplay.
- `--episodes` / `--steps` to control episode count and length.

The script demonstrates how to instantiate `PacmanEnv`, sample random actions, and close pygame cleanly.

### Reward shaping overview

`PacmanEnv` computes the per-step reward using three additive pieces @rl/env.py#122-140:

1. **Score delta (scaled)** – In-game score changes are multiplied by `RewardConfig.score_scale`; if the delta is zero we apply `RewardConfig.step_penalty` (defaults shown below).
2. **Shaping bonuses/penalties** – `_compute_shaping_rewards()` tracks pellets, power pellets, ghost multipliers, lives, and power-up activation to award or deduct points based on the `RewardConfig` fields @rl/env.py#252-283.
3. **Terminal penalty** – When Pacman dies (`game_over=True`), `RewardConfig.death_penalty` is added.

Default coefficients (tweakable when constructing the env) are defined in `RewardConfig` @rl/env.py#19-27:

```python
RewardConfig(
    score_scale=1.0,
    step_penalty=-0.1,
    pellet_reward=0.0,
    power_pellet_reward=0.0,
    ghost_reward=0.0,
    life_lost_penalty=-100.0,
    death_penalty=-500.0,
)
```

### Observation export options

The environment aggregates game state via `ObservationBuilder` @rl/observation.py#25-128. Configure the returned shape when instantiating `PacmanEnv`:

- `observation_mode`: choose between
  - `"structured"` (default) – nested dict with `pacman`, `ghosts`, counters, and optional `board_consumables` grid.
  - `"vector"` – flattened `np.float32` feature vector suitable for classic RL agents.
  - `"pack"` – returns an `ObservationPack` containing both the structured dict and vector views.
- `include_board_in_observation`: when `True`, the pellet/power-pellet grid is included (structured adds `board_consumables` and `board_shape`, vector appends the flattened grid scaled to `[0, 1]`).

For quick inspection run:

```bash
python -m examples.inspect_observation --mode pack
```

The random rollout CLI also exposes `--obs-mode`, `--no-board`, and `--log-obs` helpers for experimentation.

### Reward utilities

Reward shaping lives in `rl/reward.py` and is configured via `RewardConfig` @rl/reward.py#12-94 @rl/env.py#21-29. The `RewardCalculator` produces a `RewardBreakdown` containing the score term, step penalty, pellet bonuses, ghost multiplier bonuses, life penalties, and power-up activation rewards. `PacmanEnv.step()` now exposes the most recent breakdown through the `info["reward_breakdown"]` dict, making it easier to debug reward tuning.

### Training pipeline

Use the provided scripts to train and evaluate a baseline DQN agent:

```bash
# Train with defaults (writes checkpoints/metrics under runs/latest)
python train_agent.py --episodes 100 --device cpu

# Evaluate a checkpoint with greedy policy
python evaluate_agent.py --checkpoint runs/latest/latest.pt --episodes 10
```

- Training configuration is defined in `TrainingConfig` @rl/training.py#16-40. Override values via CLI flags or supply a JSON file with matching keys using `--config`.
- Checkpoints and metrics are saved as `.pt` and `metrics.json` files in the chosen `--output` directory.
- `evaluate_agent.py` can consume an existing checkpoint or trigger training if `--train-if-missing --config path/to/config.json` is passed.

#### Configuration reference

`TrainingConfig` supports the following JSON fields (all optional—defaults are applied when omitted):

```json
{
  "episodes": 200,
  "max_steps": 500,
  "buffer_size": 50000,
  "batch_size": 64,
  "warmup_steps": 1000,
  "target_update_interval": 1000,
  "gamma": 0.99,
  "learning_rate": 0.001,
  "tau": 1.0,
  "epsilon_start": 1.0,
  "epsilon_end": 0.05,
  "epsilon_decay_steps": 50000,
  "frame_skip": 2,
  "observation_include_board": false,
  "seed": 42,
  "device": "cpu",
  "reward_config": {
    "score_scale": 1.0,
    "step_penalty": -0.1,
    "pellet_reward": 0.0,
    "power_pellet_reward": 0.0,
    "ghost_reward": 0.0,
    "life_lost_penalty": -100.0,
    "death_penalty": -500.0
  },
  "evaluation_episodes": 5
}
```

Store the file (e.g., `configs/dqn_baseline.json`) and pass it via `--config`. CLI overrides always take precedence over values loaded from JSON.

#### CLI highlights

- `--output runs/exp01` keeps checkpoints under `runs/exp01/{best,latest}.pt` and writes per-episode metrics to `runs/exp01/metrics.json`.
- To resume training from a previous run, reuse the same config and output directory; the script always emits an updated `latest.pt` snapshot at the end of execution.
- Setting `--device cuda` allows DQN training on GPU (ensure PyTorch detects CUDA).
- Use `--eval --eval-episodes 20` during training to immediately measure average return with a greedy policy after the final episode.

#### Monitoring progress

- `metrics.json` contains a list of objects with `episode`, `reward`, `steps`, `mean_loss`, and `epsilon`. Load it into pandas, Excel, or TensorBoard.dev (via custom uploader) to visualize learning curves.
- For lightweight logging during long runs, pipe stdout to a file: `python train_agent.py ... | tee runs/exp01/train.log`.

#### Evaluating checkpoints

```bash
python evaluate_agent.py \
  --checkpoint runs/exp01/best.pt \
  --episodes 20 \
  --epsilon 0.05
```

- The evaluation script loads the agent with the same architecture hyperparameters used during training (serialized alongside the checkpoint).
- Add `--output runs/exp01` to save `evaluation.json`, containing aggregate episode stats (`average_reward`, `std_reward`, `average_length`).
- If the checkpoint is missing but a config exists, `--train-if-missing` will train using the supplied config before running evaluation.

#### Watch a checkpoint play

```bash
python play_agent.py runs/exp01/best.pt --episodes 3 --epsilon 0.05
```

- `play_agent.py` rebuilds `PacmanEnv` in `render_mode="human"` so you can view gameplay in a Pygame window.
- Pass `--episodes` to watch multiple rollouts back-to-back; the script prints the reward returned for each episode.
- `--epsilon` controls exploration during playback. Use:
  - `0.0` for a fully greedy policy (deterministic execution of the learned Q-values).
  - A small positive value (e.g., `0.05`) to occasionally sample alternate actions, useful for diagnosing behavior in uncertain states.
- Omit `--max-steps` to use the training horizon; provide a custom value to cap episode length during visualization.
