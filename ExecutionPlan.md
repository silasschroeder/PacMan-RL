# Execution Plan: Pacman Reinforcement Learning Integration

## Phase 0 – Baseline Analysis (status: completed)
1. **Map control flow for Pacman direction updates**  
   - Review `Game.check_events()` to understand how keyboard input mutates `GameEngine.direction_command` @pacman.py#33-46.  
   - Inspect `GameEngine.tick()` and `Player.move()` for how `direction_command` propagates into movement @draw/game_engine.py#69-88, @model/entity/player/player.py#101-133.  
   - Outcome: Confirmed agent should set `direction_command` before each `tick()`; engine reverts to current heading if the desired turn is blocked.

2. **Catalogue required state elements for RL observations**  
   - Identify accessible properties for Pacman/ghost positions, velocities, modes, pellet layout, score, and life counters @draw/game_engine.py#24-105, @settings.py#58-95.
   - Outcome: Drafted observation schema capturing: Pacman `(x, y)` normalized positions, direction one-hot, lives, state, power-up flag, score multiplier; per-ghost normalized position, direction one-hot, state, frightened timer proxy (`player.powerup_counter`), in-house flag; board consumables as flattened grid (`0` empty, `1` pellet, `2` power pellet) from `LevelConfig.board_definition.board`; score metrics (current score, score delta), power-up counter, `game_engine.game_over`, and optional frame count for timing features.

## Phase 1 – Environment API (status: in progress)
3. **Implement environment wrapper class** ✅  
   - `PacmanEnv` added with configurable `frame_skip`, Gym-like API, and reward shaping hooks @rl/env.py#16-293.
   - Outcome: Environment resets/steps headlessly or with human rendering and exposes structured observations.

4. **Provide smoke tests & examples** ✅  
   - Added `examples/random_rollout.py` for CLI-based random agent runs @examples/random_rollout.py#1-59.  
   - Outcome: quick validation entry point for environment behaviour.

5. **Document usage & configuration knobs** (next)  
   - Update README/ExecutionPlan with rollout instructions, reward config notes, and future training tasks.  
   - Success: Contributors can reproduce random rollout and understand environment parameters.

## Phase 2 – Observation & Reward Design (status: in progress)
5. **Implement observation builder module** ✅  
   - Consolidated state extraction into `rl/observation.py`, returning structured dicts and flattened vectors via `ObservationBuilder`.  
   - Captures Pacman/ghost kinematics, state flags, pellet counts, score, lives, timers; optional board grid and ObservationPack dual view.  
   - CLI tooling: `examples.random_rollout.py` exposes observation flags; `examples.inspect_observation.py` summarizes snapshots.  
   - Success: Observation size/dtype documented in README; unit test `tests/test_observation.py` verifies vector length across modes.

6. **Design reward function utilities**  
   - Base reward = score delta; penalties for loss of life; optional shaping toggles.  
   - Define configuration structure to tweak coefficients.  
   - Success: Reward tests covering pellet eat, ghost eat, death events.

## Phase 3 – Agent & Training Pipeline (status: pending)
7. **Build baseline DQN agent focused on Pacman policy**  
   - Use PyTorch; define network architecture taking chosen observation format.  
   - Implement replay buffer, epsilon-greedy exploration, target network sync.  
   - Place code under `rl/agents/dqn.py`.  
   - Success: Training script runs for set episodes without crashing.

8. **Create training & evaluation scripts**  
   - `train_agent.py`: orchestrates training loop, logging, checkpointing.  
   - `evaluate_agent.py`: loads checkpoints, runs fixed episodes, reports metrics.  
   - Integrate configuration files (YAML/JSON) for hyperparameters.  
   - Success: CLI usage documented in repo README.

## Phase 4 – Tooling & Documentation (status: pending)
9. **Add instrumentation & visualization**  
   - Logging via TensorBoard or CSV for rewards, loss, epsilon.  
   - Optional frame recorder for sample episodes.  
   - Success: Logs generated and stored per run.

10. **Documentation updates**  
    - Extend root README with setup instructions, training workflow, troubleshooting.  
    - Provide notebook or markdown walkthrough for experimentation.  
    - Success: New contributors can follow docs to train agent end-to-end.

## Research Notes (Key Integration Considerations)
- **Action Injection Point**: Agent should set `GameEngine.direction_command` before `move_player()` executes during chase state; frame-skipping ensures smoother movement @draw/game_engine.py#69-88.
- **Tick Granularity**: One `tick()` covers rendering and entity updates; repeated ticks per action simulate sustained direction holding.
- **State Sources**:  
  - Pacman positional data via `Player.location_x/location_y`, `direction`, `score_multiplier`.  
  - Ghost states accessible through ghost list in `GameEngine.ghosts` (mode methods: `is_frightened()`, `is_chasing()`, etc.).  
  - Pellet status tracked in `GameEngine.board` and `LevelConfig.score` @draw/game_engine.py#24-105.  
- **Reset Handling**: Use `Game.init()` to rebuild engine and reinitialize sounds; suppress audio when training headless by mocking `pygame.mixer`.  
- **Performance**: Consider decoupling rendering from training loop (skip `pygame.display.flip()` when headless) to increase step throughput.  

## Immediate Next Steps
1. Finish Phase 0 state mapping and observation schema draft.  
2. Prototype `PacmanEnv.reset()` and `step()` skeleton to validate control handoff.  
3. Decide on observation vector dimensions to inform DQN architecture.
