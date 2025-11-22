# Pacman Reinforcement Learning PRD

## Overview
Extend the existing pygame-based Pacman implementation with reinforcement learning (RL) capabilities so an agent can control Pacman autonomously. The current keyboard-driven loop raises direction change requests via `GameEngine.direction_command` and processes them inside the main tick cycle, handling rendering and movement decisions for Pacman and ghosts @draw/game_engine.py#24-105. Bridging this mechanic with an RL policy requires exposing programmatic control hooks and structured game state.

## Problem Statement
- **Manual control only**: Direction commands originate from keyboard events and set `direction_command` on the engine, with no programmatic interface for agent-driven actions @pacman.py#34-46.
- **Environment interface missing**: The game lacks `reset()` and `step()` semantics expected by RL algorithms; logic is coupled to the continuous pygame loop @pacman.py#25-61.
- **State observability**: Key data (entity positions, scores, pellet layout, ghost modes) lives across engine, level, and entity classes without a consolidated observation API @draw/game_engine.py#24-105 @model/entity/player/player.py#101-133.

## Goals
1. Provide a programmatic environment API (reset, step, render) compatible with RL workflows.
2. Implement an RL-friendly observation builder capturing Pacman, ghosts, pellets, score, lives, and terminal flags.
3. Route agent-selected actions into the existing direction command system without disrupting manual play.
4. Deliver an initial training pipeline (focus: Pacman policy) using a baseline algorithm such as DQN or Double DQN.
5. Add tooling for evaluation, logging, and reproducibility (seeding, episode summaries).

## Non-Goals
- Re-implement ghost AI behaviour; existing heuristics remain.
- Multi-agent learning for ghosts.
- Networking or multiplayer features.
- Extreme performance tuning beyond reasonable FPS adjustments for training throughput.

## Users & Use Cases
- **Researchers / Students** wanting a classic benchmark to explore RL concepts.
- **Developers** experimenting with different agents or reward schemes.
- **Educators** demonstrating RL techniques with a familiar game.

## Assumptions
- Training may run headless or with reduced rendering to boost data throughput.
- Observations can reuse internal data structures (LevelConfig, Player, Ghost) without extensive refactors.
- Python 3.x, pygame, NumPy, and RL dependencies (e.g., PyTorch) remain available per `requirements.txt`.

## Functional Requirements
1. **Environment Wrapper**
   - Expose `reset()` to reinitialize the level via existing initialization routines @pacman.py#17-23 @levels/level_content_initializer.py#69-72.
   - Expose `step(action)` advancing the game by one tick (configurable frame skip) while applying the agent-selected direction.
   - Return `(observation, reward, terminated, truncated, info)` aligned with Gymnasium conventions.

2. **Observation Builder**
   - Capture Pacman position, direction, velocity using `Player` movement helpers @model/entity/player/player.py#101-133 @model/entity/entity.py#49-102.
   - Include ghost positions, modes, and timers sourced from ghost subclasses and engine state @draw/game_engine.py#69-89.
   - Encode pellet/power-up distribution from level board definitions and consumption tracking @settings.py#58-95 @draw/game_engine.py#24-105.
   - Track episodic metrics (score, lives, step count) from `GameEngine.level` and `Player` fields.

3. **Reward Scheme**
   - Base reward: score delta per step.
   - Penalties for death or invalid/no-op moves; optional shaping for pellet proximity or survival time.
   - Configurable coefficients to support experimentation.

4. **Action Interface**
   - Discrete action space {UP, DOWN, LEFT, RIGHT, STAY}. Map to `direction_command` before movement occurs @draw/game_engine.py#37-89.
   - Support frame skipping to reduce decision frequency.

5. **Training Pipeline**
   - Implement baseline DQN pipeline (PyTorch) with replay buffer, epsilon-greedy exploration, target network.
   - Provide evaluation loop, checkpointing, and metrics logging.

6. **Instrumentation & Debugging**
   - Logging hooks for episode returns, rewards, agent decisions.
   - Optional video/screenshot capture for qualitative assessment.

## Success Criteria
- Automated episodes run with agent control and no manual input.
- Observations and rewards remain stable over long training sessions (>10k steps).
- Baseline agent improves average episode score over a random policy.
- Documentation (README updates, notebooks) guiding training and evaluation is available.

## Risks & Mitigations
- **Performance constraints**: Pygame rendering can bottleneck training; provide headless mode and FPS throttling options.
- **State synchronization bugs**: Ensure environment wrapper pulls fresh entity states post-step and resets counters (e.g., `start_counter`).
- **Non-determinism**: Seed pygame/random modules where possible; document remaining stochasticity.
- **Observation complexity**: Start with compact structured state (tile indices + entity vectors) before tackling pixel-based inputs.

## Open Questions
- Best representation for pellets: binary grid, coordinate list, or aggregated features?
- How to accelerate experience collection (multi-process runners, trajectory replay)?
- Should curriculum learning (fewer ghosts, smaller mazes) or additional shaping be introduced to aid convergence?
