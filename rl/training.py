from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from rl.agents import DQNAgent, DQNConfig, ReplayBuffer
from rl.env import PacmanEnv, RewardConfig


@dataclass
class TrainingConfig:
    episodes: int = 200
    max_steps: int = 500
    buffer_size: int = 50_000
    batch_size: int = 64
    warmup_steps: int = 1_000
    target_update_interval: int = 1_000
    gamma: float = 0.99
    learning_rate: float = 1e-3
    tau: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    frame_skip: int = 2
    observation_include_board: bool = False
    seed: Optional[int] = 42
    device: str = "cpu"
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    evaluation_episodes: int = 5


@dataclass
class TrainingResult:
    metrics: List[Dict[str, Any]]
    agent: DQNAgent
    config: TrainingConfig


def load_training_config(path: Optional[str]) -> TrainingConfig:
    if path is None:
        return TrainingConfig()

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
    config = TrainingConfig(**config_kwargs)
    config.reward_config = build_reward_config(reward_cfg)
    return config


def _prepare_env(config: TrainingConfig) -> PacmanEnv:
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


def _make_agent(config: TrainingConfig, state_dim: int, action_dim: int) -> DQNAgent:
    epsilon_decay = (
        (config.epsilon_start - config.epsilon_end) / max(1, config.epsilon_decay_steps)
    )
    dqn_config = DQNConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=(256, 256),
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        tau=config.tau,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay=epsilon_decay,
        device=config.device,
    )
    return DQNAgent(dqn_config)


def run_training(config: TrainingConfig, output_dir: Optional[str] = None) -> TrainingResult:
    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    env = _prepare_env(config)
    observation, _ = env.reset()
    state = np.asarray(observation, dtype=np.float32)
    action_dim = len(PacmanEnv.ACTION_MEANINGS)

    agent = _make_agent(config, state_dim=state.size, action_dim=action_dim)
    replay = ReplayBuffer(capacity=config.buffer_size, state_shape=(state.size,))

    metrics: List[Dict[str, Any]] = []
    global_step = 0
    best_reward = -math.inf
    output_path: Optional[Path] = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    try:
        for episode in range(1, config.episodes + 1):
            observation, _ = env.reset()
            state = np.asarray(observation, dtype=np.float32)
            episode_reward = 0.0
            losses: List[float] = []

            for step in range(1, config.max_steps + 1):
                action = agent.select_action(state)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                next_state = np.asarray(next_obs, dtype=np.float32)
                replay.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                global_step += 1

                if (
                    global_step >= config.warmup_steps
                    and len(replay) >= config.batch_size
                ):
                    loss = agent.train_step(replay, config.batch_size, gamma=config.gamma)
                    losses.append(loss)

                agent.decay_epsilon()

                if global_step % config.target_update_interval == 0:
                    agent.update_target()

                if done:
                    break

            mean_loss = float(np.mean(losses)) if losses else 0.0
            metrics.append(
                {
                    "episode": episode,
                    "reward": episode_reward,
                    "steps": step,
                    "mean_loss": mean_loss,
                    "epsilon": agent.epsilon,
                }
            )

            if output_path and episode_reward > best_reward:
                best_reward = episode_reward
                save_checkpoint(agent, config, output_path / "best.pt")
    finally:
        env.close()

    if output_path:
        history_path = output_path / "metrics.json"
        with open(history_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        save_checkpoint(agent, config, output_path / "latest.pt")

    return TrainingResult(metrics=metrics, agent=agent, config=config)


def evaluate_policy(
    agent: DQNAgent,
    config: TrainingConfig,
    episodes: Optional[int] = None,
    max_steps: Optional[int] = None,
    epsilon: float = 0.0,
) -> Dict[str, Any]:
    env = _prepare_env(config)
    original_epsilon = agent.epsilon
    agent.epsilon = epsilon

    rewards: List[float] = []
    lengths: List[int] = []
    try:
        for _ in range(episodes or config.evaluation_episodes):
            observation, _ = env.reset()
            state = np.asarray(observation, dtype=np.float32)
            total_reward = 0.0
            for step in range(1, (max_steps or config.max_steps) + 1):
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
        agent.epsilon = original_epsilon
        env.close()

    return {
        "episodes": len(rewards),
        "average_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "average_length": float(np.mean(lengths)) if lengths else 0.0,
    }


def save_checkpoint(agent: DQNAgent, config: TrainingConfig, path: Path) -> None:
    payload = {
        "agent": agent.save_state_dict(),
        "config": asdict(config),
    }
    torch.save(payload, path)


def load_checkpoint(path: str) -> tuple[DQNAgent, TrainingConfig]:
    checkpoint = torch.load(path, map_location="cpu")
    cfg_dict = checkpoint["config"].copy()
    reward_dict = cfg_dict.pop("reward_config", {})
    config = TrainingConfig(**cfg_dict)
    config.reward_config = RewardConfig(**reward_dict)

    env = _prepare_env(config)
    observation, _ = env.reset()
    state_dim = int(np.asarray(observation).size)
    env.close()

    agent = _make_agent(config, state_dim=state_dim, action_dim=len(PacmanEnv.ACTION_MEANINGS))
    agent.load_state_dict(checkpoint["agent"])
    return agent, config
