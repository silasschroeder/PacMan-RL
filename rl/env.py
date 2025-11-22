from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pygame

from levels.level_content_initializer import LevelContentInitializer
from model.board_definition import BoardDefinition
from model.direction import Direction
from model.level_config import LevelConfig
from settings import BOARD, FPS, POWER_UP_LIMIT, RESOLUTION
from rl.observation import ObservationBuilder, ObservationPack
from rl.reward import RewardBreakdown, RewardCalculator, RewardSnapshot


BACKGROUND_COLOR = (12, 2, 25)


@dataclass
class RewardConfig:
    score_scale: float = 1.0
    step_penalty: float = -0.1
    pellet_reward: float = 0.0
    power_pellet_reward: float = 0.0
    ghost_reward: float = 0.0
    life_lost_penalty: float = -100.0
    death_penalty: float = -500.0


class PacmanEnv:
    """Environment wrapper around the pygame Pacman implementation.

    The API mirrors Gymnasium's reset/step interface while remaining dependency-free.
    """

    ACTION_MEANINGS = {
        0: "STAY",
        1: "LEFT",
        2: "RIGHT",
        3: "UP",
        4: "DOWN",
    }

    ACTION_TO_DIRECTION = {
        1: Direction.LEFT,
        2: Direction.RIGHT,
        3: Direction.UP,
        4: Direction.DOWN,
    }

    def __init__(
        self,
        frame_skip: int = 1,
        render_mode: Optional[str] = None,
        reward_config: Optional[RewardConfig] = None,
        max_episode_steps: Optional[int] = None,
        observation_mode: str = "structured",
        include_board_in_observation: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.frame_skip = max(1, frame_skip)
        self.render_mode = render_mode or "none"
        self.reward_config = reward_config or RewardConfig()
        self.max_episode_steps = max_episode_steps
        self.observation_mode = observation_mode
        self._seed = seed

        if observation_mode not in {"structured", "vector", "pack"}:
            raise ValueError(
                "observation_mode must be one of {'structured', 'vector', 'pack'}; "
                f"got {observation_mode!r}"
            )

        self.screen: Optional[pygame.Surface] = None
        self.game_engine = None
        self._clock: Optional[pygame.time.Clock] = None
        self._initialized = False

        self._frame_count = 0

        self._reward_calculator = RewardCalculator(self.reward_config)
        self._reward_snapshot: Optional[RewardSnapshot] = None
        self._last_reward_breakdown: Optional[RewardBreakdown] = None

        self._observation_builder = ObservationBuilder(include_board=include_board_in_observation)

        if seed is not None:
            self.seed(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment and return the initial observation and info."""
        self._ensure_initialized()
        self._build_game_engine()

        self._frame_count = 0
        self._reward_snapshot = RewardSnapshot.from_engine(self.game_engine)
        self._last_reward_breakdown = RewardBreakdown.zero()

        observation = self._format_observation(self._build_observation())
        info = self._build_info()
        return observation, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Advance the game by one agent step."""
        if not self._initialized or self.game_engine is None:
            raise RuntimeError("Call reset() before step().")

        total_reward = 0.0
        terminated = False
        truncated = False

        direction = self.ACTION_TO_DIRECTION.get(action)
        if direction is not None:
            self.game_engine.direction_command = direction

        for _ in range(self.frame_skip):
            pygame.event.pump()
            if self.screen is not None:
                self.screen.fill(BACKGROUND_COLOR)
            self.game_engine.tick()
            self._frame_count += 1

            if self.max_episode_steps is not None and self._frame_count >= self.max_episode_steps:
                truncated = True

            if self.render_mode == "human":
                if self._clock is not None:
                    self._clock.tick(FPS)
                pygame.display.flip()

            current_snapshot = RewardSnapshot.from_engine(self.game_engine)
            breakdown = self._reward_calculator.compute(self._reward_snapshot, current_snapshot)
            shaped_reward = breakdown.total
            total_reward += shaped_reward
            self._last_reward_breakdown = breakdown

            self._reward_snapshot = current_snapshot

            if self.game_engine.game_over:
                terminated = True
                total_reward += self.reward_config.death_penalty
                break

            if truncated:
                break

        observation = self._format_observation(self._build_observation())
        info = self._build_info()
        return observation, total_reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Return an RGB array of the current screen if requested."""
        if not self._initialized or self.screen is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self.render_mode == "human":
            pygame.display.flip()
            return None

        array = pygame.surfarray.array3d(self.screen)
        # Pygame returns (width, height, channels); transpose to (height, width, channels)
        return np.transpose(array, (1, 0, 2))

    def close(self) -> None:
        if self._initialized:
            pygame.quit()
            self._initialized = False
            self.screen = None
            self.game_engine = None
            self._clock = None

    def seed(self, value: int) -> None:
        np.random.seed(value)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        if not pygame.get_init():
            pygame.init()

        if self.render_mode == "human":
            self.screen = pygame.display.set_mode(RESOLUTION)
        else:
            # Off-screen surface for headless operation
            self.screen = pygame.Surface(RESOLUTION)

        self._clock = pygame.time.Clock()
        self._initialized = True

    def _build_game_engine(self) -> None:
        board_copy = BOARD.copy()
        board_definition = BoardDefinition(board_copy)
        level_config = LevelConfig(
            board_definition=board_definition,
            wall_color="blue",
            gate_color="white",
            power_up_limit=POWER_UP_LIMIT,
        )

        initializer = LevelContentInitializer(level_config, self.screen)
        self.game_engine = initializer.init_game_engine()
        self._prev_score = self.game_engine.level.score
        self.game_engine.game_over = False

    def _build_observation(self) -> ObservationPack:
        return self._observation_builder.build(self.game_engine, self._frame_count)

    def _format_observation(self, pack: ObservationPack) -> Any:
        if self.observation_mode == "structured":
            return pack.structured
        if self.observation_mode == "vector":
            return pack.vector
        return pack

    def _build_info(self) -> Dict[str, Any]:
        player = self.game_engine.player
        info: Dict[str, Any] = {
            "score": self.game_engine.level.score,
            "lives": max(player.lives, 0),
            "powerup_active": player.powerup,
        }

        if self._last_reward_breakdown is not None:
            info["reward_breakdown"] = self._last_reward_breakdown.to_dict()
        return info
