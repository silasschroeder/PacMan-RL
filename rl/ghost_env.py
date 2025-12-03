"""Ghost RL environment wrapper.

This environment trains a shared RL agent that controls all 4 ghosts simultaneously.
Each ghost receives its own observation but they share the same policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame

from levels.level_content_initializer import LevelContentInitializer
from model.board_definition import BoardDefinition
from model.level_config import LevelConfig
from model.direction import Direction
from settings import BOARD, FPS, POWER_UP_LIMIT, RESOLUTION
from rl.ghost_observation import GhostObservationBuilder, GhostObservationPack
from rl.ghost_reward import (
    GhostRewardCalculator,
    GhostRewardConfig,
    GhostRewardSnapshot,
    GhostRewardBreakdown,
)


BACKGROUND_COLOR = (12, 2, 25)


@dataclass
class GhostEnvConfig:
    """Configuration for ghost training environment."""
    
    frame_skip: int = 1
    max_episode_steps: Optional[int] = 2000
    reward_config: GhostRewardConfig = None
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.reward_config is None:
            self.reward_config = GhostRewardConfig()


class GhostEnv:
    """Environment for training ghosts to catch Pacman.
    
    This environment:
    - Controls all 4 ghosts with a shared policy
    - Provides individual observations for each ghost
    - Calculates team-based rewards
    - Supports personality-based curriculum learning
    """
    
    ACTION_MEANINGS = {
        0: "STAY",
        1: "LEFT",
        2: "RIGHT",
        3: "UP",
        4: "DOWN",
    }
    
    def __init__(
        self,
        config: Optional[GhostEnvConfig] = None,
        render_mode: str = "none",
    ):
        self.config = config or GhostEnvConfig()
        self.render_mode = render_mode
        
        self.screen: Optional[pygame.Surface] = None
        self.game_engine = None
        self._clock: Optional[pygame.time.Clock] = None
        self._initialized = False
        
        self._frame_count = 0
        self._episode_count = 0
        self._episode_start_time = 0  # Track when episode started
        
        # Observation and reward systems
        self._obs_builder = GhostObservationBuilder(normalize_positions=True)
        self._reward_calculator = GhostRewardCalculator(self.config.reward_config)
        
        # Track reward snapshots for each ghost
        self._reward_snapshots: List[GhostRewardSnapshot] = []
        self._last_breakdowns: List[GhostRewardBreakdown] = []
        
        if self.config.seed is not None:
            self.seed(self.config.seed)
    
    @property
    def num_ghosts(self) -> int:
        """Number of ghosts in the environment."""
        return 4
    
    @property
    def observation_dim(self) -> int:
        """Dimension of each ghost's observation vector."""
        return GhostObservationBuilder.get_observation_dim()
    
    @property
    def action_dim(self) -> int:
        """Number of possible actions."""
        return len(self.ACTION_MEANINGS)
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return initial observations.
        
        Returns:
            observations: Array of shape (num_ghosts, observation_dim)
            info: Dictionary with episode information
        """
        self._ensure_initialized()
        self._build_game_engine()
        
        # Enable RL mode for all ghosts
        for ghost in self.game_engine.ghosts:
            ghost.set_rl_mode(True)
        
        self._frame_count = 0
        self._episode_count += 1
        self._episode_start_time = 0  # Reset episode timer
        
        # Update reward calculator episode for curriculum learning
        self._reward_calculator.set_episode(self._episode_count)
        
        # Initialize reward snapshots
        self._reward_snapshots = []
        screen_width, screen_height = self.screen.get_size()
        for i, ghost in enumerate(self.game_engine.ghosts):
            other_ghosts = [g for j, g in enumerate(self.game_engine.ghosts) if i != j]
            snapshot = GhostRewardCalculator.create_snapshot(
                ghost, self.game_engine.player, other_ghosts, screen_width, screen_height
            )
            self._reward_snapshots.append(snapshot)
        
        self._last_breakdowns = [GhostRewardBreakdown() for _ in range(self.num_ghosts)]
        
        # Set player to chase mode immediately (skip ready phase)
        self.game_engine.player.set_to_chase()
        
        # Get initial observations
        observations = self._get_observations()
        info = self._build_info()
        
        return observations, info
    
    def step(
        self, 
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        """Execute actions for all ghosts and advance the game.
        
        Args:
            actions: Array of shape (num_ghosts,) with action for each ghost
        
        Returns:
            observations: Array of shape (num_ghosts, observation_dim)
            rewards: Array of shape (num_ghosts,) with reward for each ghost
            terminated: Whether episode ended (Pacman caught or died)
            truncated: Whether episode was truncated (max steps)
            info: Dictionary with step information
        """
        if not self._initialized or self.game_engine is None:
            raise RuntimeError("Call reset() before step().")
        
        if len(actions) != self.num_ghosts:
            raise ValueError(f"Expected {self.num_ghosts} actions, got {len(actions)}")
        
        # Set actions for each ghost
        for ghost, action in zip(self.game_engine.ghosts, actions):
            ghost.set_rl_action(int(action))
        
        total_rewards = np.zeros(self.num_ghosts, dtype=np.float32)
        terminated = False
        truncated = False
        
        # Execute frame skips
        for _ in range(self.config.frame_skip):
            # Only pump events in human render mode
            if self.render_mode == "human":
                pygame.event.pump()
                self.screen.fill(BACKGROUND_COLOR)
            
            # Move player (deterministic AI)
            if self.game_engine.player.is_chasing():
                turned = self.game_engine.player.move(
                    self.screen, 
                    self.game_engine.direction_command
                )
                if not turned:
                    self.game_engine.direction_command = self.game_engine.player.direction
                
                eaten = self.game_engine.player.eat()
                if eaten:
                    from model.eaten_object import EatenObject
                    if eaten == EatenObject.DOT:
                        self.game_engine.level.score += 10
                    elif eaten == EatenObject.BIG_DOT:
                        self.game_engine.level.score += 50
                        self.game_engine.player.powerup_counter = self.game_engine.level.power_up_limit
                        for ghost in self.game_engine.ghosts:
                            ghost.set_to_frightened()
                
                if not self.game_engine.player.powerup:
                    for ghost in self.game_engine.ghosts:
                        if not ghost.is_eaten():
                            ghost.set_to_chase()
            
            # Move ghosts (RL controlled)
            for ghost in self.game_engine.ghosts:
                ghost.follow_target()
            
            # Check collisions
            self.game_engine.check_ghosts_and_player_collision()
            
            # Render if needed
            if self.render_mode == "human":
                self.game_engine.render_level()
                self.game_engine.draw_misc()
                self.game_engine.render_player()
                self.game_engine.render_ghosts()
                if self._clock is not None:
                    self._clock.tick(FPS)
                pygame.display.flip()
            
            self._frame_count += 1
            
            # Check termination conditions
            if self.game_engine.game_over or self.game_engine.player.is_eaten():
                terminated = True
            
            if self.config.max_episode_steps and self._frame_count >= self.config.max_episode_steps:
                truncated = True
            
            # Calculate rewards for each ghost
            screen_width, screen_height = self.screen.get_size()
            step_rewards = self._calculate_rewards(actions, screen_width, screen_height)
            total_rewards += step_rewards
            
            if terminated or truncated:
                break
        
        observations = self._get_observations()
        info = self._build_info()
        
        return observations, total_rewards, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the current state."""
        if not self._initialized or self.screen is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        if self.render_mode == "human":
            pygame.display.flip()
            return None
        
        array = pygame.surfarray.array3d(self.screen)
        return np.transpose(array, (1, 0, 2))
    
    def close(self) -> None:
        """Clean up resources."""
        if self._initialized:
            pygame.quit()
            self._initialized = False
            self.screen = None
            self.game_engine = None
            self._clock = None
    
    def seed(self, value: int) -> None:
        """Set random seed."""
        np.random.seed(value)
    
    def _ensure_initialized(self) -> None:
        """Initialize pygame and screen if not already done."""
        if self._initialized:
            return
        
        if not pygame.get_init():
            pygame.init()
            # Quit mixer to prevent sounds in headless mode
            if self.render_mode != "human":
                try:
                    pygame.mixer.quit()
                except pygame.error:
                    pass
        
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode(RESOLUTION)
            self._clock = pygame.time.Clock()
        else:
            # Headless mode - use a surface, no display needed
            self.screen = pygame.Surface(RESOLUTION)
            self._clock = None
        
        self._initialized = True
    
    def _build_game_engine(self) -> None:
        """Build a new game engine instance."""
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
        self.game_engine.game_over = False
        
        # Set player to use random movements for variety
        self.game_engine.direction_command = Direction.LEFT
    
    def _get_observations(self) -> np.ndarray:
        """Get observations for all ghosts."""
        if self.game_engine is None:
            raise RuntimeError("Game engine not initialized")
        
        screen_width, screen_height = self.screen.get_size()
        obs_packs = self._obs_builder.build_all(
            self.game_engine.ghosts,
            self.game_engine.player,
            self.game_engine.board,
            screen_width,
            screen_height,
        )
        
        # Stack observation vectors
        observations = np.stack([pack.vector for pack in obs_packs], axis=0)
        return observations
    
    def _calculate_rewards(
        self, 
        actions: np.ndarray,
        screen_width: int,
        screen_height: int,
    ) -> np.ndarray:
        """Calculate rewards for all ghosts."""
        rewards = np.zeros(self.num_ghosts, dtype=np.float32)
        
        for i, ghost in enumerate(self.game_engine.ghosts):
            # Get current snapshot
            other_ghosts = [g for j, g in enumerate(self.game_engine.ghosts) if i != j]
            curr_snapshot = GhostRewardCalculator.create_snapshot(
                ghost, self.game_engine.player, other_ghosts, screen_width, screen_height
            )
            
            # Compute reward with personality bonus
            breakdown = self._reward_calculator.compute_with_personality(
                self._reward_snapshots[i],
                curr_snapshot,
                actions[i],
                ghost.get_original_direction(),
            )
            
            rewards[i] = breakdown.total
            self._last_breakdowns[i] = breakdown
            self._reward_snapshots[i] = curr_snapshot
        
        return rewards
    
    def _build_info(self) -> Dict[str, Any]:
        """Build info dictionary."""
        info = {
            "episode": self._episode_count,
            "frame": self._frame_count,
            "score": self.game_engine.level.score if self.game_engine else 0,
            "personality_weight": self._reward_calculator.get_personality_weight(),
            "time_elapsed": self._frame_count,  # Steps since episode start
        }
        
        # Add catch metrics if Pacman was caught
        if self.game_engine and self.game_engine.player.is_eaten():
            info["pacman_caught"] = True
            info["time_to_catch"] = self._frame_count
        else:
            info["pacman_caught"] = False
        
        # Add reward breakdowns for each ghost
        ghost_names = ["blinky", "pinky", "inky", "clyde"]
        for i, (name, breakdown) in enumerate(zip(ghost_names, self._last_breakdowns)):
            info[f"{name}_reward"] = breakdown.to_dict()
        
        return info


class VectorizedGhostEnv:
    """Vectorized environment wrapper for parallel training.
    
    Runs multiple GhostEnv instances in parallel for efficient data collection.
    """
    
    def __init__(
        self,
        num_envs: int,
        config: Optional[GhostEnvConfig] = None,
        render_mode: str = "none",
        seed_offset: int = 0,
    ):
        self.num_envs = num_envs
        self.config = config or GhostEnvConfig()
        
        # Create environments with different seeds
        self.envs = []
        for i in range(num_envs):
            env_config = GhostEnvConfig(
                frame_skip=self.config.frame_skip,
                max_episode_steps=self.config.max_episode_steps,
                reward_config=self.config.reward_config,
                seed=self.config.seed + seed_offset + i if self.config.seed else None,
            )
            # Only render the first environment if requested
            mode = render_mode if i == 0 else "none"
            self.envs.append(GhostEnv(env_config, render_mode=mode))
    
    @property
    def num_ghosts(self) -> int:
        return self.envs[0].num_ghosts
    
    @property
    def observation_dim(self) -> int:
        return self.envs[0].observation_dim
    
    @property
    def action_dim(self) -> int:
        return self.envs[0].action_dim
    
    def reset(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Reset all environments.
        
        Returns:
            observations: Array of shape (num_envs, num_ghosts, observation_dim)
            infos: List of info dicts, one per environment
        """
        observations = []
        infos = []
        
        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)
        
        return np.stack(observations, axis=0), infos
    
    def step(
        self, 
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Step all environments.
        
        Args:
            actions: Array of shape (num_envs, num_ghosts)
        
        Returns:
            observations: Array of shape (num_envs, num_ghosts, observation_dim)
            rewards: Array of shape (num_envs, num_ghosts)
            terminateds: Array of shape (num_envs,)
            truncateds: Array of shape (num_envs,)
            infos: List of info dicts
        """
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for i, env in enumerate(self.envs):
            obs, rew, term, trunc, info = env.step(actions[i])
            observations.append(obs)
            rewards.append(rew)
            terminateds.append(term)
            truncateds.append(trunc)
            infos.append(info)
            
            # Auto-reset if done
            if term or trunc:
                obs, info = env.reset()
                observations[-1] = obs
                infos[-1].update(info)
        
        return (
            np.stack(observations, axis=0),
            np.stack(rewards, axis=0),
            np.array(terminateds, dtype=np.float32),
            np.array(truncateds, dtype=np.float32),
            infos,
        )
    
    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()
