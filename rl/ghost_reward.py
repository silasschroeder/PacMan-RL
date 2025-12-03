"""Ghost reward calculator for RL training.

Rewards include:
- Getting closer to Pacman (when in chase mode)
- Catching Pacman
- Avoiding Pacman when vulnerable (frightened state)
- Following original personality-based behavior (curriculum learning)
- Team coordination (staying spread out to corner Pacman)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from model.direction import Direction
    from model.entity.ghost.ghost import Ghost
    from model.entity.player.player import Player


@dataclass
class GhostRewardConfig:
    """Configuration for ghost reward shaping."""
    
    # Movement rewards
    distance_reduction_reward: float = 1.0  # Reward for getting closer to Pacman
    distance_increase_penalty: float = -0.5  # Penalty for moving away from Pacman
    
    # Outcome rewards
    catch_pacman_reward: float = 100.0  # Reward for catching Pacman
    caught_by_pacman_penalty: float = -50.0  # Penalty for being eaten
    
    # Safety rewards (when frightened)
    frightened_distance_reward: float = 2.0  # Reward for fleeing when vulnerable
    frightened_distance_penalty: float = -3.0  # Penalty for approaching when vulnerable
    
    # Personality adherence (curriculum learning)
    personality_match_reward: float = 0.5  # Reward for matching original behavior
    personality_mismatch_penalty: float = -0.2  # Penalty for deviating
    personality_weight_start: float = 1.0  # Initial weight for personality reward
    personality_weight_end: float = 0.1  # Final weight for personality reward
    personality_decay_episodes: int = 5000  # Episodes over which to decay personality weight
    
    # Team coordination
    coordination_reward: float = 0.3  # Reward for maintaining good spacing from other ghosts
    clustering_penalty: float = -0.2  # Penalty for being too close to other ghosts
    optimal_ghost_distance: float = 0.15  # Optimal normalized distance between ghosts
    
    # Step penalty
    step_penalty: float = -0.01  # Small penalty for each step to encourage efficiency


@dataclass
class GhostRewardSnapshot:
    """State snapshot for calculating reward deltas."""
    
    ghost_position: tuple[float, float]
    player_position: tuple[float, float]
    distance_to_player: float
    is_frightened: bool
    is_eaten: bool
    player_alive: bool
    ghost_alive: bool  # Not eaten
    other_ghost_positions: List[tuple[float, float]]
    

@dataclass
class GhostRewardBreakdown:
    """Detailed breakdown of reward components."""
    
    distance_reward: float = 0.0
    catch_reward: float = 0.0
    safety_reward: float = 0.0
    personality_reward: float = 0.0
    coordination_reward: float = 0.0
    step_penalty: float = 0.0
    
    @property
    def total(self) -> float:
        return (
            self.distance_reward +
            self.catch_reward +
            self.safety_reward +
            self.personality_reward +
            self.coordination_reward +
            self.step_penalty
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "distance_reward": self.distance_reward,
            "catch_reward": self.catch_reward,
            "safety_reward": self.safety_reward,
            "personality_reward": self.personality_reward,
            "coordination_reward": self.coordination_reward,
            "step_penalty": self.step_penalty,
            "total": self.total,
        }


class GhostRewardCalculator:
    """Calculates shaped rewards for ghost agents."""
    
    def __init__(self, config: GhostRewardConfig):
        self.config = config
        self._episode_count = 0
    
    def set_episode(self, episode: int) -> None:
        """Update episode counter for curriculum learning."""
        self._episode_count = episode
    
    def get_personality_weight(self) -> float:
        """Calculate current personality weight based on curriculum schedule."""
        if self._episode_count >= self.config.personality_decay_episodes:
            return self.config.personality_weight_end
        
        # Linear decay from start to end weight
        progress = self._episode_count / self.config.personality_decay_episodes
        weight = self.config.personality_weight_start - (
            self.config.personality_weight_start - self.config.personality_weight_end
        ) * progress
        
        return weight
    
    def compute(
        self,
        prev_snapshot: GhostRewardSnapshot,
        curr_snapshot: GhostRewardSnapshot,
        action_taken: int,
        original_direction: Direction,
    ) -> GhostRewardBreakdown:
        """Compute reward for a ghost's transition."""
        breakdown = GhostRewardBreakdown()
        
        # Step penalty
        breakdown.step_penalty = self.config.step_penalty
        
        # Check if Pacman was caught
        if prev_snapshot.player_alive and not curr_snapshot.player_alive:
            breakdown.catch_reward = self.config.catch_pacman_reward
        
        # Check if ghost was eaten
        if prev_snapshot.ghost_alive and not curr_snapshot.ghost_alive:
            breakdown.catch_reward = self.config.caught_by_pacman_penalty
        
        # Distance-based rewards (only if both are alive)
        if curr_snapshot.player_alive and curr_snapshot.ghost_alive:
            breakdown.distance_reward = self._compute_distance_reward(
                prev_snapshot, curr_snapshot
            )
        
        # Safety rewards when frightened
        if curr_snapshot.is_frightened and curr_snapshot.player_alive:
            breakdown.safety_reward = self._compute_safety_reward(
                prev_snapshot, curr_snapshot
            )
        
        # Coordination reward
        if curr_snapshot.player_alive and curr_snapshot.ghost_alive:
            breakdown.coordination_reward = self._compute_coordination_reward(
                curr_snapshot
            )
        
        return breakdown
    
    def compute_with_personality(
        self,
        prev_snapshot: GhostRewardSnapshot,
        curr_snapshot: GhostRewardSnapshot,
        action_taken: int,
        original_direction: Direction,
    ) -> GhostRewardBreakdown:
        """Compute reward including personality adherence bonus."""
        breakdown = self.compute(prev_snapshot, curr_snapshot, action_taken, original_direction)
        
        # Add personality reward
        personality_weight = self.get_personality_weight()
        if personality_weight > 0:
            breakdown.personality_reward = self._compute_personality_reward(
                action_taken, original_direction
            ) * personality_weight
        
        return breakdown
    
    def _compute_distance_reward(
        self,
        prev_snapshot: GhostRewardSnapshot,
        curr_snapshot: GhostRewardSnapshot,
    ) -> float:
        """Reward for getting closer to Pacman."""
        if curr_snapshot.is_frightened:
            return 0.0  # Don't reward approaching when frightened
        
        distance_delta = prev_snapshot.distance_to_player - curr_snapshot.distance_to_player
        
        if distance_delta > 0:  # Got closer
            return self.config.distance_reduction_reward * distance_delta
        else:  # Got farther
            return self.config.distance_increase_penalty * abs(distance_delta)
    
    def _compute_safety_reward(
        self,
        prev_snapshot: GhostRewardSnapshot,
        curr_snapshot: GhostRewardSnapshot,
    ) -> float:
        """Reward for fleeing when vulnerable."""
        distance_delta = curr_snapshot.distance_to_player - prev_snapshot.distance_to_player
        
        if distance_delta > 0:  # Increased distance (fleeing)
            return self.config.frightened_distance_reward * distance_delta
        else:  # Decreased distance (approaching danger)
            return self.config.frightened_distance_penalty * abs(distance_delta)
    
    def _compute_personality_reward(
        self,
        action_taken: int,
        original_direction: Direction,
    ) -> float:
        """Reward for matching original personality-based behavior."""
        # Map action to direction
        from model.direction import Direction
        action_to_direction = {
            0: None,  # STAY
            1: Direction.LEFT,
            2: Direction.RIGHT,
            3: Direction.UP,
            4: Direction.DOWN,
        }
        
        action_direction = action_to_direction.get(action_taken)
        
        if action_direction is None or original_direction is None:
            return 0.0
        
        # Exact match
        if action_direction == original_direction:
            return self.config.personality_match_reward
        
        # Opposite direction (worst case)
        opposites = {
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
        }
        
        if action_direction == opposites.get(original_direction):
            return self.config.personality_mismatch_penalty * 2.0
        
        # Perpendicular direction (medium penalty)
        return self.config.personality_mismatch_penalty
    
    def _compute_coordination_reward(
        self,
        curr_snapshot: GhostRewardSnapshot,
    ) -> float:
        """Reward for maintaining good spacing from other ghosts."""
        if not curr_snapshot.other_ghost_positions:
            return 0.0
        
        # Calculate distances to other ghosts
        ghost_pos = curr_snapshot.ghost_position
        distances = []
        for other_pos in curr_snapshot.other_ghost_positions:
            dist = np.sqrt(
                (ghost_pos[0] - other_pos[0])**2 +
                (ghost_pos[1] - other_pos[1])**2
            )
            distances.append(dist)
        
        # Reward for being near optimal distance
        # Penalty for being too close (clustering)
        total_reward = 0.0
        for dist in distances:
            if dist < self.config.optimal_ghost_distance * 0.5:
                # Too close - clustering penalty
                total_reward += self.config.clustering_penalty
            elif abs(dist - self.config.optimal_ghost_distance) < 0.05:
                # Near optimal distance
                total_reward += self.config.coordination_reward
        
        return total_reward
    
    @staticmethod
    def create_snapshot(
        ghost: Ghost,
        player: Player,
        other_ghosts: List[Ghost],
        screen_width: int,
        screen_height: int,
    ) -> GhostRewardSnapshot:
        """Create a snapshot from current game state."""
        # Normalize positions
        ghost_pos = (ghost.location_x / screen_width, ghost.location_y / screen_height)
        player_pos = (player.location_x / screen_width, player.location_y / screen_height)
        
        # Calculate distance
        distance = np.sqrt(
            (ghost.location_x - player.location_x)**2 +
            (ghost.location_y - player.location_y)**2
        )
        max_dist = np.sqrt(screen_width**2 + screen_height**2)
        normalized_distance = distance / max_dist
        
        # Other ghost positions
        other_positions = [
            (g.location_x / screen_width, g.location_y / screen_height)
            for g in other_ghosts
        ]
        
        return GhostRewardSnapshot(
            ghost_position=ghost_pos,
            player_position=player_pos,
            distance_to_player=normalized_distance,
            is_frightened=ghost.is_frightened(),
            is_eaten=ghost.is_eaten(),
            player_alive=not player.is_eaten(),
            ghost_alive=not ghost.is_eaten(),
            other_ghost_positions=other_positions,
        )
