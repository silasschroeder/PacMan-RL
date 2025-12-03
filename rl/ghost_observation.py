"""Ghost observation builder for RL training.

This module creates observations from each ghost's perspective, including:
- Ghost's own position, direction, and state
- Pacman's position, direction, and powerup status
- Other ghosts' positions and states
- Power pellet locations and distances
- Local board structure (walls, pellets)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from model.direction import Direction
from model.entity.ghost.ghost import Ghost
from model.entity.player.player import Player


@dataclass
class GhostObservationPack:
    """Container for ghost observations in different formats."""
    
    structured: Dict[str, Any]
    vector: np.ndarray


class GhostObservationBuilder:
    """Builds observations for ghost RL agents."""
    
    def __init__(self, normalize_positions: bool = True):
        self.normalize_positions = normalize_positions
    
    def build(
        self, 
        ghost: Ghost, 
        player: Player, 
        other_ghosts: List[Ghost],
        board: np.ndarray,
        screen_width: int,
        screen_height: int,
    ) -> GhostObservationPack:
        """Build observation for a single ghost."""
        
        # Ghost's own state
        ghost_struct = self._build_ghost_state(ghost, screen_width, screen_height)
        
        # Player state
        player_struct = self._build_player_state(player, screen_width, screen_height)
        
        # Other ghosts' states
        other_ghosts_struct = [
            self._build_ghost_state(g, screen_width, screen_height) 
            for g in other_ghosts
        ]
        
        # Power pellet information
        power_pellets = self._find_power_pellets(board, ghost, screen_width, screen_height)
        
        # Distance calculations
        distances = self._calculate_distances(ghost, player, other_ghosts, screen_width, screen_height)
        
        # Local grid information (walls/pellets around ghost)
        local_grid = self._get_local_grid(ghost, board, screen_width, screen_height)
        
        structured = {
            "ghost": ghost_struct,
            "player": player_struct,
            "other_ghosts": other_ghosts_struct,
            "power_pellets": power_pellets,
            "distances": distances,
            "local_grid": local_grid,
        }
        
        vector = self._build_vector(structured)
        
        return GhostObservationPack(structured=structured, vector=vector)
    
    def build_all(
        self,
        ghosts: List[Ghost],
        player: Player,
        board: np.ndarray,
        screen_width: int,
        screen_height: int,
    ) -> List[GhostObservationPack]:
        """Build observations for all ghosts."""
        observations = []
        for i, ghost in enumerate(ghosts):
            other_ghosts = [g for j, g in enumerate(ghosts) if i != j]
            obs = self.build(ghost, player, other_ghosts, board, screen_width, screen_height)
            observations.append(obs)
        return observations
    
    def _build_ghost_state(self, ghost: Ghost, screen_width: int, screen_height: int) -> Dict[str, Any]:
        """Extract ghost state information."""
        if self.normalize_positions:
            pos_x = ghost.location_x / screen_width
            pos_y = ghost.location_y / screen_height
        else:
            pos_x = ghost.location_x
            pos_y = ghost.location_y
        
        return {
            "position": [pos_x, pos_y],
            "direction": self._direction_one_hot(ghost.direction),
            "state": ghost.state.name,
            "state_id": self._ghost_state_to_id(ghost.state.name),
            "is_frightened": ghost.is_frightened(),
            "is_eaten": ghost.is_eaten(),
            "is_in_house": ghost.is_in_house(),
            "velocity": ghost.velocity / 8.0,  # Normalize by max velocity
        }
    
    def _build_player_state(self, player: Player, screen_width: int, screen_height: int) -> Dict[str, Any]:
        """Extract player state information."""
        if self.normalize_positions:
            pos_x = player.location_x / screen_width
            pos_y = player.location_y / screen_height
        else:
            pos_x = player.location_x
            pos_y = player.location_y
        
        return {
            "position": [pos_x, pos_y],
            "direction": self._direction_one_hot(player.direction),
            "powerup_active": player.powerup,
            "powerup_counter": player.powerup_counter,
            "powerup_counter_normalized": min(player.powerup_counter, 600) / 600.0,
        }
    
    def _find_power_pellets(
        self, 
        board: np.ndarray, 
        ghost: Ghost,
        screen_width: int,
        screen_height: int
    ) -> Dict[str, Any]:
        """Find power pellets and calculate distances."""
        # Power pellets are marked as 2 in the board
        pellet_positions = np.argwhere(board == 2)
        
        if len(pellet_positions) == 0:
            return {
                "count": 0,
                "nearest_distance": 1.0,  # Max normalized distance
                "nearest_position": [0.5, 0.5],  # Center as default
            }
        
        # Convert board coordinates to screen coordinates
        tile_height = screen_height // board.shape[0]
        tile_width = screen_width // board.shape[1]
        
        pellet_screen_positions = []
        for row, col in pellet_positions:
            screen_x = col * tile_width + tile_width // 2
            screen_y = row * tile_height + tile_height // 2
            pellet_screen_positions.append((screen_x, screen_y))
        
        # Find nearest pellet
        distances = []
        for px, py in pellet_screen_positions:
            dist = np.sqrt((ghost.location_x - px)**2 + (ghost.location_y - py)**2)
            distances.append(dist)
        
        nearest_idx = np.argmin(distances)
        nearest_dist = distances[nearest_idx]
        nearest_pos = pellet_screen_positions[nearest_idx]
        
        # Normalize distance (max distance is diagonal of screen)
        max_dist = np.sqrt(screen_width**2 + screen_height**2)
        normalized_dist = nearest_dist / max_dist
        
        if self.normalize_positions:
            nearest_pos = [nearest_pos[0] / screen_width, nearest_pos[1] / screen_height]
        
        return {
            "count": len(pellet_positions),
            "nearest_distance": normalized_dist,
            "nearest_position": list(nearest_pos),
        }
    
    def _calculate_distances(
        self,
        ghost: Ghost,
        player: Player,
        other_ghosts: List[Ghost],
        screen_width: int,
        screen_height: int,
    ) -> Dict[str, Any]:
        """Calculate normalized distances to key entities."""
        max_dist = np.sqrt(screen_width**2 + screen_height**2)
        
        # Distance to player
        player_dist = np.sqrt(
            (ghost.location_x - player.location_x)**2 + 
            (ghost.location_y - player.location_y)**2
        )
        
        # Distances to other ghosts
        other_ghost_dists = []
        for other in other_ghosts:
            dist = np.sqrt(
                (ghost.location_x - other.location_x)**2 + 
                (ghost.location_y - other.location_y)**2
            )
            other_ghost_dists.append(dist / max_dist)
        
        # Directional offset to player (where is player relative to ghost?)
        dx = (player.location_x - ghost.location_x) / max_dist
        dy = (player.location_y - ghost.location_y) / max_dist
        
        return {
            "to_player": player_dist / max_dist,
            "to_player_dx": dx,
            "to_player_dy": dy,
            "to_other_ghosts": other_ghost_dists,
            "min_ghost_distance": min(other_ghost_dists) if other_ghost_dists else 1.0,
        }
    
    def _get_local_grid(
        self,
        ghost: Ghost,
        board: np.ndarray,
        screen_width: int,
        screen_height: int,
        radius: int = 3,
    ) -> np.ndarray:
        """Extract local grid information around the ghost."""
        tile_height = screen_height // board.shape[0]
        tile_width = screen_width // board.shape[1]
        
        # Ghost's grid position
        grid_x = int(ghost.location_x // tile_width)
        grid_y = int(ghost.location_y // tile_height)
        
        # Extract local patch
        rows, cols = board.shape
        local = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
        
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                r = grid_y + i
                c = grid_x + j
                if 0 <= r < rows and 0 <= c < cols:
                    # 0 = empty, 1 = pellet, 2 = power pellet, 3-8 = walls, 9 = gate
                    cell_value = board[r, c]
                    # Simplify: wall (3-8, 9) -> -1, empty (0) -> 0, pellet (1) -> 0.5, power (2) -> 1
                    if cell_value >= 3:
                        local[i + radius, j + radius] = -1.0
                    elif cell_value == 2:
                        local[i + radius, j + radius] = 1.0
                    elif cell_value == 1:
                        local[i + radius, j + radius] = 0.5
                    else:
                        local[i + radius, j + radius] = 0.0
                else:
                    local[i + radius, j + radius] = -1.0  # Out of bounds treated as wall
        
        return local
    
    def _build_vector(self, structured: Dict[str, Any]) -> np.ndarray:
        """Convert structured observation to flat vector."""
        features: List[float] = []
        
        # Ghost state (8 features)
        features.extend(structured["ghost"]["position"])
        features.extend(structured["ghost"]["direction"])
        features.append(float(structured["ghost"]["is_frightened"]))
        features.append(float(structured["ghost"]["is_in_house"]))
        features.append(structured["ghost"]["velocity"])
        
        # Player state (8 features)
        features.extend(structured["player"]["position"])
        features.extend(structured["player"]["direction"])
        features.append(float(structured["player"]["powerup_active"]))
        features.append(structured["player"]["powerup_counter_normalized"])
        
        # Distance information (6 features: 3 to player + 3 to other ghosts)
        features.append(structured["distances"]["to_player"])
        features.append(structured["distances"]["to_player_dx"])
        features.append(structured["distances"]["to_player_dy"])
        features.extend(structured["distances"]["to_other_ghosts"])
        
        # Power pellet info (3 features)
        features.append(float(structured["power_pellets"]["count"]))
        features.append(structured["power_pellets"]["nearest_distance"])
        
        # Local grid (7x7 = 49 features)
        local_grid = structured["local_grid"].flatten()
        features.extend(local_grid.tolist())
        
        return np.asarray(features, dtype=np.float32)
    
    @staticmethod
    def _direction_one_hot(direction: Direction) -> List[int]:
        """Convert direction to one-hot encoding."""
        mapping = [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN]
        return [1 if direction == d else 0 for d in mapping]
    
    @staticmethod
    def _ghost_state_to_id(state_name: str) -> int:
        """Convert ghost state name to integer ID."""
        mapping = {"CHASE": 0, "FRIGHTENED": 1, "SCATTER": 2, "EATEN": 3}
        return mapping.get(state_name, 0)
    
    @staticmethod
    def get_observation_dim() -> int:
        """Return the dimension of the observation vector."""
        # 8 (ghost) + 8 (player) + 6 (distances) + 3 (power pellets) + 49 (local grid)
        return 74
