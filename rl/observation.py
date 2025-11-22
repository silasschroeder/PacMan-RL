from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from model.direction import Direction
from model.entity.ghost.ghost import Ghost
from model.entity.player.player import Player
from settings import POWER_UP_LIMIT


PACMAN_STATE_ORDER: Sequence[str] = ("READY", "CHASE", "EATEN")
GHOST_STATE_ORDER: Sequence[str] = ("CHASE", "FRIGHTENED", "SCATTER", "EATEN")
MAX_LIVES = 5
FRAME_SCALE = 1000.0
SCORE_SCALE = 1000.0
SCORE_MULTIPLIER_SCALE = 8.0


@dataclass
class ObservationPack:
    """Container bundling structured and vectorised observations."""

    structured: Dict[str, Any]
    vector: np.ndarray


class ObservationBuilder:
    """Builds structured and flat observations from the game engine state."""

    def __init__(self, include_board: bool = True, board_dtype: np.dtype = np.float32) -> None:
        self.include_board = include_board
        self.board_dtype = board_dtype

    def build(self, game_engine: Any, frame_index: int) -> ObservationPack:
        player: Player = game_engine.player
        ghosts: Iterable[Ghost] = game_engine.ghosts
        screen_width, screen_height = game_engine.screen.get_size()

        consumable_grid = self._consumable_grid(game_engine.board)
        pellets_remaining = int(np.count_nonzero(consumable_grid == 1))
        power_pellets_remaining = int(np.count_nonzero(consumable_grid == 2))

        pacman_struct = {
            "position": [player.location_x / screen_width, player.location_y / screen_height],
            "direction": self._direction_one_hot(player.direction),
            "state": player.state.name,
            "state_id": PACMAN_STATE_ORDER.index(player.state.name),
            "lives": max(player.lives, 0),
            "lives_normalised": np.clip(player.lives, 0, MAX_LIVES) / MAX_LIVES,
            "powerup_active": player.powerup,
            "score_multiplier": player.score_multiplier,
            "powerup_counter": max(player.powerup_counter, 0),
            "powerup_counter_normalised": self._normalise_powerup_counter(player.powerup_counter),
        }

        ghost_structs: List[Dict[str, Any]] = []
        for ghost in ghosts:
            ghost_structs.append(
                {
                    "name": ghost.__class__.__name__,
                    "position": [ghost.location_x / screen_width, ghost.location_y / screen_height],
                    "direction": self._direction_one_hot(ghost.direction),
                    "state": ghost.state.name,
                    "state_id": GHOST_STATE_ORDER.index(ghost.state.name),
                    "is_in_house": ghost.is_in_house(),
                    "is_frightened": ghost.is_frightened(),
                }
            )

        structured: Dict[str, Any] = {
            "frame": frame_index,
            "score": game_engine.level.score,
            "game_over": game_engine.game_over,
            "pacman": pacman_struct,
            "ghosts": ghost_structs,
            "pellets_remaining": pellets_remaining,
            "power_pellets_remaining": power_pellets_remaining,
        }

        if self.include_board:
            structured["board_consumables"] = consumable_grid.tolist()
            structured["board_shape"] = list(consumable_grid.shape)

        vector = self._build_vector(
            frame_index,
            game_engine.level.score,
            pacman_struct,
            ghost_structs,
            consumable_grid if self.include_board else None,
        )

        return ObservationPack(structured=structured, vector=vector)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_vector(
        self,
        frame_index: int,
        score: int,
        pacman_struct: Dict[str, Any],
        ghost_structs: Sequence[Dict[str, Any]],
        consumable_grid: np.ndarray | None,
    ) -> np.ndarray:
        features: List[float] = []

        features.append(frame_index / FRAME_SCALE)
        features.append(score / SCORE_SCALE)
        features.extend(pacman_struct["position"])
        features.extend(pacman_struct["direction"])
        features.extend(self._state_one_hot(pacman_struct["state"], PACMAN_STATE_ORDER))
        features.append(pacman_struct["lives_normalised"])
        features.append(1.0 if pacman_struct["powerup_active"] else 0.0)
        features.append(
            np.clip(pacman_struct["score_multiplier"], 0.0, SCORE_MULTIPLIER_SCALE) / SCORE_MULTIPLIER_SCALE
        )
        features.append(pacman_struct["powerup_counter_normalised"])

        for ghost in ghost_structs:
            features.extend(ghost["position"])
            features.extend(ghost["direction"])
            features.extend(self._state_one_hot(ghost["state"], GHOST_STATE_ORDER))
            features.append(1.0 if ghost["is_in_house"] else 0.0)
            features.append(1.0 if ghost["is_frightened"] else 0.0)

        if consumable_grid is not None:
            # Normalise to [0, 1] where pellets=0.5, power pellets=1.0
            board_vector = consumable_grid.astype(self.board_dtype).flatten() / 2.0
            features.extend(board_vector.tolist())

        return np.asarray(features, dtype=np.float32)

    @staticmethod
    def _direction_one_hot(direction: Direction) -> List[int]:
        mapping = [Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN]
        return [1 if direction == option else 0 for option in mapping]

    @staticmethod
    def _state_one_hot(state_name: str, ordering: Sequence[str]) -> List[int]:
        return [1 if state_name == candidate else 0 for candidate in ordering]

    @staticmethod
    def _consumable_grid(board: np.ndarray) -> np.ndarray:
        return np.where(board == 1, 1, np.where(board == 2, 2, 0)).astype(np.int8)

    @staticmethod
    def _normalise_powerup_counter(counter: int) -> float:
        if POWER_UP_LIMIT <= 0:
            return 0.0
        return np.clip(max(counter, 0), 0, POWER_UP_LIMIT) / POWER_UP_LIMIT
