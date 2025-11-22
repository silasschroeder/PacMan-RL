from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from rl.env import RewardConfig


@dataclass(frozen=True)
class RewardSnapshot:
    """State summary used to derive reward deltas between steps."""

    score: int
    dot_count: int
    power_pellet_count: int
    score_multiplier: int
    lives: int
    powerup_active: bool

    @staticmethod
    def from_engine(game_engine: any) -> "RewardSnapshot":
        board = game_engine.board
        dot_count = int(np.count_nonzero(board == 1))
        power_pellet_count = int(np.count_nonzero(board == 2))
        player = game_engine.player
        return RewardSnapshot(
            score=game_engine.level.score,
            dot_count=dot_count,
            power_pellet_count=power_pellet_count,
            score_multiplier=player.score_multiplier,
            lives=max(player.lives, 0),
            powerup_active=player.powerup,
        )


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components for a single agent step."""

    score_delta: int = 0
    score: float = 0.0
    step_penalty: float = 0.0
    pellets: float = 0.0
    power_pellets: float = 0.0
    ghost_multiplier: float = 0.0
    life_penalty: float = 0.0
    powerup_activation: float = 0.0

    @staticmethod
    def zero() -> "RewardBreakdown":
        return RewardBreakdown()

    @property
    def total(self) -> float:
        return (
            self.score
            + self.step_penalty
            + self.pellets
            + self.power_pellets
            + self.ghost_multiplier
            + self.life_penalty
            + self.powerup_activation
        )

    def to_dict(self, include_total: bool = True) -> Dict[str, float]:
        data: Dict[str, float] = {
            "score_delta": float(self.score_delta),
            "score": self.score,
            "step_penalty": self.step_penalty,
            "pellets": self.pellets,
            "power_pellets": self.power_pellets,
            "ghost_multiplier": self.ghost_multiplier,
            "life_penalty": self.life_penalty,
            "powerup_activation": self.powerup_activation,
        }
        if include_total:
            data["total"] = self.total
        return data

    def __add__(self, other: "RewardBreakdown") -> "RewardBreakdown":
        return RewardBreakdown(
            score_delta=self.score_delta + other.score_delta,
            score=self.score + other.score,
            step_penalty=self.step_penalty + other.step_penalty,
            pellets=self.pellets + other.pellets,
            power_pellets=self.power_pellets + other.power_pellets,
            ghost_multiplier=self.ghost_multiplier + other.ghost_multiplier,
            life_penalty=self.life_penalty + other.life_penalty,
            powerup_activation=self.powerup_activation + other.powerup_activation,
        )


class RewardCalculator:
    """Utility responsible for computing shaped rewards between snapshots."""

    def __init__(self, config: "RewardConfig") -> None:
        self._config = config

    def compute(self, prev: RewardSnapshot, curr: RewardSnapshot) -> RewardBreakdown:
        score_delta = curr.score - prev.score
        score_term = score_delta * self._config.score_scale
        step_penalty = self._config.step_penalty if score_delta == 0 else 0.0

        dots_consumed = max(prev.dot_count - curr.dot_count, 0)
        pellets_term = dots_consumed * self._config.pellet_reward

        power_pellets_consumed = max(prev.power_pellet_count - curr.power_pellet_count, 0)
        power_pellet_term = power_pellets_consumed * self._config.power_pellet_reward

        ghost_increment = max(curr.score_multiplier - prev.score_multiplier, 0)
        ghost_term = ghost_increment * self._config.ghost_reward

        lives_lost = max(prev.lives - curr.lives, 0)
        life_penalty = lives_lost * self._config.life_lost_penalty

        powerup_activation = (
            self._config.power_pellet_reward
            if curr.powerup_active and not prev.powerup_active
            else 0.0
        )

        return RewardBreakdown(
            score_delta=score_delta,
            score=score_term,
            step_penalty=step_penalty,
            pellets=pellets_term,
            power_pellets=power_pellet_term,
            ghost_multiplier=ghost_term,
            life_penalty=life_penalty,
            powerup_activation=powerup_activation,
        )
