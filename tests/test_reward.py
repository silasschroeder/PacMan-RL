from __future__ import annotations

import unittest

from rl.env import RewardConfig
from rl.reward import RewardBreakdown, RewardCalculator, RewardSnapshot


class RewardCalculatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RewardConfig(
            score_scale=2.0,
            step_penalty=-0.5,
            pellet_reward=1.0,
            power_pellet_reward=3.0,
            ghost_reward=5.0,
            life_lost_penalty=-50.0,
            death_penalty=-500.0,
        )
        self.calculator = RewardCalculator(self.config)

    def test_score_delta_scales_without_step_penalty(self) -> None:
        prev = RewardSnapshot(
            score=100,
            dot_count=10,
            power_pellet_count=2,
            score_multiplier=1,
            lives=3,
            powerup_active=False,
        )
        curr = RewardSnapshot(
            score=110,
            dot_count=10,
            power_pellet_count=2,
            score_multiplier=1,
            lives=3,
            powerup_active=False,
        )

        breakdown = self.calculator.compute(prev, curr)

        self.assertEqual(breakdown.score_delta, 10)
        self.assertEqual(breakdown.score, 20.0)
        self.assertEqual(breakdown.step_penalty, 0.0)
        self.assertEqual(breakdown.total, 20.0)

    def test_step_penalty_applied_when_score_static(self) -> None:
        prev = RewardSnapshot(100, 10, 2, 1, 3, False)
        curr = RewardSnapshot(100, 10, 2, 1, 3, False)

        breakdown = self.calculator.compute(prev, curr)

        self.assertEqual(breakdown.score, 0.0)
        self.assertEqual(breakdown.step_penalty, -0.5)
        self.assertAlmostEqual(breakdown.total, -0.5)

    def test_consumables_multiplier_life_and_powerup_rewards(self) -> None:
        prev = RewardSnapshot(200, 10, 2, 1, 3, False)
        curr = RewardSnapshot(250, 7, 0, 4, 2, True)

        breakdown = self.calculator.compute(prev, curr)

        # Score delta 50 -> scaled by 2.0
        self.assertEqual(breakdown.score_delta, 50)
        self.assertEqual(breakdown.score, 100.0)
        # Pellets eaten: 3 normal, 2 power pellets
        self.assertEqual(breakdown.pellets, 3.0)
        self.assertEqual(breakdown.power_pellets, 6.0)
        # Score multiplier increase 3 -> 5 each
        self.assertEqual(breakdown.ghost_multiplier, 15.0)
        # Life lost 1 * -50
        self.assertEqual(breakdown.life_penalty, -50.0)
        # Powerup activation bonus
        self.assertEqual(breakdown.powerup_activation, 3.0)

        expected_total = sum(
            [
                breakdown.score,
                breakdown.step_penalty,
                breakdown.pellets,
                breakdown.power_pellets,
                breakdown.ghost_multiplier,
                breakdown.life_penalty,
                breakdown.powerup_activation,
            ]
        )
        self.assertAlmostEqual(breakdown.total, expected_total)

        as_dict = breakdown.to_dict()
        self.assertIn("total", as_dict)
        self.assertAlmostEqual(as_dict["total"], expected_total)

    def test_breakdown_zero_factory(self) -> None:
        zero = RewardBreakdown.zero()
        self.assertEqual(zero.total, 0.0)
        for value in zero.to_dict().values():
            self.assertEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
