from __future__ import annotations

import unittest

import numpy as np

from rl.env import PacmanEnv
from rl.observation import GHOST_STATE_ORDER, PACMAN_STATE_ORDER


class ObservationBuilderTest(unittest.TestCase):
    def test_pack_observation_contains_structured_and_vector_views(self) -> None:
        env = PacmanEnv(
            render_mode="none",
            observation_mode="pack",
            include_board_in_observation=True,
            frame_skip=1,
        )

        try:
            observation, info = env.reset()

            self.assertTrue(hasattr(observation, "structured"))
            self.assertTrue(hasattr(observation, "vector"))

            structured = observation.structured
            vector = observation.vector

            self.assertIn("pacman", structured)
            self.assertIn("ghosts", structured)
            self.assertIn("board_consumables", structured)
            self.assertEqual(len(structured["ghosts"]), len(env.game_engine.ghosts))

            base_features = 2 + 2 + 4 + len(PACMAN_STATE_ORDER) + 1 + 1 + 1 + 1
            ghost_features = len(env.game_engine.ghosts) * (
                2 + 4 + len(GHOST_STATE_ORDER) + 1 + 1
            )
            board_size = env.game_engine.board.size
            expected_length = base_features + ghost_features + board_size

            self.assertEqual(vector.shape, (expected_length,))
            self.assertEqual(vector.dtype, np.float32)

            board_consumables = np.asarray(structured["board_consumables"], dtype=np.int8)
            self.assertEqual(board_consumables.size, board_size)
            self.assertEqual(structured.get("board_shape"), list(board_consumables.shape))
        finally:
            env.close()

    def test_vector_mode_excludes_board_when_disabled(self) -> None:
        env = PacmanEnv(
            render_mode="none",
            observation_mode="vector",
            include_board_in_observation=False,
            frame_skip=1,
        )

        try:
            observation, _ = env.reset()
            self.assertIsInstance(observation, np.ndarray)

            ghost_count = len(env.game_engine.ghosts)
            base_features = 2 + 2 + 4 + len(PACMAN_STATE_ORDER) + 1 + 1 + 1 + 1
            ghost_features = ghost_count * (2 + 4 + len(GHOST_STATE_ORDER) + 1 + 1)
            expected_length = base_features + ghost_features

            self.assertEqual(observation.shape, (expected_length,))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
