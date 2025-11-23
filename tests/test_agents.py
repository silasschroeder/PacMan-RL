from __future__ import annotations

import unittest

import numpy as np
import torch

from rl.agents import DQNAgent, DQNConfig, ReplayBuffer


class ReplayBufferTest(unittest.TestCase):
    def test_push_and_sample_shapes(self) -> None:
        buffer = ReplayBuffer(capacity=16, state_shape=(5,))

        for idx in range(12):
            buffer.push(
                state=np.full(5, idx, dtype=np.float32),
                action=idx % 4,
                reward=float(idx),
                next_state=np.full(5, idx + 1, dtype=np.float32),
                done=idx % 3 == 0,
            )

        batch = buffer.sample(batch_size=8)

        self.assertEqual(batch.states.shape, (8, 5))
        self.assertEqual(batch.next_states.shape, (8, 5))
        self.assertEqual(batch.actions.shape, (8,))
        self.assertEqual(batch.rewards.shape, (8,))
        self.assertEqual(batch.dones.shape, (8,))
        self.assertTrue(torch.all((batch.actions >= 0) & (batch.actions < 4)))

    def test_sample_raises_when_insufficient(self) -> None:
        buffer = ReplayBuffer(capacity=4, state_shape=(3,))
        with self.assertRaises(ValueError):
            buffer.sample(batch_size=2)


class DQNAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)

    def test_select_action_greedy_and_train_step(self) -> None:
        config = DQNConfig(state_dim=3, action_dim=2, hidden_sizes=(32,), epsilon_start=0.0)
        agent = DQNAgent(config)

        buffer = ReplayBuffer(capacity=64, state_shape=(3,))
        for _ in range(32):
            state = np.random.rand(3).astype(np.float32)
            action = np.random.randint(0, config.action_dim)
            reward = np.random.randn()
            next_state = np.random.rand(3).astype(np.float32)
            done = bool(np.random.rand() < 0.1)
            buffer.push(state, action, reward, next_state, done)

        action = agent.select_action(np.zeros(3, dtype=np.float32))
        self.assertIn(action, range(config.action_dim))

        loss = agent.train_step(buffer, batch_size=16)
        self.assertIsInstance(loss, float)
        self.assertTrue(np.isfinite(loss))

    def test_target_network_updates(self) -> None:
        config = DQNConfig(state_dim=4, action_dim=3, hidden_sizes=(16,), tau=0.5)
        agent = DQNAgent(config)

        for param in agent.policy_net.parameters():
            nn = param.data
            nn.fill_(1.0)
        for param in agent.target_net.parameters():
            param.data.zero_()

        agent.update_target(soft=False)
        for target_param, policy_param in zip(agent.target_net.parameters(), agent.policy_net.parameters()):
            self.assertTrue(torch.allclose(target_param, policy_param))

        for param in agent.target_net.parameters():
            param.data.zero_()
        agent.update_target(soft=True)
        for target_param in agent.target_net.parameters():
            self.assertTrue(torch.allclose(target_param, torch.full_like(target_param, 0.5)))


if __name__ == "__main__":
    unittest.main()
