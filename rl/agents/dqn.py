from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.agents.replay_buffer import ReplayBuffer, ReplayBatch


@dataclass
class DQNConfig:
    state_dim: int
    action_dim: int
    hidden_sizes: Sequence[int] = (256, 256)
    learning_rate: float = 1e-3
    gamma: float = 0.99
    tau: float = 1.0  # 1.0 -> hard update
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 5e-5
    device: str = "cpu"


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = state_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.ReLU())
            last_dim = hidden
        layers.append(nn.Linear(last_dim, action_dim))
        self.model = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=np.sqrt(5))
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -0.1, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DQNAgent:
    """Baseline DQN agent with optional soft target updates."""

    def __init__(self, config: DQNConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

        self.policy_net = QNetwork(config.state_dim, config.action_dim, config.hidden_sizes).to(self.device)
        self.target_net = QNetwork(config.state_dim, config.action_dim, config.hidden_sizes).to(self.device)
        self.update_target(soft=False)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.gamma = config.gamma

        self.epsilon = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay

        self.training_steps = 0

    def select_action(self, state: np.ndarray | torch.Tensor, epsilon: float | None = None) -> int:
        eps = self.epsilon if epsilon is None else epsilon
        if np.random.rand() < eps:
            return int(np.random.randint(0, self.config.action_dim))

        state_tensor = self._to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        action = int(torch.argmax(q_values, dim=1).item())
        return action

    def decay_epsilon(self) -> None:
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon - self.epsilon_decay,
            )

    def update_target(self, soft: bool | None = None) -> None:
        use_soft = self.config.tau < 1.0 if soft is None else soft
        tau = self.config.tau if use_soft else 1.0
        with torch.no_grad():
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                if use_soft:
                    target_param.data.mul_(1.0 - tau)
                    target_param.data.add_(tau * param.data)
                else:
                    target_param.data.copy_(param.data)

    def train_step(
        self,
        buffer: ReplayBuffer,
        batch_size: int,
        gamma: float | None = None,
        device: str | torch.device | None = None,
    ) -> float:
        if len(buffer) < batch_size:
            raise ValueError("ReplayBuffer does not contain enough samples for training")

        batch_device = device or self.device
        batch: ReplayBatch = buffer.sample(batch_size, device=batch_device)

        current_q = self.policy_net(batch.states).gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(batch.next_states)
            max_next_q = next_q_values.max(dim=1).values
            targets = batch.rewards + (1.0 - batch.dones) * (gamma or self.gamma) * max_next_q

        loss = F.mse_loss(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.training_steps += 1
        return float(loss.item())

    def save_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": torch.tensor(self.epsilon),
            "steps": torch.tensor(self.training_steps),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.policy_net.load_state_dict(state["policy"])
        self.target_net.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.epsilon = float(state["epsilon"].item())
        self.training_steps = int(state["steps"].item())

    def _to_tensor(self, array_like: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(array_like, torch.Tensor):
            return array_like.to(self.device, dtype=torch.float32)
        return torch.as_tensor(array_like, device=self.device, dtype=torch.float32)
