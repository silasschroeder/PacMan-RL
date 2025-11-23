from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class ReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Uniform replay buffer storing fixed-size transitions."""

    def __init__(self, capacity: int, state_shape: Tuple[int, ...]) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = int(capacity)
        self._state_shape = tuple(state_shape)

        self._states = np.zeros((self._capacity, *self._state_shape), dtype=np.float32)
        self._actions = np.zeros(self._capacity, dtype=np.int64)
        self._rewards = np.zeros(self._capacity, dtype=np.float32)
        self._next_states = np.zeros((self._capacity, *self._state_shape), dtype=np.float32)
        self._dones = np.zeros(self._capacity, dtype=np.bool_)

        self._index = 0
        self._filled = 0

    def __len__(self) -> int:
        return self._filled

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        idx = self._index
        self._states[idx] = np.asarray(state, dtype=np.float32)
        self._actions[idx] = int(action)
        self._rewards[idx] = float(reward)
        self._next_states[idx] = np.asarray(next_state, dtype=np.float32)
        self._dones[idx] = bool(done)

        self._index = (self._index + 1) % self._capacity
        self._filled = min(self._filled + 1, self._capacity)

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> ReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._filled < batch_size:
            raise ValueError("not enough samples to draw the requested batch")

        indices = np.random.choice(self._filled, size=batch_size, replace=False)
        device = torch.device(device)

        states = torch.as_tensor(self._states[indices], device=device)
        actions = torch.as_tensor(self._actions[indices], device=device)
        rewards = torch.as_tensor(self._rewards[indices], device=device)
        next_states = torch.as_tensor(self._next_states[indices], device=device)
        dones = torch.as_tensor(self._dones[indices].astype(np.float32), device=device)

        return ReplayBatch(states, actions, rewards, next_states, dones)
