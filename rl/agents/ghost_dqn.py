"""Specialized DQN agent for ghost training.

This is a thin wrapper around the base DQN agent with ghost-specific functionality.
The main difference is that this agent handles multiple ghost observations per step.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from rl.agents.dqn import DQNAgent, DQNConfig
from rl.agents.replay_buffer import ReplayBuffer


class GhostDQNAgent:
    """DQN agent for controlling ghosts.
    
    This agent uses a shared policy for all ghosts. Each ghost gets its own
    observation but they all use the same Q-network.
    """
    
    def __init__(self, config: DQNConfig):
        self.config = config
        self.base_agent = DQNAgent(config)
    
    def select_actions(
        self, 
        states: np.ndarray, 
        epsilon: Optional[float] = None
    ) -> np.ndarray:
        """Select actions for multiple ghosts.
        
        Args:
            states: Array of shape (num_ghosts, state_dim)
            epsilon: Optional epsilon override
        
        Returns:
            actions: Array of shape (num_ghosts,)
        """
        eps = epsilon if epsilon is not None else self.base_agent.epsilon
        actions = np.zeros(len(states), dtype=np.int64)
        
        for i, state in enumerate(states):
            if np.random.rand() < eps:
                actions[i] = np.random.randint(0, self.config.action_dim)
            else:
                state_tensor = self.base_agent._to_tensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.base_agent.policy_net(state_tensor)
                actions[i] = int(torch.argmax(q_values, dim=1).item())
        
        return actions
    
    def train_step(
        self,
        buffer: ReplayBuffer,
        batch_size: int,
        gamma: Optional[float] = None,
    ) -> float:
        """Perform a training step."""
        return self.base_agent.train_step(buffer, batch_size, gamma)
    
    def decay_epsilon(self) -> None:
        """Decay epsilon for exploration."""
        self.base_agent.decay_epsilon()
    
    def update_target(self, soft: Optional[bool] = None) -> None:
        """Update target network."""
        self.base_agent.update_target(soft)
    
    @property
    def epsilon(self) -> float:
        """Current epsilon value."""
        return self.base_agent.epsilon
    
    @epsilon.setter
    def epsilon(self, value: float) -> None:
        """Set epsilon value."""
        self.base_agent.epsilon = value
    
    @property
    def training_steps(self) -> int:
        """Number of training steps."""
        return self.base_agent.training_steps
    
    def save_state_dict(self) -> dict[str, torch.Tensor]:
        """Save agent state."""
        return self.base_agent.save_state_dict()
    
    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load agent state."""
        self.base_agent.load_state_dict(state)


class GhostReplayBuffer:
    """Replay buffer for ghost training.
    
    Handles storage of experiences from multiple ghosts per step.
    Each ghost's experience is stored independently.
    """
    
    def __init__(
        self, 
        capacity: int, 
        state_shape: tuple[int, ...],
        num_ghosts: int = 4,
    ):
        self.capacity = capacity
        self.state_shape = state_shape
        self.num_ghosts = num_ghosts
        
        # Store experiences from all ghosts in a single buffer
        self.buffer = ReplayBuffer(capacity, state_shape)
    
    def push_multi(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Push experiences from multiple ghosts.
        
        Args:
            states: Array of shape (num_ghosts, state_dim)
            actions: Array of shape (num_ghosts,)
            rewards: Array of shape (num_ghosts,)
            next_states: Array of shape (num_ghosts, state_dim)
            dones: Array of shape (num_ghosts,)
        """
        # Add each ghost's experience individually
        for i in range(len(states)):
            self.buffer.push(
                states[i],
                int(actions[i]),
                float(rewards[i]),
                next_states[i],
                bool(dones[i]),
            )
    
    def sample(self, batch_size: int, device: str | torch.device = "cpu"):
        """Sample a batch from the buffer."""
        return self.buffer.sample(batch_size, device)
    
    def __len__(self) -> int:
        return len(self.buffer)


class VectorizedGhostReplayBuffer:
    """Replay buffer for vectorized ghost environments.
    
    Handles experiences from multiple parallel environments, each with multiple ghosts.
    """
    
    def __init__(
        self,
        capacity: int,
        state_shape: tuple[int, ...],
        num_ghosts: int = 4,
    ):
        self.capacity = capacity
        self.state_shape = state_shape
        self.num_ghosts = num_ghosts
        
        # Single shared buffer for all environments and ghosts
        self.buffer = ReplayBuffer(capacity, state_shape)
    
    def push_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Push experiences from vectorized environments.
        
        Args:
            states: Array of shape (num_envs, num_ghosts, state_dim)
            actions: Array of shape (num_envs, num_ghosts)
            rewards: Array of shape (num_envs, num_ghosts)
            next_states: Array of shape (num_envs, num_ghosts, state_dim)
            dones: Array of shape (num_envs,) - same done flag for all ghosts in an env
        """
        num_envs = states.shape[0]
        
        for env_idx in range(num_envs):
            done = bool(dones[env_idx])
            for ghost_idx in range(self.num_ghosts):
                self.buffer.push(
                    states[env_idx, ghost_idx],
                    int(actions[env_idx, ghost_idx]),
                    float(rewards[env_idx, ghost_idx]),
                    next_states[env_idx, ghost_idx],
                    done,
                )
    
    def sample(self, batch_size: int, device: str | torch.device = "cpu"):
        """Sample a batch from the buffer."""
        return self.buffer.sample(batch_size, device)
    
    def __len__(self) -> int:
        return len(self.buffer)
