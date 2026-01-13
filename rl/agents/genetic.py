from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from rl.agents.dqn import QNetwork


@dataclass
class GeneticConfig:
    """Configuration for Genetic Algorithm agent."""

    state_dim: int
    action_dim: int
    hidden_sizes: Sequence[int] = (256, 256)
    population_size: int = 100
    elite_fraction: float = 0.1  # Top 10% preserved
    tournament_size: int = 5
    crossover_rate: float = 0.8
    mutation_rate: float = 0.05
    mutation_std: float = 0.1
    device: str = "cpu"


class GeneticAgent:
    """Genetic Algorithm agent using neuroevolution with fixed topology.

    Uses the same network architecture as DQN for fair comparison, but
    optimizes weights via evolutionary operators instead of backpropagation.
    """

    def __init__(self, config: GeneticConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

        # Policy network (no target network needed for GA)
        self.policy_net = QNetwork(
            config.state_dim, config.action_dim, config.hidden_sizes
        ).to(self.device)

        self.generation = 0

    def select_action(
        self, state: np.ndarray | torch.Tensor, epsilon: float = 0.0
    ) -> int:
        """Select action using current policy network.

        Args:
            state: Current state observation
            epsilon: Exploration rate (for compatibility; typically 0 for GA)

        Returns:
            Selected action index
        """
        # Optional epsilon-greedy for evaluation
        if epsilon > 0 and np.random.rand() < epsilon:
            return int(np.random.randint(0, self.config.action_dim))

        state_tensor = self._to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        action = int(torch.argmax(q_values, dim=1).item())
        return action

    def get_weights(self) -> np.ndarray:
        """Extract all network weights as a flat numpy array (genome)."""
        weights = []
        for param in self.policy_net.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

    def set_weights(self, weights: np.ndarray) -> None:
        """Load weights from a flat numpy array into the network."""
        offset = 0
        for param in self.policy_net.parameters():
            param_size = param.numel()
            param_data = weights[offset : offset + param_size].reshape(param.shape)
            param.data.copy_(torch.from_numpy(param_data).to(param.device))
            offset += param_size

    def save_state_dict(self) -> dict[str, torch.Tensor]:
        """Save agent state for checkpointing."""
        return {
            "policy": self.policy_net.state_dict(),
            "generation": torch.tensor(self.generation),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load agent state from checkpoint."""
        self.policy_net.load_state_dict(state["policy"])
        self.generation = int(state["generation"].item())

    def _to_tensor(self, array_like: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert array to tensor on the correct device."""
        if isinstance(array_like, torch.Tensor):
            return array_like.to(self.device, dtype=torch.float32)
        return torch.as_tensor(array_like, device=self.device, dtype=torch.float32)


class GeneticPopulation:
    """Manages a population of genomes for genetic algorithm evolution."""

    def __init__(self, config: GeneticConfig, template_agent: GeneticAgent) -> None:
        self.config = config
        self.genome_size = len(template_agent.get_weights())

        # Initialize population with random weights
        self.genomes: List[np.ndarray] = []
        self.fitnesses: List[float] = []

        # Initialize population from template + noise
        template_weights = template_agent.get_weights()
        for _ in range(config.population_size):
            noise = np.random.randn(self.genome_size) * 0.1
            genome = template_weights + noise
            self.genomes.append(genome)

        self.fitnesses = [0.0] * config.population_size
        self.best_genome: np.ndarray | None = None
        self.best_fitness: float = -float("inf")

    def update_fitness(self, index: int, fitness: float) -> None:
        """Update fitness for a specific individual."""
        self.fitnesses[index] = fitness

        # Track best individual
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_genome = self.genomes[index].copy()

    def evolve(self) -> None:
        """Create next generation via selection, crossover, and mutation."""
        elite_count = max(1, int(self.config.elite_fraction * self.config.population_size))
        new_genomes: List[np.ndarray] = []

        # Sort by fitness (descending)
        sorted_indices = np.argsort(self.fitnesses)[::-1]
        sorted_genomes = [self.genomes[i] for i in sorted_indices]
        sorted_fitnesses = [self.fitnesses[i] for i in sorted_indices]

        # Elitism: preserve top performers
        for i in range(elite_count):
            new_genomes.append(sorted_genomes[i].copy())

        # Generate offspring to fill remaining slots
        while len(new_genomes) < self.config.population_size:
            # Tournament selection for parents
            parent1 = self._tournament_selection(sorted_genomes, sorted_fitnesses)
            parent2 = self._tournament_selection(sorted_genomes, sorted_fitnesses)

            # Crossover
            if np.random.rand() < self.config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            child1 = self._mutate(child1)
            if len(new_genomes) + 1 < self.config.population_size:
                child2 = self._mutate(child2)
                new_genomes.extend([child1, child2])
            else:
                new_genomes.append(child1)

        self.genomes = new_genomes[: self.config.population_size]
        self.fitnesses = [0.0] * self.config.population_size

    def _tournament_selection(
        self, sorted_genomes: List[np.ndarray], sorted_fitnesses: List[float]
    ) -> np.ndarray:
        """Select a genome via rank-based tournament selection."""
        # Sample tournament_size individuals
        tournament_size = min(self.config.tournament_size, len(sorted_genomes))
        # Use rank-based selection: better ranks have exponentially higher probability
        ranks = np.arange(len(sorted_genomes))
        # Higher rank = better fitness (since sorted descending)
        probabilities = np.exp(-ranks / (len(sorted_genomes) / 4))
        probabilities = probabilities / probabilities.sum()

        selected_indices = np.random.choice(
            len(sorted_genomes), size=tournament_size, replace=False, p=probabilities
        )
        # Choose the best from tournament (lowest index = highest fitness)
        winner_index = selected_indices[np.argmin(selected_indices)]
        return sorted_genomes[winner_index].copy()

    def _crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform uniform crossover between two parent genomes."""
        mask = np.random.rand(self.genome_size) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        """Apply Gaussian mutation to genome."""
        mutation_mask = np.random.rand(self.genome_size) < self.config.mutation_rate
        noise = np.random.randn(self.genome_size) * self.config.mutation_std
        genome = genome + mutation_mask * noise
        return genome

    def get_population_stats(self) -> dict[str, float]:
        """Return statistics about current population fitness."""
        fitnesses = np.array(self.fitnesses)
        return {
            "best": float(np.max(fitnesses)),
            "worst": float(np.min(fitnesses)),
            "mean": float(np.mean(fitnesses)),
            "std": float(np.std(fitnesses)),
            "median": float(np.median(fitnesses)),
        }
