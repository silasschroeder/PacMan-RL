"""Unit tests for genetic algorithm agent."""

from __future__ import annotations

import numpy as np
import torch

from rl.agents.genetic import GeneticAgent, GeneticConfig, GeneticPopulation


def test_genetic_agent_creation():
    """Test that GeneticAgent can be created with valid config."""
    config = GeneticConfig(
        state_dim=10,
        action_dim=5,
        hidden_sizes=(64, 64),
        population_size=20,
    )
    agent = GeneticAgent(config)
    
    assert agent.config.state_dim == 10
    assert agent.config.action_dim == 5
    assert agent.generation == 0
    print("✓ GeneticAgent creation test passed")


def test_genetic_agent_action_selection():
    """Test that agent can select actions from states."""
    config = GeneticConfig(state_dim=10, action_dim=5)
    agent = GeneticAgent(config)
    
    state = np.random.randn(10).astype(np.float32)
    
    # Test greedy action selection
    action = agent.select_action(state, epsilon=0.0)
    assert isinstance(action, int)
    assert 0 <= action < 5
    
    # Test with exploration
    action = agent.select_action(state, epsilon=0.5)
    assert isinstance(action, int)
    assert 0 <= action < 5
    
    print("✓ GeneticAgent action selection test passed")


def test_genetic_agent_weight_operations():
    """Test getting and setting weights."""
    config = GeneticConfig(state_dim=10, action_dim=5, hidden_sizes=(32,))
    agent = GeneticAgent(config)
    
    # Get weights
    weights = agent.get_weights()
    assert isinstance(weights, np.ndarray)
    assert weights.ndim == 1
    original_size = len(weights)
    
    # Modify weights
    new_weights = np.random.randn(original_size).astype(np.float32)
    agent.set_weights(new_weights)
    
    # Verify weights were updated
    retrieved_weights = agent.get_weights()
    np.testing.assert_allclose(retrieved_weights, new_weights, rtol=1e-5)
    
    print("✓ GeneticAgent weight operations test passed")


def test_genetic_agent_checkpoint():
    """Test saving and loading agent state."""
    config = GeneticConfig(state_dim=10, action_dim=5)
    agent1 = GeneticAgent(config)
    agent1.generation = 42
    
    # Save state
    state_dict = agent1.save_state_dict()
    
    # Create new agent and load state
    agent2 = GeneticAgent(config)
    agent2.load_state_dict(state_dict)
    
    assert agent2.generation == 42
    
    # Verify weights match
    weights1 = agent1.get_weights()
    weights2 = agent2.get_weights()
    np.testing.assert_allclose(weights1, weights2, rtol=1e-5)
    
    print("✓ GeneticAgent checkpoint test passed")


def test_population_initialization():
    """Test that population initializes correctly."""
    config = GeneticConfig(
        state_dim=10,
        action_dim=5,
        population_size=50,
    )
    agent = GeneticAgent(config)
    population = GeneticPopulation(config, agent)
    
    assert len(population.genomes) == 50
    assert len(population.fitnesses) == 50
    assert population.best_fitness == -float("inf")
    
    print("✓ GeneticPopulation initialization test passed")


def test_population_fitness_update():
    """Test updating fitness and tracking best individual."""
    config = GeneticConfig(state_dim=10, action_dim=5, population_size=10)
    agent = GeneticAgent(config)
    population = GeneticPopulation(config, agent)
    
    # Update fitnesses
    population.update_fitness(0, 100.0)
    population.update_fitness(1, 200.0)
    population.update_fitness(2, 150.0)
    
    assert population.fitnesses[0] == 100.0
    assert population.fitnesses[1] == 200.0
    assert population.best_fitness == 200.0
    assert population.best_genome is not None
    
    print("✓ GeneticPopulation fitness update test passed")


def test_population_evolution():
    """Test that evolution creates a new generation."""
    config = GeneticConfig(
        state_dim=10,
        action_dim=5,
        population_size=20,
        elite_fraction=0.2,
        crossover_rate=0.8,
        mutation_rate=0.05,
    )
    agent = GeneticAgent(config)
    population = GeneticPopulation(config, agent)
    
    # Set some fitnesses
    for i in range(20):
        population.update_fitness(i, float(i * 10))
    
    # Store original genomes
    original_genomes = [g.copy() for g in population.genomes]
    
    # Evolve
    population.evolve()
    
    # Check that we still have the same population size
    assert len(population.genomes) == 20
    
    # Check that at least some genomes changed (not all elite)
    changed_count = sum(
        not np.allclose(population.genomes[i], original_genomes[i])
        for i in range(20)
    )
    assert changed_count > 0  # At least some should have changed
    
    print("✓ GeneticPopulation evolution test passed")


def test_population_crossover():
    """Test crossover operation."""
    config = GeneticConfig(state_dim=10, action_dim=5, population_size=10)
    agent = GeneticAgent(config)
    population = GeneticPopulation(config, agent)
    
    # Use the actual genome size from the population
    genome_size = population.genome_size
    parent1 = np.ones(genome_size)
    parent2 = np.zeros(genome_size)
    
    child1, child2 = population._crossover(parent1, parent2)
    
    # Children should be different from parents (unless very unlucky)
    assert not np.allclose(child1, parent1)
    assert not np.allclose(child2, parent2)
    
    # Children should contain mix of parent genes
    assert np.any(child1 == 1.0) and np.any(child1 == 0.0)
    assert np.any(child2 == 1.0) and np.any(child2 == 0.0)
    
    print("✓ GeneticPopulation crossover test passed")


def test_population_mutation():
    """Test mutation operation."""
    config = GeneticConfig(
        state_dim=10,
        action_dim=5,
        population_size=10,
        mutation_rate=0.5,  # High rate for testing
        mutation_std=0.1,
    )
    agent = GeneticAgent(config)
    population = GeneticPopulation(config, agent)
    
    # Use the actual genome size from the population
    genome_size = population.genome_size
    genome = np.zeros(genome_size)
    mutated = population._mutate(genome)
    
    # Some values should have changed due to mutation
    assert not np.allclose(mutated, genome)
    
    # But many should still be close to zero
    assert np.mean(np.abs(mutated)) < 0.5
    
    print("✓ GeneticPopulation mutation test passed")


def test_population_stats():
    """Test population statistics calculation."""
    config = GeneticConfig(state_dim=10, action_dim=5, population_size=10)
    agent = GeneticAgent(config)
    population = GeneticPopulation(config, agent)
    
    # Set known fitnesses
    fitnesses = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    for i, fitness in enumerate(fitnesses):
        population.update_fitness(i, fitness)
    
    stats = population.get_population_stats()
    
    assert stats["best"] == 100.0
    assert stats["worst"] == 10.0
    assert stats["mean"] == 55.0
    assert stats["median"] == 55.0
    assert stats["std"] > 0
    
    print("✓ GeneticPopulation stats test passed")


def run_all_tests():
    """Run all genetic algorithm tests."""
    print("Running Genetic Algorithm Tests")
    print("=" * 60)
    
    test_genetic_agent_creation()
    test_genetic_agent_action_selection()
    test_genetic_agent_weight_operations()
    test_genetic_agent_checkpoint()
    test_population_initialization()
    test_population_fitness_update()
    test_population_evolution()
    test_population_crossover()
    test_population_mutation()
    test_population_stats()
    
    print("=" * 60)
    print("All tests passed! ✓")


if __name__ == "__main__":
    run_all_tests()
