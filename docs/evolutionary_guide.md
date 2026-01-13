# Evolutionary Algorithm Implementation Guide

## Overview

This document provides a technical overview of the Genetic Algorithm (GA) implementation for training Pacman agents through neuroevolution.

## Architecture

### Core Components

1. **`rl/agents/genetic.py`** - GA agent and population management

   - `GeneticAgent`: Neural network agent with weight extraction/loading
   - `GeneticConfig`: Configuration dataclass for GA parameters
   - `GeneticPopulation`: Population manager handling evolution operators

2. **`rl/evolutionary_training.py`** - Training pipeline

   - `EvolutionaryConfig`: Training configuration
   - `run_evolutionary_training()`: Main training loop
   - `evaluate_individual()`: Fitness evaluation function

3. **Training Scripts**
   - `train_evolutionary.py`: CLI for training
   - `evaluate_evolutionary.py`: CLI for evaluation
   - `play_evolutionary.py`: CLI for visualization
   - `compare_results.py`: Compare DQN vs GA results

## Algorithm Details

### Network Architecture

Uses identical architecture to DQN for fair comparison:

- Input: State vector (flattened observations)
- Hidden: 256 → 256 neurons with ReLU
- Output: Q-values for 5 actions (STAY, LEFT, RIGHT, UP, DOWN)

Total parameters: ~134,000 weights

### Evolutionary Process

**1. Initialization (Generation 0)**

```
For each of 100 individuals:
  genome[i] = template_weights + random_noise
```

**2. Fitness Evaluation**

```
For each genome in population:
  fitness = average_reward over 3 episodes
  Track best genome across entire population
```

**3. Selection (Rank-Based Tournament)**

```
tournament_selection():
  Sample 5 candidates with rank-based probability
  Higher fitness rank → higher selection probability
  Return best from tournament
```

**4. Reproduction**

```
Elitism (10%):
  Copy top 10 individuals unchanged

Crossover (80% rate):
  parent1, parent2 = tournament_selection()
  child1, child2 = uniform_crossover(parent1, parent2)

Mutation (5% per weight):
  For each weight in genome:
    if random() < 0.05:
      weight += N(0, 0.1)  # Gaussian noise
```

**5. Next Generation**

```
Replace population with elite + offspring
Reset fitness scores
Repeat from step 2
```

## Configuration Parameters

### Population Parameters

- `population_size` (100): Number of individuals per generation
- `elite_fraction` (0.1): Top percentage preserved unchanged
- `tournament_size` (5): Candidates per tournament selection

### Genetic Operators

- `crossover_rate` (0.8): Probability of breeding vs. cloning
- `mutation_rate` (0.05): Per-weight mutation probability
- `mutation_std` (0.1): Gaussian noise standard deviation

### Environment Parameters

- `max_steps` (1500): Steps per episode
- `fitness_episodes` (3): Episodes averaged for fitness
- `frame_skip` (2): Frames skipped per action
- `reward_config`: Same as DQN reward shaping

### Training Parameters

- `generations` (50): Number of evolutionary generations
- `seed` (42): Random seed for reproducibility
- `device` ("cpu"): Torch device (cpu/cuda)

## Usage Examples

### Basic Training

```bash
# Train with defaults
python train_evolutionary.py --output runs/ga_exp01

# Train with custom parameters
python train_evolutionary.py \
  --generations 100 \
  --population-size 150 \
  --max-steps 2000 \
  --fitness-episodes 5 \
  --output runs/ga_exp02
```

### Using Config Files

```bash
# Create config
cat > my_ga_config.json << EOF
{
  "generations": 75,
  "population_size": 120,
  "max_steps": 1500,
  "fitness_episodes": 3,
  "mutation_rate": 0.03,
  "seed": 123
}
EOF

# Train with config
python train_evolutionary.py \
  --config my_ga_config.json \
  --output runs/ga_custom
```

### Evaluation

```bash
# Evaluate best checkpoint
python evaluate_evolutionary.py \
  --checkpoint runs/ga_exp01/best.pt \
  --episodes 50 \
  --output runs/ga_exp01

# Watch agent play
python play_evolutionary.py \
  runs/ga_exp01/best.pt \
  --episodes 5
```

### Comparing with DQN

```bash
# Train both
python train_agent.py --episodes 200 --output runs/dqn_exp
python train_evolutionary.py --generations 50 --output runs/ga_exp

# Compare results
python compare_results.py \
  --dqn-metrics runs/dqn_exp/metrics.json \
  --ga-metrics runs/ga_exp/metrics.json \
  --output comparison.json
```

## Metrics & Monitoring

### Tracked Metrics

Each generation logs:

- `generation`: Current generation number
- `best_fitness`: Best reward in current generation
- `mean_fitness`: Average fitness across population
- `worst_fitness`: Minimum fitness in population
- `std_fitness`: Fitness standard deviation (diversity)
- `median_fitness`: Median fitness
- `best_fitness_ever`: Best fitness across all generations

### Output Files

```
runs/ga_exp01/
├── best.pt           # Best genome checkpoint
├── latest.pt         # Final generation checkpoint
└── metrics.json      # Per-generation statistics
```

### Checkpoint Contents

```python
checkpoint = {
  "agent": {
    "policy": state_dict,      # Network weights
    "generation": int,         # Generation number
  },
  "config": {
    "generations": 50,
    "population_size": 100,
    # ... all config parameters
  }
}
```

## Performance Considerations

### Computational Cost

**Per Generation:**

- Population evaluations: 100 individuals × 3 episodes × 1500 steps
- Total environment steps: ~450,000 per generation
- Wall time: ~5-15 minutes (CPU, no rendering)

**Parallelization Opportunities:**

- Individual fitness evaluations are independent
- Can parallelize across population using multiprocessing
- Current implementation: sequential (simple, deterministic)

### Memory Usage

- Population genomes: 100 × 134K weights × 4 bytes ≈ 54 MB
- Replay buffer: Not needed (no gradient storage)
- Peak RAM: ~500 MB (including pygame, environment)

## Comparison: DQN vs GA

| Aspect                 | DQN                             | Genetic Algorithm           |
| ---------------------- | ------------------------------- | --------------------------- |
| **Optimization**       | Gradient descent                | Evolutionary operators      |
| **Memory**             | Replay buffer (50K transitions) | Population (100 genomes)    |
| **Sample efficiency**  | High (reuses transitions)       | Low (each episode fresh)    |
| **Parallelization**    | Sequential training             | Embarrassingly parallel     |
| **Hyperparameters**    | Learning rate, epsilon decay    | Mutation rate, crossover    |
| **Convergence**        | Can be unstable                 | More robust to local optima |
| **Computational cost** | Lower per update                | Higher per generation       |

## Advanced Techniques (Not Implemented)

Potential enhancements:

- **Novelty search**: Reward behavioral diversity
- **Multi-objective**: Optimize reward + other metrics
- **Adaptive mutation**: Adjust rates based on population diversity
- **Co-evolution**: Evolve ghost behaviors alongside Pacman
- **Hybrid GA-DQN**: Use GA for hyperparameter optimization

## Troubleshooting

### Low Fitness / No Learning

- Increase `fitness_episodes` for more stable fitness estimates
- Reduce `mutation_rate` if population is too chaotic
- Increase `population_size` for better exploration
- Check reward config matches DQN setup

### Population Convergence (Loss of Diversity)

- Decrease `elite_fraction` to reduce selection pressure
- Increase `mutation_rate` or `mutation_std`
- Increase `tournament_size` for more competitive selection
- Monitor `std_fitness` metric

### Slow Training

- Reduce `population_size` or `fitness_episodes`
- Reduce `max_steps` per episode
- Increase `frame_skip`
- Consider parallelizing fitness evaluations

## References

- **DQN**: Mnih et al. (2015) "Human-level control through deep RL"
- **Neuroevolution**: Stanley & Miikkulainen (2002) "Evolving Neural Networks through Augmenting Topologies"
- **ES**: Salimans et al. (2017) "Evolution Strategies as a Scalable Alternative to RL"
