# Evolutionary Algorithm Quick Start Guide

## Prerequisites

Ensure dependencies are installed:

```bash
pip install -r requirements.txt
```

Required packages: `pygame`, `numpy`, `torch`

## 1. Quick Demo (2-3 minutes)

Run a minimal example to verify the implementation:

```bash
python -m examples.quick_evolutionary_demo
```

This trains a small population (20) for 5 generations and shows fitness progression.

## 2. Full Training Run (~15-30 minutes)

Train a GA agent with recommended settings:

```bash
python train_evolutionary.py \
  --generations 50 \
  --population-size 100 \
  --output runs/my_first_ga
```

**What happens:**

- 50 generations of evolution
- 100 individuals per generation
- Each individual evaluated over 3 episodes (1500 steps each)
- Best genome saved to `runs/my_first_ga/best.pt`
- Metrics logged to `runs/my_first_ga/metrics.json`

**Expected output:**

```
Starting evolutionary training: 50 generations, population 100

=== Generation 1/50 ===
  Evaluated 100/100 individuals
  Fitness: best=-245.30, mean=-512.15, worst=-892.40, std=145.23
  ✓ New best fitness: -245.30

=== Generation 2/50 ===
...
```

## 3. Evaluate Performance

After training, evaluate the best agent:

```bash
python evaluate_evolutionary.py \
  --checkpoint runs/my_first_ga/best.pt \
  --episodes 20
```

**Output:**

```json
{
  "episodes": 20,
  "average_reward": 150.5,
  "std_reward": 45.2,
  "min_reward": 80.0,
  "max_reward": 220.0,
  "average_length": 450.5
}
```

## 4. Watch the Agent Play

Visualize the trained agent in action:

```bash
python play_evolutionary.py runs/my_first_ga/best.pt --episodes 3
```

A pygame window will open showing Pacman controlled by your evolved agent.

## 5. Compare with DQN

Train a DQN agent for comparison:

```bash
python train_agent.py \
  --episodes 200 \
  --output runs/my_first_dqn
```

Then compare results:

```bash
python compare_results.py \
  --dqn-metrics runs/my_first_dqn/metrics.json \
  --ga-metrics runs/my_first_ga/metrics.json
```

**Output:**

```
============================================================
TRAINING COMPARISON: DQN vs. Genetic Algorithm
============================================================

DQN Statistics:
  Total episodes:     200
  Final reward:       180.50
  Max reward:         220.30
  ...

Genetic Algorithm Statistics:
  Total generations:  50
  Final best fitness: 165.20
  Max fitness ever:   195.40
  ...

Comparison:
  Winner (final):     DQN
  Winner (peak):      DQN
============================================================
```

## 6. Visualize Learning Curves (Optional)

If you have matplotlib installed:

```bash
pip install matplotlib

python plot_comparison.py \
  --dqn runs/my_first_dqn/metrics.json \
  --ga runs/my_first_ga/metrics.json \
  --output learning_curves.png
```

## Configuration Files

Create custom configurations for repeated experiments:

**`my_config.json`:**

```json
{
  "generations": 75,
  "population_size": 150,
  "max_steps": 2000,
  "fitness_episodes": 5,
  "mutation_rate": 0.03,
  "seed": 123
}
```

Train with config:

```bash
python train_evolutionary.py \
  --config my_config.json \
  --output runs/custom_ga
```

## Common CLI Flags

### Training (`train_evolutionary.py`)

- `--generations N`: Number of generations (default: 50)
- `--population-size N`: Population size (default: 100)
- `--max-steps N`: Steps per episode (default: 1500)
- `--fitness-episodes N`: Episodes per evaluation (default: 3)
- `--mutation-rate F`: Mutation rate (default: 0.05)
- `--output DIR`: Output directory for checkpoints
- `--eval`: Run evaluation after training
- `--seed N`: Random seed for reproducibility

### Evaluation (`evaluate_evolutionary.py`)

- `--checkpoint PATH`: Path to .pt file
- `--episodes N`: Number of episodes (default: 10)
- `--epsilon F`: Exploration rate (default: 0.0)
- `--output DIR`: Save evaluation JSON

### Playback (`play_evolutionary.py`)

- `checkpoint`: Path to .pt file (positional)
- `--episodes N`: Number of episodes to render (default: 1)
- `--epsilon F`: Exploration during playback (default: 0.0)

## Troubleshooting

### "ModuleNotFoundError: No module named 'pygame'"

Install dependencies: `pip install -r requirements.txt`

### Training is very slow

- Reduce `--population-size` (try 50)
- Reduce `--fitness-episodes` (try 2)
- Reduce `--max-steps` (try 1000)
- Increase `--frame-skip` (try 3 or 4)

### Low/negative fitness scores

- Check reward configuration in config file
- Ensure `reward_config` matches DQN setup
- Increase `--fitness-episodes` for more stable estimates
- Try different `--seed` values

### Population not improving

- Increase `--generations` (try 100)
- Adjust mutation rate (try 0.03 or 0.08)
- Check that best fitness is being tracked correctly
- Monitor population diversity (`std_fitness` metric)

## Next Steps

1. **Experiment with hyperparameters**: Try different mutation rates, population sizes, and selection pressures
2. **Modify reward function**: Edit `reward_config` in your JSON config
3. **Compare multiple runs**: Train with different seeds and aggregate results
4. **Implement enhancements**: Add novelty search, multi-objective optimization, or adaptive mutation
5. **Run unit tests**: `python tests/test_genetic.py` to verify implementation

## File Structure Reference

```
PacMan-RL/
├── rl/
│   ├── agents/
│   │   ├── genetic.py          # GA agent implementation
│   │   └── __init__.py         # Exports GA classes
│   └── evolutionary_training.py # Training pipeline
├── configs/
│   └── evolutionary_default.json # Default config
├── docs/
│   └── evolutionary_guide.md    # Detailed guide
├── tests/
│   └── test_genetic.py         # Unit tests
├── examples/
│   └── quick_evolutionary_demo.py
├── train_evolutionary.py       # Training script
├── evaluate_evolutionary.py    # Evaluation script
├── play_evolutionary.py        # Visualization script
├── compare_results.py          # DQN vs GA comparison
└── plot_comparison.py          # Learning curve plotting
```

## Resources

- **Detailed guide**: See `docs/evolutionary_guide.md`
- **Implementation summary**: See `IMPLEMENTATION_SUMMARY.md`
- **DQN comparison**: See README.md section on "Comparing DQN vs. Genetic Algorithm"
- **Unit tests**: Run `python tests/test_genetic.py`

## Support

For issues or questions:

1. Check `docs/evolutionary_guide.md` for detailed explanations
2. Review unit tests in `tests/test_genetic.py` for usage examples
3. Examine `examples/quick_evolutionary_demo.py` for minimal working example
