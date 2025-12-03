# Ghost Agent Performance Metrics

## Primary Metrics

### 1. **Time-to-Catch (Steps)** ⭐

The number of steps it takes for ghosts to catch Pacman in successful episodes.

**Why it's important:**

- Direct measure of ghost efficiency
- Lower is better (faster catches = better performance)
- Only measured when ghosts actually catch Pacman

**Reported statistics:**

- `mean_time_to_catch`: Average across successful catches
- `std_time_to_catch`: Consistency (lower = more consistent)
- `min_time_to_catch`: Fastest catch (best case)
- `max_time_to_catch`: Slowest catch (worst case)
- `median_time_to_catch`: Typical performance (less affected by outliers)

**Example output:**

```
Time-to-Catch Metrics (when successful):
  Mean: 450.3 steps
  Std:  120.5 steps
  Min:  180 steps (fastest catch)
  Max:  890 steps (slowest catch)
  Median: 420.0 steps
```

**Interpretation:**

- **< 400 steps**: Excellent - ghosts catch Pacman quickly
- **400-600 steps**: Good - reasonable catching speed
- **600-800 steps**: Moderate - catching takes time
- **> 800 steps**: Slow - ghosts struggle to catch Pacman

---

### 2. **Catch Rate (Percentage)**

The proportion of episodes where ghosts successfully catch Pacman.

**Why it's important:**

- Measures success rate
- Higher is better
- Complements time-to-catch (you want both high catch rate AND low time)

**Example:**

- `catch_rate: 0.65` = 65% of games end with ghosts catching Pacman

**Interpretation:**

- **> 70%**: Excellent - ghosts dominate
- **50-70%**: Good - balanced gameplay
- **30-50%**: Moderate - ghosts need improvement
- **< 30%**: Poor - ghosts rarely catch Pacman

---

### 3. **Episode Length**

Total number of steps in an episode (whether caught or not).

**Why it's important:**

- Shows game duration
- Shorter episodes with high catch rate = efficient ghosts
- Long episodes with low catch rate = ineffective ghosts

---

## Secondary Metrics

### 4. **Mean Reward**

Total accumulated reward during evaluation.

**What it measures:**

- Combined performance across all reward components
- Includes distance reduction, catch bonuses, coordination, personality, etc.

**Use case:** Track learning progress over time

---

### 5. **Ghost Deaths Per Episode**

How many times ghosts are eaten by Pacman (when vulnerable).

**Why it's important:**

- Lower is better
- Shows if ghosts flee properly when frightened
- High death rate = poor safety behavior

---

### 6. **Per-Ghost Metrics**

Individual performance for each ghost (Blinky, Pinky, Inky, Clyde).

**Includes:**

- Individual rewards
- Personality adherence scores

**Use case:** Identify if specific ghosts underperform

---

## How Metrics Work Together

### Best Case Scenario

```
Catch Rate: 75%
Mean Time-to-Catch: 380 steps
Ghost Deaths: 0.5 per episode
Mean Reward: 450.0
```

**Interpretation:** Ghosts catch Pacman frequently and quickly, rarely die, and work well together.

---

### Training Progress Example

```
Episode 1000:  Catch Rate: 20%, TTC: 950±200
Episode 5000:  Catch Rate: 45%, TTC: 620±150
Episode 10000: Catch Rate: 65%, TTC: 420±80
```

**Interpretation:** Both catch rate increases AND time-to-catch decreases = successful training!

---

### Red Flags

❌ **High catch rate but long time-to-catch**

- Example: 80% catch rate, 1200 steps average
- Problem: Ghosts eventually win but very slowly
- Solution: Increase distance reduction rewards

❌ **Low catch rate with short episode length**

- Example: 15% catch rate, 300 steps average
- Problem: Pacman wins quickly (collects all pellets or ghosts fail)
- Solution: Increase catch rewards, improve coordination

❌ **High ghost deaths**

- Example: 3+ deaths per episode
- Problem: Ghosts don't flee when vulnerable
- Solution: Increase frightened safety rewards

---

## Using Metrics for Tuning

### To Reduce Time-to-Catch

1. Increase `distance_reduction_reward` (1.0 → 2.0)
2. Increase `coordination_reward` (0.3 → 0.5)
3. Reduce `step_penalty` (-0.01 → -0.005)

### To Improve Catch Rate

1. Increase `catch_pacman_reward` (100 → 150)
2. Train longer (more timesteps)
3. Use more parallel environments

### To Improve Efficiency (Both Metrics)

1. Balance rewards carefully
2. Ensure coordination is working
3. Consider larger network architecture

---

## Logging During Training

Training output includes time-to-catch:

```
Step 50,000/1,000,000 | Ep 1250 | Reward: 145.30 ± 45.20 | Catch: 45.0% | TTC: 580±120 | ε: 0.450
```

- `TTC: 580±120` = Mean time-to-catch is 580 steps with std of 120
- If no catches: TTC is omitted from display

---

## Evaluation Output

Running `evaluate_ghost_agent.py` shows detailed metrics:

```json
{
  "episodes": 20,
  "mean_total_reward": 425.5,
  "pacman_catch_rate": 0.65,
  "total_catches": 13,
  "mean_time_to_catch": 420.3,
  "std_time_to_catch": 95.7,
  "min_time_to_catch": 240.0,
  "max_time_to_catch": 680.0,
  "median_time_to_catch": 410.0,
  "mean_episode_length": 512.8,
  "avg_ghost_deaths_per_episode": 0.85
}
```

---

## Best Practices

1. **Track trends over time**: Don't judge from single evaluations
2. **Compare with baselines**: Original deterministic ghosts as reference
3. **Consider trade-offs**: Fast catches might mean more deaths
4. **Balance metrics**: Don't optimize only time-to-catch at expense of catch rate
5. **Use median**: More robust than mean for time-to-catch (avoids outlier influence)

---

## Additional Metrics You Could Add

**If you want more detailed analysis:**

1. **Distance to Pacman over time**: Track how distance changes
2. **Ghost spread/coordination**: Measure spacing between ghosts
3. **Power pellet collection**: How often Pacman gets power-ups
4. **Zone control**: Which areas ghosts dominate
5. **Personality retention**: How much each ghost follows original patterns

These are not currently implemented but could be added to the info dict.
