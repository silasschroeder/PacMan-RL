Option A: Increase DQN to match GA

```sh
python train_agent.py --episodes 15000 --output runs/dqn_exp_matched
```

Option B: Decrease GA to match DQN

```sh
python train_evolutionary.py --generations 20 --population-size 10 --fitness-episodes 1 --output runs/ga_exp_matched
```

Option C: Reasonable middle ground

```sh
python train_agent.py --episodes 6000 --output runs/dqn_exp_fair
python train_evolutionary.py --generations 20 --population-size 100 --fitness-episodes 3 --output runs/ga_exp_fair
```
