# Policy Gradient (REINFORCE + baseline) for Airbag Triggering

This is a policy-gradient counterpart to the offline Q-learning pipeline
(`training_v2.py` / `testing_v2.py`), built to compare against it on the
same dataset without touching any Q-learning code, checkpoints, or outputs.
It lives entirely under `policy_gradient/` — its own checkpoints, plots,
and results directories — so the two methods never collide.

## Dataset

Trained on `data/aggregated/rl_trajectories_b15.0_c15.0_b25.0_c25.0_c35.0_eta0.15.pkl`
— the "all 5.0" reward configuration (`b1=c1=b2=c2=c3=5.0`, `eta=0.15`), same
trajectories used for `checkpoints/model/q_net_b15.0_c15.0_b25.0_c25.0_c35.0_eta0.15.pth`.
State/action space is identical to the Q-learning setup:

- State (4-dim): `walker_vel_ms`, `ego_vel_ms`, `dx`, `dy`
- Action (2-way): `0 = wait`, `1 = trigger`
- Reward: 0 everywhere except the terminal step of each trajectory (first
  trigger, or collision), where it's `+5`, `-5`, or `0` depending on the
  trigger/collision/injury-risk outcome (see `data/transform_reward.py`).

## Method

Because reward is zero everywhere except the terminal transition, the
discounted Monte-Carlo return-to-go collapses to an exact, bootstrap-free
credit assignment: `G_t = gamma^(T-t) * r_T`. That makes vanilla REINFORCE
a natural fit here (no need for TD bootstrapping the way DQN needs it):

- **Actor** (`PolicyNetwork`): outputs `{wait, trigger}` logits, updated via
  policy gradient `-log π(a_t|s_t) * A_t`, plus an entropy bonus
  (`--entropy_coef`, default `0.01`) so the policy doesn't collapse to
  always-wait — trigger actions are only ~0.4% of transitions in the
  offline data (2069 / 503099).
- **Critic** (`ValueNetwork`): a state-value baseline `V(s)` regressed
  toward `G_t` via MSE, used only to form the advantage
  `A_t = G_t - V(s_t)` (normalized per batch) for variance reduction —
  it is not bootstrapped and does not affect the actor's target directly.

**Caveat:** the offline dataset was collected under a fixed random
trigger policy (`trigger_prob=0.005` in `data/transform_reward.py`), not
the policy being learned. This REINFORCE implementation reuses the logged
`(state, action, return)` tuples directly without importance-sampling
correction for that mismatch — the same simplifying assumption the
existing fitted-Q pipeline already makes by treating the offline
transitions as valid Bellman-backup data. Worth keeping in mind when
comparing sample efficiency/bias between the two methods, not just
end-metric error rates.

Evaluation uses the exact same Type-I / Type-II error definitions as the
Q-learning pipeline (`evaluate_errors` in `training_v2.py`), so numbers
from the two methods are directly comparable:

- **Type-I** (false trigger): triggered on a trajectory with no collision,
  or a low-injury-risk collision.
- **Type-II** (missed trigger): never triggered on a high-injury-risk
  collision.

## Files

- `train.py` — trains actor + critic on the aggregated trajectories,
  tracks Type-I/II error each epoch on a held-out split, saves a
  checkpoint and a loss/error plot.
- `test.py` — rolls the trained policy out on `data/simulation/`
  trajectories frame-by-frame (mirrors `testing_v2.py`), computing true
  injury risk via the Ford pedestrian GP model and saving per-trajectory
  results to CSV.

## Usage

```bash
conda activate RL   # or whichever env has torch/pandas/tqdm

# Train (defaults to the b1=c1=b2=c2=c3=5.0, eta=0.15 dataset)
python policy_gradient/train.py

# Evaluate the trained policy on simulation trajectories
python policy_gradient/test.py
```

Both scripts accept the same `--b1 --c1 --b2 --c2 --c3 --eta` flags as
`training_v2.py`/`testing_v2.py` to point at a different reward config —
just make sure the corresponding `data/aggregated/rl_trajectories_*.pkl`
already exists (generate it with `data/transform_reward.py` if not).

Outputs:

- `policy_gradient/checkpoints/model/pg_net_{suffix}.pth`
- `policy_gradient/plots/PG_loss_{suffix}.png`
- `policy_gradient/results/evaluation_results_{suffix}.csv`

### Performance note

On a CPU-only shared login node, PyTorch defaults to using every visible
core for even tiny per-trajectory forward passes, which causes severe
thread-contention slowdowns (observed: ~15 min/epoch uncapped vs. ~26
sec/epoch after capping). Both scripts call `torch.set_num_threads(8)`
when no GPU is visible to avoid this. For a full run, prefer a GPU node:

```bash
conda activate RL
sbatch jobs/policy_gradient/train_pg.sh
```

This mirrors `jobs/sensitivity/j0.sh` (same reward config, same
`--hidden_dim 256 --batch_size 20000`, 1000 epochs), just pointed at
`policy_gradient/train.py` and `policy_gradient/test.py` instead of the
Q-learning scripts. Logs go to `checkpoints/logs/policy_gradient_train.log`
alongside the existing Q-learning job logs (distinct filename, so nothing
existing is overwritten).

## Comparing to Q-learning

Both pipelines report Type-I/Type-II error with identical semantics, so
the fastest comparison is:

```bash
python training_v2.py --b1 5.0 --c1 5.0 --b2 5.0 --c2 5.0 --c3 5.0 --eta 0.15   # if not already trained
python testing_v2.py  --b1 5.0 --c1 5.0 --b2 5.0 --c2 5.0 --c3 5.0 --eta 0.15

python policy_gradient/train.py
python policy_gradient/test.py
```

then diff `results/evaluation_results_b15.0_c15.0_b25.0_c25.0_c35.0_eta0.15.csv`
against `policy_gradient/results/evaluation_results_b15.0_c15.0_b25.0_c25.0_c35.0_eta0.15.csv`.
