# Example trajectories

Two trajectories pulled from `data/simulation/`, selected using
`results/evaluation_results_b15.0_c16.0_b25.0_c25.0_c35.0_eta0.2.csv`
(checkpoint: `q_net_b15.0_c16.0_b25.0_c25.0_c35.0_eta0.2.pth`).

## trigger_case_102/  (originally `data/simulation/102/`)

Model correctly triggers ahead of a genuine high-risk collision.

- collision: True, at frame 96
- triggered: True, at frame 22 (TTC at trigger = 74 frames)
- pjoint (injury risk): 0.9993  (> eta=0.2 -> high-injury collision)
- speed at trigger: 65.1 km/h, speed at collision: 51.4 km/h
- Q(wait) = 6.21, Q(trigger) = 6.23 at the triggering frame

## no_trigger_case_2/  (originally `data/simulation/2/`)

Model correctly withholds the trigger — no collision ever occurs.

- collision: False
- triggered: False
- No `collision.csv` for this trajectory (no impact).

## Files

Each folder keeps the original `measurments/measurements.csv` (state columns
used by the model: `walker_vel_ms`, `ego_vel_ms`, plus `w_location_x/y` and
`e_location_x/y` used to derive `dx`/`dy`) and, for the trigger case,
`measurments/collision.csv` (collision frame).

To replay either through the standalone module:

```python
import pandas as pd
from RL_trigger_module.trigger_controller import TriggerController

df = pd.read_csv("example/trigger_case_102/measurments/measurements.csv")
df["dx"] = df["w_location_x"] - df["e_location_x"]
df["dy"] = df["w_location_y"] - df["e_location_y"]
states = df[["walker_vel_ms", "ego_vel_ms", "dx", "dy"]].values

controller = TriggerController()
result = controller.predict_trajectory(states)
print(result)
```
