# RL_trigger_module

Self-contained inference wrapper around a trained airbag-trigger Q-network.
Only depends on `torch` and `numpy` — no other files from the parent repo are needed.

Bundled checkpoint: `checkpoints/q_net_b15.0_c16.0_b25.0_c25.0_c35.0_eta0.2.pth`
(reward params `B1=5.0, C1=6.0, B2=5.0, C2=5.0, C3=5.0`, `ETA=0.2`).

## State format

`[walker_vel_ms, ego_vel_ms, dx, dy]` (raw, unnormalized — the module handles
normalization internally using stats stored in the checkpoint).

## Usage

```python
from trigger_controller import TriggerController

controller = TriggerController()  # uses bundled checkpoint by default

# 1) Single-state prediction
result = controller.predict([3.0, 12.0, 15.0, 1.0])
# {"action": 0 or 1, "trigger": bool, "q_values": [q_no_trigger, q_trigger]}

# 2) Trajectory evaluation (real-time style — no ground truth needed)
traj_result = controller.predict_trajectory([
    [3.0, 12.0, 30.0, 2.0],
    [3.0, 12.0, 20.0, 1.5],
    [3.0, 12.0, 8.0,  0.5],
    [3.0, 12.0, 2.0,  0.2],
])
# {"trigger": bool, "trigger_step": int or None, "actions": [...], "q_values": [...]}
```

`trigger_step` is the first index at which the model would fire the airbag,
mirroring deployment: once triggered, the decision is final.

To use a different checkpoint, pass its path explicitly:

```python
controller = TriggerController(checkpoint_path="/path/to/other_q_net.pth")
```
