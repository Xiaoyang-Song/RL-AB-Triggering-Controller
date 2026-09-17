"""
Standalone airbag-trigger inference module.

Wraps a trained Q-network checkpoint (state -> trigger / no-trigger) so it can
be queried in two ways:

  1. predict(state)            -> decision for a single state observation.
  2. predict_trajectory(states) -> real-time-style evaluation of a sequence of
                                    states, reporting whether/when a trigger
                                    fires. No ground-truth labels are assumed
                                    or required (unlike offline evaluation).

State layout (order matters): [walker_vel_ms, ego_vel_ms, dx, dy]
Action: 0 = no trigger, 1 = trigger airbag.

Checkpoint parameters (reward shaping + eta) baked into the bundled weights:
    B1=5.0  C1=6.0  B2=5.0  C2=5.0  C3=5.0  ETA=0.20
"""

import os
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

_DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints",
    "q_net_b15.0_c16.0_b25.0_c25.0_c35.0_eta0.2.pth",
)

STATE_COLUMNS = ["walker_vel_ms", "ego_vel_ms", "dx", "dy"]


class QNetwork(nn.Module):
    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class TriggerController:
    """Loads a Q-network checkpoint and exposes single-state / trajectory inference."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        self.checkpoint_path = checkpoint_path or _DEFAULT_CHECKPOINT
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        hidden_dim = checkpoint.get("hyperparameters", {}).get("hidden_dim", 64)

        self.q_net = QNetwork(hidden_dim=hidden_dim).to(self.device)
        self.q_net.load_state_dict(checkpoint["model_state_dict"])
        self.q_net.eval()

        self.state_mean = checkpoint["state_mean"].to(self.device)
        self.state_std = checkpoint["state_std"].to(self.device)
        self.hyperparameters = checkpoint.get("hyperparameters", {})

    def _normalize(self, states: torch.Tensor) -> torch.Tensor:
        return (states - self.state_mean) / self.state_std

    @torch.no_grad()
    def _forward(self, states: torch.Tensor) -> torch.Tensor:
        """states: (N, 4) raw (unnormalized) tensor -> (N, action_dim) Q-values."""
        states = states.to(self.device)
        normalized = self._normalize(states)
        return self.q_net(normalized)

    def predict(self, state: Union[Sequence[float], np.ndarray]) -> dict:
        """
        Single-state inference.

        Args:
            state: length-4 array-like [walker_vel_ms, ego_vel_ms, dx, dy].

        Returns:
            dict with keys:
                action    : int (0 or 1)
                trigger   : bool (True if action == 1)
                q_values  : list[float], length 2, Q(no-trigger), Q(trigger)
        """
        arr = np.asarray(state, dtype=np.float32).reshape(1, -1)
        if arr.shape[1] != 4:
            raise ValueError(f"Expected state of length 4 {STATE_COLUMNS}, got shape {arr.shape}")

        q_values = self._forward(torch.from_numpy(arr))
        action = int(torch.argmax(q_values, dim=1).item())

        return {
            "action": action,
            "trigger": action == 1,
            "q_values": q_values.squeeze(0).cpu().tolist(),
        }

    def predict_trajectory(self, states: Iterable[Sequence[float]]) -> dict:
        """
        Real-time-style trajectory evaluation: no ground truth is assumed.
        Evaluates each state in sequence and reports whether/when the model
        would fire the trigger, mirroring deployment (airbag fires on the
        first trigger action encountered).

        Args:
            states: iterable of length-4 states, in time order.

        Returns:
            dict with keys:
                trigger       : bool, True if any step triggers
                trigger_step  : int or None, index (0-based) of first trigger
                actions       : list[int], per-step chosen action
                q_values      : list[list[float]], per-step Q-values
        """
        arr = np.asarray(states, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError(f"Expected states of shape (T, 4) {STATE_COLUMNS}, got shape {arr.shape}")

        q_values = self._forward(torch.from_numpy(arr))
        actions: List[int] = torch.argmax(q_values, dim=1).cpu().tolist()

        trigger_step = next((i for i, a in enumerate(actions) if a == 1), None)

        return {
            "trigger": trigger_step is not None,
            "trigger_step": trigger_step,
            "actions": actions,
            "q_values": q_values.cpu().tolist(),
        }


if __name__ == "__main__":
    controller = TriggerController()

    example_state = [3.0, 12.0, 15.0, 1.0]
    result = controller.predict(example_state)
    print(f"Single-state input {example_state} -> {result}")

    example_traj = [
        [3.0, 12.0, 30.0, 2.0],
        [3.0, 12.0, 20.0, 1.5],
        [3.0, 12.0, 8.0, 0.5],
        [3.0, 12.0, 2.0, 0.2],
    ]
    traj_result = controller.predict_trajectory(example_traj)
    print(f"Trajectory input -> {traj_result}")
