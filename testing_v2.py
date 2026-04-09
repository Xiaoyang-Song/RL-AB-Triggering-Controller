import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

current_dir = os.path.dirname(os.path.abspath(__file__))
gp_path = os.path.abspath(os.path.join(current_dir, "GP_pred"))
sys.path.append(gp_path)

from ford_ped_backend_single import *

# =========================
# Reproducibility
# =========================
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

checkpoint_path = os.path.join("checkpoints", "model", "q_net.pth")
root_folder = "data/simulation"

# =========================
# Q-network
# =========================
class QNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# Load checkpoint
# =========================
checkpoint = torch.load(checkpoint_path, map_location=device)

hidden_dim = checkpoint.get("hyperparameters", {}).get("hidden_dim", 64)

q_net = QNetwork(hidden_dim=hidden_dim).to(device)
q_net.load_state_dict(checkpoint["model_state_dict"])
q_net.eval()

state_mean = checkpoint["state_mean"].to(device)
state_std = checkpoint["state_std"].to(device)

print("Checkpoint loaded successfully.")
print(f"Hidden dim: {hidden_dim}")

# =========================
# Read simulation trajectories
# =========================
cols_to_use = [
    "frame", "walker_vel_ms", "ego_vel_ms",
    "w_location_x", "w_location_y",
    "e_location_x", "e_location_y"
]

all_trajectories = []
collisions = []

traj_ids = sorted(
    [d for d in os.listdir(root_folder) if d.isdigit()],
    key=lambda x: int(x)
)

for traj_id_str in traj_ids:
    traj_id = int(traj_id_str)
    traj_folder = os.path.join(root_folder, traj_id_str, "measurments")
    file_path = os.path.join(traj_folder, "measurements.csv")
    collision_path = os.path.join(traj_folder, "collision.csv")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path, usecols=cols_to_use)

        df["dx"] = df["w_location_x"] - df["e_location_x"]
        df["dy"] = df["w_location_y"] - df["e_location_y"]
        df["trajectory_id"] = traj_id

        all_trajectories.append(df)
        collisions.append(os.path.exists(collision_path))
    else:
        print(f"File not found: {file_path}")

# =========================
# Injury predictor
# =========================
injury_service = ford_ped_calc_service()

# =========================
# Evaluation
# =========================
results = []
data = list(zip(all_trajectories, collisions))

for df, has_collision in tqdm(data, desc="Evaluating trajectories"):
    traj_id = int(df["trajectory_id"].iloc[0])

    collision_frame = None
    p_joint = None

    if has_collision:
        collision_file = os.path.join(
            root_folder, str(traj_id), "measurments", "collision.csv"
        )
        col_df = pd.read_csv(collision_file)
        collision_frame = int(col_df["frame"].iloc[0])

        collision_rows = df.loc[df["frame"] == collision_frame, "ego_vel_ms"].values
        if len(collision_rows) > 0:
            collision_v_ego = float(collision_rows[0]) * 3.6
        else:
            collision_v_ego = float(df["ego_vel_ms"].iloc[-1]) * 3.6

        collision_v_ped = 1.2 * 3.6

        injury_input = np.array([
            0, 1.65, 'F', float(27.13), float(0), float(-90),
            float(collision_v_ped), 0.956, 'SUV', 30, float(collision_v_ego)
        ], dtype=object)

        ir = injury_service.predict_injury(injury_input)
        p_joint = pjoint(ir)

    triggered = False
    trigger_frame = None
    ttc_at_trigger = None
    q_wait_at_trigger = None
    q_trigger_at_trigger = None

    for _, row in df.iterrows():
        current_frame = int(row["frame"])

        # Optional safety stop:
        # if collision already happened and no trigger yet, stop rollout
        if collision_frame is not None and current_frame >= collision_frame:
            break

        state = np.array([
            row["walker_vel_ms"],
            row["ego_vel_ms"],
            row["dx"],
            row["dy"],
        ], dtype=np.float32)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        state_tensor = (state_tensor - state_mean) / state_std

        with torch.no_grad():
            q_values = q_net(state_tensor)
            action = torch.argmax(q_values, dim=1).item()

        if action == 1:
            triggered = True
            trigger_frame = current_frame
            q_wait_at_trigger = float(q_values[0, 0].item())
            q_trigger_at_trigger = float(q_values[0, 1].item())

            if collision_frame is not None:
                ttc_at_trigger = collision_frame - trigger_frame

            break

    # case: collision occurs before any trigger
    missed_collision = bool(has_collision and not triggered)

    # low/high injury flags if collision exists
    high_injury_collision = bool(has_collision and (p_joint is not None) and (p_joint > 0.2))
    low_injury_collision = bool(has_collision and (p_joint is not None) and (p_joint <= 0.2))

    results.append({
        "trajectory_id": traj_id,
        "collision": bool(has_collision),
        "collision_frame": collision_frame,
        "triggered": triggered,
        "trigger_frame": trigger_frame,
        "ttc_at_trigger": ttc_at_trigger,
        "pjoint": p_joint,
        "missed_collision": missed_collision,
        "high_injury_collision": high_injury_collision,
        "low_injury_collision": low_injury_collision,
        "q_wait_at_trigger": q_wait_at_trigger,
        "q_trigger_at_trigger": q_trigger_at_trigger,
    })

# =========================
# Save results
# =========================
results_df = pd.DataFrame(results)
results_df.to_csv("evaluation_results.csv", index=False)

print("Evaluation finished.")
print(results_df.head())

# =========================
# Simple summaries
# =========================
print("\nSummary:")
print(f"Total trajectories: {len(results_df)}")
print(f"Triggered: {results_df['triggered'].sum()}")
print(f"Collisions: {results_df['collision'].sum()}")
print(f"Triggered before collision: {len(results_df[(results_df['triggered']) & (results_df['collision'])])}")
print(f"Missed collisions: {results_df['missed_collision'].sum()}")

if "high_injury_collision" in results_df.columns:
    print(f"High-injury collisions: {results_df['high_injury_collision'].sum()}")
    print(
        "Triggered in high-injury collisions:",
        len(results_df[(results_df["high_injury_collision"]) & (results_df["triggered"])])
    )

trigger_collision_cases = results_df.loc[
    (results_df["triggered"]) & (results_df["collision"])
].reset_index(drop=True)

print("\nTriggered + collision cases:")
print(trigger_collision_cases.head())