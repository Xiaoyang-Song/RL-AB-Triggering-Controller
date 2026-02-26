import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
current_dir = os.path.dirname(os.path.abspath(__file__))
gp_path = os.path.abspath(os.path.join(current_dir, "GP_pred"))
sys.path.append(gp_path)
from ford_ped_backend_single import *

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --- Define Q-network ---
class QNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=32):
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

q_net = QNetwork().to(device)
q_net.load_state_dict(torch.load(os.path.join('checkpoints', 'model', 'q_net.pth')))
print("Pretrained checkpoints loaded successfully.")

root_folder = "data/simulation"

# Columns to read
cols_to_use = [
    "frame", "walker_vel_ms", "ego_vel_ms",
    "w_location_x", "w_location_y",
    "e_location_x", "e_location_y"
]

# List to store all trajectories
all_trajectories = []
collisions = []

for traj_id in range(1, len(os.listdir(root_folder))+1):
    traj_folder = os.path.join(root_folder, str(traj_id), "measurments")
    file_path = os.path.join(traj_folder, "measurements.csv")
    collision_path = os.path.join(traj_folder, "collision.csv")
    
    if os.path.exists(file_path):
        # Read CSV with only required columns
        df = pd.read_csv(file_path, usecols=cols_to_use)
        
        # Compute relative distance dx, dy
        df["dx"] = df["w_location_x"] - df["e_location_x"]
        df["dy"] = df["w_location_y"] - df["e_location_y"]
        df["trajectory_id"] = traj_id
        
        all_trajectories.append(df)
        collisions.append(os.path.exists(collision_path))
    else:
        print(f"File not found: {file_path}")


results = []
t=ford_ped_calc_service()
data = list(zip(all_trajectories, collisions))  # limit to 100 trajectories for quick evaluation

for df, has_collision in tqdm(data):

    traj_id = df["trajectory_id"].iloc[0]
    
    # Get collision frame if exists
    collision_frame = None
    p_joint = None
    if has_collision:
        collision_file = os.path.join(
            root_folder, str(traj_id), "measurments", "collision.csv"
        )
        col_df = pd.read_csv(collision_file)
        collision_frame = col_df["frame"].iloc[0]
        collision_v_ego = df.loc[df["frame"] == collision_frame, "ego_vel_ms"].values[0] * 3.6
        collision_v_ped = 1.2 * 3.6
        # --height=1.65 --sex=F --bmi=27.13 --offset=0 --orientation=-90 --vped=5 --vego=60 --vstiffness=0.956 --vtype=SUV --age=30
        data = np.array([0, 1.65, 'F', float(27.13), float(0), float(-90), 
                            float(collision_v_ped), 0.956, 'SUV', 30, float(collision_v_ego)])
        ir=t.predict_injury(data)
        # print(f"Predicted Injury Risks [head chest femur tibia]\n>> {ir}")
        p_joint = pjoint(ir)

    triggered = False
    trigger_frame = None
    ttc_at_trigger = None

    for idx, row in df.iterrows():

        state = np.array([
            row["dx"],
            row["dy"],
            row["walker_vel_ms"],
            row["ego_vel_ms"]
        ], dtype=np.float32)

        state_tensor = torch.tensor(state, device=device).unsqueeze(0)

        with torch.no_grad():
            q_values = q_net(state_tensor)
            action = torch.argmax(q_values, dim=1).item()

        # If model decides to trigger
        if action == 1:
            triggered = True
            trigger_frame = row["frame"]

            if collision_frame is not None:
                ttc_at_trigger = collision_frame - trigger_frame

            break   # stop trajectory once triggered

    # Save results
    results.append({
        "trajectory_id": traj_id,
        "collision": has_collision,
        "collision_frame": collision_frame,
        "triggered": triggered,
        "trigger_frame": trigger_frame,
        "ttc_at_trigger": ttc_at_trigger,
        "pjoint": p_joint
    })

results_df = pd.DataFrame(results)
results_df.to_csv("evaluation_results.csv", index=False)
print("Evaluation finished.")
print(results_df.head())

trigger_collision_cases = (
    results_df.loc[
        (results_df["triggered"]) & (results_df["collision"])
    ]
    .reset_index(drop=True)
)

print(trigger_collision_cases.head())