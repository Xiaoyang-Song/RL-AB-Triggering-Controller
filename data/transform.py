import numpy as np
import pickle
import pandas as pd
import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
gp_path = os.path.abspath(os.path.join(current_dir, "..", "GP_pred"))
sys.path.append(gp_path)
from ford_ped_backend_single import *

# Root folder
root_folder = "data/simulation"

# Columns to read
cols_to_use = [
    "frame", "walker_vel_ms", "ego_vel_ms",
    "w_location_x", "w_location_y",
    "e_location_x", "e_location_y"
]

trigger_prob = 1/200
C_1 = 1
C_2 = 1

# param_root = "../GP_pred/"

# # load pretrained model parameters
# GP_hic = load_param_set(os.path.join(param_root, "GP_hic.json"))
# GP_lacc = load_param_set(os.path.join(param_root, "GP_lacc.json"))
# GP_ln = load_param_set(os.path.join(param_root, "GP_ln.json"))
# GP_pelvis = load_param_set(os.path.join(param_root, "GP_pelvis.json"))
# GP_racc = load_param_set(os.path.join(param_root, "GP_racc.json"))
# GP_rn = load_param_set(os.path.join(param_root, "GP_rn.json"))
# GP_sternum = load_param_set(os.path.join(param_root, "GP_sternum.json"))

t=ford_ped_calc_service()

# List to store all trajectories
all_trajectories = []
print(f"number of trajectories found: {len(os.listdir(root_folder))}")
count_trigger = 0
count_case_1, count_case_2, count_case_3 = 0, 0, 0

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

        # Check if collision eventually happens
        eventual_collision = os.path.exists(collision_path)

        # Generate random triggers
        random_vals = np.random.rand(len(df))
        df["action_trigger"] = (random_vals < trigger_prob).astype(int)

        # Find first trigger index
        trigger_indices = df.index[df["action_trigger"] == 1]

        df["reward"] = 0.0

        if len(trigger_indices) > 0:
            first_trigger = trigger_indices[0]
            count_trigger += 1

            # Case 2 & 3 (trigger case)
            if eventual_collision:
                count_case_2 += 1
                # Case 2: triggered and eventual collision → blank for now
                collision_frame = pd.read_csv(collision_path)["frame"].iloc[0]
                collision_v_ego = df.loc[df["frame"] == collision_frame, "ego_vel_ms"].values[0] * 3.6
                collision_v_ped = 1.2 * 3.6
                # --height=1.65 --sex=F --bmi=27.13 --offset=0 --orientation=-90 --vped=5 --vego=60 --vstiffness=0.956 --vtype=SUV --age=30
                data = np.array([0, 1.65, 'F', float(27.13), float(0), float(-90), 
                                 float(collision_v_ped), 0.956, 'SUV', 30, float(collision_v_ego)])
                ir=t.predict_injury(data)
                # print(f"Predicted Injury Risks [head chest femur tibia]\n>> {ir}")
                iir = pjoint(ir)
                # print(iir)
                df.loc[first_trigger, "reward"] = iir - 1
            else:
                # Balance training cases
                if count_case_3 >= 300:
                    continue
                count_case_3 += 1
                # Case 3: triggered and no collision → -C2
                df.loc[first_trigger, "reward"] = -C_2

            # Truncate trajectory after trigger
            df = df.loc[:first_trigger]

        # Case 1:
        # No trigger but collision happens eventually
        elif eventual_collision and len(trigger_indices) == 0:
            count_case_1 += 1
            # collision occurs but agent never triggered
            # penalize final time step
            df.loc[df.index[-1], "reward"] = -C_1
        
        # Optionally, add a column for trajectory ID
        df["trajectory_id"] = traj_id
        
        all_trajectories.append(df)
    else:
        print(f"File not found: {file_path}")
# Check the result
# List to store per-trajectory RL sequences
rl_trajectories = []

selected_cols = [
    "frame",
    "walker_vel_ms",
    "ego_vel_ms",
    "dx",
    "dy",
    "action_trigger",
    "reward",
]

for traj_df in all_trajectories:
    # Keep only relevant columns
    traj_rl = traj_df[selected_cols].copy()
    rl_trajectories.append(traj_rl)


print(f"count_trigger: {count_trigger}")
print(f"count_case_1 (no trigger but collision): {count_case_1}")
print(f"count_case_2 (trigger and collision): {count_case_2}")
print(f"count_case_3 (trigger but no collision): {count_case_3}")
print("Length of first trajectory:", len(rl_trajectories[0]))
print(rl_trajectories[0].head())

print(f"Total trajectories processed: {len(rl_trajectories)}")

with open('data/aggregated/rl_trajectories.pkl', 'wb') as f:
    pickle.dump(rl_trajectories, f)

# Load it back
# with open('aggregated/rl_trajectories.pkl', 'rb') as f:
#     loaded_list = pickle.load(f)
