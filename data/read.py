import pandas as pd
import os

# Root folder
root_folder = "simulation_data"

# Columns to read
cols_to_use = [
    "frame", "walker_vel_ms", "ego_vel_ms",
    "w_location_x", "w_location_y",
    "e_location_x", "e_location_y"
]

# List to store all trajectories
all_trajectories = []

for traj_id in range(1, 11):
    traj_folder = os.path.join(root_folder, str(traj_id), "measurments")
    file_path = os.path.join(traj_folder, "measurements.csv")
    
    if os.path.exists(file_path):
        # Read CSV with only required columns
        df = pd.read_csv(file_path, usecols=cols_to_use)
        
        # Compute relative distance dx, dy
        df["dx"] = df["w_location_x"] - df["e_location_x"]
        df["dy"] = df["w_location_y"] - df["e_location_y"]
        
        # Optionally, add a column for trajectory ID
        df["trajectory_id"] = traj_id
        
        all_trajectories.append(df)
    else:
        print(f"File not found: {file_path}")

# Combine all trajectories into one DataFrame
final_df = pd.concat(all_trajectories, ignore_index=True)

# Check the result
print(final_df.head())