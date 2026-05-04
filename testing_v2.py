import argparse
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
# Argparse
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Q-network on simulation trajectories.")

    # Paths
    parser.add_argument("--root_folder", type=str, default="data/simulation/")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/model")
    parser.add_argument("--output_dir", type=str, default="results")

    # Reward / eta parameters (used only for resolving checkpoint filename)
    parser.add_argument("--b1", type=float, default=2.0)
    parser.add_argument("--c1", type=float, default=1.5)
    parser.add_argument("--b2", type=float, default=1.0)
    parser.add_argument("--c2", type=float, default=3.0)
    parser.add_argument("--c3", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.2)

    # Demographic / injury model inputs
    parser.add_argument("--height", type=float, default=1.65)
    parser.add_argument("--sex", type=str, default="F")
    parser.add_argument("--bmi", type=float, default=27.13)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--orientation", type=float, default=-90.0)
    parser.add_argument("--vstiffness", type=float, default=0.956)
    parser.add_argument("--vtype", type=str, default="SUV")
    parser.add_argument("--age", type=float, default=30.0)

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def build_param_suffix(args):
    return f"b1{args.b1}_c1{args.c1}_b2{args.b2}_c2{args.c2}_c3{args.c3}_eta{args.eta}"


def build_paths(args):
    suffix = build_param_suffix(args)
    checkpoint_path = os.path.join(args.checkpoint_dir, f"q_net_{suffix}.pth")
    output_path = os.path.join(args.output_dir, f"evaluation_results_{suffix}.csv")
    return checkpoint_path, output_path


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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Main
# =========================
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    checkpoint_path, output_path = build_paths(args)
    print(f"Loading checkpoint from: {checkpoint_path}")

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
        [d for d in os.listdir(args.root_folder) if d.isdigit()],
        key=lambda x: int(x)
    )

    for traj_id_str in traj_ids:
        traj_id = int(traj_id_str)
        traj_folder = os.path.join(args.root_folder, traj_id_str, "measurments")
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

    for df, has_collision in tqdm(zip(all_trajectories, collisions), desc="Evaluating trajectories"):
        traj_id = int(df["trajectory_id"].iloc[0])

        collision_frame = None
        p_joint = None
        collision_v_ego = None

        if has_collision:
            collision_file = os.path.join(
                args.root_folder, str(traj_id), "measurments", "collision.csv"
            )
            col_df = pd.read_csv(collision_file)
            collision_frame = int(col_df["frame"].iloc[0])

            collision_rows = df.loc[df["frame"] == collision_frame, "ego_vel_ms"].values
            collision_v_ego = (
                float(collision_rows[0]) * 3.6
                if len(collision_rows) > 0
                else float(df["ego_vel_ms"].iloc[-1]) * 3.6
            )
            # collision_v_ped = float(df["walker_vel_ms"].iloc[-1]) * 3.6
            collision_v_ped = 1.2 * 3.6

            injury_input = np.array([
                0,
                args.height,
                args.sex,
                float(args.bmi),
                float(args.offset),
                float(args.orientation),
                float(collision_v_ped),
                float(args.vstiffness),
                args.vtype,
                float(args.age),
                float(collision_v_ego),
            ], dtype=object)

            ir = injury_service.predict_injury(injury_input)
            p_joint = float(pjoint(ir))

        triggered = False
        trigger_frame = None
        ttc_at_trigger = None
        q_wait_at_trigger = None
        q_trigger_at_trigger = None
        speed_at_trigger = None

        for _, row in df.iterrows():
            current_frame = int(row["frame"])

            # Stop rollout at collision frame — can't trigger after impact
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
                ttc_at_trigger = collision_frame - trigger_frame if collision_frame is not None else None
                speed_at_trigger = float(row["ego_vel_ms"]) * 3.6
                break

        missed_collision = bool(has_collision and not triggered)
        high_injury_collision = bool(has_collision and p_joint is not None and p_joint > args.eta)
        low_injury_collision = bool(has_collision and p_joint is not None and p_joint <= args.eta)

        # Type-I: triggered when shouldn't have (no collision, or low-risk collision)
        type1_error = bool(triggered and not (has_collision and high_injury_collision))
        # Type-II: didn't trigger when should have (high-risk collision, no trigger)
        type2_error = bool(not triggered and high_injury_collision)

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
            "type1_error": type1_error,
            "type2_error": type2_error,
            "q_wait_at_trigger": q_wait_at_trigger,
            "q_trigger_at_trigger": q_trigger_at_trigger,
            "speed_at_trigger": speed_at_trigger,
            "speed_at_collision": collision_v_ego,
        })

    # =========================
    # Save results
    # =========================
    os.makedirs(args.output_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")

    # =========================
    # Summary
    # =========================
    n = len(results_df)
    n_collision = results_df["collision"].sum()
    n_triggered = results_df["triggered"].sum()
    n_high = results_df["high_injury_collision"].sum()
    n_low = results_df["low_injury_collision"].sum()
    n_type1 = results_df["type1_error"].sum()
    n_type2 = results_df["type2_error"].sum()

    # Denominators mirror the training evaluation logic
    type1_denom = n - n_high          # trajectories where trigger is wrong
    type2_denom = int(n_high)         # trajectories where trigger is needed

    print("\n=== Summary ===")
    print(f"Total trajectories:              {n}")
    print(f"Collisions:                      {n_collision}")
    print(f"  High-injury collisions:        {n_high}")
    print(f"  Low-injury collisions:         {n_low}")
    print(f"Triggered:                       {n_triggered}")
    print(f"  Triggered before collision:    {len(results_df[results_df['triggered'] & results_df['collision']])}")
    print(f"  Missed collisions:             {results_df['missed_collision'].sum()}")
    print(f"Type-I  error rate:              {n_type1}/{type1_denom} = {n_type1 / max(type1_denom, 1):.4f}")
    print(f"Type-II error rate:              {n_type2}/{type2_denom} = {n_type2 / max(type2_denom, 1):.4f}")

    print("\nTriggered + collision cases:")
    print(results_df[results_df["triggered"] & results_df["collision"]].head())


if __name__ == "__main__":
    main()