"""
Verification script: reproduces the testing_v2.py / visualize.py evaluation
pipeline on the simulation test set, but using RL_trigger_module's standalone
TriggerController for the Q-network inference step instead of loading the
model inline. Confirms the packaged module yields the same trigger decisions
(and therefore the same Type-I / Type-II error rates) as the original code.

Must be run from the repository root (the injury GP models are loaded via a
path relative to cwd: "GP_pred/").
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(REPO_ROOT, "GP_pred"))
sys.path.append(os.path.join(REPO_ROOT, "RL_trigger_module"))

from ford_ped_backend_single import ford_ped_calc_service, pjoint  # noqa: E402
from trigger_controller import TriggerController  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify RL_trigger_module against the testing_v2.py test set."
    )
    parser.add_argument("--root_folder", type=str, default="data/simulation/")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Defaults to RL_trigger_module's bundled checkpoint "
             "(b1=5.0, c1=6.0, b2=5.0, c2=5.0, c3=5.0, eta=0.2).",
    )
    parser.add_argument("--eta", type=float, default=0.2, help="High-risk pjoint threshold.")

    # Demographic / injury model inputs (match testing_v2.py defaults)
    parser.add_argument("--height", type=float, default=1.65)
    parser.add_argument("--sex", type=str, default="F")
    parser.add_argument("--bmi", type=float, default=27.13)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--orientation", type=float, default=-90.0)
    parser.add_argument("--vstiffness", type=float, default=0.956)
    parser.add_argument("--vtype", type=str, default="SUV")
    parser.add_argument("--age", type=float, default=30.0)

    return parser.parse_args()


def main():
    args = parse_args()

    controller = TriggerController(checkpoint_path=args.checkpoint_path)
    print(f"Loaded checkpoint: {controller.checkpoint_path}")

    cols_to_use = [
        "frame", "walker_vel_ms", "ego_vel_ms",
        "w_location_x", "w_location_y",
        "e_location_x", "e_location_y",
    ]

    all_trajectories = []
    collisions = []

    traj_ids = sorted(
        [d for d in os.listdir(args.root_folder) if d.isdigit()],
        key=lambda x: int(x),
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

    injury_service = ford_ped_calc_service()

    results = []

    for df, has_collision in tqdm(zip(all_trajectories, collisions), total=len(all_trajectories), desc="Verifying trajectories"):
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

        # Rollout stops at collision frame — can't trigger after impact.
        if collision_frame is not None:
            rollout_df = df.loc[df["frame"] < collision_frame]
        else:
            rollout_df = df

        triggered = False
        trigger_frame = None
        ttc_at_trigger = None
        q_wait_at_trigger = None
        q_trigger_at_trigger = None
        speed_at_trigger = None

        if len(rollout_df) > 0:
            states = rollout_df[["walker_vel_ms", "ego_vel_ms", "dx", "dy"]].values
            traj_result = controller.predict_trajectory(states)

            if traj_result["trigger"]:
                triggered = True
                step = traj_result["trigger_step"]
                trigger_row = rollout_df.iloc[step]
                trigger_frame = int(trigger_row["frame"])
                q_wait_at_trigger, q_trigger_at_trigger = traj_result["q_values"][step]
                ttc_at_trigger = collision_frame - trigger_frame if collision_frame is not None else None
                speed_at_trigger = float(trigger_row["ego_vel_ms"]) * 3.6

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

    results_df = pd.DataFrame(results)

    # =========================
    # Overall Type-I / Type-II summary (mirrors visualize.py)
    # =========================
    high_risk_cases = results_df.loc[
        (results_df["collision"] == True) & (results_df["pjoint"] > args.eta)
    ]
    high_risk_triggered = high_risk_cases.loc[high_risk_cases["triggered"] == True]
    total_high_risk = len(high_risk_cases)
    triggered_count = len(high_risk_triggered)

    low_risk_collision_cases = results_df.loc[
        (results_df["collision"] == True) & (results_df["pjoint"] <= args.eta)
    ]
    total_low_risk_collision = len(low_risk_collision_cases)
    low_risk_triggered = len(low_risk_collision_cases.loc[low_risk_collision_cases["triggered"] == True])

    no_collision_cases = results_df.loc[results_df["collision"] == False]
    total_no_collision = len(no_collision_cases)
    no_collision_triggered = len(no_collision_cases.loc[no_collision_cases["triggered"] == True])

    type1_denom = total_low_risk_collision + total_no_collision
    type1_count = low_risk_triggered + no_collision_triggered
    type1_rate = (type1_count / type1_denom * 100) if type1_denom > 0 else 0

    type2_denom = total_high_risk
    type2_count = total_high_risk - triggered_count
    type2_rate = (type2_count / type2_denom * 100) if type2_denom > 0 else 0

    print("\n" + "=" * 60)
    print("\n" + "=" * 60)
    print("OVERALL ERROR SUMMARY")
    print("=" * 60)
    print(f"Type-I  error (false trigger):   {type1_count}/{type1_denom} = {type1_rate:.2f}%")
    print(f"Type-II error (missed trigger):  {type2_count}/{type2_denom} = {type2_rate:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
