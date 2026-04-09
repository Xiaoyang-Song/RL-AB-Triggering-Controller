import argparse
import numpy as np
import pickle
import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
gp_path = os.path.abspath(os.path.join(current_dir, "..", "GP_pred"))
sys.path.append(gp_path)

from ford_ped_backend_single import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate RL trajectories for airbag triggering."
    )

    # Paths
    parser.add_argument("--root_folder", type=str, default="data/simulation")
    parser.add_argument("--output_path", type=str, default="data/aggregated/rl_trajectories.pkl")

    # Random trigger generation
    parser.add_argument("--trigger_prob", type=float, default=1 / 200)

    # Reward parameters
    parser.add_argument("--b1", type=float, default=2.0,
                        help="Reward for correct trigger under high injury risk")
    parser.add_argument("--c1", type=float, default=1.5,
                        help="Penalty for trigger under low injury risk")
    parser.add_argument("--b2", type=float, default=1.0,
                        help="Reward for correct no-trigger under low injury risk")
    parser.add_argument("--c2", type=float, default=3.0,
                        help="Penalty for no-trigger under high injury risk")
    parser.add_argument("--c3", type=float, default=2.0,
                        help="Penalty for trigger when no collision happens")

    # Injury threshold
    parser.add_argument("--eta", type=float, default=0.2,
                        help="Injury risk threshold")

    # Optional dataset balancing
    parser.add_argument("--max_case3", type=int, default=300,
                        help="Optional cap on number of no-collision trigger trajectories")

    # Demographic / model inputs
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


COLS_TO_USE = [
    "frame", "walker_vel_ms", "ego_vel_ms",
    "w_location_x", "w_location_y",
    "e_location_x", "e_location_y"
]


def compute_joint_injury_risk(service, row, args):
    """
    Compute current predicted joint injury risk from the current state row.
    Assumes the backend model can use current ego/pedestrian speeds as proxies.
    """
    collision_v_ego = float(row["ego_vel_ms"]) * 3.6
    collision_v_ped = float(row["walker_vel_ms"]) * 3.6

    data = np.array([
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

    ir = service.predict_injury(data)
    return float(pjoint(ir))


def load_collision_frame(collision_path):
    if not os.path.exists(collision_path):
        return None

    collision_df = pd.read_csv(collision_path)
    if "frame" not in collision_df.columns or len(collision_df) == 0:
        return None

    return collision_df["frame"].iloc[0]


def assign_terminal_reward(triggered, collision_happens, high_risk, args):
    """
    Reward table:

    trigger & future collision & high risk     -> +b1
    trigger & future collision & low risk      -> -c1
    no trigger & collision & low risk          -> +b2
    no trigger & collision & high risk         -> -c2
    trigger & no future collision              -> -c3
    no trigger & no collision                  -> 0
    """
    if triggered:
        if collision_happens:
            return args.b1 if high_risk else -args.c1
        return -args.c3
    else:
        if collision_happens:
            return -args.c2 if high_risk else args.b2
        return 0.0


def main():
    args = parse_args()
    np.random.seed(args.seed)

    service = ford_ped_calc_service()

    all_trajectories = []
    rl_trajectories = []

    if not os.path.exists(args.root_folder):
        raise FileNotFoundError(f"Root folder not found: {args.root_folder}")

    traj_names = sorted(
        [x for x in os.listdir(args.root_folder) if os.path.isdir(os.path.join(args.root_folder, x))],
        key=lambda x: int(x) if x.isdigit() else x
    )

    print(f"number of trajectories found: {len(traj_names)}")

    count_trigger = 0
    count_case_1 = 0  # no trigger + collision
    count_case_2 = 0  # trigger + collision
    count_case_3 = 0  # trigger + no collision
    count_case_4 = 0  # no trigger + no collision

    for traj_name in traj_names:
        traj_id = traj_name
        traj_folder = os.path.join(args.root_folder, traj_name, "measurments")
        file_path = os.path.join(traj_folder, "measurements.csv")
        collision_path = os.path.join(traj_folder, "collision.csv")

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        df = pd.read_csv(file_path, usecols=COLS_TO_USE).copy()

        # Relative position
        df["dx"] = df["w_location_x"] - df["e_location_x"]
        df["dy"] = df["w_location_y"] - df["e_location_y"]

        # Initialize
        df["action_trigger"] = 0
        df["reward"] = 0.0
        df["trajectory_id"] = traj_id

        collision_frame = load_collision_frame(collision_path)
        eventual_collision = collision_frame is not None

        # Only allow trigger before collision frame.
        # This prevents impossible "trigger after collision" samples.
        if eventual_collision:
            valid_action_mask = df["frame"] < collision_frame
        else:
            valid_action_mask = np.ones(len(df), dtype=bool)

        valid_indices = df.index[valid_action_mask].to_numpy()

        # Random trigger generation only on valid pre-collision steps
        if len(valid_indices) > 0:
            random_vals = np.random.rand(len(valid_indices))
            trigger_flags = (random_vals < args.trigger_prob).astype(int)
            df.loc[valid_indices, "action_trigger"] = trigger_flags

        trigger_indices = df.index[df["action_trigger"] == 1].to_list()
        first_trigger = trigger_indices[0] if len(trigger_indices) > 0 else None

        # Determine terminal event: first trigger or collision, whichever comes first
        if eventual_collision:
            collision_candidates = df.index[df["frame"] == collision_frame].to_list()
            if len(collision_candidates) == 0:
                # fallback: if collision frame is missing from measurements, use last frame
                collision_index = df.index[-1]
            else:
                collision_index = collision_candidates[0]
        else:
            collision_index = None

        if first_trigger is not None:
            # Trigger happens before any collision by construction
            trigger_row = df.loc[first_trigger]
            risk = compute_joint_injury_risk(service, trigger_row, args)
            high_risk = risk >= args.eta

            if eventual_collision:
                terminal_reward = assign_terminal_reward(
                    triggered=True,
                    collision_happens=True,
                    high_risk=high_risk,
                    args=args
                )
                count_case_2 += 1
            else:
                if args.max_case3 is not None and count_case_3 >= args.max_case3:
                    continue
                terminal_reward = assign_terminal_reward(
                    triggered=True,
                    collision_happens=False,
                    high_risk=False,
                    args=args
                )
                count_case_3 += 1

            df.loc[first_trigger, "reward"] = terminal_reward
            df = df.loc[:first_trigger].copy()
            count_trigger += 1

        elif eventual_collision:
            # No trigger before collision; terminate at collision
            collision_row = df.loc[collision_index]
            risk = compute_joint_injury_risk(service, collision_row, args)
            high_risk = risk >= args.eta

            terminal_reward = assign_terminal_reward(
                triggered=False,
                collision_happens=True,
                high_risk=high_risk,
                args=args
            )
            df.loc[collision_index, "reward"] = terminal_reward
            df = df.loc[:collision_index].copy()
            count_case_1 += 1

        else:
            # No trigger and no collision: keep full trajectory, all rewards 0
            count_case_4 += 1

        all_trajectories.append(df)

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
        traj_rl = traj_df[selected_cols].copy()
        rl_trajectories.append(traj_rl)

    print(f"count_trigger: {count_trigger}")
    print(f"count_case_1 (no trigger + collision): {count_case_1}")
    print(f"count_case_2 (trigger + collision): {count_case_2}")
    print(f"count_case_3 (trigger + no collision): {count_case_3}")
    print(f"count_case_4 (no trigger + no collision): {count_case_4}")

    if len(rl_trajectories) > 0:
        print("Length of first trajectory:", len(rl_trajectories[0]))
        print(rl_trajectories[0].head())

    print(f"Total trajectories processed: {len(rl_trajectories)}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(rl_trajectories, f)

    print(f"Saved trajectories to: {args.output_path}")


if __name__ == "__main__":
    main()