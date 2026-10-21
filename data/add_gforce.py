"""
Augments an existing rl_trajectories_*.pkl (built by transform_reward.py) with
a constant per-trajectory gforce feature, without rerunning the full pipeline.

transform_reward.py's saved pickle drops "trajectory_id", so this script
replicates its exact trajectory selection/skip logic (file-not-found skips,
the max_case3 cap, and the action_trigger random draws — in the same order,
with the same seed) to recover which original data/simulation/<id> folder
produced each entry, WITHOUT importing the injury-risk GP models (unneeded:
reward/high_risk are already baked into the existing pickle). This avoids the
~1.2GB of GP json files that OOM-kill on the login node.

Trajectories missing a gforce line in simulation_log.txt are dropped from the
augmented output (same policy as transform_reward.py --include_gforce).
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

COLS_TO_USE = [
    "frame", "walker_vel_ms", "ego_vel_ms",
    "w_location_x", "w_location_y",
    "e_location_x", "e_location_y",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attach gforce to an existing rl_trajectories pickle."
    )
    parser.add_argument("--pkl_path", type=str, required=True)
    parser.add_argument("--root_folder", type=str, default="data/simulation/")
    parser.add_argument("--output_path", type=str, default=None,
                         help="Defaults to <pkl_path without .pkl>_gforce.pkl")

    # Must match the values used to originally build --pkl_path.
    parser.add_argument("--trigger_prob", type=float, default=1 / 200)
    parser.add_argument("--max_case3", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_gforce(traj_dir):
    log_path = os.path.join(traj_dir, "simulation_log.txt")
    if not os.path.exists(log_path):
        return None

    with open(log_path, "r", encoding="latin-1", errors="ignore") as f:
        for line in f:
            if line.strip().startswith("gforce:"):
                try:
                    return float(line.split("gforce:")[1].strip())
                except ValueError:
                    return None
    return None


def load_collision_frame(collision_path):
    if not os.path.exists(collision_path):
        return None
    collision_df = pd.read_csv(collision_path)
    if "frame" not in collision_df.columns or len(collision_df) == 0:
        return None
    return collision_df["frame"].iloc[0]


def recover_kept_trajectory_ids(args):
    """
    Replays transform_reward.py's main() loop just far enough to reproduce,
    in order, which trajectory ids ended up kept (and in which order) —
    without touching the injury-risk model, since reward/high_risk values are
    not being recomputed here.
    """
    np.random.seed(args.seed)

    traj_names = sorted(
        [x for x in os.listdir(args.root_folder) if os.path.isdir(os.path.join(args.root_folder, x))],
        key=lambda x: int(x) if x.isdigit() else x,
    )

    kept_ids = []
    count_case_3 = 0

    for traj_name in traj_names:
        traj_folder = os.path.join(args.root_folder, traj_name, "measurments")
        file_path = os.path.join(traj_folder, "measurements.csv")
        collision_path = os.path.join(traj_folder, "collision.csv")

        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path, usecols=COLS_TO_USE).copy()

        collision_frame = load_collision_frame(collision_path)
        eventual_collision = collision_frame is not None

        if eventual_collision:
            valid_action_mask = df["frame"] < collision_frame
        else:
            valid_action_mask = np.ones(len(df), dtype=bool)
        valid_indices = df.index[valid_action_mask].to_numpy()

        action_trigger = np.zeros(len(df), dtype=int)
        if len(valid_indices) > 0:
            random_vals = np.random.rand(len(valid_indices))
            trigger_flags = (random_vals < args.trigger_prob).astype(int)
            action_trigger[valid_indices] = trigger_flags

        trigger_indices = np.where(action_trigger == 1)[0]
        first_trigger = trigger_indices[0] if len(trigger_indices) > 0 else None

        if first_trigger is not None:
            if not eventual_collision:
                if args.max_case3 is not None and count_case_3 >= args.max_case3:
                    continue
                count_case_3 += 1
            kept_ids.append(traj_name)
        else:
            # case_1 (no trigger + collision) or case_4 (no trigger + no collision): always kept
            kept_ids.append(traj_name)

    return kept_ids


def main():
    args = parse_args()

    with open(args.pkl_path, "rb") as f:
        trajectories = pickle.load(f)

    kept_ids = recover_kept_trajectory_ids(args)

    if len(kept_ids) != len(trajectories):
        raise RuntimeError(
            f"Recovered {len(kept_ids)} trajectory ids but the pickle has "
            f"{len(trajectories)} trajectories — selection logic did not "
            f"realign (check --trigger_prob/--max_case3/--seed match the "
            f"values used to build {args.pkl_path})."
        )

    # Sanity check alignment on first/last trajectories before trusting the rest.
    for idx in (0, len(trajectories) - 1):
        traj_dir = os.path.join(args.root_folder, kept_ids[idx], "measurments")
        raw = pd.read_csv(os.path.join(traj_dir, "measurements.csv"), usecols=COLS_TO_USE)
        saved_first_row = trajectories[idx].iloc[0]
        if not np.isclose(raw["walker_vel_ms"].iloc[0], saved_first_row["walker_vel_ms"]) or \
           not np.isclose(raw["ego_vel_ms"].iloc[0], saved_first_row["ego_vel_ms"]):
            raise RuntimeError(
                f"Alignment check failed at index {idx} (trajectory id "
                f"{kept_ids[idx]}) — refusing to attach possibly-mismatched gforce."
            )

    augmented = []
    skipped_missing_gforce = []

    for traj_id, traj_df in zip(kept_ids, trajectories):
        gforce = load_gforce(os.path.join(args.root_folder, traj_id))
        if gforce is None:
            skipped_missing_gforce.append(traj_id)
            continue
        traj_df = traj_df.copy()
        traj_df["gforce"] = gforce
        augmented.append(traj_df)

    print(f"Input trajectories: {len(trajectories)}")
    print(f"Dropped (missing gforce): {len(skipped_missing_gforce)} -> {skipped_missing_gforce}")
    print(f"Output trajectories: {len(augmented)}")

    output_path = args.output_path
    if output_path is None:
        base, ext = os.path.splitext(args.pkl_path)
        output_path = f"{base}_gforce{ext}"

    with open(output_path, "wb") as f:
        pickle.dump(augmented, f)
    print(f"Saved augmented trajectories to: {output_path}")


if __name__ == "__main__":
    main()
