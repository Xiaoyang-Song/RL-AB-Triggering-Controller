"""
Compares all runs from jobs/gforce_sweep/{j0..j9}.sh: for each sweep tag,
reads the hyperparameters from its best-val-loss checkpoint and the
Type-I/Type-II summary from its evaluation_results_..._best.csv, and prints
one table sorted by combined error.

Usage:
    python summarize_gforce_sweep.py
"""

import argparse
import glob
import os

import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Summarise the gforce hyperparameter sweep.")
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints/model/sweep")
    parser.add_argument("--results_root", type=str, default="results/sweep")
    parser.add_argument("--eta", type=float, default=0.2)
    return parser.parse_args()


def compute_overall_errors(df, eta):
    high_risk = df[(df["collision"] == True) & (df["pjoint"] > eta)]
    low_risk_collision = df[(df["collision"] == True) & (df["pjoint"] <= eta)]
    no_collision = df[df["collision"] == False]

    type1_denom = len(low_risk_collision) + len(no_collision)
    type1_count = (
        (low_risk_collision["triggered"] == True).sum()
        + (no_collision["triggered"] == True).sum()
    )
    type2_denom = len(high_risk)
    type2_count = type2_denom - (high_risk["triggered"] == True).sum()

    type1_rate = type1_count / type1_denom * 100 if type1_denom > 0 else 0.0
    type2_rate = type2_count / type2_denom * 100 if type2_denom > 0 else 0.0
    return type1_count, type1_denom, type1_rate, type2_count, type2_denom, type2_rate


def main():
    args = parse_args()

    tags = sorted(
        [os.path.basename(p) for p in glob.glob(os.path.join(args.checkpoint_root, "j*"))],
        key=lambda x: int(x[1:]) if x[1:].isdigit() else x,
    )

    if not tags:
        print(f"No sweep checkpoints found under {args.checkpoint_root}")
        return

    rows = []
    for tag in tags:
        ckpt_candidates = glob.glob(os.path.join(args.checkpoint_root, tag, "q_net_*_best.pth"))
        results_candidates = glob.glob(os.path.join(args.results_root, tag, "evaluation_results_*_best.csv"))

        if not ckpt_candidates or not results_candidates:
            print(f"{tag}: missing checkpoint or results CSV, skipping (job may still be running)")
            continue

        hp = torch.load(ckpt_candidates[0], map_location="cpu", weights_only=False)["hyperparameters"]
        df = pd.read_csv(results_candidates[0])
        t1c, t1d, t1r, t2c, t2d, t2r = compute_overall_errors(df, args.eta)

        rows.append({
            "tag": tag,
            "lr": hp.get("lr"),
            "hidden_dim": hp.get("hidden_dim"),
            "target_update_freq": hp.get("target_update_freq"),
            "lr_step_size": hp.get("lr_step_size"),
            "lr_gamma": hp.get("lr_gamma"),
            "type1_error_%": round(t1r, 2),
            "type1_frac": f"{t1c}/{t1d}",
            "type2_error_%": round(t2r, 2),
            "type2_frac": f"{t2c}/{t2d}",
            "combined_%": round(t1r + t2r, 2),
        })

    if not rows:
        print("No completed sweep runs to summarize yet.")
        return

    summary_df = pd.DataFrame(rows).sort_values("combined_%").reset_index(drop=True)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(summary_df.to_string(index=False))

    print(f"\nBest config by combined Type-I + Type-II error: {summary_df.iloc[0]['tag']}")


if __name__ == "__main__":
    main()
