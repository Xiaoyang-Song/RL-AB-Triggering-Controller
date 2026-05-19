"""
Reads all replication CSVs produced by run_uncertainty.sh and reports
mean ± std for every statistic that visualize.py computes, aggregated
across replications.

Usage:
    python summarize_uncertainty.py \
        --results_dir results/uncertainty \
        --pattern "unc_mag0.1_rep" \
        --eta 0.2
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Summarise uncertainty-analysis replications.")
    parser.add_argument("--results_dir", type=str, default="results/uncertainty")
    parser.add_argument("--pattern", type=str, default="unc_mag*_rep",
                        help="Substring that appears in the label part of target CSV filenames")
    parser.add_argument("--eta", type=float, default=0.2,
                        help="Injury threshold (must match the value used during evaluation)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-replication stats before the summary")
    return parser.parse_args()


def compute_stats(df, eta):
    """Mirror every metric printed by visualize.py, return as a flat dict."""
    n = len(df)

    # --- case splits (same logic as visualize.py) ---
    high_risk      = df[(df["collision"] == True)  & (df["pjoint"] > eta)]
    low_risk_col   = df[(df["collision"] == True)  & (df["pjoint"] <= eta)]
    no_col         = df[df["collision"] == False]

    total_high       = len(high_risk)
    total_low_col    = len(low_risk_col)
    total_no_col     = len(no_col)

    triggered_high    = (high_risk["triggered"]  == True).sum()
    triggered_low_col = (low_risk_col["triggered"] == True).sum()
    triggered_no_col  = (no_col["triggered"]      == True).sum()

    not_triggered_low_col = total_low_col - triggered_low_col
    not_triggered_no_col  = total_no_col  - triggered_no_col

    # --- SUCCESS TRIGGER RATE (Type-II complement) ---
    success_trigger_rate = triggered_high / total_high * 100 if total_high > 0 else float("nan")
    type2_rate           = 100 - success_trigger_rate

    # --- LOW INJURY RISK WITH COLLISION ---
    low_risk_trigger_rate    = triggered_low_col    / total_low_col * 100 if total_low_col > 0 else float("nan")
    low_risk_no_trigger_rate = not_triggered_low_col / total_low_col * 100 if total_low_col > 0 else float("nan")

    # --- NO COLLISION CASES ---
    no_col_trigger_rate    = triggered_no_col    / total_no_col * 100 if total_no_col > 0 else float("nan")
    no_col_no_trigger_rate = not_triggered_no_col / total_no_col * 100 if total_no_col > 0 else float("nan")

    # --- OVERALL TYPE-I / TYPE-II ---
    type1_denom = total_low_col + total_no_col
    type1_count = triggered_low_col + triggered_no_col
    type1_rate  = type1_count / type1_denom * 100 if type1_denom > 0 else float("nan")

    type2_count = total_high - triggered_high
    type2_rate2 = type2_count / total_high * 100 if total_high > 0 else float("nan")

    # --- additional numeric details ---
    triggered_all = (df["triggered"] == True).sum()
    ttc_vals      = df.loc[df["triggered"] == True, "ttc_at_trigger"].dropna()
    speed_vals    = df.loc[df["triggered"] == True, "speed_at_trigger"].dropna()

    return {
        # counts (fixed across reps, sanity check)
        "n_trajectories":               n,
        "n_high_risk":                  int(total_high),
        "n_low_risk_collision":         int(total_low_col),
        "n_no_collision":               int(total_no_col),
        "n_triggered":                  int(triggered_all),

        # SUCCESS TRIGGER RATE
        "success_trigger_rate_%":       success_trigger_rate,

        # LOW INJURY RISK WITH COLLISION
        "low_risk_triggered_%":         low_risk_trigger_rate,
        "low_risk_not_triggered_%":     low_risk_no_trigger_rate,

        # NO COLLISION CASES
        "no_collision_triggered_%":     no_col_trigger_rate,
        "no_collision_not_triggered_%": no_col_no_trigger_rate,

        # OVERALL ERROR SUMMARY
        "type1_rate_%":                 type1_rate,
        "type2_rate_%":                 type2_rate2,

        # NUMERIC DETAILS
        "mean_ttc_at_trigger":          ttc_vals.mean()   if len(ttc_vals)   > 0 else float("nan"),
        "mean_speed_at_trigger_kph":    speed_vals.mean() if len(speed_vals) > 0 else float("nan"),
    }


def _fmt(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


def main():
    args = parse_args()

    csv_files = sorted(glob.glob(
        os.path.join(args.results_dir, f"*{args.pattern}*.csv")
    ))

    if not csv_files:
        print(f"No CSV files found in '{args.results_dir}' matching pattern '{args.pattern}'")
        return

    print(f"Found {len(csv_files)} replication file(s) matching '{args.pattern}'.")

    per_rep = []
    for fpath in csv_files:
        df = pd.read_csv(fpath)
        stats = compute_stats(df, args.eta)
        stats["file"] = os.path.basename(fpath)
        per_rep.append(stats)

    summary_df = pd.DataFrame(per_rep)

    if args.verbose:
        print("\n=== Per-replication stats ===")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(summary_df.drop(columns="file").to_string(index=False))

    numeric_cols = [c for c in summary_df.columns if c not in ("file",)]
    means = summary_df[numeric_cols].mean()
    stds  = summary_df[numeric_cols].std()

    sep = "=" * 62

    print(f"\n{sep}")
    print(f"UNCERTAINTY SUMMARY  ({len(csv_files)} reps,  eta={args.eta})")
    print(sep)

    sections = [
        ("Trajectory counts",
         ["n_trajectories", "n_high_risk", "n_low_risk_collision",
          "n_no_collision", "n_triggered"]),
        ("Success trigger rate  (Type-II complement)",
         ["success_trigger_rate_%"]),
        ("Low injury risk with collision",
         ["low_risk_triggered_%", "low_risk_not_triggered_%"]),
        ("No collision cases",
         ["no_collision_triggered_%", "no_collision_not_triggered_%"]),
        ("Overall error summary",
         ["type1_rate_%", "type2_rate_%"]),
        ("Numeric details",
         ["mean_ttc_at_trigger", "mean_speed_at_trigger_kph"]),
    ]

    col_w = max(len(c) for c in numeric_cols) + 2
    for title, cols in sections:
        print(f"\n  {title}")
        print(f"  {'-' * (col_w + 28)}")
        for col in cols:
            print(f"    {col:<{col_w}}  {_fmt(means[col], stds[col])}")

    print(f"\n{sep}")

    # Save summary CSV
    out_path = os.path.join(args.results_dir, f"summary_{args.pattern.replace('*','')}.csv")
    row = {c: _fmt(means[c], stds[c]) for c in numeric_cols}
    row["n_replications"] = len(csv_files)
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"Summary saved to: {out_path}")


if __name__ == "__main__":
    main()
