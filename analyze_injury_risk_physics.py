"""
Like analyze_injury_risk_bins.py, but bins Type-I/Type-II error by a
PHYSICS-ESTIMATED injury risk (something the controller could estimate in
real time) instead of the true, retrospective pjoint.

Every trajectory gets exactly one anchor frame and one estimated collision
speed there:

  - triggered            -> anchor = trigger_frame (the actual decision frame)
  - collision, untriggered -> anchor = last frame before collision_frame
                              (outcome already confirmed)
  - never triggered, never collided -> anchor = frame of closest approach
                              (cheapest proxy for "this trajectory's worst
                              moment"; always triggered=False, so it only
                              ever contributes to the Type-I denominator —
                              without it, bins at low/zero predicted risk
                              would trivially show 100% Type-I error, since
                              only false triggers could ever land there)

At the anchor frame, distance = sqrt(dx^2+dy^2) between pedestrian and
vehicle decides how the collision speed is estimated:

  - "collision" anchor: use current ego_vel_ms directly (no projection needed).
  - distance < CONTACT_THRESHOLD_M: already touching: current ego_vel_ms IS
    the impact speed, happening right now.
  - dx <= 0, or the pedestrian is essentially stationary (walker_vel_ms <
    PED_STATIONARY_MPS): physics predicts NO collision (diverging paths, or a
    pedestrian who can't walk into the car's path) — these go in a separate
    "predicted: no collision" bucket, not the continuous risk axis.
  - otherwise: project the vehicle's longitudinal motion forward (fitting its
    recent deceleration trend over the last DECEL_WINDOW frames) to estimate
    its speed when it reaches the pedestrian's position.

The collision speed (plus the same FIXED pedestrian speed, 1.2 m/s, used by
the real pjoint computation) is fed through the GP injury model to get a
predicted_pjoint. Because that model loads ~1.2GB of parameter JSON at
import, this script must run on a compute node, not the login node.

Usage:
    python analyze_injury_risk_physics.py \
        --results_csv results/sweep_allfive/j1/evaluation_results_b15.0_c15.0_b25.0_c25.0_c35.0_eta0.2_gforce_best.csv \
        --root_folder data/simulation/ \
        --eta 0.2 \
        --title "gforce sweep_allfive j1 — physics-estimated risk" \
        --out plots/sweep_allfive/j1/injury_risk_physics_curve.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DT_SECONDS = 0.016  # empirically measured simulation tick duration
CONTACT_THRESHOLD_M = 3.5
DECEL_WINDOW = 10
PED_STATIONARY_MPS = 0.1
FIXED_PED_SPEED_KMH = 1.2 * 3.6

COLOR_TYPE1 = "#fab219"  # dataviz skill status palette: "warning"
COLOR_TYPE2 = "#d03b3b"  # dataviz skill status palette: "critical"
COLOR_TYPE1_SMOOTH = "#b9790a"
COLOR_TYPE2_SMOOTH = "#8c1f1f"


def parse_args():
    parser = argparse.ArgumentParser(description="Bin Type-I/Type-II error by physics-estimated injury risk.")
    parser.add_argument("--results_csv", type=str, default=None,
                         help="Required unless --data_csv is given (skips GP recomputation).")
    parser.add_argument("--data_csv", type=str, default=None,
                         help="A previously-saved *_data.csv from this script — replot only, no GP model needed.")
    parser.add_argument("--root_folder", type=str, default="data/simulation/")
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--bin_width", type=float, default=0.02)
    parser.add_argument("--smooth_window_bins", type=int, default=7)
    parser.add_argument("--contact_threshold_m", type=float, default=CONTACT_THRESHOLD_M)
    parser.add_argument("--decel_window", type=int, default=DECEL_WINDOW)
    parser.add_argument("--ped_stationary_mps", type=float, default=PED_STATIONARY_MPS)

    # Demographic inputs — must match the values used to produce --results_csv.
    parser.add_argument("--height", type=float, default=1.65)
    parser.add_argument("--sex", type=str, default="F")
    parser.add_argument("--bmi", type=float, default=27.13)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--orientation", type=float, default=-90.0)
    parser.add_argument("--vstiffness", type=float, default=0.956)
    parser.add_argument("--vtype", type=str, default="SUV")
    parser.add_argument("--age", type=float, default=30.0)

    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def load_traj_frames(root_folder, traj_id):
    path = os.path.join(root_folder, str(traj_id), "measurments", "measurements.csv")
    cols = ["frame", "w_location_x", "w_location_y", "e_location_x", "e_location_y", "ego_vel_ms", "walker_vel_ms"]
    df = pd.read_csv(path, usecols=cols).sort_values("frame").reset_index(drop=True)
    df["dx"] = df["w_location_x"] - df["e_location_x"]
    df["dy"] = df["w_location_y"] - df["e_location_y"]
    return df


def find_closest_approach_frame(df):
    distances = np.hypot(df["dx"], df["dy"])
    return int(df.loc[distances.idxmin(), "frame"])


def estimate_collision_speed(df, anchor_frame, anchor_type, args):
    row = df[df["frame"] == anchor_frame]
    if len(row) == 0:
        return None
    row = row.iloc[0]
    dx, dy, ego_vel, walker_vel = row["dx"], row["dy"], row["ego_vel_ms"], row["walker_vel_ms"]
    distance = float(np.hypot(dx, dy))

    if anchor_type == "collision":
        return dict(speed=ego_vel, distance=distance, method="current (collision confirmed)",
                    no_collision_predicted=False)

    if distance < args.contact_threshold_m:
        return dict(speed=ego_vel, distance=distance, method="current (already in contact)",
                    no_collision_predicted=False)

    if dx <= 0 or walker_vel < args.ped_stationary_mps:
        return dict(speed=None, distance=distance, method="no collision predicted",
                    no_collision_predicted=True)

    # Project the vehicle's longitudinal motion forward using its recent
    # deceleration trend, to estimate speed when it reaches the pedestrian's
    # x position.
    window = df[df["frame"] <= anchor_frame].tail(args.decel_window)
    if len(window) >= 3:
        slope_per_frame = np.polyfit(window["frame"].values, window["ego_vel_ms"].values, 1)[0]
        decel_per_sec = slope_per_frame / DT_SECONDS
    else:
        decel_per_sec = 0.0

    # Solve ego_vel*t + 0.5*decel*t^2 = dx for the smallest positive root.
    a, b, c = 0.5 * decel_per_sec, ego_vel, -dx
    if abs(a) < 1e-9:
        t = dx / ego_vel if ego_vel > 0 else None
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            t = None
        else:
            sqrt_disc = np.sqrt(disc)
            roots = [(-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)]
            positive_roots = [r for r in roots if r > 0]
            t = min(positive_roots) if positive_roots else None

    if t is None:
        t = dx / ego_vel if ego_vel > 0 else 0.0
        decel_per_sec = 0.0

    predicted_speed = max(0.0, ego_vel + decel_per_sec * t)
    return dict(speed=predicted_speed, distance=distance,
                method=f"projected (t={t:.2f}s, decel={decel_per_sec:.2f} m/s^2)",
                no_collision_predicted=False)


def compute_physics_data(args):
    # Imported lazily: this module loads ~1.2GB of GP parameter JSON at import
    # time, which OOM-kills on the login node. The replot-only path (--data_csv)
    # must never trigger it.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gp_path = os.path.abspath(os.path.join(current_dir, "GP_pred"))
    if gp_path not in sys.path:
        sys.path.append(gp_path)
    from ford_ped_backend_single import ford_ped_calc_service, pjoint

    results = pd.read_csv(args.results_csv)

    main_rows = results[(results["triggered"] == True) | (results["collision"] == True)].copy()
    main_rows["anchor_type"] = np.where(main_rows["triggered"] == True, "trigger", "collision")

    excluded_rows = results[(results["triggered"] == False) & (results["collision"] == False)].copy()
    excluded_rows["anchor_type"] = "closest_approach"

    included = pd.concat([main_rows, excluded_rows], ignore_index=True)
    print(f"Total trajectories: {len(results)} | trigger/collision anchors: {len(main_rows)} "
          f"| closest-approach anchors: {len(excluded_rows)}")

    injury_service = ford_ped_calc_service()

    predicted_speeds, distances, methods, no_collision_flags = [], [], [], []
    for _, row in included.iterrows():
        traj_id = int(row["trajectory_id"])
        anchor_type = row["anchor_type"]
        df = load_traj_frames(args.root_folder, traj_id)

        if anchor_type == "trigger":
            anchor_frame = int(row["trigger_frame"])
        elif anchor_type == "collision":
            anchor_frame = int(row["collision_frame"]) - 1
        else:
            anchor_frame = find_closest_approach_frame(df)

        result = estimate_collision_speed(df, anchor_frame, anchor_type, args)
        if result is None:
            predicted_speeds.append(np.nan)
            distances.append(np.nan)
            methods.append("missing anchor frame")
            no_collision_flags.append(False)
            continue

        predicted_speeds.append(result["speed"] if result["speed"] is not None else np.nan)
        distances.append(result["distance"])
        methods.append(result["method"])
        no_collision_flags.append(result["no_collision_predicted"])

    included["predicted_collision_v_ego"] = predicted_speeds
    included["anchor_distance_m"] = distances
    included["anchor_method"] = methods
    included["no_collision_predicted"] = no_collision_flags

    predicted_pjoints = []
    for _, row in included.iterrows():
        if row["no_collision_predicted"] or np.isnan(row["predicted_collision_v_ego"]):
            predicted_pjoints.append(np.nan)
            continue
        injury_input = np.array([
            0, args.height, args.sex, float(args.bmi), float(args.offset), float(args.orientation),
            float(FIXED_PED_SPEED_KMH), float(args.vstiffness), args.vtype, float(args.age),
            float(row["predicted_collision_v_ego"] * 3.6),
        ], dtype=object)
        ir = injury_service.predict_injury(injury_input)
        predicted_pjoints.append(float(pjoint(ir)))

    included["predicted_pjoint"] = predicted_pjoints

    out_csv = args.out.rsplit(".", 1)[0] + "_data.csv"
    included.to_csv(out_csv, index=False)
    print(f"Saved per-trajectory physics-estimated data to: {out_csv}")
    return included


def wilson_interval(count, n, z=1.96):
    """95% Wilson score interval for a binomial rate (vectorized) — its width
    IS the sample-size signal: wide at small n, tight at large n."""
    count = np.asarray(count, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        phat = count / n
        denom = 1 + z**2 / n
        center = (phat + z**2 / (2 * n)) / denom
        halfwidth = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    lo = np.clip(center - halfwidth, 0, 1) * 100
    hi = np.clip(center + halfwidth, 0, 1) * 100
    lo[n == 0] = np.nan
    hi[n == 0] = np.nan
    return lo, hi


def make_plot(included, args):
    # "Predicted: no collision" trajectories are dropped entirely — they
    # contribute no numeric risk value to plot.
    continuum = included[included["no_collision_predicted"] == False]
    edges = np.arange(0.0, 1.0 + args.bin_width / 2, args.bin_width)

    xs, type1_n, type1_c, type2_n, type2_c = [], [], [], [], []

    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = continuum[(continuum["predicted_pjoint"] >= lo) & (continuum["predicted_pjoint"] < hi + 1e-9)]
        should_not = in_bin[in_bin["high_injury_collision"] == False]
        should = in_bin[in_bin["high_injury_collision"] == True]

        xs.append((lo + hi) / 2)
        type1_n.append(len(should_not))
        type1_c.append(int((should_not["triggered"] == True).sum()))
        type2_n.append(len(should))
        type2_c.append(int((should["triggered"] == False).sum()))

    xs = np.array(xs)

    def rolling_rate_and_band(counts, totals):
        """Aggregate counts/totals over the same window used for the smoothed
        line, so the trend line and its confidence band reflect the same
        effective sample size — otherwise a smoothed line sits awkwardly
        inside a band computed from raw, much-noisier single-bin counts."""
        c = pd.Series(counts, dtype=float).rolling(args.smooth_window_bins, center=True, min_periods=1).sum()
        n = pd.Series(totals, dtype=float).rolling(args.smooth_window_bins, center=True, min_periods=1).sum()
        rate = (100.0 * c / n).to_numpy()
        lo, hi = wilson_interval(c.to_numpy(), n.to_numpy())
        rate[n.to_numpy() == 0] = np.nan
        return rate, lo, hi, n.to_numpy()

    type1_smooth, type1_lo, type1_hi, type1_eff_n = rolling_rate_and_band(type1_c, type1_n)
    type2_smooth, type2_lo, type2_hi, type2_eff_n = rolling_rate_and_band(type2_c, type2_n)

    # =========================
    # Plot
    # =========================
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
    })

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Confidence bands (95% Wilson interval) — width shows how much to trust
    # each region: wide band = few samples, don't over-read it.
    ax.fill_between(xs, type1_lo, type1_hi, color=COLOR_TYPE1, alpha=0.18, linewidth=0, zorder=1)
    ax.fill_between(xs, type2_lo, type2_hi, color=COLOR_TYPE2, alpha=0.18, linewidth=0, zorder=1)

    # Smoothed trend lines through the bands.
    ax.plot(xs, type1_smooth, color=COLOR_TYPE1_SMOOTH, linewidth=2.6, zorder=3,
            label="Type-I (false trigger)")
    ax.plot(xs, type2_smooth, color=COLOR_TYPE2_SMOOTH, linewidth=2.6, zorder=3,
            label="Type-II (missed trigger)")

    ax.axvline(args.eta, color="#999999", linestyle=":", linewidth=1.2, ymax=0.95, zorder=2)
    ax.text(args.eta, 101, f"injury risk threshold = {args.eta}", ha="center", va="bottom",
            fontsize=9.5, color="#666666")

    # Flag the most prominent peak that's actually backed by few samples —
    # a concrete example to anchor the general "wide band = unreliable" note.
    def flag_unreliable_peak(xs, smooth, eff_n, lo_band, hi_band, color, va_offset):
        valid = ~np.isnan(smooth)
        if not valid.any():
            return
        peak_idx = np.nanargmax(np.where(valid, smooth, -np.inf))
        n_at_peak = eff_n[peak_idx]
        band_width = hi_band[peak_idx] - lo_band[peak_idx]
        if band_width < 40:  # band isn't unusually wide here — nothing to flag
            return
        ax.annotate(
            f"n≈{int(n_at_peak)} here — wide CI,\nnot a reliable peak",
            xy=(xs[peak_idx], smooth[peak_idx]), xytext=(xs[peak_idx] + 0.12, smooth[peak_idx] + va_offset),
            fontsize=8.5, color=color, ha="left",
            arrowprops=dict(arrowstyle="->", color=color, linewidth=1.1, shrinkA=2, shrinkB=6),
        )

    flag_unreliable_peak(xs, type1_smooth, type1_eff_n, type1_lo, type1_hi, COLOR_TYPE1_SMOOTH, 12)
    flag_unreliable_peak(xs, type2_smooth, type2_eff_n, type2_lo, type2_hi, COLOR_TYPE2_SMOOTH, 10)

    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xticklabels([f"{v:.1f}" for v in np.arange(0, 1.01, 0.1)])
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Error rate (%)", fontsize=12)
    ax.set_xlabel("Physics-estimated injury risk at the decision frame", fontsize=12)

    fig.suptitle(args.title or "Error Rate vs. Physics-Estimated Risk at Trigger or Collision",
                 fontsize=15, fontweight="bold", y=0.975)
    ax.set_title("Quantifying the reliability of the RL controller's trigger / no-trigger output",
                 fontsize=10.5, color="#666666", pad=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", frameon=False, fontsize=10.5, ncol=2, bbox_to_anchor=(0.5, 0.93))

    fig.text(0.5, 0.012,
             "Shaded bands = 95% confidence interval (Wilson score). Wide bands mark regions with few samples — "
             "treat peaks there as noise, not signal.",
             ha="center", fontsize=8.5, color="#888888")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(args.out, dpi=180)
    print(f"Saved figure to: {args.out}")


def main():
    args = parse_args()
    if args.data_csv:
        included = pd.read_csv(args.data_csv)
    else:
        if not args.results_csv:
            raise ValueError("Pass either --results_csv (full recompute) or --data_csv (replot only).")
        included = compute_physics_data(args)
    make_plot(included, args)


if __name__ == "__main__":
    main()
