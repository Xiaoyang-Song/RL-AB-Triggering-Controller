"""
Breaks Type-I/Type-II error down by injury-risk level (pjoint), instead of
the single low/high split at eta that visualize.py reports. Bins trajectories
by their real, simulation-derived pjoint (no-collision trajectories get their
own bucket), and for each bin reports:
  - below eta (or no-collision): Type-I rate = fraction wrongly triggered
  - at/above eta: Type-II rate = fraction wrongly left untriggered

Usage:
    python analyze_injury_risk_bins.py \
        --results_csv results/sweep_allfive/j1/evaluation_results_b15.0_c15.0_b25.0_c25.0_c35.0_eta0.2_gforce_best.csv \
        --eta 0.2 \
        --title "gforce sweep (all-5.0 reward), j1: lr=0.002 hidden_dim=256" \
        --out plots/sweep_allfive/j1/injury_risk_bins.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Bin Type-I/Type-II error by injury risk level.")
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--bin_width", type=float, default=0.05)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.results_csv)

    assert abs(args.eta / args.bin_width - round(args.eta / args.bin_width)) < 1e-9, (
        f"--eta ({args.eta}) must land exactly on a bin edge for --bin_width ({args.bin_width})"
    )

    edges = np.arange(0.0, 1.0 + args.bin_width / 2, args.bin_width)
    collided = df[df["collision"] == True]
    no_collision = df[df["collision"] == False]

    # x = -bin_width for "no collision" (true pjoint = 0, qualitatively distinct
    # from the collision continuum, so it's offset rather than fused into bin 0).
    no_collision_x = -args.bin_width
    xs, rates, counts, err_types = [], [], [], []

    n_total = len(no_collision)
    n_triggered = int((no_collision["triggered"] == True).sum())
    xs.append(no_collision_x)
    rates.append(100.0 * n_triggered / n_total if n_total else np.nan)
    counts.append(n_total)
    err_types.append("Type-I")

    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = collided[(collided["pjoint"] >= lo) & (collided["pjoint"] < hi + 1e-9)]
        n_total = len(in_bin)
        n_triggered = int((in_bin["triggered"] == True).sum())

        is_high_risk = lo >= args.eta - 1e-9
        if is_high_risk:
            n_wrong = n_total - n_triggered  # missed trigger
            err_type = "Type-II"
        else:
            n_wrong = n_triggered  # false trigger
            err_type = "Type-I"

        xs.append((lo + hi) / 2)
        rates.append(100.0 * n_wrong / n_total if n_total else np.nan)
        counts.append(n_total)
        err_types.append(err_type)

    xs = np.array(xs)
    rates = np.array(rates)
    color_map = {"Type-I": "orange", "Type-II": "red"}
    colors = [color_map[t] for t in err_types]

    fig, ax = plt.subplots(figsize=(13, 6.5))

    # Thin neutral connector so the curve's shape reads across both error
    # regimes; identity (Type-I vs Type-II) still carries entirely in marker color.
    ax.plot(xs, rates, color="lightgray", linewidth=1.5, zorder=1)
    for err_type, color in color_map.items():
        mask = np.array([t == err_type for t in err_types])
        ax.scatter(
            xs[mask], rates[mask], color=color, s=45, zorder=3,
            label=f"{err_type} ({'false trigger' if err_type == 'Type-I' else 'missed trigger'})",
        )

    for xi, rate, n_total in zip(xs, rates, counts):
        if n_total == 0 or np.isnan(rate):
            continue
        ax.annotate(
            f"n={n_total}", (xi, rate), textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=7.5, color="dimgray",
        )

    ax.axvline(args.eta, color="dimgray", linestyle=":", linewidth=1, ymax=0.93)
    ax.text(args.eta, 100, f"eta = {args.eta}", ha="center", va="bottom", fontsize=9, color="dimgray")

    ax.set_xlim(no_collision_x - args.bin_width, 1.0 + args.bin_width)
    ax.set_xticks([no_collision_x] + list(np.arange(0, 1.01, 0.1)))
    ax.set_xticklabels(["No\ncollision"] + [f"{v:.1f}" for v in np.arange(0, 1.01, 0.1)])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Error rate (%)")
    ax.set_xlabel("True injury risk (pjoint, from the actual recorded collision)")

    title = args.title or "Type-I / Type-II error vs. true injury risk"
    ax.set_title(title)
    ax.grid(axis="y", color="lightgray", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved figure to: {args.out}")

    print("\nBin breakdown:")
    for xi, rate, n_total, err_type in zip(xs, rates, counts, err_types):
        x_label = "No collision" if xi == no_collision_x else f"{xi:.3f}"
        if n_total == 0:
            print(f"  {x_label:>14}: n=0")
        else:
            n_wrong = round(rate / 100.0 * n_total)
            print(f"  {x_label:>14} [{err_type}]: {n_wrong}/{n_total} = {rate:.2f}%")


if __name__ == "__main__":
    main()
