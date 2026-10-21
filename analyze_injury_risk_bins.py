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
    parser.add_argument("--bin_width", type=float, default=0.1)
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

    labels, rates, counts, fracs, colors = [], [], [], [], []
    eta_boundary_idx = None

    # No-collision bucket: triggering here is always a false trigger (Type-I).
    n_total = len(no_collision)
    n_triggered = int((no_collision["triggered"] == True).sum())
    labels.append("No\ncollision")
    rates.append(100.0 * n_triggered / n_total if n_total else 0.0)
    counts.append((n_triggered, n_total))
    fracs.append("Type-I")
    colors.append("orange")

    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = collided[(collided["pjoint"] >= lo) & (collided["pjoint"] < hi + 1e-9)]
        n_total = len(in_bin)
        n_triggered = int((in_bin["triggered"] == True).sum())

        is_high_risk = lo >= args.eta - 1e-9
        if is_high_risk:
            if eta_boundary_idx is None:
                eta_boundary_idx = len(labels) - 0.5
            n_wrong = n_total - n_triggered  # missed trigger
            err_type = "Type-II"
            color = "red"
        else:
            n_wrong = n_triggered  # false trigger
            err_type = "Type-I"
            color = "orange"

        labels.append(f"[{lo:.1f},{hi:.1f})")
        rates.append(100.0 * n_wrong / n_total if n_total else np.nan)
        counts.append((n_wrong, n_total))
        fracs.append(err_type)
        colors.append(color)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, rates, color=colors, width=0.65, edgecolor="white", linewidth=1)

    for xi, rate, (n_wrong, n_total) in zip(x, rates, counts):
        if n_total == 0:
            ax.text(xi, 1.5, "n=0", ha="center", va="bottom", fontsize=9, color="gray")
            continue
        label_y = rate + 1.5 if not np.isnan(rate) else 1.5
        ax.text(xi, label_y, f"{n_wrong}/{n_total}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Error rate (%)")
    ax.set_xlabel("Injury risk level (pjoint)")
    if eta_boundary_idx is not None:
        ax.axvline(eta_boundary_idx, color="dimgray", linestyle=":", linewidth=1, ymax=0.93)
        ax.text(eta_boundary_idx, 100, f"eta = {args.eta}", ha="center", va="bottom", fontsize=9, color="dimgray")

    title = args.title or "Type-I / Type-II error by injury risk level"
    ax.set_title(title)
    ax.grid(axis="y", color="lightgray", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="orange", label="Type-I (false trigger)"),
        plt.Rectangle((0, 0), 1, 1, color="red", label="Type-II (missed trigger)"),
    ]
    ax.legend(handles=handles, loc="upper center")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved figure to: {args.out}")

    print("\nBin breakdown:")
    for label, rate, (n_wrong, n_total), err_type in zip(labels, rates, counts, fracs):
        print(f"  {label:>14} [{err_type}]: {n_wrong}/{n_total} = {rate:.2f}%" if n_total else f"  {label:>14}: n=0")


if __name__ == "__main__":
    main()
