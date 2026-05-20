"""
Plot Type-I/II error rates AND per-scenario success rates vs. noise magnitude
with 2-sigma uncertainty bands from results/uncertainty/mag**/summary_rep.csv.

Figure layout  (1 row × 2 col):
  Left  – Aggregate errors:   Type-I (false trigger) + Type-II (missed trigger)
  Right – Per-scenario rates: successful trigger (high-risk)
                              successful no-trigger (low-risk collision)
                              successful no-trigger (no-collision)
"""

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── Global style ───────────────────────────────────────────────────────────────
plt.style.use("seaborn-whitegrid")

mpl.rcParams.update({
    "font.family":          "DejaVu Sans",
    "font.size":            12,
    "axes.titlesize":       13,
    "axes.titleweight":     "bold",
    "axes.labelsize":       12,
    "axes.labelweight":     "bold",
    "legend.fontsize":      10.5,
    "legend.framealpha":    0.92,
    "legend.edgecolor":     "#cccccc",
    "xtick.labelsize":      10.5,
    "ytick.labelsize":      10.5,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.spines.left":     True,
    "axes.spines.bottom":   True,
    "axes.linewidth":       1.2,
    "xtick.major.size":     4,
    "ytick.major.size":     4,
    "lines.linewidth":      2.2,
    "lines.markersize":     8,
    "grid.color":           "#e0e0e0",
    "grid.linewidth":       0.8,
    "figure.dpi":           150,
})

SIGMA = 2          # band half-width in standard deviations

# ── Palette (colorblind-friendly) ──────────────────────────────────────────────
C_T1 = "#D62728"   # vivid red    – Type-I error
C_T2 = "#1F77B4"   # steel blue   – Type-II error
C_ST = "#2CA02C"   # forest green – successful trigger
C_SL = "#FF7F0E"   # amber        – successful no-trigger (LRC)
C_SN = "#9467BD"   # violet       – successful no-trigger (NC)

# ── Helpers ────────────────────────────────────────────────────────────────────
def parse_mean_std(cell: str):
    m = re.match(r"([\d.+-]+)\s*±\s*([\d.+-]+)", cell.strip())
    if m:
        return float(m.group(1)), float(m.group(2))
    try:
        return float(cell.strip()), 0.0
    except ValueError:
        raise ValueError(f"Cannot parse: {cell!r}")


def read_summary(csv_path: str) -> dict:
    with open(csv_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    header, data_row = None, None
    for line in lines:
        if line.startswith("n_trajectories"):
            header = line.split(",")
        else:
            data_row = line.split(",")
    if header is None or data_row is None:
        raise ValueError(f"Unexpected format: {csv_path}")
    return {col: parse_mean_std(val) for col, val in zip(header, data_row)}


def vec(key, sub, records):
    return (np.array([r[key][0] for r in records]),
            np.array([r[key][1] for r in records]))


def draw(ax, x, mean, std, color, marker, label, sigma=SIGMA):
    ax.plot(x, mean, marker=marker, color=color, linewidth=2.2,
            markersize=8, label=label, zorder=3,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.fill_between(x, mean - sigma * std, mean + sigma * std,
                    color=color, alpha=0.15, zorder=2)


def style_ax(ax, title, ylabel, mags):
    ax.set_title(title, pad=10)
    ax.set_xlabel("Noise Magnitude", labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_xticks(mags)
    ax.set_xticklabels([f"{m:g}" for m in mags])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.tick_params(axis="both", direction="out", length=4)
    ax.set_xlim(mags[0] - 0.01, mags[-1] + 0.01)
    ax.spines["left"].set_color("#aaaaaa")
    ax.spines["bottom"].set_color("#aaaaaa")


# ── Collect data ───────────────────────────────────────────────────────────────
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "uncertainty")
pattern = os.path.join(base_dir, "mag*", "summary_rep.csv")

records = []
for path in sorted(glob.glob(pattern)):
    mag = float(re.search(r"mag([\d.]+)", path).group(1))
    stats = read_summary(path)
    records.append({
        "mag":      mag,
        "t1":       stats["type1_rate_%"],
        "t2":       stats["type2_rate_%"],
        "suc_trig": stats["success_trigger_rate_%"],
        "suc_lrc":  stats["low_risk_not_triggered_%"],
        "suc_nc":   stats["no_collision_not_triggered_%"],
    })

records.sort(key=lambda r: r["mag"])
mags = np.array([r["mag"] for r in records])

t1_m, t1_s = vec("t1",       None, records)
t2_m, t2_s = vec("t2",       None, records)
st_m, st_s = vec("suc_trig", None, records)
sl_m, sl_s = vec("suc_lrc",  None, records)
sn_m, sn_s = vec("suc_nc",   None, records)

# ── Print summary ──────────────────────────────────────────────────────────────
print(f"{'Mag':>6}  {'Type-I':>14}  {'Type-II':>14}  "
      f"{'SucTrig':>14}  {'SucLRC':>14}  {'SucNC':>14}")
for i, m in enumerate(mags):
    print(f"{m:>6.2f}  "
          f"{t1_m[i]:>6.2f}±{t1_s[i]:.2f}  "
          f"{t2_m[i]:>6.2f}±{t2_s[i]:.2f}  "
          f"{st_m[i]:>6.2f}±{st_s[i]:.2f}  "
          f"{sl_m[i]:>6.2f}±{sl_s[i]:.2f}  "
          f"{sn_m[i]:>6.2f}±{sn_s[i]:.2f}")

# ── Figure: 1 row × 2 columns ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2),
                                gridspec_kw={"wspace": 0.30})

# ── Left panel: aggregate errors ──────────────────────────────────────────────
draw(ax1, mags, t1_m, t1_s, C_T1, "o", "Type-I Error  (false trigger)")
draw(ax1, mags, t2_m, t2_s, C_T2, "s", "Type-II Error (missed trigger)")
style_ax(ax1, "Aggregate Error Rates", "Error Rate", mags)
ax1.legend(loc="upper left")

# ── Right panel: per-scenario success rates ───────────────────────────────────
draw(ax2, mags, st_m, st_s, C_ST, "o", "Successful Trigger\n(high-risk correctly triggered)")
draw(ax2, mags, sl_m, sl_s, C_SL, "s", "Successful No-Trigger\n(low-risk collision skipped)")
draw(ax2, mags, sn_m, sn_s, C_SN, "^", "Successful No-Trigger\n(no-collision skipped)")
style_ax(ax2, "Per-Scenario Success Rates", "Success Rate", mags)
ax2.legend(loc="lower left")

# ── Shared panel labels (a / b) ───────────────────────────────────────────────
for ax, lbl in zip((ax1, ax2), ("(a)", "(b)")):
    ax.text(-0.08, 1.04, lbl, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")

# ── Figure-level caption ──────────────────────────────────────────────────────
fig.text(0.5, -0.02,
         f"Shaded bands = mean ± {SIGMA}σ  |  20 replications per noise level",
         ha="center", va="top", fontsize=10, color="#555555",
         style="italic")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(base_dir, "error_vs_noise.png")
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
print(f"\nSaved → {out_path}")
plt.show()
