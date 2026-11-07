#!/usr/bin/env python3
"""One figure for "how the check, the finder and the points to replace were chosen" (page 9.3): three comparisons made
while the repair was being settled, each as truth-frame outcome (top) and whole-video shares (bottom).

    python method_selection_figure.py

Reads results/flag_and_repair/all_videos/acceptance_by_finder.csv (seven checks, finder = fitting, pupil points only),
finders.csv (three finders, check = label range, pupil points only) and full_repair.csv (three replacement scopes,
finder = either, check = label range). Writes results/flag_and_repair/all_videos/method_selection.png. Nothing existing
is modified."""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_peak_repair import RES  # noqa: E402

A = RES / "flag_and_repair" / "all_videos"
BLUE, ORANGE, RED, GREY = "#0072b2", "#e69f00", "#d55e00", "#9a9a9a"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888",
                     "axes.labelcolor": "#222", "xtick.color": "#444", "ytick.color": "#444", "axes.titlesize": 10.5})


def panel(ax_top, ax_bot, labels, repaired, worse, rep_pct, still_pct, used, title):
    x = np.arange(len(labels))
    w = 0.38
    ax_top.bar(x - w / 2, repaired, w * 0.92, color=BLUE, label="wrong truth frames repaired (of 23)", edgecolor="white", linewidth=0.5)
    ax_top.bar(x + w / 2, worse, w * 0.92, color=RED, label="right truth frames made worse", edgecolor="white", linewidth=0.5)
    for xi, y in zip(x - w / 2, repaired):
        ax_top.text(xi, y + 0.3, f"{y:.0f}", ha="center", va="bottom", fontsize=8, color="#222")
    for xi, y in zip(x + w / 2, worse):
        ax_top.text(xi, y + 0.3, f"{y:.0f}", ha="center", va="bottom", fontsize=8, color="#222")
    ax_top.set_ylim(0, 19)
    ax_top.set_title(title)
    ax_bot.bar(x - w / 2, rep_pct, w * 0.92, color=BLUE, label="frames repaired, % of all frames", edgecolor="white", linewidth=0.5)
    ax_bot.bar(x + w / 2, still_pct, w * 0.92, color=ORANGE, label="frames flagged and not repaired, % of all frames", edgecolor="white", linewidth=0.5)
    for xi, y in zip(x - w / 2, rep_pct):
        ax_bot.text(xi, y + 0.03, f"{y:.2f}", ha="center", va="bottom", fontsize=7.5, color="#222")
    for xi, y in zip(x + w / 2, still_pct):
        ax_bot.text(xi, y + 0.03, f"{y:.2f}", ha="center", va="bottom", fontsize=7.5, color="#222")
    ax_bot.set_ylim(0, 2.7)
    for ax in (ax_top, ax_bot):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.axvspan(used - 0.5, used + 0.5, color="#dff0d8", zorder=0)
    ax_top.text(used, 18.3, "used", ha="center", va="top", fontsize=8.5, color="#2e7d32", fontweight="bold")


def main():
    C = pd.read_csv(A / "acceptance_by_finder.csv")
    C = C[C.finder == "fit"].set_index("accept")
    checks = [("labelregion", "(a) label\nrange"), ("geometry", "(b) geometry\nrule"), ("fit", "(c) fitting\nscore"),
              ("conf", "(d) confi-\ndence"), ("jump", "(e) jump"), ("labelgeo", "(f) range,\nno margin"), ("labelregion0", "(g) range,\nno margin,\ncheck first")]
    F = pd.read_csv(A / "finders.csv").set_index("finder")
    finders = [("fit", "fitting"), ("pca", "PCA"), ("either", "either")]
    R = pd.read_csv(A / "full_repair.csv").set_index("run")
    scopes = [("pupil", "pupil points\nonly"), ("all", "all 8,\nwhole\nreplacement"), ("smallest", "all 8,\nsmallest\nsubset")]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.6), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    panel(axes[0, 0], axes[1, 0], [l for _, l in checks], [C.loc[k, "wrong_repaired_to_good"] for k, _ in checks],
          [C.loc[k, "good_made_worse"] for k, _ in checks], [C.loc[k, "repaired_pct"] for k, _ in checks],
          [C.loc[k, "flagged_not_repaired_pct"] for k, _ in checks], 0, "1. the check (fitting, pupil points only)")
    panel(axes[0, 1], axes[1, 1], [l for _, l in finders], [F.loc[k, "wrong_repaired_to_good"] for k, _ in finders],
          [F.loc[k, "good_made_worse"] for k, _ in finders], [F.loc[k, "repaired_pct"] for k, _ in finders],
          [F.loc[k, "flagged_not_repaired_pct"] for k, _ in finders], 2, "2. the finder (label range, pupil only)")
    panel(axes[0, 2], axes[1, 2], [l for _, l in scopes], [R.loc[k, "wrong_repaired_to_good"] for k, _ in scopes],
          [R.loc[k, "good_made_worse"] for k, _ in scopes], [R.loc[k, "repaired_pct"] for k, _ in scopes],
          [R.loc[k, "flagged_not_repaired_pct"] for k, _ in scopes], 2, "3. points replaced (either, label range)")
    axes[0, 0].set_ylabel("truth frames")
    axes[1, 0].set_ylabel("% of all 1.31 M frames")
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = A / "method_selection.png"
    fig.savefig(out, dpi=170)
    print("written", out)


if __name__ == "__main__":
    main()
