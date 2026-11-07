#!/usr/bin/env python3
"""Pupil points only against all eight points (smallest subset): the repair on the truth frames and on the whole
videos, for the report (page 9.3).

    python repair_scope_figure.py

Reads results/flag_and_repair/all_videos/full_repair.csv (rows pupil / smallest) and the per-video whole_video.csv of
results/flag_and_repair/videoNN_either_labelregion/ (pupil points only) and videoNN_either_labelregion_all_smallest/
(all eight points). Writes results/flag_and_repair/all_videos/repair_scope.png. Nothing existing is modified."""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_peak_repair import RES  # noqa: E402

FAR = RES / "flag_and_repair"
RUNS = {"pupil": ("either_labelregion", "pupil points only", "#8c8c8c"),
        "smallest": ("either_labelregion_all_smallest", "all eight points, smallest subset", "#0072b2")}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888",
                     "axes.labelcolor": "#222", "xtick.color": "#444", "ytick.color": "#444", "axes.titlesize": 11})


def per_video(tag):
    rows = []
    for f in sorted(FAR.glob(f"video*_{tag}/whole_video.csv")):
        d = pd.read_csv(f).iloc[0]
        rows.append({"video": int(d.video), "repaired": d.repaired_pct, "still": d.flagged_not_repaired_pct,
                     "plausible": 100 * d.flagged_passes_check / d.frames})
    return pd.DataFrame(rows).set_index("video")


def main():
    F = pd.read_csv(FAR / "all_videos" / "full_repair.csv").set_index("run")
    V = {k: per_video(tag) for k, (tag, _, _) in RUNS.items()}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), gridspec_kw={"width_ratios": [1.3, 2, 2]})
    # left: truth frames
    cats = [("wrong_repaired_to_good", "wrong,\nrepaired"), ("wrong_still_flagged", "wrong,\nstill flagged"),
            ("good_made_worse", "right,\nmade worse"), ("good_passes_check", "right,\nleft alone")]
    x = np.arange(len(cats))
    w = 0.38
    ax = axes[0]
    for j, (k, (_, label, col)) in enumerate(RUNS.items()):
        ys = [float(F.loc[k, c]) for c, _ in cats]
        ax.bar(x + (j - 0.5) * w, ys, w * 0.92, color=col, label=label, edgecolor="white", linewidth=0.5)
        for xi, y in zip(x + (j - 0.5) * w, ys):
            ax.text(xi, y, f"{y:.0f}", ha="center", va="bottom", fontsize=8, color="#222")
    ax.set_xticks(x)
    ax.set_xticklabels([c for _, c in cats], fontsize=8)
    ax.set_ylabel("truth frames (23 wrong, 40 right flagged)")
    ax.set_ylim(0, 44)
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
    ax.set_axisbelow(True)
    # middle and right: per video
    vids = V["pupil"].index.tolist()
    xv = np.arange(len(vids))
    for ax, col, ylabel in ((axes[1], "repaired", "frames repaired, % of the video"),
                            (axes[2], "still", "frames flagged and not repaired, % of the video")):
        for j, (k, (_, label, c)) in enumerate(RUNS.items()):
            ys = V[k].loc[vids, col].values
            ax.bar(xv + (j - 0.5) * w, ys, w * 0.92, color=c, label=label, edgecolor="white", linewidth=0.5)
        ax.set_xticks(xv)
        ax.set_xticklabels([str(v) for v in vids], fontsize=8)
        ax.set_xlabel("video")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
        ax.set_axisbelow(True)
    for ax, col in ((axes[1], "repaired"), (axes[2], "still")):
        tot = {k: float(F.loc[k, "repaired_pct" if col == "repaired" else "flagged_not_repaired_pct"]) for k in RUNS}
        ax.text(0.98, 0.95, f"all videos: {tot['pupil']:.2f}% → {tot['smallest']:.2f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="#222")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FAR / "all_videos" / "repair_scope.png"
    fig.savefig(out, dpi=170)
    print("written", out)


if __name__ == "__main__":
    main()
