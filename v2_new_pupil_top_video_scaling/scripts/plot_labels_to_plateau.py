#!/usr/bin/env python3
"""Labels needed per video (user request 2026-09-30): the plateau point of every video in the sequence, as one figure.

    python plot_labels_to_plateau.py

Reads results/labels_to_plateau.csv (one row per video: the plateau point decided by the stopping rule, with the
user's decision where the rule needed one - the same numbers as the table in results/README.md) and writes
results/labels_to_plateau.png. Left: labels of the video's own frames at the plateau. Right: the error at the plateau
and the 0-label error (the previous model applied unchanged). Videos still in progress are drawn hollow."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
R = HERE / "results"
d = pd.read_csv(R / "labels_to_plateau.csv")
x = np.arange(len(d))
lab = [f"video {r.video}\n{r.date}" for r in d.itertuples()]
done = d.status == "final"
BLUE, RED, GRAY = "#2F6B9A", "#D1495B", "#888888"

fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), constrained_layout=True)
ax = axes[0]
ax.bar(x[done], d.labels_at_plateau[done], color=BLUE, width=0.6, label="labels of this video at the plateau (final)")
if (~done).any():
    ax.bar(x[~done], d.labels_at_plateau[~done], color="white", edgecolor=BLUE, hatch="//", width=0.6, label="in progress (best step so far)")
for i, r in d.iterrows():
    ax.annotate(str(int(r.labels_at_plateau)), (i, r.labels_at_plateau), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8.5); ax.set_ylabel("labels of the video's own frames at the plateau")
ax.set_ylim(0, max(70, d.labels_at_plateau.max() * 1.25)); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8.5, loc="upper right")
ax.set_title("How many labels each video needed (stopping rule: two 20-label steps with <= 3% gain)", fontsize=10.5)
ax = axes[1]
ax.plot(x, d.zero_label_px.clip(upper=60), "o--", color=GRAY, ms=6, label="0 labels: previous model applied unchanged")
ax.plot(x[done], d.plateau_median_px[done], "o-", color=RED, ms=7, label="error at the plateau (median frame RMSE)")
if (~done).any():
    ax.plot(x[~done], d.plateau_median_px[~done], "o", color=RED, mfc="white", ms=7, label="in progress (best so far)")
for i, r in d.iterrows():
    ax.annotate(f"{r.plateau_median_px:.1f}", (i, r.plateau_median_px), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8.5, color=RED)
    if np.isfinite(r.zero_label_px) and r.zero_label_px > 60:
        ax.annotate(f"{r.zero_label_px:.0f} (off scale)", (i, 60), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8, color=GRAY)
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8.5); ax.set_ylabel("median frame RMSE on the 50 frozen test frames (px)")
ax.set_ylim(0, 62); ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc="upper right")
ax.set_title("Error at the plateau vs error before any label of that video", fontsize=10.5)
fig.suptitle("Labels needed per video along the sequence (mouse A: video 0; Pluto: one recording day each, 2025-10-31 -> 10-24)", fontsize=12)
fig.savefig(R / "labels_to_plateau.png", dpi=130)
print("saved", R / "labels_to_plateau.png")
