#!/usr/bin/env python3
"""Convergence figure of the active-learning experiment (rounds 0-11), one error formula per panel for all rounds.

    python plot_convergence.py

Reads ../results/convergence.csv and writes ../results/01_convergence_curves.png. Panels: mean RMSE over all test
points (sensitive to a few bad frames), median over the 100 test frames of the per-frame RMSE over the 8 keypoints
(primary), and median of the per-frame mean absolute error (alternative). The labelling rule was the same in every
round, so no round is marked."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
d = pd.read_csv(R / "convergence.csv")
COL = {"fitting": "#2a9d8f", "jump": "#d1495b", "uncertain": "#2f6b9a"}
PANELS = [("external_overall_rmse_px", "Mean RMSE (sensitive to a few bad frames)"),
          ("median_frame_rmse_px", "Median frame RMSE over 8 keypoints (primary)"),
          ("median_frame_mean_abs_px", "Median frame mean absolute error (alternative)")]
fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), constrained_layout=True)
for ax, (col, title) in zip(axes, PANELS):
    for b in ("fitting", "jump", "uncertain"):
        g = d[d.branch == b].sort_values("round")
        ax.plot(g.cumulative_training_frames, g[col], "o-", color=COL[b], lw=2, ms=6, label=b)
    ax.set_title(title, fontsize=12); ax.set_xlabel("cumulative human-reviewed training frames"); ax.set_ylabel("error on the 100 test frames (px)")
    ax.grid(alpha=0.3); ax.legend()
fig.suptitle("Active learning, rounds 0-11: every round scored with the same formula in each panel; the curves are flat from 80 to 300 frames", fontsize=13)
fig.savefig(R / "01_convergence_curves.png", dpi=200)
print("saved", R / "01_convergence_curves.png")
