#!/usr/bin/env python3
"""Scale curve of the new-video experiment: error on the frozen Pluto test set vs number of
Pluto training labels. Headline = final snapshot; other rules/seeds shown as secondary marks."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

UNIT = Path(__file__).resolve().parents[1]
d = pd.read_csv(UNIT / "results/scale_curve.csv")
j = pd.read_csv(UNIT / "results/jump_flagged.csv")
x = "pluto_training_frames"
is_seed2 = d["label"].str.contains("seed43")
head = d[(d["snapshot_rule"].str.startswith("final") | d["snapshot_rule"].str.startswith("n/a")) & ~is_seed2].sort_values(x)
rep = d[d["snapshot_rule"].str.startswith("final") & is_seed2]
rob = d[d["snapshot_rule"].str.contains(">= 80")]
legacy = d[d["snapshot_rule"].str.contains("->")]
BLUE, RED, GREEN, GRAY = "#2F6B9A", "#D1495B", "#2A9D8F", "#888888"

fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), constrained_layout=True)
ax = axes[0]
ax.plot(head[x], head["median_frame_rmse_px"], "o-", color=BLUE, lw=2.2, ms=8, label="final snapshot (headline)")
for _, r in head.iterrows():
    ax.annotate(f"{r['median_frame_rmse_px']:.1f}", (r[x], r["median_frame_rmse_px"]), textcoords="offset points",
                xytext=(6, 8), fontsize=10, color=BLUE)
if len(rep):
    ax.plot(rep[x], rep["median_frame_rmse_px"], "o", mfc="none", mec=BLUE, ms=10, mew=1.8, label="second seed (run-to-run noise)")
if len(rob):
    ax.plot(rob[x], rob["median_frame_rmse_px"], "s", color=GRAY, ms=5, label="best val mAP among epochs >= 80")
if len(legacy):
    ax.plot(legacy[x], legacy["median_frame_rmse_px"], "x", color=RED, ms=10, mew=2.2,
            label="DLC default: best val mAP over ALL epochs")
ax.set_yscale("log"); ax.set_yticks([10, 20, 50, 100, 200]); ax.set_yticklabels(["10", "20", "50", "100", "200"])
ax.set_xlabel("Pluto frames labeled for training"); ax.set_ylabel("median frame RMSE on the frozen test set (px, log)")
ax.set_title("Typical-frame error", fontsize=12); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8.5)

ax = axes[1]
ax.plot(head[x], head["p90_frame_rmse_px"], "o-", color=RED, lw=2.2, ms=8, label="90th percentile of frame RMSE")
if len(rep):
    ax.plot(rep[x], rep["p90_frame_rmse_px"], "o", mfc="none", mec=RED, ms=10, mew=1.8, label="second seed")
ax.set_yscale("log"); ax.set_yticks([20, 50, 100, 200]); ax.set_yticklabels(["20", "50", "100", "200"])
ax.set_xlabel("Pluto frames labeled for training"); ax.set_ylabel("px (log)"); ax.grid(alpha=0.3, which="both")
ax2 = ax.twinx()
ax2.plot(head[x], head["frac_frames_rmse_gt_50px"] * 100, "s--", color=GRAY, lw=1.5, ms=6, label="% of test frames with RMSE > 50 px")
ax2.set_ylabel("% of test frames", color=GRAY); ax2.set_ylim(0, 100)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper right")
ax.set_title("Error tail (the frames that still fail)", fontsize=12)

ax = axes[2]
ax.plot(head[x], head["frac_points_conf_ge_0.6"] * 100, "o-", color=GREEN, lw=2.2, ms=8, label="test keypoints with confidence >= 0.6")
ax.plot(j["model_pluto_training_frames"], j["jump_flagged_pool_frames"] / j["pool_frames"] * 100, "s-", color="#9B1D64", lw=2,
        ms=7, label="training-pool frames flagged by the jump detector")
ax.set_ylim(0, 100); ax.set_xlabel("Pluto frames labeled for training"); ax.set_ylabel("%"); ax.grid(alpha=0.3)
ax.set_title("Label-free health signals (whole 16-min pool)", fontsize=12); ax.legend(fontsize=8.5, loc="center right")
n = int(d["n_test_frames"].iloc[0])
fig.suptitle(f"New mouse, new video (Pluto): how many labels does adaptation cost?  Training = 676 old-mouse frames + N Pluto "
             f"frames, from scratch; test = {n} held-out Pluto frames, never used for any decision", fontsize=12)
fig.savefig(UNIT / "results/03_scale_curve.png", dpi=140)
print(head[[x, "median_frame_rmse_px", "p90_frame_rmse_px", "frac_frames_rmse_gt_50px", "frac_points_conf_ge_0.6"]].to_string(index=False))
