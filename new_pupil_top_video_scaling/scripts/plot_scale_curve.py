#!/usr/bin/env python3
"""Scale curve of one video (new pupil standard): test error vs labels from this video.
    python plot_scale_curve.py --unit first5minvedio
Reads <unit>/results/scale_curve.csv (final-snapshot rows = headline) and, if present, <unit>/results/jump_flagged.csv."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--unit", required=True); a = ap.parse_args()
R = HERE / a.unit / "results"
d = pd.read_csv(R / "scale_curve.csv")
x = "training_frames_this_video"
head = d[d["label"].str.endswith("_final")].sort_values(x)
late = d[d["label"].str.endswith("_mAPlate")].sort_values(x)
BLUE, RED, GREEN, GRAY, PLUM = "#2F6B9A", "#D1495B", "#2A9D8F", "#888888", "#9B1D64"
fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), constrained_layout=True)
ax = axes[0]
ax.plot(head[x], head["median_frame_rmse_px"], "o-", color=BLUE, lw=2.2, ms=8, label="final snapshot (epoch 120) - headline")
ax.plot(late[x], late["median_frame_rmse_px"], "s", color=GRAY, ms=6, label="best validation mAP among epochs >= 96")
for _, r in head.iterrows():
    ax.annotate(f"{r['median_frame_rmse_px']:.1f}", (r[x], r["median_frame_rmse_px"]), textcoords="offset points", xytext=(6, 8), color=BLUE)
best = head["median_frame_rmse_px"].cummin()
ax.step(head[x], best, where="post", color=BLUE, lw=1, ls=":", label="running best (the stopping rule looks at this)")
ax.set_ylim(0, max(16, head["median_frame_rmse_px"].max() * 1.4)); ax.set_xticks(head[x])
ax.set_xlabel("labels from this video"); ax.set_ylabel("median frame RMSE on the frozen test set (px)")
ax.set_title("Typical-frame error (y axis from 0)"); ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc="lower left")
ax = axes[1]
for bp, c, ls in (("pupil_top", RED, "-"), ("pupil_left", "#E08E45", "-"), ("pupil_right", "#5B8E7D", "-"), ("pupil_bottom", BLUE, "-"),
                  ("eyelid_top", GRAY, "--"), ("eyelid_bottom", "#B0B0B0", "--"), ("eye_nasal_corner", PLUM, ":"), ("eye_temporal_corner", "#C77DB0", ":")):
    ax.plot(head[x], head[f"median_{bp}_px"], "o", ls=ls, color=c, lw=1.8, ms=6, label=bp)
ax.set_ylim(0, None); ax.set_xticks(head[x]); ax.set_xlabel("labels from this video"); ax.set_ylabel("median error of the keypoint (px)")
ax.set_title("Which keypoints carry the error (solid = pupil, dashed = eyelids, dotted = eye corners)"); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
ax = axes[2]
ax.plot(head[x], head["frac_points_conf_ge_0.6"] * 100, "o-", color=GREEN, lw=2.2, ms=8, label="test keypoints with confidence >= 0.6 (%)")
ax.plot(head[x], head["frac_frames_rmse_gt_50px"] * 100, "s--", color=RED, lw=1.6, ms=6, label="test frames with RMSE > 50 px (%)")
ax.set_ylim(0, 100); ax.set_xticks(head[x]); ax.set_xlabel("labels from this video"); ax.set_ylabel("%"); ax.grid(alpha=0.3)
jf = R / "jump_flagged.csv"
if jf.exists():
    j = pd.read_csv(jf); ax2 = ax.twinx()
    ax2.plot(j["model_labels"], j["jump_flagged_pool_frames"] / j["pool_frames"] * 100, "^-", color=PLUM, lw=1.8, ms=7, label="pool frames flagged by the jump rule (%, right axis)")
    ax2.set_ylim(0, 10); ax2.set_ylabel("% of pool frames flagged", color=PLUM)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="center right")
else:
    ax.legend(fontsize=8.5, loc="center right")
ax.set_title("Tail and label-free health signals")
n = int(d["n_test_frames"].iloc[0])
fig.suptitle(f"{a.unit}: labels needed on this video (new pupil standard, training from scratch on this video's labels only; "
             f"test = {n} frozen frames, never used for any decision)", fontsize=12)
fig.savefig(R / "scale_curve.png", dpi=140)
print(head[[x, "median_frame_rmse_px", "p90_frame_rmse_px", "frac_frames_rmse_gt_50px", "frac_points_conf_ge_0.6"]].to_string(index=False))
