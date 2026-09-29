#!/usr/bin/env python3
"""Fewer-labels study figure (user request 2026-09-27): test error of one video at 0 / 5 / 10 / 20 / 40 own labels.
    python plot_fewer_labels.py --unit 5_20251029_Pluto_spont_1
Reads <unit>/results/scale_curve.csv: the regular steps (labels "x..._stepNN_final") and the subset runs
("x..._step01_final_subKKs", from make_label_subsets.py + scale_step.py --subset). Final snapshot only (the headline).
Writes <unit>/results/fewer_labels.png and fewer_labels.csv."""
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
d = d[d["label"].str.contains("_final")].copy()
d["run"] = d["label"].str.extract(r"_final_(sub\d+[ab])$")[0].fillna("regular step")
cols = ["label", "run", x, "median_frame_rmse_px", "p90_frame_rmse_px", "frac_frames_rmse_gt_50px", "frac_points_conf_ge_0.6"]
t = d[cols].sort_values([x, "run"])
t.to_csv(R / "fewer_labels.csv", index=False)
print(t.to_string(index=False))

reg = t[t.run == "regular step"]
sub = t[t.run != "regular step"]
BLUE, ORANGE = "#2F6B9A", "#E08E45"
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
for ax, col, name in zip(axes, ["median_frame_rmse_px", "p90_frame_rmse_px", "frac_frames_rmse_gt_50px"],
                         ["median frame RMSE (px)", "90th-percentile frame RMSE (px)", "share of test frames with RMSE > 50 px"]):
    y = reg[col] * (100 if col.startswith("frac") else 1)
    ax.plot(reg[x], y, "o-", color=BLUE, lw=2, ms=8, label="regular steps (batches of 20)")
    for _, r in reg.iterrows():
        v = r[col] * (100 if col.startswith("frac") else 1)
        ax.annotate(f"{v:.1f}", (r[x], v), textcoords="offset points", xytext=(6, 6), color=BLUE, fontsize=9)
    for s, m in (("a", "D"), ("b", "s")):
        ss = sub[sub.run.str.endswith(s)]
        v = ss[col] * (100 if col.startswith("frac") else 1)
        ax.plot(ss[x], v, m, color=ORANGE, ms=8, mfc="white" if s == "b" else ORANGE, ls="none",
                label=f"subset {s} of batch01 (5 / 10 of its 20 frames)")
        for (_, r), vv in zip(ss.iterrows(), v):
            ax.annotate(f"{vv:.1f}", (r[x], vv), textcoords="offset points", xytext=(6, -12), color=ORANGE, fontsize=9)
    if col == "median_frame_rmse_px":
        ax.set_ylim(0, max(20, y[reg[x] > 0].max() * 1.5, (sub[col].max() if len(sub) else 0) * 1.2))
    elif col == "p90_frame_rmse_px":
        ax.set_yscale("log")
    ax.set_xticks(sorted(t[x].unique())); ax.set_xlabel("labels from this video")
    ax.set_title(name + (" (%)" if col.startswith("frac") else "")); ax.grid(alpha=0.3)
axes[0].legend(fontsize=8.5, loc="lower left")
fig.suptitle(f"{a.unit}: fewer labels - is 5 or 10 enough? (final snapshot, 50 frozen test frames; 0 = earlier videos' model unchanged)", fontsize=12)
fig.savefig(R / "fewer_labels.png", dpi=110)
print("saved", R / "fewer_labels.png")
