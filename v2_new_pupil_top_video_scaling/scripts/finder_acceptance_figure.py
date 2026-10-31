#!/usr/bin/env python3
"""Two figures for the report. (1) results/fit_threshold/finder_curves.png: for the three DeepLabCut outlier
algorithms and the variant used here, the share of the 23 wrong frames found against the share of the 1307 right frames
flagged, over every threshold (one curve per algorithm, log x); the dot marks the threshold used (14 % of eye width).
(2) results/flag_and_repair/all_videos/acceptance_bars.png: the seven acceptance tests side by side: wrong frames
repaired / still flagged / left alone (closed eye), right frames repaired, made worse, still flagged. Nothing existing is
modified."""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from second_peak_repair import RES

line = float(pd.read_csv(RES / "error_tier" / "tier.csv").iloc[0].line_eye_widths)
S = pd.read_csv(RES / "unreliable_frames_sweep" / "signals_all_frames.csv")
S = S.drop(columns=[c for c in S.columns if c == "eye_width_px"])
E = pd.read_csv(RES / "error_tier" / "truth_errors.csv")[["video", "frame", "error_rel", "eye_width_px"]]
M = E.merge(S, on=["video", "frame"])
wrong = (M.error_rel > line).to_numpy()
signals = [("jump: largest move since the previous frame / eye width", M.jump.to_numpy(), "#9bb7d4", None),
           ("uncertain: lowest confidence of the 8 points (low = suspicious)", (-M.min_conf).to_numpy(), "#e0a458", None),
           ("fitting, DLC default: mean deviation of the 8 points / eye width", (M.fitting / M.eye_width_px).to_numpy(), "#7fb77e", None),
           ("fitting, used here: largest deviation of the 8 points / eye width", (M.fit_o200_max / M.eye_width_px).to_numpy(), "#2f6db3", line)]
fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
for name, v, col, mark in signals:
    ts = np.unique(v)
    fs = np.array([(v[~wrong] > t).mean() for t in ts])
    rc = np.array([(v[wrong] > t).mean() for t in ts])
    order = np.argsort(fs)
    ax.plot(100 * np.maximum(fs[order], 2e-4), 100 * rc[order], color=col, lw=2.2 if mark else 1.8, label=name)
    if mark is not None:
        ax.plot([100 * (v[~wrong] > mark).mean()], [100 * (v[wrong] > mark).mean()], "o", color=col, ms=8)
        ax.annotate(f"threshold used: {100 * mark:.0f}% of eye width\n{int((v[wrong] > mark).sum())} of 23 found, {100 * (v[~wrong] > mark).mean():.1f}% of right frames flagged",
                    (100 * (v[~wrong] > mark).mean(), 100 * (v[wrong] > mark).mean()), xytext=(0.35, 60), textcoords="data", fontsize=9, color="0.25",
                    arrowprops=dict(arrowstyle="-", color="0.6"))
ax.set_xscale("log")
ax.set_xlim(0.02, 100)
ax.set_ylim(0, 103)
ax.set_xlabel("right frames flagged (% of 1307), log scale")
ax.set_ylabel("wrong frames found (% of 23)")
ax.set_title("Finding the 23 wrong frames: the three DeepLabCut algorithms, every threshold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="0.92", which="both")
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
fig.savefig(RES / "fit_threshold" / "finder_curves.png", dpi=150)

T = pd.read_csv(RES / "flag_and_repair" / "all_videos" / "totals.csv")
names = {"geometry": "geometry rule\n(hand-made)", "fit": "same fitting\nscore", "conf": "second-peak\nconfidence", "jump": "jump", "labelgeo": "label range,\none step",
         "labelregion0": "label range,\ntwo steps", "labelregion": "label range,\ntwo steps,\nwith margin"}
order = ["labelregion", "geometry", "labelregion0", "conf", "labelgeo", "jump", "fit"]
T = T.set_index("accept").loc[order]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), constrained_layout=True)
x = np.arange(len(T))
ax = axes[0]
ax.bar(x, T.wrong_repaired_to_good, color="#2f6db3", label="repaired")
ax.bar(x, T.wrong_flagged_not_repaired, bottom=T.wrong_repaired_to_good, color="#e0a458", label="still flagged")
ax.bar(x, T.wrong_left_untouched, bottom=T.wrong_repaired_to_good + T.wrong_flagged_not_repaired, color="#cfcfcf", label="left alone (eye judged closed)")
for i, r in enumerate(T.itertuples()):
    ax.text(i, r.wrong_repaired_to_good / 2, f"{int(r.wrong_repaired_to_good)}", ha="center", va="center", color="white", fontsize=10)
ax.set_title("the 23 wrong frames")
ax.set_ylabel("frames")
ax = axes[1]
bw = 0.28
ax.bar(x - bw, T.good_repaired, width=bw - 0.03, color="#2f6db3", label="repaired")
ax.bar(x, T.good_made_worse, width=bw - 0.03, color="#d1495b", label="made worse")
ax.bar(x + bw, T.good_flagged_not_repaired, width=bw - 0.03, color="#e0a458", label="still flagged")
ax.set_title("the 28 right frames that were flagged")
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels([names[a] for a in T.index], fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.92")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5)
fig.suptitle("Seven acceptance tests for the second-peak repair, same 19 videos, same flagged frames", fontsize=11)
fig.savefig(RES / "flag_and_repair" / "all_videos" / "acceptance_bars.png", dpi=150)
print("saved finder_curves.png, acceptance_bars.png")
