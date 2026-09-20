#!/usr/bin/env python3
"""Selection figure with the keypoint time series underneath (advisor request, 2026-09-20).

    python viz_selection_timeline.py --unit first5minvedio --batch-no N [--pred-h5 FILE --pred-desc TEXT]

One figure per batch selection, saved to <unit>/results/ (advisor: every selection gets its cluster view AND the
time series on one sheet).

Top: the k-means view of the batch (appearance space, cluster sizes) as in viz_kmeans.py.
Below, on ONE shared time axis: clusters over time with the picked frames, then the keypoint
time series of the same stretch of video (pupil centre x / y, eye opening, pupil area, confidence,
frame-to-frame jump). Vertical lines mark the 20 picked frames in every row, so it can be read off
directly what the eye and the model were doing at the moments that were chosen for labeling.

The traces come from the predictions that SELECTED the batch (the model trained on the previous
step). Batch 1 was chosen before any adapted model existed, so its traces use the latest model
instead and say so in the title. Pupil centre/area use the current method (3-point endpoint
ellipse, blink-bridged: time-series-analysis/scripts/pupil_trace.py).
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pupil_trace import load, pupil_trace  # noqa: E402

EPS_FRAC = 0.03                       # same jump threshold as select_scale_frames.py
ap = argparse.ArgumentParser()
ap.add_argument("--unit", required=True)
ap.add_argument("--batch-no", type=int, required=True)
ap.add_argument("--pred-h5", default=None, help="predictions for the time series; default: <unit>/training-data/predictions_step<N-1>/*.h5")
ap.add_argument("--video-frames", type=int, default=None, help="total frames of the video when the predictions cover only part of it")
ap.add_argument("--pred-desc", default=None, help="one line saying which model made those predictions")
a = ap.parse_args()
UNIT = HERE / a.unit
TD = UNIT / "training-data"
name = f"batch{a.batch_no:02d}"
st = np.load(TD / "labels" / name / "kmeans_state.npz")
feats, labels, fidx, picks, k = st["feats"], st["labels"], st["frame_idx"], np.sort(st["picks"]), int(st["k"])

h5 = a.pred_h5 or sorted((TD / f"predictions_step{a.batch_no - 1:02d}").glob("*.h5"))[-1]
who = a.pred_desc or f"model trained on {20 * (a.batch_no - 1)} labels of this video (final snapshot)"
if a.batch_no == 1:
    who += " - batch 1 itself used NO model, traces shown for orientation only"
d = load(h5)
raw, fil, _, info = pupil_trace(d)
fps = 60.0
t_all = np.arange(len(d)) / fps
nvid = a.video_frames or len(d)
dur = nvid / fps                                       # same split rule as select_frames.py
seg = max(0.10 * dur, 60.0)
pool_end = (nvid - 2 * int(round(seg * fps)) - 2 * int(round(2.0 * fps))) / fps
if a.batch_no == 1:
    pool_end = min(pool_end, len(d) / fps)
bps = d.columns.get_level_values(0).unique()
width = np.hypot(d["eye_temporal_corner"]["x"] - d["eye_nasal_corner"]["x"], d["eye_temporal_corner"]["y"] - d["eye_nasal_corner"]["y"])
eps = EPS_FRAC * float(np.nanmedian(width))
jump = np.zeros(len(d))
for bp in bps:
    jump = np.fmax(jump, np.nan_to_num(np.hypot(np.diff(d[bp]["x"], prepend=np.nan), np.diff(d[bp]["y"], prepend=np.nan))))
opening = (d["eyelid_bottom"]["y"] - d["eyelid_top"]["y"]).to_numpy(float)
conf = np.min([d[bp]["likelihood"].to_numpy(float) for bp in ("pupil_left", "pupil_right", "pupil_bottom")], axis=0)
n_flag = int((jump[: int(pool_end * fps)] > eps).sum())

# ---- appearance space ------------------------------------------------------------------------
mu = feats.mean(0)
_, S, Vt = np.linalg.svd(feats - mu, full_matrices=False)
proj = (feats - mu) @ Vt[:2].T
expl = (S[:2] ** 2 / (S ** 2).sum()).sum()
sizes = np.bincount(labels, minlength=k)
pos = np.array([int(np.where(fidx == p)[0][0]) for p in picks])
pick_cluster = labels[pos]
cmap = plt.get_cmap("tab20", k)

fig = plt.figure(figsize=(18, 24))
gs = fig.add_gridspec(8, 2, width_ratios=[1.5, 1], height_ratios=[3.2, 1.9, 1, 1, 1, 1, 1, 1], hspace=0.42, wspace=0.18)
ax = fig.add_subplot(gs[0, 0])
ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap=cmap, s=4, alpha=0.45, vmin=0, vmax=k - 1, rasterized=True)
ax.scatter(proj[pos, 0], proj[pos, 1], marker="*", s=260, c="black", zorder=5)
for pp, cl in zip(pos, pick_cluster):
    ax.annotate(str(cl), (proj[pp, 0], proj[pp, 1]), fontsize=8, color="white", ha="center", va="center", zorder=6)
pool_txt = "every 5th frame of the training pool" if a.batch_no == 1 else "frames flagged by the jump rule"
ax.set_title(f"{len(feats):,} candidate frames ({pool_txt}) in appearance space; colors = {k} k-means clusters, "
             f"stars = the {len(picks)} picked frames", fontsize=10.5)
ax.set_xlabel(f"PC1 (first two components show {expl:.0%} of the variance)"); ax.set_ylabel("PC2")
ax = fig.add_subplot(gs[0, 1])
order = np.argsort(-sizes)
ax.barh(range(k), sizes[order], color=[cmap(i) for i in order])
ax.set_yticks(range(k)); ax.set_yticklabels([str(i) for i in order], fontsize=8); ax.invert_yaxis()
ax.set_xlabel("frames in cluster"); ax.set_ylabel("cluster id"); ax.set_title("Cluster sizes: every cluster gives one frame", fontsize=10.5)

# ---- shared time axis ------------------------------------------------------------------------
axc = fig.add_subplot(gs[1, :])
axc.scatter(fidx / fps, labels, c=labels, cmap=cmap, s=3, alpha=0.6, vmin=0, vmax=k - 1, rasterized=True)
axc.scatter(picks / fps, pick_cluster, marker="*", s=200, c="black", zorder=5)
axc.set_yticks(range(0, k, 2)); axc.set_ylabel("cluster id")
axc.set_title("Clusters over time (stars = picked frames). Everything below shares this time axis; "
              "thin vertical lines = the picked frames", fontsize=10.5)


def robust(v, lo=0.5, hi=99.5, pad=0.08):
    v = v[np.isfinite(v)]
    a_, b_ = np.percentile(v, [lo, hi])
    return a_ - pad * (b_ - a_), b_ + pad * (b_ - a_)


m = t_all <= pool_end
rows = [("pupil centre x (px)", raw["center_x"].to_numpy(), fil["center_x"].to_numpy(), "#2F6B9A"),
        ("pupil centre y (px)", raw["center_y"].to_numpy(), fil["center_y"].to_numpy(), "#2F6B9A"),
        ("eye opening (px)", opening, None, "#D1495B"),
        ("pupil area (px$^2$)", raw["area"].to_numpy(), fil["area"].to_numpy(), "#7A5195"),
        ("min confidence of L/R/B", conf, None, "#2A9D8F"),
        ("largest keypoint jump (px)", jump, None, "#555555")]
for r, (lab, y, yf, col) in enumerate(rows):
    axr = fig.add_subplot(gs[2 + r, :], sharex=axc)
    if yf is not None:
        axr.plot(t_all[m], y[m], lw=0.4, color="0.7", rasterized=True)
        axr.plot(t_all[m], yf[m], lw=0.6, color=col, rasterized=True)
        axr.set_ylim(*robust(yf[m]))
    else:
        axr.plot(t_all[m], y[m], lw=0.5, color=col, rasterized=True)
        axr.set_ylim(*((0, 1.02) if "confidence" in lab else robust(y[m], 0.5, 99.8)))
    if "jump" in lab:
        axr.axhline(eps, color="#D1495B", lw=1, ls="--")
        axr.set_yscale("symlog", linthresh=eps); axr.set_ylim(0, max(np.nanpercentile(y[m], 99.9), eps * 4))
        axr.set_title(f"dashed = jump threshold {eps:.1f} px (3% of eye width); {n_flag:,} pool frames above it "
                      f"({n_flag / (pool_end * fps):.0%})" + (" = the frames this batch was drawn from" if a.batch_no > 1 else " (orientation only)"), fontsize=9, color="#D1495B", loc="left", pad=3)
    if "confidence" in lab:
        axr.axhline(0.6, color="0.4", lw=0.8, ls=":")
    for p in picks:
        axr.axvline(p / fps, color="k", lw=0.6, alpha=0.55)
    axr.set_ylabel(lab, fontsize=9)
    if r < len(rows) - 1:
        plt.setp(axr.get_xticklabels(), visible=False)
    else:
        axr.set_xlabel("time in video (s)")
for p in picks:
    axc.axvline(p / fps, color="k", lw=0.6, alpha=0.35)
axc.set_xlim(0, pool_end)
plt.setp(axc.get_xticklabels(), visible=False)
how = "k-means over the whole training pool (no model yet)" if a.batch_no == 1 else f"jump rule on the {who}, then k-means among the flagged frames"
fig.suptitle(f"{a.unit} / {name}: how the 20 frames were chosen - {how}\n"
             f"Time series below: {who}.\n"
             "Grey = computed on every frame, color = untrusted frames bridged (3-point endpoint pupil).", fontsize=12, y=0.925)
out = UNIT / "results" / f"selection_{20 * a.batch_no:03d}_frames.png"      # named by the cumulative labels this batch brings the video to
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"{name}: {len(feats)} candidates, picks {picks.min() / fps:.0f}-{picks.max() / fps:.0f} s, jump eps {eps:.1f} px, "
      f"{n_flag} pool frames flagged -> {out.name}")
