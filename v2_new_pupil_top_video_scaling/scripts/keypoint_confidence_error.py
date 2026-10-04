#!/usr/bin/env python3
"""Error and model confidence of each of the 8 keypoints, per video and for all videos (user request 2026-10-04), read-only.

    python keypoint_confidence_error.py

Data: the per-point errors and confidences that scale_step.py saved when it scored each step on the video's 50 frozen
test frames (<unit>/training-data/eval/xLLL_stepNN_final/per_frame_errors.csv; final snapshot, no confidence cut-off,
points without a human label excluded). Error of a point = Euclidean distance (px) between prediction and human label;
confidence = the model's likelihood of that point (0-1).

Per video (finished videos of results/labels_to_plateau.csv), figure keypoint_confidence_error_videoNN.png:
  left    median error (px) of each keypoint at each step (20, 40, ... labels of this video)
  middle  mean confidence of each keypoint at each step
  right   every test point of the plateau-point model: confidence (x) against error (y, log scale), coloured by keypoint
All videos, figure keypoint_confidence_error_all_videos.png: the same two tables with one column per video (model at the
plateau point; video 7, plateau at 0 labels: its 20-label model) plus the pooled column, and the median error of the
points within each confidence bin (0-0.2, ..., 0.8-1), per keypoint.
Tables: keypoint_confidence_error_all_videos.csv (video, step, keypoint: n, median / p90 error, mean confidence, share
of points with confidence >= 0.6, Spearman correlation of confidence and error) and
keypoint_confidence_error_bins_all_videos.csv. Outputs in results/keypoint_confidence_error/. Nothing else is modified."""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_step import HERE  # noqa: E402

KP = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right", "eyelid_top", "eyelid_bottom", "eye_nasal_corner", "eye_temporal_corner"]
NAME = [k.replace("eye_", "").replace("_", " ") for k in KP]
COL = ["#E63946", "#F4A261", "#E9C46A", "#B5838D", "#2A9D8F", "#1D3557", "#457B9D", "#8D99AE"]
BINS = [0, 0.2, 0.4, 0.6, 0.8, 1.0000001]
OUT = HERE / "results" / "keypoint_confidence_error"
OUT.mkdir(exist_ok=True)
PL = pd.read_csv(HERE / "results" / "labels_to_plateau.csv")
PL = PL[PL.status == "final"]


def stats(e, c):
    ok = np.isfinite(e)
    e, c = e[ok], c[ok]
    rho = spearmanr(c, e)[0] if len(e) > 2 and np.ptp(c) > 0 and np.ptp(e) > 0 else np.nan
    return {"n": len(e), "median_error_px": round(float(np.median(e)), 2), "p90_error_px": round(float(np.percentile(e, 90)), 2),
            "mean_confidence": round(float(c.mean()), 3), "share_conf_ge_0.6": round(float((c >= 0.6).mean()), 3),
            "spearman_conf_error": round(float(rho), 2)}


def heat(ax, M, cols, title, cmap, fmt, vmin, vmax):
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(range(8)); ax.set_yticklabels(NAME, fontsize=10)
    mid = (vmin + vmax) / 2
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                dark = (M[i, j] > mid)
                ax.text(j, i, fmt % M[i, j], ha="center", va="center", fontsize=8.5, color="w" if dark else "k")
    ax.set_title(title, fontsize=11)
    return im


rows, pooled = [], []
for r in PL.itertuples():
    ev = HERE / r.unit / "training-data" / "eval"
    steps = sorted((int(m.group(1)), int(m.group(2)), d) for d in ev.iterdir() if (m := re.fullmatch(r"x(\d{3})_step(\d{2})_final", d.name)))
    pstep = max(1, int(r.labels_at_plateau) // 20)
    E, C, labs, plate = [], [], [], None
    for labels, step, d in steps:
        t = pd.read_csv(d / "per_frame_errors.csv", index_col=0)
        E.append([]); C.append([]); labs.append(labels)
        for k in KP:
            e, c = t[f"{k}_error_px"].to_numpy(float), t[f"{k}_likelihood"].to_numpy(float)
            s = stats(e, c)
            rows.append({"video": r.video, "unit": r.unit, "labels_this_video": labels, "step": step, "plateau_model": step == pstep, "keypoint": k, **s})
            E[-1].append(s["median_error_px"]); C[-1].append(s["mean_confidence"])
        if step == pstep:
            plate = t
            for k in KP:
                pooled.append(pd.DataFrame({"video": r.video, "keypoint": k, "error": t[f"{k}_error_px"].to_numpy(float), "conf": t[f"{k}_likelihood"].to_numpy(float)}))
    E, C = np.array(E).T, np.array(C).T
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.6), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1, 1.25]})
    cols = [f"{l}" + ("\n(plateau)" if l // 20 == pstep else "") for l in labs]
    fig.colorbar(heat(ax[0], E, cols, "Median error of each keypoint (px)", "YlOrRd", "%.1f", 0, max(20, np.nanmax(E))), ax=ax[0], shrink=0.8)
    fig.colorbar(heat(ax[1], C, cols, "Mean confidence of each keypoint", "YlGnBu", "%.2f", 0, 1), ax=ax[1], shrink=0.8)
    for a in ax[:2]:
        a.set_xlabel("labels of this video in the training set")
    for k, n, c in zip(KP, NAME, COL):
        ax[2].scatter(plate[f"{k}_likelihood"], plate[f"{k}_error_px"].clip(lower=0.5), s=16, color=c, alpha=0.75, label=n, lw=0)
    ax[2].axvline(0.6, color="0.4", ls="--", lw=0.8); ax[2].set_yscale("log"); ax[2].set_xlim(-0.02, 1.02)
    ax[2].set_xlabel("confidence"); ax[2].set_ylabel("error (px, log scale)"); ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8.5, ncol=2)
    ax[2].set_title(f"Every test point of the plateau-point model ({pstep * 20} labels); dashed = 0.6", fontsize=11)
    fig.suptitle(f"Video {r.video} ({r.unit}): error and confidence by keypoint on its 50 test frames", fontsize=12.5)
    fig.savefig(OUT / f"keypoint_confidence_error_video{int(r.video):02d}.png", dpi=120)
    plt.close(fig)

P = pd.concat(pooled, ignore_index=True)
P = P[np.isfinite(P.error)]
for k in KP:
    g = P[P.keypoint == k]
    rows.append({"video": "all", "unit": "all videos pooled (plateau-point models)", "labels_this_video": "", "step": "", "plateau_model": True, "keypoint": k,
                 **stats(g.error.to_numpy(), g.conf.to_numpy())})
T = pd.DataFrame(rows)
T.to_csv(OUT / "keypoint_confidence_error_all_videos.csv", index=False)
P["bin"] = pd.cut(P.conf, BINS, right=False, labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1"])
Bn = P.groupby(["keypoint", "bin"], observed=False).error.agg(n="size", median_error_px="median", share_error_gt_20px=lambda x: (x > 20).mean()).round(3).reset_index()
Bn.to_csv(OUT / "keypoint_confidence_error_bins_all_videos.csv", index=False)

pm = T[T.plateau_model.astype(bool)]
vids = list(PL.video) + ["all"]
E = np.array([[pm[(pm.video.astype(str) == str(v)) & (pm.keypoint == k)].median_error_px.iloc[0] for v in vids] for k in KP])
C = np.array([[pm[(pm.video.astype(str) == str(v)) & (pm.keypoint == k)].mean_confidence.iloc[0] for v in vids] for k in KP])
fig, ax = plt.subplots(1, 3, figsize=(24, 5.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1.25, 1]})
cols = [str(v) for v in PL.video] + ["all"]
fig.colorbar(heat(ax[0], E, cols, "Median error of each keypoint (px)", "YlOrRd", "%.1f", 0, 30), ax=ax[0], shrink=0.8)
fig.colorbar(heat(ax[1], C, cols, "Mean confidence of each keypoint", "YlGnBu", "%.2f", 0, 1), ax=ax[1], shrink=0.8)
for a in ax[:2]:
    a.set_xlabel("video (model at its plateau point); all = pooled")
for k, n, c in zip(KP, NAME, COL):
    g = Bn[(Bn.keypoint == k) & (Bn.n >= 5)]
    ax[2].plot(g["bin"].cat.codes, g.median_error_px, "o-", color=c, label=n, lw=1.6, ms=5)
ax[2].set_xticks(range(5)); ax[2].set_xticklabels(["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1"])
ax[2].set_yscale("log"); ax[2].set_xlabel("confidence of the point"); ax[2].set_ylabel("median error of the points in the bin (px, log scale)")
ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8.5, ncol=2); ax[2].set_title("Error against confidence, all videos pooled (bins with >= 5 points)", fontsize=11)
fig.suptitle(f"All {len(PL)} videos: error and confidence by keypoint on the test frames ({len(P)} points)", fontsize=12.5)
fig.savefig(OUT / "keypoint_confidence_error_all_videos.png", dpi=120)
pd.set_option("display.width", 220)
print(T[T.video.astype(str) == "all"].drop(columns=["unit", "labels_this_video", "step", "plateau_model"]).to_string(index=False))
print(Bn.pivot(index="keypoint", columns="bin", values="median_error_px").loc[KP].to_string())
print(Bn.pivot(index="keypoint", columns="bin", values="n").loc[KP].to_string())
