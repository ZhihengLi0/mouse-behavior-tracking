#!/usr/bin/env python3
"""Why did every branch's mean error spike at 120 training frames (round 2)?

For each branch: rank test frames by how much the round-2 model got WORSE
than the round-0 baseline on them, overlay human labels vs both models'
predictions on the top culprits, and show how many frames the whole spike
actually lives in (mean error recomputed with top-k culprits removed).
"""
import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Users/lizhiheng/Desktop/生物")
TEST = ROOT / "local_data/test_sets/eye_last_minute_100"
OUT = ROOT / "active-learning/analysis-120frame-spike/results"
OUT.mkdir(parents=True, exist_ok=True)
BPS = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right",
       "eyelid_top", "eyelid_bottom", "eye_nasal_corner", "eye_temporal_corner"]
ERR = [f"{b}_error_px" for b in BPS]
N_SHOW = 3

GT = pd.read_hdf(TEST / "dlc_label_project/labeled-data/eye_last_minute_100/CollectedData_Zhiheng.h5")
GT.columns = GT.columns.droplevel(0)


def load(label):
    d = TEST / f"predictions_100train_{label}"
    pfe = pd.read_csv(d / "eye_test_per_frame_errors.csv", index_col=0)
    pred = pd.read_hdf(glob.glob(str(d / "image_predictions_*.h5"))[0])
    pred.columns = pred.columns.droplevel(0)
    while pred.columns.nlevels > 2:
        pred.columns = pred.columns.droplevel(0)
    return pfe, pred


def row_for(df, name):
    idx = [i for i in df.index
           if os.path.basename(i if isinstance(i, str) else i[-1]) == name]
    return df.loc[idx[0]] if idx else None


pfe0, pred0 = load("al_r0_baseline")
base_frame_mean = pfe0[ERR].mean(axis=1)

summary = {}
for branch in ["uncertain", "jump", "fitting"]:
    pfe2, pred2 = load(f"al_{branch}_r2")
    fm2 = pfe2[ERR].mean(axis=1)
    delta = (fm2 - base_frame_mean).sort_values(ascending=False)
    culprits = delta.head(N_SHOW)

    # ---- overlay: GT vs baseline vs round-2 on the top culprits ----------
    fig, axes = plt.subplots(1, N_SHOW, figsize=(5.6 * N_SHOW, 5.8))
    for ax, (name, dv) in zip(np.atleast_1d(axes), culprits.items()):
        ax.imshow(mpimg.imread(TEST / "frames" / name))
        g, p0, p2 = row_for(GT, name), row_for(pred0, name), row_for(pred2, name)
        r2row = pfe2.loc[name]
        for bp in BPS:
            if g is not None and np.isfinite(g[bp]["x"]):
                ax.plot(g[bp]["x"], g[bp]["y"], "x", color="lime", ms=13, mew=3)
            if p0 is not None and np.isfinite(p0[bp]["x"]):
                ax.plot(p0[bp]["x"], p0[bp]["y"], "s", color="#4FA3E3", ms=7,
                        mfc="none", mew=2)
            if p2 is not None and np.isfinite(p2[bp]["x"]):
                ax.plot(p2[bp]["x"], p2[bp]["y"], "o", color="red", ms=8, alpha=0.8)
                e2 = r2row.get(f"{bp}_error_px", np.nan)
                if np.isfinite(e2) and e2 > 60:
                    ax.annotate(f"{bp}\n{e2:.0f}px", (float(p2[bp]["x"]), float(p2[bp]["y"])),
                                fontsize=9, color="yellow", fontweight="bold")
        fnum = int(name.split("frame")[1].split(".")[0])
        ax.set_title(f"{name}  t={fnum / 60:.1f}s\n"
                     f"baseline {base_frame_mean[name]:.0f}px -> r2 {fm2[name]:.0f}px  (+{dv:.0f})",
                     fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{branch}: top {N_SHOW} frames driving the round-2 spike  "
                 "(green X = human, blue square = round-0 baseline, red dot = round-2)",
                 fontsize=13)
    fig.savefig(OUT / f"02_culprits_{branch}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- how concentrated is the spike? ----------------------------------
    means = [fm2.mean()]
    for k in range(1, 11):
        means.append(fm2.drop(delta.head(k).index).mean())
    summary[branch] = {"delta": delta, "means_k": means, "fm2": fm2}
    print(branch, "spike concentration:", [f"{m:.1f}" for m in means[:6]],
          "| baseline", f"{base_frame_mean.mean():.1f}")

# ---- figure 1: concentration curves --------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.5))
colors = {"uncertain": "#2F6B9A", "jump": "#D1495B", "fitting": "#2A9D8F"}
for b, s in summary.items():
    ax.plot(range(11), s["means_k"], "o-", color=colors[b], label=b)
ax.axhline(base_frame_mean.mean(), color="gray", ls="--",
           label=f"round-0 baseline ({base_frame_mean.mean():.1f}px)")
ax.set(xlabel="worst frames removed (k)", ylabel="mean frame error, remaining 100-k test frames (px)",
       title="The round-2 'regression' lives in a handful of frames")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "01_spike_concentration.png", dpi=150)

# ---- culprit table --------------------------------------------------------
rows = []
for b, s in summary.items():
    for name, dv in s["delta"].head(5).items():
        pfe2 = s["fm2"]
        rows.append({"branch": b, "frame": name,
                     "t_s": round(int(name.split("frame")[1].split(".")[0]) / 60, 2),
                     "baseline_px": round(base_frame_mean[name], 1),
                     "round2_px": round(pfe2[name], 1),
                     "delta_px": round(dv, 1)})
pd.DataFrame(rows).to_csv(OUT / "culprit_frames.csv", index=False)
print("saved to", OUT)
