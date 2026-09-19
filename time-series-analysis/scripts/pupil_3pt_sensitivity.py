#!/usr/bin/env python3
"""Sensitivity test (advisor request, 2026-09-18): pupil size and location
from THREE keypoints (left, right, bottom) versus all FOUR.

pupil_top is often hidden under the upper eyelid and is the least reliable
pupil point. Three estimators are compared:
  4pt        center = (mid(L,R).x, mid(T,B).y); area = pi/4 * |R-L| * |B-T|
  3pt-circle circle through L, R, B (circumcircle): center, area = pi r^2
  3pt-ratio  width from L,R; height = k * width, k = the video's own median
             height/width ratio measured on frames where pupil_top is
             confident; center y = B.y - height/2
Two questions:
  A. On the old video's 4-minute clip (production model, 14,400 frames): how
     far apart are the estimators, overall and when pupil_top is unreliable?
  B. On the 100 human-labeled test frames: which model-derived estimate is
     closer to the HUMAN 4-point pupil?
"""
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "time-series-analysis" / "results"
CLIP_H5 = sorted(glob.glob(str(ROOT / "active-learning/training-data/face_first4minDLC*shuffle60*.h5")))[-1]
TEST = ROOT / "local_data/test_sets/eye_last_minute_100"
GT_H5 = TEST / "dlc_label_project/labeled-data/eye_last_minute_100/CollectedData_Zhiheng.h5"
PRED_TEST = sorted(glob.glob(str(TEST / "predictions_100train_al_production_v1/image_predictions_*.h5")))[-1]
PCUT = 0.6


def flat(p):
    d = pd.read_hdf(p)
    d.columns = d.columns.droplevel(0)
    while d.columns.nlevels > 2:
        d.columns = d.columns.droplevel(0)
    d.index = [i[-1] if isinstance(i, tuple) else Path(str(i)).name for i in d.index]
    return d


def pts(d):
    g = lambda b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)]
    return g("pupil_left"), g("pupil_right"), g("pupil_top"), g("pupil_bottom")


def four(L, R, T, B):
    w = np.abs(R[:, 0] - L[:, 0]); h = np.abs(B[:, 1] - T[:, 1])
    return np.c_[(L[:, 0] + R[:, 0]) / 2, (T[:, 1] + B[:, 1]) / 2], np.pi / 4 * w * h, w, h


def circle3(L, R, B):
    ax, ay, bx, by, cx, cy = L[:, 0], L[:, 1], R[:, 0], R[:, 1], B[:, 0], B[:, 1]
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    d = np.where(np.abs(d) < 1e-6, np.nan, d)
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    r = np.hypot(ax - ux, ay - uy)
    return np.c_[ux, uy], np.pi * r**2


def ratio3(L, R, B, k):
    w = np.abs(R[:, 0] - L[:, 0]); h = k * w
    return np.c_[(L[:, 0] + R[:, 0]) / 2, B[:, 1] - h / 2], np.pi / 4 * w * h


# ---------------- A. the 4-minute clip ------------------------------------
clip = flat(CLIP_H5)
L, R, T, B = pts(clip)
lk = {b: clip[b]["likelihood"].to_numpy(float) for b in ["pupil_left", "pupil_right", "pupil_top", "pupil_bottom"]}
c4, a4, w, h = four(L, R, T, B)
good_top = lk["pupil_top"] >= PCUT
lrb_ok = (lk["pupil_left"] >= PCUT) & (lk["pupil_right"] >= PCUT) & (lk["pupil_bottom"] >= PCUT)
k = float(np.nanmedian((h / w)[good_top & lrb_ok]))
cc, ac = circle3(L, R, B)
cr, ar = ratio3(L, R, B, k)
t = np.arange(len(clip)) / 60.0

print(f"clip: {len(clip)} frames | confident (>= {PCUT}): top {good_top.mean():.0%}, left {np.mean(lk['pupil_left']>=PCUT):.0%}, "
      f"right {np.mean(lk['pupil_right']>=PCUT):.0%}, bottom {np.mean(lk['pupil_bottom']>=PCUT):.0%}")
print(f"median height/width ratio when top is confident: k = {k:.3f}")
for name, c3, a3 in [("3pt-circle", cc, ac), ("3pt-ratio ", cr, ar)]:
    for lab, m in [("top confident ", good_top & lrb_ok), ("top UNreliable", ~good_top & lrb_ok)]:
        da = (a3[m] - a4[m]) / a4[m] * 100
        dc = np.hypot(*(c3[m] - c4[m]).T)
        r_ = np.corrcoef(a3[m], a4[m])[0, 1]
        print(f"  {name} vs 4pt | {lab} (n={m.sum():5d}): area diff median {np.nanmedian(da):+5.1f}% "
              f"(|diff| {np.nanmedian(np.abs(da)):4.1f}%), corr {r_:.3f}; center shift median {np.nanmedian(dc):4.1f} px")

# ---------------- B. against human labels ---------------------------------
gt, pr = flat(GT_H5), flat(PRED_TEST)
pr = pr.loc[gt.index]
gL, gR, gT, gB = pts(gt)
hc, ha, _, _ = four(gL, gR, gT, gB)                 # human 4-point pupil = reference
mL, mR, mT, mB = pts(pr)
m4c, m4a, _, _ = four(mL, mR, mT, mB)
mcc, mca = circle3(mL, mR, mB)
mrc, mra = ratio3(mL, mR, mB, k)
print(f"\ntest set: {len(gt)} human-labeled frames; reference = human 4-point pupil")
rows = []
for name, c, a_ in [("model 4pt", m4c, m4a), ("model 3pt-circle", mcc, mca), ("model 3pt-ratio", mrc, mra)]:
    ea = np.abs(a_ - ha) / ha * 100
    ec = np.hypot(*(c - hc).T)
    rows.append((name, np.nanmedian(ea), np.nanpercentile(ea, 90), np.nanmedian(ec), np.nanpercentile(ec, 90)))
    print(f"  {name:17s}: area error median {rows[-1][1]:5.1f}% (p90 {rows[-1][2]:5.1f}%) | "
          f"center error median {rows[-1][3]:4.1f} px (p90 {rows[-1][4]:5.1f} px)")
hk = np.nanmedian(np.abs(gB[:, 1] - gT[:, 1]) / np.abs(gR[:, 0] - gL[:, 0]))
print(f"  human height/width ratio on the test frames: {hk:.3f} (model-derived k = {k:.3f})")

# ---------------- figure ---------------------------------------------------
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.1], hspace=0.38, wspace=0.25)
sm = lambda v: pd.Series(v).rolling(15, center=True, min_periods=1).median().to_numpy()
ax = fig.add_subplot(gs[0, :])
ax.plot(t, sm(a4), lw=0.8, color="#2F6B9A", label="4 points")
ax.plot(t, sm(ar), lw=0.8, color="#D1495B", alpha=0.85, label=f"3 points (L,R,B) + fixed ratio k={k:.2f}")
ax.plot(t, sm(ac), lw=0.8, color="#2A9D8F", alpha=0.7, label="3 points, circle fit")
ax.fill_between(t, 0, 1, where=~good_top, transform=ax.get_xaxis_transform(), color="gray", alpha=0.18,
                label="pupil_top confidence < 0.6")
ax.set_ylabel("pupil area (px$^2$)"); ax.legend(fontsize=9, ncol=4, loc="upper right")
ax.set_title("Pupil area over the 4-minute clip: four keypoints vs three (left, right, bottom)", fontsize=12)
ax = fig.add_subplot(gs[1, :], sharex=ax)
ax.plot(t, sm(c4[:, 1]), lw=0.8, color="#2F6B9A", label="4 points")
ax.plot(t, sm(cr[:, 1]), lw=0.8, color="#D1495B", alpha=0.85, label="3 points + fixed ratio")
ax.plot(t, sm(cc[:, 1]), lw=0.8, color="#2A9D8F", alpha=0.7, label="3 points, circle fit")
ax.fill_between(t, 0, 1, where=~good_top, transform=ax.get_xaxis_transform(), color="gray", alpha=0.18)
ax.set_ylabel("pupil center y (px)"); ax.set_xlabel("time (s)"); ax.legend(fontsize=9, ncol=3, loc="upper right")
ax.set_title("Pupil vertical position (center x is identical by construction: all use the L-R midpoint)", fontsize=12)

ax = fig.add_subplot(gs[2, 0])
m = good_top & lrb_ok
ax.scatter(a4[m][::5], ar[m][::5], s=3, alpha=0.3, color="#D1495B", label="ratio")
ax.scatter(a4[m][::5], ac[m][::5], s=3, alpha=0.3, color="#2A9D8F", label="circle")
lim = [np.nanpercentile(a4, 1), np.nanpercentile(a4, 99)]
ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("4-point area"); ax.set_ylabel("3-point area"); ax.legend(fontsize=9)
ax.set_title("Agreement when pupil_top is confident", fontsize=11)

ax = fig.add_subplot(gs[2, 1])
names = [r[0].replace("model ", "") for r in rows]
x = np.arange(len(rows))
ax.bar(x - 0.2, [r[1] for r in rows], 0.4, color="#2F6B9A", label="median")
ax.bar(x + 0.2, [r[2] for r in rows], 0.4, color="#9ec3dd", label="90th pct")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9); ax.set_ylabel("area error vs human (%)"); ax.legend(fontsize=9)
ax.set_title("Pupil AREA error against human labels (100 test frames)", fontsize=11)

ax = fig.add_subplot(gs[2, 2])
ax.bar(x - 0.2, [r[3] for r in rows], 0.4, color="#2F6B9A", label="median")
ax.bar(x + 0.2, [r[4] for r in rows], 0.4, color="#9ec3dd", label="90th pct")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9); ax.set_ylabel("center error vs human (px)"); ax.legend(fontsize=9)
ax.set_title("Pupil CENTER error against human labels", fontsize=11)
fig.savefig(OUT / "06_pupil_3pt_vs_4pt.png", dpi=125, bbox_inches="tight")
print("\nsaved", OUT / "06_pupil_3pt_vs_4pt.png")
