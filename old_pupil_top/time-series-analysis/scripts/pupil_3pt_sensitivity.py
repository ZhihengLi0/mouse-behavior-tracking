#!/usr/bin/env python3
"""Sensitivity test (advisor request, 2026-09-18): pupil size and location
from THREE keypoints (left, right, bottom) versus all FOUR.

pupil_top is often hidden under the upper eyelid and is the least reliable
pupil point. The pupil is modeled as an axis-aligned ELLIPSE (advisor,
2026-09-12 meeting). Estimators:
  4pt          ellipse from all four points: area = pi/4 * |R-L| * |B-T|,
               center y = (T.y + B.y)/2. Measures the VISIBLE pupil.
  3pt-endpoint left/right are the ENDPOINTS of the horizontal axis, so the
               center height is their mean y; the bottom point gives the
               half-height: h = 2 * (B.y - cy). No constant, nothing to
               transfer between mice. The hidden upper half is reconstructed
               by symmetry, so this measures the WHOLE pupil.  <- adopted
  3pt-circle   circle through L, R, B (reference: a perfectly round pupil).
  width        |R-L| alone, the simplest size measure.
An earlier variant fixed the aspect ratio at the video's median (k = 0.68);
it was dropped 2026-09-19: the constant is mouse/camera specific and is itself
learned from the occluded top point (see results/README.md).

  A. 4-minute clip (production model, 14,400 frames): offsets between
     estimators, frame-to-frame noise, coupling to eye opening, blink behavior.
  B. 100 human-labeled test frames: each estimator computed from the model's
     points against the SAME estimator computed from the human's points.
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


def endpoint3(L, R, B):
    """Left/right are the ENDPOINTS of the horizontal axis, so the center height is their mean y;
    the bottom point gives the half-height. No fixed ratio."""
    w = np.abs(R[:, 0] - L[:, 0]); cy = (L[:, 1] + R[:, 1]) / 2; h = 2 * (B[:, 1] - cy)
    return np.c_[(L[:, 0] + R[:, 0]) / 2, cy], np.pi / 4 * w * h, h / w


# ---------------- A. the 4-minute clip ------------------------------------
clip = flat(CLIP_H5)
L, R, T, B = pts(clip)
PUP = ["pupil_left", "pupil_right", "pupil_top", "pupil_bottom"]
lk = {b: clip[b]["likelihood"].to_numpy(float) for b in PUP + ["eyelid_top", "eyelid_bottom"]}
c4, a4, w, h = four(L, R, T, B)
cc, ac = circle3(L, R, B)
ce, ae, ke = endpoint3(L, R, B)
good_top = lk["pupil_top"] >= PCUT
lrb_ok = (lk["pupil_left"] >= PCUT) & (lk["pupil_right"] >= PCUT) & (lk["pupil_bottom"] >= PCUT)
allok = good_top & lrb_ok
t = np.arange(len(clip)) / 60.0
roll = lambda v, n: pd.Series(v).rolling(n, center=True, min_periods=1).median().to_numpy()

opening = (clip["eyelid_bottom"]["y"] - clip["eyelid_top"]["y"]).to_numpy(float)
rel_open = opening / roll(opening, 300)                      # vs 5-s baseline, as in eye_pipeline.py
lids_ok = (lk["eyelid_top"] >= PCUT) & (lk["eyelid_bottom"] >= PCUT)
state = {"open (>= 95%)": (rel_open >= 0.95), "partial blink (70-85%)": (rel_open >= 0.70) & (rel_open < 0.85),
         "deep blink (< 70%)": rel_open < 0.70}

EST = {"4-pt ellipse (visible)": (a4, c4[:, 1]), "3-pt endpoint ellipse": (ae, ce[:, 1]),
       "3-pt circle": (ac, cc[:, 1]), "width only": (w, None)}
print(f"clip: {len(clip)} frames; all four pupil points confident in {allok.sum()}")
print(f"labeled-point geometry: top point {np.nanmedian((ce[:,1]-T[:,1])[allok]):.0f} px above the L-R midline, bottom "
      f"{np.nanmedian((B[:,1]-ce[:,1])[allok]):.0f} px below; L is {np.nanmedian((R[:,1]-L[:,1])[allok]):.0f} px higher than R")
print(f"height/width: 4pt {np.nanmedian((h/w)[allok]):.3f} | endpoint {np.nanmedian(ke[allok]):.3f} | circle 1.000")
blink = {}
for name, (size, cy) in EST.items():
    m = allok & state["open (>= 95%)"]
    jit = np.nanmedian(np.abs(size - roll(size, 15))[m] / size[m]) * 100
    r_open = np.corrcoef(size[lrb_ok & lids_ok], opening[lrb_ok & lids_ok])[0, 1]
    line = f"  {name:24s} size jitter {jit:4.2f}% | corr(size, eye opening) {r_open:+.2f}"
    if cy is not None:
        line += f" | center-y jitter {np.nanmedian(np.abs(cy - roll(cy, 15))[m]):.2f} px"
    base_s, base_y = roll(size, 300), (roll(cy, 300) if cy is not None else None)
    blink[name] = {}
    for st, ms in state.items():
        mm = ms & lrb_ok          # eyelid confidence is not gated: it drops precisely during blinks
        ds = np.nanmedian((size / base_s - 1)[mm]) * 100
        dy = np.nanmedian((cy - base_y)[mm]) if cy is not None else np.nan
        blink[name][st] = (ds, dy, int(mm.sum()))
    line += " | blink size change partial/deep " + "/".join(f"{blink[name][s_][0]:+.0f}%" for s_ in list(state)[1:])
    if cy is not None:
        line += " | center-y shift partial/deep " + "/".join(f"{blink[name][s_][1]:+.0f}px" for s_ in list(state)[1:])
    print(line)
print("  frames per state:", {s_: blink["width only"][s_][2] for s_ in state})
da = (ae[allok] - a4[allok]) / a4[allok] * 100
print(f"  endpoint vs 4pt: area {np.nanmedian(da):+.1f}% , center y {np.nanmedian((ce[:,1]-c4[:,1])[allok]):+.1f} px (negative = higher in the image)")

# ---------------- B. against human labels ---------------------------------
gt, pr = flat(GT_H5), flat(PRED_TEST)
pr = pr.loc[gt.index]
gL, gR, gT, gB = pts(gt)
mL, mR, mT, mB = pts(pr)
h4c, h4a, hw, hh = four(gL, gR, gT, gB); m4c, m4a, mw, _ = four(mL, mR, mT, mB)
hec, hea, hke = endpoint3(gL, gR, gB); mec, mea, _ = endpoint3(mL, mR, mB)
hcc, hca = circle3(gL, gR, gB); mcc, mca = circle3(mL, mR, mB)
print(f"\ntest set: {len(gt)} human-labeled frames; each estimator from MODEL points vs the same estimator from HUMAN points")
print(f"  human labels: top point {np.nanmedian(hec[:,1]-gT[:,1]):.0f} px above the L-R midline, bottom {np.nanmedian(gB[:,1]-hec[:,1]):.0f} px below; "
      f"height/width 4pt {np.nanmedian(hh/hw):.3f}, endpoint {np.nanmedian(hke):.3f}")
rows = []
for name, mc_, ma_, hc_, ha_ in [("4-pt ellipse", m4c, m4a, h4c, h4a), ("3-pt endpoint\nellipse", mec, mea, hec, hea),
                                 ("3-pt circle", mcc, mca, hcc, hca)]:
    ea = np.abs(ma_ - ha_) / ha_ * 100
    ec = np.hypot(*(mc_ - hc_).T)
    rows.append((name, np.nanmedian(ea), np.nanpercentile(ea, 90), np.nanmedian(ec), np.nanpercentile(ec, 90)))
    print(f"  {name.replace(chr(10), ' '):22s}: area error median {rows[-1][1]:5.1f}% (p90 {rows[-1][2]:5.1f}%) | "
          f"center error median {rows[-1][3]:4.1f} px (p90 {rows[-1][4]:5.1f} px)")
ew = np.abs(mw - hw) / hw * 100
print(f"  width only            : error median {np.nanmedian(ew):5.1f}% (p90 {np.nanpercentile(ew, 90):5.1f}%)")

# ---------------- figure ---------------------------------------------------
BLUE, RED, GREEN, PURPLE = "#2F6B9A", "#D1495B", "#2A9D8F", "#7A5195"
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.1], hspace=0.38, wspace=0.25)
sm = lambda v: roll(v, 15)
ax = fig.add_subplot(gs[0, :])
ax.plot(t, sm(a4), lw=0.8, color=BLUE, label="4-point ellipse (visible pupil)")
ax.plot(t, sm(ae), lw=0.8, color=RED, alpha=0.85, label="3-point endpoint ellipse (L,R = axis endpoints; whole pupil)")
ax.plot(t, sm(ac), lw=0.8, color=GREEN, alpha=0.6, label="circle through L,R,B (reference)")
ax.fill_between(t, 0, 1, where=~good_top, transform=ax.get_xaxis_transform(), color="gray", alpha=0.18,
                label="pupil_top confidence < 0.6")
ax.set_ylabel("pupil area (px$^2$)"); ax.legend(fontsize=9, ncol=4, loc="upper right")
ax.set_title("Pupil area over the 4-minute clip: four keypoints vs three (left, right, bottom), no fixed ratio", fontsize=12)
m4, me, mc = np.nanmedian(a4), np.nanmedian(ae), np.nanmedian(ac)
ax.text(0.005, 0.04, f"median area:  4-pt {m4:,.0f} px$^2$   |   3-pt endpoint {me:,.0f} px$^2$ ({(me/m4-1)*100:+.0f}%)"
        f"   |   circle {mc:,.0f} px$^2$ ({(mc/m4-1)*100:+.0f}%)", transform=ax.transAxes, fontsize=10.5,
        bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.9))
ax = fig.add_subplot(gs[1, :], sharex=ax)
ax.plot(t, sm(c4[:, 1]), lw=0.8, color=BLUE, label="4-point: midpoint of top and bottom")
ax.plot(t, sm(ce[:, 1]), lw=0.8, color=RED, alpha=0.85, label="3-point endpoint: mean y of left and right")
ax.plot(t, sm(cc[:, 1]), lw=0.8, color=GREEN, alpha=0.6, label="circle center (reference)")
ax.fill_between(t, 0, 1, where=~good_top, transform=ax.get_xaxis_transform(), color="gray", alpha=0.18)
ax.set_ylabel("pupil center y (px, larger = lower in image)"); ax.set_xlabel("time (s)"); ax.legend(fontsize=9, ncol=3, loc="upper right")
ax.set_title("Pupil vertical position (center x of both ellipses is the same left-right midpoint)", fontsize=12)
y4, ye, yc = np.nanmedian(c4[:, 1]), np.nanmedian(ce[:, 1]), np.nanmedian(cc[:, 1])
ax.text(0.005, 0.04, f"median center y:  4-pt {y4:.0f} px   |   3-pt endpoint {ye:.0f} px ({ye-y4:+.0f} px = higher)"
        f"   |   circle {yc:.0f} px ({yc-y4:+.0f} px)", transform=ax.transAxes, fontsize=10.5,
        bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.9))

ax = fig.add_subplot(gs[2, 0])
sts = list(state)[1:]
names_b = list(EST)
xb = np.arange(len(names_b))
for i_, (st, col) in enumerate(zip(sts, ["#f0a35e", "#b5462f"])):
    bars = ax.bar(xb + (i_ - 0.5) * 0.38, [blink[n_][st][0] for n_ in names_b], 0.38, color=col,
                  label=f"{st}, n={blink[names_b[0]][st][2]} frames")
    ax.bar_label(bars, fmt="%+.0f%%", fontsize=8.5)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(xb); ax.set_xticklabels([n_.replace(" (visible)", "").replace(" ellipse", "\nellipse") for n_ in names_b], fontsize=8.5)
ax.set_ylabel("size change vs 5-s baseline (%)"); ax.set_ylim(-68, 0); ax.legend(fontsize=8, loc="lower center", ncol=1)
ax.set_title("Apparent pupil size change during blinks (0 = unaffected)", fontsize=11)

ax = fig.add_subplot(gs[2, 1])
names = [r[0] for r in rows]
x = np.arange(len(rows))
b1 = ax.bar(x - 0.2, [r[1] for r in rows], 0.4, color=BLUE, label="median (typical frame)")
b2 = ax.bar(x + 0.2, [r[2] for r in rows], 0.4, color="#9ec3dd", label="90th pct (worst 10% of frames start here)")
ax.bar_label(b1, fmt="%.1f%%", fontsize=9); ax.bar_label(b2, fmt="%.1f%%", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9); ax.set_ylabel("area error (%)"); ax.legend(fontsize=7.5, loc="upper left")
ax.set_ylim(0, max(r[2] for r in rows) * 1.45)
ax.set_title("AREA: model points vs human points, same estimator (100 test frames)", fontsize=10.5)

ax = fig.add_subplot(gs[2, 2])
b1 = ax.bar(x - 0.2, [r[3] for r in rows], 0.4, color=BLUE, label="median (typical frame)")
b2 = ax.bar(x + 0.2, [r[4] for r in rows], 0.4, color="#9ec3dd", label="90th pct (worst 10% of frames start here)")
ax.bar_label(b1, fmt="%.1f px", fontsize=9); ax.bar_label(b2, fmt="%.1f px", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9); ax.set_ylabel("center error (px)"); ax.legend(fontsize=7.5, loc="upper left")
ax.set_ylim(0, max(r[4] for r in rows) * 1.45)
ax.set_title("CENTER: model points vs human points, same estimator", fontsize=10.5)
fig.savefig(OUT / "06_pupil_3pt_vs_4pt.png", dpi=125, bbox_inches="tight")
print("\nsaved", OUT / "06_pupil_3pt_vs_4pt.png")
