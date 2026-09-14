#!/usr/bin/env python3
"""Keypoint-vs-time analysis on the first-4-minute clip (lab meeting
2026-09-12, task 2): plot trajectories, derive eye opening and pupil area,
flag frames whose residual from a rolling-median fit is large, and dump the
flagged intervals with representative frame snapshots.

Uses an existing analyze_videos output (no GPU work): the jump round-10 model
(shuffle 50), the current strongest generation.
Derived statistics:
  eye_opening = |eyelid_top - eyelid_bottom|            (px)
  pupil_area  = pi/4 * |p_left-p_right| * |p_top-p_bottom|   (px^2, ellipse)
Residual rule (pre-declared): |value - rolling_median(1 s)| > 5 * MAD of the
residual trace -> anomalous frame; consecutive anomalous frames merge into
events with a 0.25 s gap tolerance.
"""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
UNIT = ROOT / "time-series-analysis"
PRED = (ROOT / "active-learning/training-data/"
        "face_first4minDLC_Resnet50_EyePupilBlinkAug17shuffle50_snapshot_best-50.h5")
CLIP = ROOT / "active-learning/training-data/face_first4min.mp4"
FPS = 60.0
PCUT = 0.6
WIN = 61          # rolling window, ~1 s
MAD_K = 5.0       # anomaly threshold in MADs
GAP_S = 0.25      # merge tolerance

df = pd.read_hdf(PRED)
df.columns = df.columns.droplevel(0)
while df.columns.nlevels > 2:
    df.columns = df.columns.droplevel(0)
t = np.arange(len(df)) / FPS


def xy(bp):
    return df[bp]["x"].to_numpy(), df[bp]["y"].to_numpy()


def lk(bp):
    return df[bp]["likelihood"].to_numpy()


plx, _ = xy("pupil_left"); prx, _ = xy("pupil_right")
_, pty = xy("pupil_top"); _, pby = xy("pupil_bottom")
_, ety = xy("eyelid_top"); _, eby = xy("eyelid_bottom")
pcx = (plx + prx) / 2
pcy = (pty + pby) / 2
eye_opening = np.abs(eby - ety)
pupil_area = np.pi / 4 * np.abs(prx - plx) * np.abs(pby - pty)
pupil_lk = np.min([lk(b) for b in
                   ["pupil_left", "pupil_right", "pupil_top", "pupil_bottom"]], axis=0)

traces = {"pupil_center_x": pcx, "pupil_center_y": pcy,
          "eye_opening": eye_opening, "pupil_area": pupil_area}


def anomalies(v):
    s = pd.Series(v)
    med = s.rolling(WIN, center=True, min_periods=1).median()
    resid = (s - med).to_numpy()
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid))) or 1.0
    return np.abs(resid) > MAD_K * 1.4826 * mad, med.to_numpy(), resid


flag_union = np.zeros(len(df), bool)
fits = {}
for name, v in traces.items():
    fl, med, _ = anomalies(v)
    fits[name] = (fl, med)
    flag_union |= fl

low_conf = pupil_lk < PCUT


def to_events(mask):
    ev, start = [], None
    gap = int(GAP_S * FPS)
    last = -10 * gap
    for i in np.where(mask)[0]:
        if i - last > gap:
            if start is not None:
                ev.append((start, last))
            start = i
        last = i
    if start is not None:
        ev.append((start, last))
    return ev


events = to_events(flag_union | low_conf)

# ---- figure 1: the four traces with anomaly marks -------------------------
fig, axes = plt.subplots(5, 1, figsize=(16, 13), sharex=True)
for ax, (name, v) in zip(axes, traces.items()):
    fl, med = fits[name]
    ax.plot(t, v, lw=0.5, color="#2F6B9A", label="predicted")
    ax.plot(t, med, lw=1.0, color="black", alpha=0.6, label="rolling median (1 s)")
    ax.plot(t[fl], v[fl], ".", color="red", ms=3, label="residual anomaly")
    ax.set_ylabel(name)
    ax.legend(loc="upper right", fontsize=8)
axes[4].plot(t, pupil_lk, lw=0.5, color="#2A9D8F")
axes[4].axhline(PCUT, color="red", ls="--", lw=1)
axes[4].fill_between(t, 0, 1, where=low_conf, color="red", alpha=0.15,
                     label=f"min pupil likelihood < {PCUT}")
axes[4].set_ylabel("min pupil likelihood")
axes[4].set_xlabel("time (s)")
axes[4].legend(loc="upper right", fontsize=8)
fig.suptitle("Keypoint time series, first 4 minutes (jump r10 model, shuffle 50) - "
             f"{len(events)} flagged events", fontsize=14)
fig.tight_layout()
fig.savefig(UNIT / "results/01_keypoint_timeseries.png", dpi=130)
plt.close(fig)

# ---- event table ----------------------------------------------------------
rows = []
for s, e in events:
    seg = slice(s, e + 1)
    rows.append({
        "start_s": round(s / FPS, 2), "end_s": round(e / FPS, 2),
        "n_frames": e - s + 1,
        "min_pupil_likelihood": round(float(np.min(pupil_lk[seg])), 3),
        "min_eye_opening_px": round(float(np.min(eye_opening[seg])), 1),
        "max_pupil_area_px2": round(float(np.max(pupil_area[seg])), 0),
        "low_confidence": bool(low_conf[seg].any()),
        "likely_blink": bool(low_conf[seg].any()
                             and np.min(eye_opening[seg]) < np.nanpercentile(eye_opening, 5)),
    })
ev_df = pd.DataFrame(rows)
ev_df.to_csv(UNIT / "results/flagged_events.csv", index=False)

# ---- figure 2: snapshots of the longest events ----------------------------
top = ev_df.nlargest(8, "n_frames")
cap = cv2.VideoCapture(str(CLIP))
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, (_, r) in zip(axes.ravel(), top.iterrows()):
    mid = int((r.start_s + r.end_s) / 2 * FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, im = cap.read()
    if ok:
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        for bp in df.columns.get_level_values(0).unique():
            x, y = df[bp]["x"].iloc[mid], df[bp]["y"].iloc[mid]
            c = "red" if df[bp]["likelihood"].iloc[mid] < PCUT else "lime"
            ax.plot(x, y, "o", color=c, ms=5)
    tag = "BLINK?" if r.likely_blink else ("low-conf" if r.low_confidence else "residual")
    ax.set_title(f"{r.start_s:.1f}-{r.end_s:.1f}s  {tag}", fontsize=10)
    ax.axis("off")
cap.release()
fig.suptitle("Longest flagged events, middle frame "
             "(green = confident point, red = likelihood < 0.6)", fontsize=13)
fig.tight_layout()
fig.savefig(UNIT / "results/02_flagged_event_frames.png", dpi=130)

n_blink = int(ev_df["likely_blink"].sum())
print(f"events: {len(ev_df)}  likely blinks: {n_blink}  "
      f"flagged frames: {int((flag_union | low_conf).sum())}/{len(df)}")
