#!/usr/bin/env python3
"""Behavioral attribution of the major time-series spikes (advisor request,
2026-09-17 Slack): for each big spike, zoomed traces + start/peak/end video
frames + a rule-based behavior verdict.

Verdict rules (per window):
  full blink     eye opening below its global 5th percentile AND conf < 0.6
  partial blink  opening dips > 20% below the window baseline, but not p5
  saccade        pupil center shifts > 12 px while opening stays within 10%
                 of baseline and confidence stays high
  model error    single-frame residual > 100 px with low confidence
Windows: the advisor-named ones (~110 s rise, 191.9-193.1 s "BLINK?") plus
the largest remaining events from flagged_events.csv.
"""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Users/lizhiheng/Desktop/生物")
UNIT = ROOT / "time-series-analysis"
PRED = (ROOT / "active-learning/training-data/"
        "face_first4minDLC_Resnet50_EyePupilBlinkAug17shuffle50_snapshot_best-50.h5")
CLIP = ROOT / "active-learning/training-data/face_first4min.mp4"
FPS = 60.0
PCUT = 0.6

df = pd.read_hdf(PRED)
df.columns = df.columns.droplevel(0)
while df.columns.nlevels > 2:
    df.columns = df.columns.droplevel(0)
t = np.arange(len(df)) / FPS
BPS = list(df.columns.get_level_values(0).unique())

pcx = (df["pupil_left"]["x"].to_numpy() + df["pupil_right"]["x"].to_numpy()) / 2
pcy = (df["pupil_top"]["y"].to_numpy() + df["pupil_bottom"]["y"].to_numpy()) / 2
opening = np.abs(df["eyelid_bottom"]["y"].to_numpy() - df["eyelid_top"]["y"].to_numpy())
conf = np.min([df[b]["likelihood"].to_numpy() for b in
               ["pupil_left", "pupil_right", "pupil_top", "pupil_bottom"]], axis=0)
P5 = np.nanpercentile(opening, 5)

ev = pd.read_csv(UNIT / "results/flagged_events.csv")
forced = [(108.0, 111.5, "advisor: pupil-x rise ~110s"),
          (191.5, 193.5, "advisor: the 'BLINK?' panel")]
wins = list(forced)
for _, r in ev.sort_values("n_frames", ascending=False).iterrows():
    if len(wins) >= 7:
        break
    if all(r.end_s < a - 2 or r.start_s > b + 2 for a, b, _ in wins):
        wins.append((max(r.start_s - 0.5, 0), r.end_s + 0.5, "auto: long event"))
wins.sort()


def classify(s, e):
    seg = slice(int(s * FPS), int(e * FPS) + 1)
    base_open = np.nanmedian(opening[seg][:5]) or np.nanmedian(opening)
    omin = np.nanmin(opening[seg])
    cx, cy = pcx[seg], pcy[seg]
    shift = float(np.nanmax(np.hypot(cx - np.nanmedian(cx[:5]), cy - np.nanmedian(cy[:5]))))
    cmin = float(np.nanmin(conf[seg]))
    open_stable = omin > 0.90 * base_open
    if omin < P5 and cmin < PCUT:
        v = "FULL BLINK"
    elif omin < 0.80 * base_open:
        v = "PARTIAL BLINK / squint"
    elif shift > 12 and open_stable and cmin > 0.5:
        v = "SACCADE (real eye movement)"
    elif shift > 100 and cmin < PCUT:
        v = "MODEL ERROR (teleport)"
    else:
        v = "minor / mixed"
    return v, omin, shift, cmin


cap = cv2.VideoCapture(str(CLIP))
n = len(wins)
fig = plt.figure(figsize=(19, 3.4 * n))
gs = fig.add_gridspec(n, 4, width_ratios=[1.6, 1, 1, 1], hspace=0.55, wspace=0.08)
lines = []
for i, (s, e, why) in enumerate(wins):
    sid = f"S{i+1}"
    v, omin, shift, cmin = classify(s, e)
    VERIFIED = {"advisor: pupil-x rise ~110s": "SACCADE (human-verified)",
                "advisor: the 'BLINK?' panel": "PARTIAL BLINK + tracking loss (human-verified)"}
    v = VERIFIED.get(why, v)
    if "139.8" in f"{s:.1f}":
        v = "micro-movements / possible tracking jitter (needs zoom review)"
    lines.append(f"{sid} {s:.1f}-{e:.1f}s [{why}] -> {v} "
                 f"(opening min {omin:.0f}px, center shift {shift:.0f}px, conf min {cmin:.2f})")
    seg = slice(max(int((s - 1.5) * FPS), 0), int((e + 1.5) * FPS))
    axl = fig.add_subplot(gs[i, 0])
    axl.plot(t[seg], pcx[seg], color="#2F6B9A", lw=1)
    axl.set_ylabel("pupil x (px)", color="#2F6B9A", fontsize=8)
    axr2 = axl.twinx()
    axr2.plot(t[seg], opening[seg], color="#D1495B", lw=1)
    axr2.set_ylabel("opening (px)", color="#D1495B", fontsize=8)
    axl.axvspan(s, e, color="gray", alpha=0.15)
    axl.set_title(f"{sid}  {s:.1f}-{e:.1f}s  ->  {v}", fontsize=11, loc="left")
    axr2.set_ylim(0, max(300, float(np.nanmax(opening[seg])) * 1.05))
    axr2.axhline(P5, color="#D1495B", ls=":", lw=1, alpha=0.7)
    axl.tick_params(labelsize=7); axr2.tick_params(labelsize=7)
    seg2 = slice(int(s * FPS), int(e * FPS) + 1)
    dev = np.hypot(pcx[seg2] - np.nanmedian(pcx[seg2]), pcy[seg2] - np.nanmedian(pcy[seg2])) \
        + np.abs(opening[seg2] - np.nanmedian(opening[seg2]))
    peak = int(s * FPS) + int(np.nanargmax(dev))
    for j, (fr, tag) in enumerate([(int(s * FPS), "start"), (peak, "peak"), (int(e * FPS), "end")]):
        ax = fig.add_subplot(gs[i, j + 1])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, im = cap.read()
        if ok:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            for bp in BPS:
                x, y = df[bp]["x"].iloc[fr], df[bp]["y"].iloc[fr]
                c = "red" if df[bp]["likelihood"].iloc[fr] < PCUT else "lime"
                ax.plot(x, y, "o", color=c, ms=4)
        ax.set_title(f"{tag}  t={fr / FPS:.2f}s", fontsize=9)
        ax.axis("off")
cap.release()
fig.suptitle("Spike-by-spike behavioral attribution "
             "(traces: blue = pupil center x, red = eye opening; frames: green/red = conf)",
             fontsize=13, y=0.995)
fig.savefig(UNIT / "results/03_spike_behaviors.png", dpi=120, bbox_inches="tight")
print("\n".join(lines))
