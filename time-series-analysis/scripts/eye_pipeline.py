#!/usr/bin/env python3
"""One-command eye pipeline for a NEW video (next-phase deliverable).

    python eye_pipeline.py --video /path/to/new_eye_video.mp4 [--skip-analyze]

Steps:
  1. analyze the video with the production model (shuffle 60, ResNet-50
     trained on all 596 reviewed frames) unless predictions already exist;
  2. derive pupil center / eye opening / pupil area / confidence traces;
  3. classifier v2 events:
       BLINK        opening dips below 70% of its local baseline
       PARTIAL      opening dips to 70-85% of baseline
       SACCADE      center shift > 10 px OR velocity > 4 px/frame, opening >= 90%
       TRACK-LOSS   min pupil confidence < 0.6 (tracking failure flag,
                    NOT blink evidence - advisor-agreed demotion)
       residual     any trace leaves its 1-s rolling median by > 5 MAD
     overlapping flags merge (0.25 s gap); blink depth wins over saccade.
  4. outputs into <video_dir>/pipeline_<stem>/:
       events.csv, timeseries.png, review_montage.png (flagged events with
       frames, for the human error-review pass), summary.txt
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Users/lizhiheng/Desktop/生物")
CONFIG = ROOT / "dlc_projects/EyePupilBlink-Zhiheng-2026-08-17/config.yaml"
PY = "/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python"
SHUFFLE, TSI = 60, 11  # production model
FPS_DEFAULT = 60.0
PCUT, WIN, MAD_K, GAP_S = 0.6, 61, 5.0, 0.25

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--skip-analyze", action="store_true")
ap.add_argument("--fps", type=float, default=None)
a = ap.parse_args()
video = Path(a.video).resolve()
out = video.parent / f"pipeline_{video.stem}"
out.mkdir(exist_ok=True)

h5s = sorted(video.parent.glob(f"{video.stem}DLC*shuffle{SHUFFLE}*.h5"))
if not h5s and not a.skip_analyze:
    code = (f"import deeplabcut,os; os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'; "
            f"deeplabcut.analyze_videos(r'{CONFIG}', [r'{video}'], shuffle={SHUFFLE}, "
            f"trainingsetindex={TSI}, device='mps', batch_size=32)")
    subprocess.run([PY, "-c", code], check=True)
    h5s = sorted(video.parent.glob(f"{video.stem}DLC*shuffle{SHUFFLE}*.h5"))
if not h5s:
    sys.exit("no predictions found")
df = pd.read_hdf(h5s[-1])
df.columns = df.columns.droplevel(0)
while df.columns.nlevels > 2:
    df.columns = df.columns.droplevel(0)

cap = cv2.VideoCapture(str(video))
fps = a.fps or cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
t = np.arange(len(df)) / fps

g = lambda bp, c: df[bp][c].to_numpy()
pcx = (g("pupil_left", "x") + g("pupil_right", "x")) / 2
pcy = (g("pupil_top", "y") + g("pupil_bottom", "y")) / 2
opening = np.abs(g("eyelid_bottom", "y") - g("eyelid_top", "y"))
area = np.pi / 4 * np.abs(g("pupil_right", "x") - g("pupil_left", "x")) \
    * np.abs(g("pupil_bottom", "y") - g("pupil_top", "y"))
conf = np.min([g(b, "likelihood") for b in
               ["pupil_left", "pupil_right", "pupil_top", "pupil_bottom"]], axis=0)

roll = lambda v, w=WIN: pd.Series(v).rolling(w, center=True, min_periods=1).median().to_numpy()
open_base = roll(opening, int(5 * fps) | 1)          # slow baseline (~5 s)
open_frac = opening / np.maximum(open_base, 1)
center_med = roll(pcx), roll(pcy)
center_shift = np.hypot(pcx - center_med[0], pcy - center_med[1])
vel = np.hypot(np.gradient(pcx), np.gradient(pcy))  # px/frame, catches held saccades


def residual_mask(v):
    r = v - roll(v)
    mad = np.nanmedian(np.abs(r - np.nanmedian(r))) or 1.0
    return np.abs(r) > MAD_K * 1.4826 * mad


masks = {
    "BLINK": open_frac < 0.70,
    "PARTIAL_BLINK": (open_frac >= 0.70) & (open_frac < 0.85),
    "SACCADE": ((center_shift > 10) | (vel > 4)) & (open_frac >= 0.90),
    "TRACK_LOSS": conf < PCUT,
    "residual": residual_mask(pcx) | residual_mask(pcy) | residual_mask(opening) | residual_mask(area),
}
union = np.zeros(len(df), bool)
for m in masks.values():
    union |= m


def to_events(mask):
    ev, start, last = [], None, -10 ** 9
    gap = int(GAP_S * fps)
    for i in np.where(mask)[0]:
        if i - last > gap:
            if start is not None:
                ev.append((start, last))
            start = i
        last = i
    if start is not None:
        ev.append((start, last))
    return ev


PRIORITY = ["BLINK", "PARTIAL_BLINK", "SACCADE", "TRACK_LOSS", "residual"]
rows = []
for s, e in to_events(union):
    seg = slice(s, e + 1)
    label = next((k for k in PRIORITY if masks[k][seg].any()), "residual")
    rows.append({"start_s": round(s / fps, 2), "end_s": round(e / fps, 2),
                 "n_frames": e - s + 1, "class": label,
                 "min_opening_frac": round(float(np.nanmin(open_frac[seg])), 2),
                 "max_center_shift_px": round(float(np.nanmax(center_shift[seg])), 1),
                 "min_pupil_conf": round(float(np.nanmin(conf[seg])), 3),
                 "needs_human": bool(label in ("TRACK_LOSS", "residual")
                                     or masks["TRACK_LOSS"][seg].any())})
ev = pd.DataFrame(rows)
ev.to_csv(out / "events.csv", index=False)

# ---- timeseries figure ----------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(16, 10.5), sharex=True)
for ax, (name, v) in zip(axes, [("pupil center x (px)", pcx),
                                ("eye opening (px)", opening),
                                ("pupil area (px^2)", area)]):
    ax.plot(t, v, lw=0.5, color="#2F6B9A")
    ax.set_ylabel(name, fontsize=9)
axes[3].plot(t, conf, lw=0.5, color="#2A9D8F")
axes[3].axhline(PCUT, color="red", ls="--", lw=1)
axes[3].set_ylabel("min pupil conf", fontsize=9)
axes[3].set_xlabel("time (s)")
colors = {"BLINK": "#D1495B", "PARTIAL_BLINK": "#E8A87C", "SACCADE": "#2F6B9A",
          "TRACK_LOSS": "#9B1D64", "residual": "gray"}
for _, r in ev.iterrows():
    for ax in axes:
        ax.axvspan(r.start_s, r.end_s, color=colors[r["class"]], alpha=0.18)
handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5) for c in colors.values()]
axes[0].legend(handles, colors.keys(), fontsize=8, ncol=5, loc="upper right")
fig.suptitle(f"{video.name} - production model (shuffle {SHUFFLE}) - "
             f"{len(ev)} events, {int(union.sum())}/{len(df)} frames flagged", fontsize=13)
fig.tight_layout()
fig.savefig(out / "timeseries.png", dpi=130)
plt.close(fig)

# ---- review montage: events needing human eyes ----------------------------
need = ev[ev.needs_human].nlargest(12, "n_frames")
if len(need):
    ncol = 4
    nrow = int(np.ceil(len(need) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.2 * nrow))
    for ax, (_, r) in zip(np.ravel(axs), need.iterrows()):
        mid = int((r.start_s + r.end_s) / 2 * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, im = cap.read()
        if ok:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            for bp in df.columns.get_level_values(0).unique():
                c = "red" if df[bp]["likelihood"].iloc[mid] < PCUT else "lime"
                ax.plot(df[bp]["x"].iloc[mid], df[bp]["y"].iloc[mid], "o", color=c, ms=4)
        ax.set_title(f"{r.start_s:.1f}-{r.end_s:.1f}s {r['class']}", fontsize=9)
        ax.axis("off")
    for ax in np.ravel(axs)[len(need):]:
        ax.axis("off")
    fig.suptitle("Events for human review (middle frame; green/red = confidence)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "review_montage.png", dpi=120)
cap.release()

counts = ev["class"].value_counts().to_dict()
summary = (f"video: {video.name}\nframes: {len(df)}  fps: {fps:.0f}\n"
           f"events: {len(ev)}  flagged frames: {int(union.sum())} "
           f"({union.mean():.1%})\nby class: {counts}\n"
           f"needs human review: {int(ev.needs_human.sum())} events\n")
(out / "summary.txt").write_text(summary)
print(summary)
