#!/usr/bin/env python3
"""Per-video eye time series (user request 2026-09-25): the metrics of the earlier time-series study
(old_pupil_top/time-series-analysis) on the newest whole-video prediction of one video, with the pupil area
computed by the 4-point ellipse rule (required) and the current 3-point rule for reference.

    python eye_timeseries.py --unit 4_20251030_Pluto_spont_1 [--pred-h5 FILE]

Per frame
  pupil centre      x = mean(left.x, right.x), y = mean(top.y, bottom.y)
  pupil area 4-pt   pi/4 * |right.x - left.x| * |bottom.y - top.y|        (all four points are ellipse endpoints)
  pupil area 3-pt   pi/4 * width * 2 (bottom.y - mean(left.y, right.y))   (pupil_trace.py, top unused)
  eye opening       eyelid_bottom.y - eyelid_top.y
  pupil confidence  lowest likelihood of the four pupil points
Flags (drawn as rasters, counted in the summary)
  untrusted         the production rule of pupil_trace.py (left/right/bottom conf < 0.6 or opening < 95% of its
                    5-s median, widened by 3 frames) - unchanged
  residual          earlier study's rule: |value - 1-s rolling median| > 5 robust MADs, per signal
Outputs <unit>/results/blink_area_analysis/timeseries.png and timeseries_summary.csv. Nothing else is modified."""
import argparse
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pupil_trace as pt  # noqa: E402
from scale_step import HERE  # noqa: E402

FPS = 60.0
WIN = 61
MAD_K = 5.0
PCUT = 0.6


def newest_prediction(unit):
    dirs = sorted((HERE / unit / "training-data").glob("predictions_*"))
    for d in reversed(dirs):
        h5 = sorted(glob.glob(str(d / "*snapshot*120*.h5")))
        if h5:
            return Path(h5[-1])
    sys.exit(f"no whole-video prediction for {unit}")


def residual_flags(v):
    s = pd.Series(v)
    res = s - s.rolling(WIN, center=True, min_periods=1).median()
    mad = np.nanmedian(np.abs(res - np.nanmedian(res))) * 1.4826
    return (np.abs(res) > MAD_K * mad).to_numpy() if mad > 0 else np.zeros(len(v), bool)


ap = argparse.ArgumentParser()
ap.add_argument("--unit", required=True)
ap.add_argument("--pred-h5", default=None)
a = ap.parse_args()
h5 = Path(a.pred_h5) if a.pred_h5 else newest_prediction(a.unit)
d = pt.load(h5)
n = len(d)
t = np.arange(n) / FPS / 60.0                      # minutes
L, R, T, B = (d[b] for b in ("pupil_left", "pupil_right", "pupil_top", "pupil_bottom"))
cx = ((L["x"] + R["x"]) / 2).to_numpy(float)
cy = ((T["y"] + B["y"]) / 2).to_numpy(float)
width = (R["x"] - L["x"]).abs().to_numpy(float)
height4 = (B["y"] - T["y"]).abs().to_numpy(float)
area4 = np.pi / 4 * width * height4
raw3, fil3, _, info = pt.pupil_trace(d, fps=FPS)
area3 = raw3["area"].to_numpy(float)
opening = (d["eyelid_bottom"]["y"] - d["eyelid_top"]["y"]).to_numpy(float)
minconf = np.min(np.stack([d[b]["likelihood"].to_numpy(float) for b in ("pupil_left", "pupil_right", "pupil_top", "pupil_bottom")]), axis=0)
untrusted = info["bad"]
flags = {"area 4-pt": residual_flags(area4), "eye opening": residual_flags(opening),
         "centre x": residual_flags(cx), "centre y": residual_flags(cy)}
any_res = np.any(np.stack(list(flags.values())), axis=0)
low_conf = minconf < PCUT

runs = np.cumsum(np.r_[True, untrusted[1:] != untrusted[:-1]])
n_runs = int(len(np.unique(runs[untrusted])))
ok = ~untrusted & np.isfinite(area4) & np.isfinite(area3)
summary = pd.DataFrame([{
    "unit": a.unit, "prediction": h5.name, "frames": n, "minutes": round(n / FPS / 60, 2),
    "median_area_4pt_px2": round(float(np.nanmedian(area4[ok])), 0), "median_area_3pt_px2": round(float(np.nanmedian(area3[ok])), 0),
    "median_area_3pt_over_4pt_minus_1_pct": round(float(np.nanmedian(area3[ok] / area4[ok] - 1) * 100), 1),
    "corr_area_3pt_4pt_trusted": round(float(np.corrcoef(area3[ok], area4[ok])[0, 1]), 3),
    "median_eye_opening_px": round(float(np.nanmedian(opening)), 1),
    "frac_untrusted_production_rule": round(float(untrusted.mean()), 4), "untrusted_runs": n_runs,
    "frac_pupil_minconf_lt_0.6": round(float(low_conf.mean()), 4),
    **{f"frac_residual_{k.replace(' ', '_').replace('-', '')}": round(float(v.mean()), 4) for k, v in flags.items()},
    "frac_residual_any": round(float(any_res.mean()), 4),
}])
OUT = HERE / a.unit / "results" / "blink_area_analysis"
OUT.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUT / "timeseries_summary.csv", index=False)
print(summary.T.to_string())

# ---------------- figure ----------------
BLUE, RED, GRAY, GREEN, ORANGE, PURPLE = "#2F6B9A", "#D1495B", "0.6", "#2A9D8F", "#E39B3C", "#7B4EA3"
# zoom: the 60-s window with the most untrusted frames
w = int(60 * FPS)
csum = np.convolve(untrusted.astype(float), np.ones(w), mode="valid") if n > w else np.array([untrusted.sum()])
z0 = int(np.argmax(csum)); z1 = min(n, z0 + w)
fig = plt.figure(figsize=(18, 17), constrained_layout=True)
gs = fig.add_gridspec(7, 1, height_ratios=[1, 1.2, 1, 0.8, 0.9, 0.2, 1.4])
axs = [fig.add_subplot(gs[i]) for i in range(7)]
for ax in axs[:5]:
    ax.fill_between(t, 0, 1, where=untrusted, transform=ax.get_xaxis_transform(), color=ORANGE, alpha=0.18, lw=0)
    ax.axvspan(t[z0], t[z1 - 1], color="k", alpha=0.05)
ax = axs[0]
ax.plot(t, cx - np.nanmedian(cx), lw=0.4, color=BLUE, label="centre x - median")
ax.plot(t, cy - np.nanmedian(cy), lw=0.4, color=RED, label="centre y - median")
lim = np.nanpercentile(np.abs(np.r_[cx - np.nanmedian(cx), cy - np.nanmedian(cy)]), 99.5) * 1.3
ax.set_ylim(-lim, lim); ax.set_ylabel("pupil centre (px)"); ax.legend(fontsize=8, loc="upper right", ncol=2)
ax = axs[1]
ax.plot(t, area3, lw=0.4, color=GRAY, label="3-point area (current production rule, top unused)")
ax.plot(t, area4, lw=0.5, color=GREEN, label="4-point area (ellipse endpoints)")
ax.set_ylim(0, np.nanpercentile(area4, 99.5) * 1.25); ax.set_ylabel("pupil area (px$^2$)"); ax.legend(fontsize=8, loc="upper right", ncol=2)
ax = axs[2]
ax.plot(t, opening, lw=0.4, color=PURPLE)
ax.set_ylim(max(0, np.nanpercentile(opening, 0.2) * 0.8), np.nanpercentile(opening, 99.8) * 1.1); ax.set_ylabel("eye opening (px)")
ax = axs[3]
ax.plot(t, minconf, lw=0.3, color="k"); ax.axhline(PCUT, color=RED, lw=0.8, ls="--")
ax.set_ylim(0, 1.02); ax.set_ylabel("lowest pupil\nconfidence")
ax = axs[4]
rows = [("untrusted (production rule)", untrusted, ORANGE), ("pupil conf < 0.6", low_conf, "k")] + \
       [(f"residual: {k}", v, c) for (k, v), c in zip(flags.items(), (GREEN, PURPLE, BLUE, RED))]
for i, (name, f, c) in enumerate(rows):
    idx = np.where(f)[0]
    ax.vlines(t[idx], i + 0.1, i + 0.9, color=c, lw=0.3)
ax.set_yticks(np.arange(len(rows)) + 0.5); ax.set_yticklabels([f"{r[0]} ({r[1].mean() * 100:.1f}%)" for r in rows], fontsize=8)
ax.set_ylim(0, len(rows)); ax.invert_yaxis(); ax.set_xlabel("time (min)")
for ax in axs[:5]:
    ax.set_xlim(0, t[-1]); ax.grid(alpha=0.2)
axs[5].axis("off")
axs[5].text(0.5, 0.5, f"zoom below: {t[z0]:.2f}-{t[z1 - 1]:.2f} min (the 60-s window with the most untrusted frames, grey band above)",
            ha="center", va="center", fontsize=10, transform=axs[5].transAxes)
ax = axs[6]
ts = np.arange(z0, z1) / FPS
ax.fill_between(ts, 0, 1, where=untrusted[z0:z1], transform=ax.get_xaxis_transform(), color=ORANGE, alpha=0.25, lw=0, label="untrusted")
ax.plot(ts, area3[z0:z1], lw=0.7, color=GRAY, label="3-point area")
ax.plot(ts, area4[z0:z1], lw=0.9, color=GREEN, label="4-point area")
ax.set_ylabel("pupil area (px$^2$)"); ax.set_xlabel("time (s)")
ax.set_ylim(0, np.nanpercentile(area4, 99.5) * 1.25)
ax2 = ax.twinx(); ax2.plot(ts, opening[z0:z1], lw=0.8, color=PURPLE, label="eye opening"); ax2.set_ylabel("eye opening (px)", color=PURPLE)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right", ncol=4); ax.grid(alpha=0.2); ax.set_xlim(ts[0], ts[-1])
fig.suptitle(f"{a.unit}: eye time series ({n} frames, {n / FPS / 60:.1f} min) - model {h5.name.split('DLC_')[-1].replace('.h5', '')}\n"
             f"orange = untrusted by the production blink rule ({untrusted.mean() * 100:.1f}% of frames, {n_runs} runs); "
             f"4-point vs 3-point area on trusted frames: corr {summary.corr_area_3pt_4pt_trusted[0]:.3f}, "
             f"3-point is {summary.median_area_3pt_over_4pt_minus_1_pct[0]:+.1f}% (median)", fontsize=12)
fig.savefig(OUT / "timeseries.png", dpi=110)
print("saved", OUT / "timeseries.png")
