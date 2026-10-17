"""Time series of every frame of every video, for section 11 of the page: no rule, no flag, no threshold, just the traces.

For each video the model at its own plateau point (results/labels_to_plateau.csv; video 7 uses its 20-label model) is run
on every frame of the whole video (the <unit>/training-data/predictions_stepNN/*.h5 written in step 3). Per frame:
  four-point area    = pi/4 * |x(pupil_right) - x(pupil_left)| * |y(pupil_bottom) - y(pupil_top)|   (px^2)
  eye opening O      = y(eyelid_bottom) - y(eyelid_top)                                             (px)
  lowest pupil confidence = min of the four pupil points' likelihood
  pupil centre       = ((x_left + x_right)/2, (y_top + y_bottom)/2), shown relative to its median over the video
Outputs in results/area_timeseries/:
  area_timeseries_videoNN.png   one figure per video, four panels (area, opening, lowest confidence, centre offset) against time
  area_timeseries_all_videos.png  one row per video: four-point area divided by the video's median area
  area_timeseries_all_videos.csv  per video: frames, minutes, model step, median area, median opening
"""
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
OUT = HERE / "results" / "area_timeseries"
OUT.mkdir(exist_ok=True)
PL = pd.read_csv(HERE / "results" / "labels_to_plateau.csv")
PL = PL[PL.status == "final"]


def whole_pred(unit, step):
    h = sorted(glob.glob(str(HERE / unit / "training-data" / f"predictions_step{step:02d}" / "*snapshot*120*.h5")))
    return h[-1] if h else None


rows, traces = [], []
for r in PL.itertuples():
    step = max(1, int(r.labels_at_plateau) // 20)
    d = pt.load(whole_pred(r.unit, step))
    g = lambda b, c: d[b][c].to_numpy(float)
    area = np.pi / 4 * np.abs(g("pupil_right", "x") - g("pupil_left", "x")) * np.abs(g("pupil_bottom", "y") - g("pupil_top", "y"))
    opening = g("eyelid_bottom", "y") - g("eyelid_top", "y")
    conf = np.min(np.stack([g(b, "likelihood") for b in ("pupil_top", "pupil_bottom", "pupil_left", "pupil_right")], axis=1), axis=1)
    cx = (g("pupil_left", "x") + g("pupil_right", "x")) / 2
    cy = (g("pupil_top", "y") + g("pupil_bottom", "y")) / 2
    t = np.arange(len(d)) / FPS
    med = float(np.nanmedian(area))
    rows.append({"video": r.video, "unit": r.unit, "mouse": r.mouse, "date": r.date, "model_step": step, "frames": len(d),
                 "minutes": round(len(d) / FPS / 60, 2), "area_median_px2": round(med, 0), "opening_median_px": round(float(np.nanmedian(opening)), 1)})
    traces.append((r, t, area / med))

    fig, ax = plt.subplots(4, 1, figsize=(20, 10), sharex=True, constrained_layout=True)
    ax[0].plot(t, area, lw=0.5, color="#1D3557")
    ax[0].set_ylim(0, np.nanpercentile(area, 99.9) * 1.15); ax[0].set_ylabel("four-point pupil area (px$^2$)")
    ax[1].plot(t, opening, lw=0.5, color="#2A9D8F"); ax[1].set_ylim(bottom=0); ax[1].set_ylabel("eye opening O (px)")
    ax[2].plot(t, conf, lw=0.5, color="0.3"); ax[2].set_ylim(0, 1.02); ax[2].set_ylabel("lowest pupil confidence")
    ax[3].plot(t, cx - np.nanmedian(cx), lw=0.5, color="#457B9D", label="x")
    ax[3].plot(t, cy - np.nanmedian(cy), lw=0.5, color="#E9C46A", label="y")
    ax[3].set_ylim(-80, 80); ax[3].set_ylabel("pupil centre - median (px)"); ax[3].legend(loc="upper right", ncol=2, fontsize=9); ax[3].set_xlabel("time (s)")
    fig.suptitle(f"Video {r.video} ({r.mouse}, {r.date}), model at the plateau point ({int(r.labels_at_plateau)} labels): "
                 f"{len(d):,} frames, {len(d) / FPS / 60:.1f} min, every frame", fontsize=12)
    fig.savefig(OUT / f"area_timeseries_video{int(r.video):02d}.png", dpi=110)
    plt.close(fig)

pd.DataFrame(rows).to_csv(OUT / "area_timeseries_all_videos.csv", index=False)

fig, axes = plt.subplots(len(traces), 1, figsize=(20, 1.1 * len(traces) + 1), sharex=True, constrained_layout=True)
for a, (r, t, rel) in zip(axes, traces):
    a.plot(t, rel, lw=0.4, color="#1D3557")
    a.set_ylim(0, 2); a.set_yticks([0, 1, 2]); a.set_ylabel(f"v{r.video}\n{r.mouse}", rotation=0, ha="right", va="center", fontsize=9)
axes[-1].set_xlabel("time (s)")
fig.suptitle("Four-point pupil area of every frame, divided by the video's median area (one row per video; video 0 is 5 min, the others 19.9 min)", fontsize=12)
fig.savefig(OUT / "area_timeseries_all_videos.png", dpi=110)
plt.close(fig)
print(pd.DataFrame(rows).to_string(index=False))
