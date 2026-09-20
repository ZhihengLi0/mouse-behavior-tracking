#!/usr/bin/env python3
"""Pupil trace with blink handling (2026-09-19). Snapshot of old_pupil_top/time-series-analysis/scripts/pupil_trace.py
(only change: no default prediction file, so it can be imported from anywhere).

    python pupil_trace.py [predictions.h5]

Per frame (3-point ENDPOINT ellipse, no constant, pupil_top unused):
    center x = mean x of left/right      center y = mean y of left/right
    width    = |R.x - L.x|               height   = 2 * (B.y - center y)
    area     = pi/4 * width * height

A frame is UNTRUSTED when
    a) any of left/right/bottom has likelihood < PCUT, or
    b) eye opening is below OPEN_FRAC of its own 5-s rolling median.
(b) is needed because keypoint confidence does NOT drop in blinks: the lower
lid pushes pupil_bottom up while its likelihood stays ~0.9 (16 of 14,400
frames have pupil_bottom < 0.6, yet 492 frames are biased by more than ~5%).
OPEN_FRAC = 0.95 is where the measured area bias crosses ~5%.
Untrusted runs are widened by PAD frames on both sides (blink on/offset) and
then filled by linear interpolation between the last trusted frame before and
the first trusted frame after. Runs longer than MAX_GAP_S stay NaN: nothing
is invented across long closures. `hold` is the causal variant (keep the last
trusted value), for real-time use.
"""
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path.cwd()
H5 = sys.argv[1] if len(sys.argv) > 1 else None          # pass a prediction h5 when run as a script
PCUT, OPEN_FRAC, PAD, MAX_GAP_S, BASE_S = 0.6, 0.95, 3, 1.0, 5.0
FPS = 60.0


def load(p):
    d = pd.read_hdf(p)
    d.columns = d.columns.droplevel(0)
    while d.columns.nlevels > 2:
        d.columns = d.columns.droplevel(0)
    return d


def pupil_trace(d, fps=FPS):
    xy = lambda b: (d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float))
    (lx, ly), (rx, ry), (_, by) = xy("pupil_left"), xy("pupil_right"), xy("pupil_bottom")
    cx, cy, width = (lx + rx) / 2, (ly + ry) / 2, np.abs(rx - lx)
    height = 2 * (by - cy)
    raw = pd.DataFrame({"center_x": cx, "center_y": cy, "width": width, "height": height,
                        "area": np.pi / 4 * width * height})
    opening = (d["eyelid_bottom"]["y"] - d["eyelid_top"]["y"]).to_numpy(float)
    n = int(round(BASE_S * fps))
    rel_open = opening / pd.Series(opening).rolling(n, center=True, min_periods=1).median().to_numpy()
    low_conf = np.any([d[b]["likelihood"].to_numpy(float) < PCUT for b in ("pupil_left", "pupil_right", "pupil_bottom")], axis=0)
    closing = rel_open < OPEN_FRAC
    bad = low_conf | closing | ~np.isfinite(raw["area"].to_numpy())
    bad = pd.Series(bad).rolling(2 * PAD + 1, center=True, min_periods=1).max().to_numpy().astype(bool)
    run_id = np.cumsum(np.r_[True, bad[1:] != bad[:-1]])
    run_len = pd.Series(bad).groupby(run_id).transform("size").to_numpy()
    too_long = bad & (run_len > MAX_GAP_S * fps)
    masked = raw.where(pd.Series(~bad), axis=0)
    filled = masked.interpolate(method="linear", limit_area="inside").where(pd.Series(~too_long), axis=0)
    hold = masked.ffill().where(pd.Series(~too_long), axis=0)
    info = dict(rel_open=rel_open, low_conf=low_conf, closing=closing, bad=bad, too_long=too_long,
                n_runs=int(len(np.unique(run_id[bad]))), longest=int(run_len[bad].max()) if bad.any() else 0)
    return raw, filled, hold, info


if __name__ == "__main__":
    d = load(H5)
    raw, fil, hold, I = pupil_trace(d)
    t = np.arange(len(d)) / FPS
    bad = I["bad"]
    print(f"{len(d)} frames | untrusted {bad.sum()} ({bad.mean():.1%}) in {I['n_runs']} runs, longest {I['longest'] / FPS:.2f} s | "
          f"by confidence {I['low_conf'].sum()}, by eye opening {I['closing'].sum()}, left empty (> {MAX_GAP_S} s) {I['too_long'].sum()}")
    op = I["rel_open"]
    for name in ("area", "width", "center_y"):
        base = raw[name].rolling(300, center=True, min_periods=1).median()
        dip_raw = (raw[name] / base - 1)[bad] * 100 if name != "center_y" else (raw[name] - base)[bad]
        dip_fil = (fil[name] / base - 1)[bad] * 100 if name != "center_y" else (fil[name] - base)[bad]
        unit = "%" if name != "center_y" else " px"
        ok = np.isfinite(fil[name]) & np.isfinite(op)
        print(f"  {name:9s} in untrusted frames: raw median {np.nanmedian(dip_raw):+.1f}{unit} (worst {np.nanmin(dip_raw):+.0f}{unit}) -> "
              f"filled {np.nanmedian(dip_fil):+.1f}{unit} (worst {np.nanmin(dip_fil):+.0f}{unit}) | corr with eye opening "
              f"raw {np.corrcoef(raw[name][ok], op[ok])[0, 1]:+.2f} -> filled {np.corrcoef(fil[name][ok], op[ok])[0, 1]:+.2f}")
    dh = (hold["area"] - fil["area"]).abs()[bad] / fil["area"][bad] * 100
    print(f"  hold-last vs linear fill, area in filled frames: median |diff| {np.nanmedian(dh):.1f}%, p90 {np.nanpercentile(dh, 90):.1f}%")

    # longest filled runs, for the zoom panels
    rid = np.cumsum(np.r_[True, bad[1:] != bad[:-1]])
    runs = pd.DataFrame({"rid": rid, "bad": bad, "i": np.arange(len(bad))}).query("bad").groupby("rid")["i"].agg(["min", "max"])
    runs["depth"] = [np.nanmin(op[a:b + 1]) for a, b in zip(runs["min"], runs["max"])]
    zoom = runs.sort_values("depth").head(3)

    fig = plt.figure(figsize=(18, 10.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 0.8, 1], hspace=0.42, wspace=0.22)
    BLUE, RED, GRAY = "#2F6B9A", "#D1495B", "0.55"
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, raw["area"], lw=0.6, color=GRAY, label="raw 3-point endpoint area (every frame)")
    ax.plot(t, fil["area"], lw=1.0, color=RED, label="blink-filled (untrusted frames replaced by interpolation)")
    ax.fill_between(t, 0, 1, where=bad, transform=ax.get_xaxis_transform(), color="#f0a35e", alpha=0.35, label="untrusted frames")
    ax.set_ylim(0.45 * np.nanmin(fil["area"]), 1.25 * np.nanmax(fil["area"]))   # raw garbage frames fall outside on purpose
    ax.set_ylabel("pupil area (px$^2$)"); ax.legend(fontsize=9, ncol=3, loc="upper right")
    ax.set_title(f"Pupil area, 3-point endpoint ellipse with blink handling: {bad.sum()} of {len(d)} frames ({bad.mean():.1%}) "
                 f"untrusted in {I['n_runs']} runs, longest {I['longest'] / FPS:.2f} s", fontsize=12)
    ax2 = fig.add_subplot(gs[1, :], sharex=ax)
    ax2.plot(t, op, lw=0.6, color=BLUE)
    ax2.axhline(OPEN_FRAC, color="k", ls="--", lw=0.8)
    ax2.fill_between(t, 0, 1, where=bad, transform=ax2.get_xaxis_transform(), color="#f0a35e", alpha=0.35)
    ax2.set_ylim(0, 1.15); ax2.set_ylabel("eye opening / its 5-s median"); ax2.set_xlabel("time (s)")
    ax2.set_title(f"Trigger: eye opening below {OPEN_FRAC:.0%} of its own 5-s baseline (dashed), or left/right/bottom confidence < {PCUT}; "
                  f"widened by {PAD} frames each side", fontsize=11)
    for j, (_, r) in enumerate(zoom.iterrows()):
        a, b = int(r["min"]), int(r["max"])
        s_, e_ = max(a - 45, 0), min(b + 45, len(d) - 1)
        axz = fig.add_subplot(gs[2, j])
        axz.set_ylim(0.5 * np.nanmin(fil["area"][s_:e_]), 1.15 * np.nanmax(fil["area"][s_:e_]))
        axz.plot(t[s_:e_], raw["area"][s_:e_], "o-", ms=2.5, lw=0.8, color=GRAY, label="raw")
        axz.plot(t[s_:e_], fil["area"][s_:e_], lw=1.8, color=RED, label="filled")
        axz.plot(t[s_:e_], hold["area"][s_:e_], lw=1.0, ls=":", color="k", label="hold-last (real-time variant)")
        axz.axvspan(t[a], t[b], color="#f0a35e", alpha=0.35)
        axz.set_title(f"Blink at {t[(a + b) // 2]:.1f} s: eye opening falls to {max(r['depth'], 0):.0%}, {(b - a + 1) / FPS * 1000:.0f} ms replaced", fontsize=10.5)
        axz.set_xlabel("time (s)"); axz.set_ylabel("area (px$^2$)" if j == 0 else "")
        if j == 0:
            axz.legend(fontsize=8, loc="lower left")
    fig.savefig(OUT / "07_pupil_trace_blink_filled.png", dpi=125, bbox_inches="tight")
    print("saved", OUT / "07_pupil_trace_blink_filled.png")
