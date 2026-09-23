#!/usr/bin/env python3
"""User hypothesis (2026-09-22): when the eye closes, the predicted pupil points scatter or jump a lot, so the
pupil geometry itself could signal a blink - independently of the eyelid distance. This script only MEASURES how
well such signals agree with the eyelid-based trigger; the pipeline (pupil_trace.py) is not changed.

    python blink_pupil_dispersion.py --unit 1_20251031_Pluto_spont_1 --pred-h5 <whole-video predictions>

Signals per frame (from the predicted keypoints):
    lid_rel      eye opening / its 5-s rolling median            (the existing trigger: < 0.95 = closing)
    aspect       pupil height / width (4-point)                   (a closing lid squashes the fitted ellipse)
    centre_off   |mean(T,B).x - mean(L,R).x| + |mean(T,B).y - mean(L,R).y|, in units of pupil width
                 (the two axes of a real ellipse cross at the same centre; scattered points do not)
    jump         largest frame-to-frame move of the 4 pupil points, in units of pupil width
    min_conf     lowest confidence among the 4 pupil points
    area_rel     4-pt area / its 5-s rolling median               (what a blink does to the measured area)
Reference = frames the eyelid trigger marks as closing (lid_rel < 0.95, not padded). For every pupil signal the
script reports the AUC for separating closing from open frames and, at the threshold that flags the same number
of frames as the eyelid trigger, the overlap (precision/recall against the eyelid trigger).
Writes <unit>/results/blink_pupil_dispersion.csv and .png."""
import argparse
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

ap = argparse.ArgumentParser()
ap.add_argument("--unit", required=True)
ap.add_argument("--pred-h5", required=True)
ap.add_argument("--fps", type=float, default=60.0)
a = ap.parse_args()
R = HERE / a.unit / "results"
d = pt.load(a.pred_h5)
n = len(d); fps = a.fps; win = int(round(5 * fps))
xy = lambda b: (d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float))
(tx, ty), (bx, by), (lx, ly), (rx, ry) = xy("pupil_top"), xy("pupil_bottom"), xy("pupil_left"), xy("pupil_right")
width = np.abs(rx - lx); height = np.abs(by - ty)
opening = np.hypot(d["eyelid_top"]["x"] - d["eyelid_bottom"]["x"], d["eyelid_top"]["y"] - d["eyelid_bottom"]["y"]).to_numpy(float)
roll = lambda v: pd.Series(v).rolling(win, center=True, min_periods=1).median().to_numpy()
sig = pd.DataFrame({
    "lid_rel": opening / roll(opening),
    "aspect": height / width,
    "centre_off": (np.abs((tx + bx) / 2 - (lx + rx) / 2) + np.abs((ty + by) / 2 - (ly + ry) / 2)) / width,
    "jump": np.r_[0, np.max([np.hypot(np.diff(v[0]), np.diff(v[1])) for v in ((tx, ty), (bx, by), (lx, ly), (rx, ry))], axis=0)] / width,
    "min_conf": np.min([d[b]["likelihood"].to_numpy(float) for b in ("pupil_top", "pupil_bottom", "pupil_left", "pupil_right")], axis=0),
    "area_rel": (width * height) / roll(width * height),
})
closing = sig["lid_rel"] < pt.OPEN_FRAC
print(f"{n} frames; eyelid trigger marks {closing.sum()} ({closing.mean():.1%}) as closing")


def auc(score, truth):
    s = pd.Series(score).rank().to_numpy(); t = truth.to_numpy()
    return (s[t].mean() - (t.sum() + 1) / 2) / (~t).sum()          # Mann-Whitney AUC


rows = []
# direction: a blink should push aspect DOWN, centre_off UP, jump UP, min_conf DOWN, area_rel DOWN
for name, direction in (("aspect", -1), ("centre_off", 1), ("jump", 1), ("min_conf", -1), ("area_rel", -1)):
    score = direction * sig[name].fillna(sig[name].median())
    A = auc(score, closing)
    k = int(closing.sum()); thr = np.sort(score.to_numpy())[-k]; flag = score >= thr
    tp = int((flag & closing).sum())
    rows.append({"signal": name, "direction": "low" if direction < 0 else "high", "AUC_vs_eyelid_trigger": round(A, 3),
                 "flagged_at_matched_count": int(flag.sum()), "overlap_with_eyelid_trigger": tp,
                 "precision": round(tp / max(flag.sum(), 1), 3), "recall": round(tp / max(k, 1), 3),
                 "median_open": round(float(sig[name][~closing].median()), 3), "median_closing": round(float(sig[name][closing].median()), 3)})
out = pd.DataFrame(rows); out.to_csv(R / "blink_pupil_dispersion.csv", index=False)
pd.set_option("display.width", 200); print(out.to_string(index=False))

# figure: distributions open vs closing, and a 20-s example strip
fig, axes = plt.subplots(2, 3, figsize=(17, 8.5), constrained_layout=True)
BLUE, RED = "#2F6B9A", "#D1495B"
for ax, name in zip(axes[0], ("aspect", "centre_off", "jump")):
    v = sig[name].replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = np.nanpercentile(v, [0.5, 99.5]); bins = np.linspace(lo, hi, 60)
    ax.hist(sig[name][~closing].clip(lo, hi), bins=bins, density=True, alpha=0.6, color=BLUE, label="eye open (eyelid trigger)")
    ax.hist(sig[name][closing].clip(lo, hi), bins=bins, density=True, alpha=0.6, color=RED, label="closing (eyelid trigger)")
    r = out.set_index("signal").loc[name]
    ax.set_title(f"{name}: AUC {r['AUC_vs_eyelid_trigger']:.2f}, overlap at matched count {r['recall']:.0%}", fontsize=10.5)
    ax.set_xlabel(name); ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
for ax, name in zip(axes[1][:2], ("min_conf", "area_rel")):
    v = sig[name].replace([np.inf, -np.inf], np.nan).dropna(); lo, hi = np.nanpercentile(v, [0.5, 99.5]); bins = np.linspace(lo, hi, 60)
    ax.hist(sig[name][~closing].clip(lo, hi), bins=bins, density=True, alpha=0.6, color=BLUE); ax.hist(sig[name][closing].clip(lo, hi), bins=bins, density=True, alpha=0.6, color=RED)
    r = out.set_index("signal").loc[name]; ax.set_title(f"{name}: AUC {r['AUC_vs_eyelid_trigger']:.2f}, overlap {r['recall']:.0%}", fontsize=10.5); ax.set_xlabel(name); ax.grid(alpha=0.3)
# example strip around the deepest blink
i0 = int(np.nanargmin(sig["lid_rel"].to_numpy())); s_, e_ = max(i0 - 10 * int(fps), 0), min(i0 + 10 * int(fps), n)
t = np.arange(s_, e_) / fps; ax = axes[1][2]
ax.plot(t, sig["lid_rel"][s_:e_], color=BLUE, lw=1.2, label="eye opening / 5-s median")
ax.plot(t, sig["aspect"][s_:e_] / np.nanmedian(sig["aspect"]), color="#E08E45", lw=1, label="pupil aspect / its median")
ax.plot(t, sig["centre_off"][s_:e_].clip(0, 2), color=RED, lw=1, label="centre offset (pupil widths)")
ax.plot(t, sig["jump"][s_:e_].clip(0, 2), color="0.4", lw=0.8, label="largest pupil-point jump (pupil widths)")
ax.axhline(pt.OPEN_FRAC, color="k", ls="--", lw=0.8); ax.set_ylim(0, 2); ax.set_xlabel("time (s)"); ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
ax.set_title(f"20 s around the deepest closure ({i0 / fps:.1f} s)", fontsize=10.5)
fig.suptitle(f"{a.unit}: do the predicted pupil points betray a blink? pupil-geometry signals vs the eyelid trigger (opening < 95% of 5-s median)", fontsize=12)
fig.savefig(R / "blink_pupil_dispersion.png", dpi=130); print("saved", R / "blink_pupil_dispersion.png")

# ---------------- ground-truth check on the labeled frames of ALL units ----------------
# frames where the labeler left >= 1 pupil point empty (eye closed / pupil not visible) vs frames with all 4 pupil
# points labeled; the signals are computed from the PRE-LABELS (model output) of the same frames
import glob  # noqa: E402

rows = []
for h5 in sorted(glob.glob(str(HERE / "*/training-data/labels/*/CollectedData_Zhiheng.h5"))):
    d_ = Path(h5).parent
    m = d_ / "machinelabels.h5"
    if not m.exists():
        m = d_ / "_old_prelabels" / "machinelabels.h5"
    if not m.exists():
        continue
    h = pd.read_hdf(h5); mm = pd.read_hdf(m)
    key = lambda i: i[-1] if isinstance(i, tuple) else str(i).split("/")[-1]
    h.index = [key(i) for i in h.index]; mm.index = [key(i) for i in mm.index]; mm = mm.reindex(h.index)
    bps = list(h.columns.get_level_values("bodyparts").unique())
    HX = h.xs("x", axis=1, level="coords"); HX.columns = bps
    closed_lab = HX[["pupil_top", "pupil_bottom", "pupil_left", "pupil_right"]].isna().any(axis=1)
    X = mm.xs("x", axis=1, level="coords"); Y = mm.xs("y", axis=1, level="coords"); Lk = mm.xs("likelihood", axis=1, level="coords")
    X.columns = Y.columns = Lk.columns = bps
    w = (X.pupil_right - X.pupil_left).abs(); hgt = (Y.pupil_bottom - Y.pupil_top).abs()
    for i in h.index:
        rows.append({"set": str(d_.relative_to(HERE)), "frame": i, "closed_by_labeler": bool(closed_lab[i]),
                     "aspect": hgt[i] / w[i],
                     "centre_off": (abs((X.pupil_top[i] + X.pupil_bottom[i]) / 2 - (X.pupil_left[i] + X.pupil_right[i]) / 2)
                                    + abs((Y.pupil_top[i] + Y.pupil_bottom[i]) / 2 - (Y.pupil_left[i] + Y.pupil_right[i]) / 2)) / w[i],
                     "min_conf": Lk[["pupil_top", "pupil_bottom", "pupil_left", "pupil_right"]].loc[i].min(),
                     "opening_px": np.hypot(X.eyelid_top[i] - X.eyelid_bottom[i], Y.eyelid_top[i] - Y.eyelid_bottom[i])})
lab = pd.DataFrame(rows)
lab.to_csv(R / "blink_pupil_dispersion_labeled_frames.csv", index=False)
print(f"\nlabeled frames with pre-labels: {len(lab)} | closed by the labeler: {int(lab.closed_by_labeler.sum())}")
for name, sgn in (("aspect", -1), ("centre_off", 1), ("min_conf", -1), ("opening_px", -1)):
    v = lab[name].fillna(lab[name].median())
    print(f"  {name:10s} AUC vs labeler-closed {auc(sgn * v, lab.closed_by_labeler):.2f} | median open {lab[name][~lab.closed_by_labeler].median():.3f}, closed {lab[name][lab.closed_by_labeler].median():.3f}")
