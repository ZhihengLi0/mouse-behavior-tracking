#!/usr/bin/env python3
"""Does the earlier pupil-area / blink machinery still fit the new label standard, and is there a more accurate
variant now that pupil_top is a real ellipse endpoint?

    python compare_pupil_methods.py --unit 20251031_Pluto_spont_1 --test-pred <eval h5 of the last step> [--pred-h5 <whole-video h5>]

A. Accuracy on the frozen test frames (human labels = truth):
     area_3pt   old rule (2026-09-19): centre = mean of left/right, width = |R.x - L.x|, height = 2 (B.y - centre.y),
                pupil_top unused (it was the visible edge under the old standard)
     area_4pt   new rule: width = |R.x - L.x|, height = |B.y - T.y| (both are ellipse endpoints now)
     opening    eye opening = distance between the two lid points
   For each: value from the human labels vs from the model prediction, relative error per frame.
   Also: how much area_3pt and area_4pt disagree on the SAME human labels (bias of the old rule under the new standard).
B. Whole video (optional): 3-pt vs 4-pt trace agreement, and the blink trigger of pupil_trace.py
   (eye opening < 95% of its 5-s median, or left/right/bottom confidence < 0.6) versus a 4-point variant that also
   requires pupil_top confidence.
Outputs <unit>/results/pupil_methods_comparison.csv and .png; nothing else is modified."""
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
from scale_step import HERE, flat, human_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--unit", required=True)
ap.add_argument("--test-pred", required=True, help="image_predictions_*.h5 of the step to judge (eval folder)")
ap.add_argument("--pred-h5", default=None, help="whole-video predictions of the same model (optional)")
ap.add_argument("--fps", type=float, default=60.0)
a = ap.parse_args()
R = HERE / a.unit / "results" / "blink_area_analysis"; R.mkdir(parents=True, exist_ok=True)


def geometry(d):
    """Per-frame pupil geometry under both rules + eye opening. d: flat frame table (bodypart, coord)."""
    g = pd.DataFrame(index=d.index)
    L, Rr, T, B = (d[b] for b in ("pupil_left", "pupil_right", "pupil_top", "pupil_bottom"))
    g["width"] = (Rr["x"] - L["x"]).abs()
    cy3 = (L["y"] + Rr["y"]) / 2
    g["height_3pt"] = 2 * (B["y"] - cy3)
    g["height_4pt"] = (B["y"] - T["y"]).abs()
    g["area_3pt"] = np.pi / 4 * g["width"] * g["height_3pt"]
    g["area_4pt"] = np.pi / 4 * g["width"] * g["height_4pt"]
    g["opening"] = np.hypot(d["eyelid_top"]["x"] - d["eyelid_bottom"]["x"], d["eyelid_top"]["y"] - d["eyelid_bottom"]["y"])
    return g


rows = []
# ---------------- A. test frames ----------------
gt = flat(pd.read_hdf(HERE / a.unit / "training-data" / "labels" / "test_frozen" / "test50_labels.h5"))
pr = flat(pd.read_hdf(a.test_pred)).loc[gt.index]
G, P = geometry(gt), geometry(pr)
ok = G[["area_3pt", "area_4pt", "opening"]].notna().all(axis=1) & (G["area_4pt"] > 0)
G, P = G[ok], P[ok]
for q in ("area_3pt", "area_4pt", "width", "height_3pt", "height_4pt", "opening"):
    rel = (P[q] - G[q]) / G[q] * 100
    rows.append({"quantity": q, "what": "model vs human, frozen test frames", "n": int(len(rel)),
                 "median_abs_rel_err_pct": round(float(rel.abs().median()), 1), "p90_abs_rel_err_pct": round(float(rel.abs().quantile(0.9)), 1),
                 "median_signed_err_pct": round(float(rel.median()), 1), "median_abs_err_px": round(float((P[q] - G[q]).abs().median()), 1)})
bias = (G["area_3pt"] / G["area_4pt"] - 1) * 100
rows.append({"quantity": "area_3pt / area_4pt - 1", "what": "same human labels: old rule vs new rule", "n": int(len(bias)),
             "median_abs_rel_err_pct": round(float(bias.abs().median()), 1), "p90_abs_rel_err_pct": round(float(bias.abs().quantile(0.9)), 1),
             "median_signed_err_pct": round(float(bias.median()), 1), "median_abs_err_px": np.nan})
# where does the 3-pt height come from under the new standard? top-to-centre vs centre-to-bottom on human labels
cy = (gt.loc[G.index, "pupil_left"]["y"] + gt.loc[G.index, "pupil_right"]["y"]) / 2
up, down = cy - gt.loc[G.index, "pupil_top"]["y"], gt.loc[G.index, "pupil_bottom"]["y"] - cy
rows.append({"quantity": "top-to-centre / centre-to-bottom", "what": "human labels: vertical symmetry of the labeled pupil", "n": int(len(up)),
             "median_abs_rel_err_pct": np.nan, "p90_abs_rel_err_pct": np.nan, "median_signed_err_pct": round(float(((up / down) - 1).median() * 100), 1),
             "median_abs_err_px": round(float((up - down).abs().median()), 1)})

# ---------------- B. whole video ----------------
trace = None
if a.pred_h5:
    d = pt.load(a.pred_h5)
    raw3, fil3, _, I3 = pt.pupil_trace(d, fps=a.fps)                       # the earlier machinery, unchanged
    g = geometry(d)
    lk = {b: d[b]["likelihood"].to_numpy(float) for b in ("pupil_top", "pupil_bottom", "pupil_left", "pupil_right")}
    # 4-point variant of the same blink rule: also distrust frames where pupil_top is uncertain
    bad4 = I3["bad"] | (lk["pupil_top"] < pt.PCUT)
    bad4 = pd.Series(bad4).rolling(2 * pt.PAD + 1, center=True, min_periods=1).max().to_numpy().astype(bool)
    masked4 = g[["area_4pt", "width", "height_4pt"]].where(pd.Series(~bad4), axis=0)
    fil4 = masked4.interpolate(method="linear", limit_area="inside")
    good = ~I3["bad"] & ~bad4 & np.isfinite(g["area_3pt"]) & np.isfinite(g["area_4pt"])
    ratio = (g["area_3pt"] / g["area_4pt"])[good]
    rows.append({"quantity": "area_3pt / area_4pt - 1", "what": "whole video, trusted frames", "n": int(good.sum()),
                 "median_abs_rel_err_pct": round(float(((ratio - 1) * 100).abs().median()), 1), "p90_abs_rel_err_pct": round(float(((ratio - 1) * 100).abs().quantile(0.9)), 1),
                 "median_signed_err_pct": round(float(((ratio - 1) * 100).median()), 1), "median_abs_err_px": np.nan})
    rows.append({"quantity": "corr(area_3pt, area_4pt)", "what": "whole video, trusted frames", "n": int(good.sum()),
                 "median_abs_rel_err_pct": np.nan, "p90_abs_rel_err_pct": np.nan,
                 "median_signed_err_pct": round(float(np.corrcoef(g["area_3pt"][good], g["area_4pt"][good])[0, 1]), 3), "median_abs_err_px": np.nan})
    for name, bad in (("blink rule 2026-09-19 (opening<95% | L/R/B conf<0.6)", I3["bad"]), ("+ pupil_top conf<0.6", bad4)):
        rid = np.cumsum(np.r_[True, bad[1:] != bad[:-1]]); runs = pd.Series(bad).groupby(rid).agg(["first", "size"]); runs = runs[runs["first"]]
        rows.append({"quantity": name, "what": "whole video: untrusted frames / runs / longest run (s)", "n": int(bad.sum()),
                     "median_abs_rel_err_pct": round(float(bad.mean() * 100), 2), "p90_abs_rel_err_pct": int(len(runs)),
                     "median_signed_err_pct": round(float(runs["size"].max() / a.fps), 2) if len(runs) else 0, "median_abs_err_px": np.nan})
    only_conf = I3["low_conf"] & ~I3["closing"]; only_open = I3["closing"] & ~I3["low_conf"]
    rows.append({"quantity": "trigger overlap", "what": "frames flagged only by confidence / only by eye opening / by both", "n": int(len(d)),
                 "median_abs_rel_err_pct": int(only_conf.sum()), "p90_abs_rel_err_pct": int(only_open.sum()),
                 "median_signed_err_pct": int((I3["low_conf"] & I3["closing"]).sum()), "median_abs_err_px": np.nan})
    trace = (g, raw3, fil3, fil4, I3, bad4)

out = pd.DataFrame(rows)
out.to_csv(R / "pupil_methods_comparison.csv", index=False)
pd.set_option("display.width", 220)
print(out.to_string(index=False))

# ---------------- figure ----------------
ncol = 3 if trace is not None else 2
fig, axes = plt.subplots(1 if trace is None else 2, ncol, figsize=(6.2 * ncol, 5.2 if trace is None else 10), constrained_layout=True)
axes = np.atleast_2d(axes)
BLUE, RED, GRAY, GREEN = "#2F6B9A", "#D1495B", "0.55", "#2A9D8F"
ax = axes[0, 0]
for q, c, lab in (("area_3pt", GRAY, "old 3-point rule (top unused)"), ("area_4pt", RED, "new 4-point rule (top = ellipse endpoint)")):
    ax.scatter(G[q], P[q], s=22, color=c, alpha=0.8, label=lab)
lim = [0, max(G["area_4pt"].max(), P["area_4pt"].max()) * 1.05]
ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("pupil area from HUMAN labels (px$^2$)"); ax.set_ylabel("pupil area from the MODEL (px$^2$)")
e3, e4 = out.set_index("quantity").loc["area_3pt", "median_abs_rel_err_pct"], out.set_index("quantity").loc["area_4pt", "median_abs_rel_err_pct"]
ax.set_title(f"Frozen test frames (n={len(G)}): area error median {e3:.1f}% (3-pt) vs {e4:.1f}% (4-pt)", fontsize=10.5)
ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=0.3)
ax = axes[0, 1]
ax.scatter(G["area_4pt"], bias, s=22, color=BLUE)
ax.axhline(0, color="k", lw=0.8); ax.axhline(float(bias.median()), color=RED, lw=1.2, ls="--", label=f"median {bias.median():+.1f}%")
ax.set_xlabel("pupil area from human labels, 4-pt (px$^2$)"); ax.set_ylabel("old rule / new rule - 1 (%)")
ax.set_title("Same human labels: how far the old 3-point rule is from the 4-point area", fontsize=10.5); ax.legend(fontsize=9); ax.grid(alpha=0.3)
if trace is None:
    if ncol > 2:
        axes[0, 2].axis("off")
else:
    g, raw3, fil3, fil4, I3, bad4 = trace
    t = np.arange(len(g)) / a.fps
    ax = axes[0, 2]
    good = ~I3["bad"] & ~bad4
    ax.scatter(g["area_4pt"][good][::20], g["area_3pt"][good][::20], s=4, color=GRAY, alpha=0.5)
    lim = [0, np.nanpercentile(g["area_4pt"][good], 99.5) * 1.1]; ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("4-pt area (px$^2$)"); ax.set_ylabel("3-pt area (px$^2$)")
    r = out.set_index("quantity").loc["corr(area_3pt, area_4pt)", "median_signed_err_pct"]
    ax.set_title(f"Whole video, trusted frames (every 20th shown): corr {r:.3f}", fontsize=10.5); ax.grid(alpha=0.3)
    ax = axes[1, 0]; ax.set_position(ax.get_position())
    gs = axes[1, 0].get_gridspec()
    for a_ in axes[1, :]:
        a_.remove()
    ax = fig.add_subplot(gs[1, :])
    ax.plot(t, raw3["area"], lw=0.5, color=GRAY, label="raw 3-pt area")
    ax.plot(t, fil3["area"], lw=0.9, color=BLUE, label="3-pt, blink-filled (rule of 2026-09-19)")
    ax.plot(t, fil4["area_4pt"], lw=0.9, color=RED, label="4-pt, blink-filled (+ pupil_top confidence)")
    ax.fill_between(t, 0, 1, where=bad4, transform=ax.get_xaxis_transform(), color="#f0a35e", alpha=0.35, label="untrusted (4-pt rule)")
    ax.set_ylim(0.45 * np.nanmin(fil4["area_4pt"]), 1.25 * np.nanmax(fil4["area_4pt"]))
    ax.set_xlabel("time (s)"); ax.set_ylabel("pupil area (px$^2$)"); ax.legend(fontsize=8.5, ncol=4, loc="upper right"); ax.grid(alpha=0.2)
    ax.set_title(f"Whole video ({len(g)} frames): untrusted {I3['bad'].sum()} frames by the old rule, {bad4.sum()} with the pupil_top condition added", fontsize=10.5)
fig.suptitle(f"{a.unit}: pupil-area and blink machinery under the new label standard", fontsize=12)
fig.savefig(R / "pupil_methods_comparison.png", dpi=130)
print("saved", R / "pupil_methods_comparison.png")
