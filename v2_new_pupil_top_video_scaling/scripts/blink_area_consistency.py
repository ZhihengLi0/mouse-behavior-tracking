#!/usr/bin/env python3
"""Are the blink and pupil-area findings consistent across all videos? (user request 2026-09-25), read-only.

    python blink_area_consistency.py

A. Pupil area on every frozen test set (human 4-point ellipse area = truth), model = the newest model trained with
   that video's own labels (videos 0-4) or the newest model overall (videos 5-7, whose test pre-labels came from a
   model of the same line -> anchored, flagged in the table). Rules: current 3-point (top unused), 4-point.
B. Blink signals against HUMAN closed/open judgements. Closed = labeled frame whose four pupil points were all left
   empty; open = all four pupil points labeled. Only frames the predicting model never trained on: test and val
   frames (newest whole-video prediction) and training batches N >= 2 (the prediction of step N-1, the model that
   selected them). Videos with a whole-video prediction only (0-4). Signals per frame:
     lid_rel     eye opening / its 5-s rolling median (production trigger: < 0.95)
     open_rel    eye opening / the video's median opening
     min_conf    lowest confidence of the four pupil points (production also uses left/right/bottom < 0.6)
     centre_off  distance between the centres of the two pupil axes, in pupil widths
     aspect      pupil height / width (4-point)
     untrusted   the production rule of pupil_trace.py (flag)
   AUC (closed vs open) per signal, pooled and per video, and the production rule's hit rate on closed frames and
   false-alarm rate on open frames.
C. Area vs eye opening on the whole video (trusted frames): correlation of the 4-point area with the eye opening.
Outputs results/blink_area_consistency_{area,blink,coupling}.csv and results/blink_area_consistency.png."""
import glob
import re
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

FPS = 60.0
P4 = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right"]
# (unit, label, model shuffle for the test set, anchored?)
VIDEOS = [("0_first5minvedio", "0 (mouse A)", 115, False), ("1_20251031_Pluto_spont_1", "1 (10-31)", 217, False),
          ("2_20251031_pluton2", "2 (10-31)", 324, False), ("3_20251031_pluto3", "3 (10-31)", 423, False),
          ("4_20251030_Pluto_spont_1", "4 (10-30, poor)", 515, False), ("5_20251029_Pluto_spont_1", "5 (10-29)", 515, True),
          ("6_20251028_Pluto_spont_1", "6 (10-28)", 515, True), ("7_20251027_Pluto_spont1", "7 (10-27)", 515, True)]


def fidx(name):
    return int(re.findall(r"\d+", Path(str(name)).stem)[0])


def areas(d):
    T, B, L, R = (d[b] for b in P4)
    w = (R["x"] - L["x"]).abs()
    a3 = np.pi / 4 * w * 2 * (B["y"] - (L["y"] + R["y"]) / 2)
    a4 = np.pi / 4 * w * (B["y"] - T["y"]).abs()
    return a3, a4


def test_pred(unit, shuffle):
    own = sorted(glob.glob(str(HERE / unit / "training-data" / "eval" / "cross" / f"shuffle{shuffle}" / "image_predictions_*.h5")))
    return own[-1] if own else None


def whole_pred(unit, step=None):
    d = HERE / unit / "training-data"
    dirs = [d / f"predictions_step{step:02d}"] if step is not None else sorted(d.glob("predictions_step*"))
    for dd in reversed(dirs):
        h = sorted(glob.glob(str(dd / "*snapshot*120*.h5")))
        if h:
            return h[-1]
    return None


def signals(d):
    L, R, T, B = (d[b] for b in ("pupil_left", "pupil_right", "pupil_top", "pupil_bottom"))
    w = (R["x"] - L["x"]).abs().to_numpy(float)
    h = (B["y"] - T["y"]).abs().to_numpy(float)
    c1 = np.c_[(L["x"] + R["x"]) / 2, (L["y"] + R["y"]) / 2]
    c2 = np.c_[(T["x"] + B["x"]) / 2, (T["y"] + B["y"]) / 2]
    opening = (d["eyelid_bottom"]["y"] - d["eyelid_top"]["y"]).to_numpy(float)
    _, _, _, info = pt.pupil_trace(d, fps=FPS)
    return pd.DataFrame({"lid_rel": info["rel_open"], "open_rel": opening / np.nanmedian(opening),
                         "min_conf": np.min(np.stack([d[b]["likelihood"].to_numpy(float) for b in P4]), axis=0),
                         "centre_off": np.hypot(*(c1 - c2).T) / np.where(w > 1, w, np.nan), "aspect": h / np.where(w > 1, w, np.nan),
                         "untrusted": info["bad"].astype(float), "opening": opening})


def auc(closed, opened, low_means_closed):
    c, o = np.asarray(closed, float), np.asarray(opened, float)
    c, o = c[np.isfinite(c)], o[np.isfinite(o)]
    if not len(c) or not len(o):
        return np.nan
    if low_means_closed:
        c, o = -c, -o
    return float(((c[:, None] > o[None, :]).sum() + 0.5 * (c[:, None] == o[None, :]).sum()) / (len(c) * len(o)))


# ---------------- A. area on every test set ----------------
rowsA = []
for unit, lab, sh, anchored in VIDEOS:
    gtf = HERE / unit / "training-data" / "labels" / "test_frozen" / "test50_labels.h5"
    ph = test_pred(unit, sh)
    if not gtf.exists() or not ph:
        continue
    gt = flat(pd.read_hdf(gtf)); pr = flat(pd.read_hdf(ph)).loc[gt.index]
    ok = pd.concat([gt[(b, c)].notna() for b in P4 for c in ("x", "y")], axis=1).all(axis=1)
    g3, g4 = areas(gt[ok]); p3, p4 = areas(pr[ok])
    e3, e4 = (p3 - g4) / g4 * 100, (p4 - g4) / g4 * 100
    rowsA.append({"video": lab, "model_shuffle": sh, "anchored_prelabels": anchored, "n": int(ok.sum()),
                  "err_3pt_median_abs_pct": round(float(e3.abs().median()), 1), "err_4pt_median_abs_pct": round(float(e4.abs().median()), 1),
                  "bias_3pt_median_pct": round(float(e3.median()), 1), "bias_4pt_median_pct": round(float(e4.median()), 1),
                  "human_3pt_vs_4pt_median_pct": round(float(((g3 / g4) - 1).median() * 100), 1),
                  "better": "4-point" if e4.abs().median() < e3.abs().median() else "3-point"})
A = pd.DataFrame(rowsA)

# ---------------- B. blink signals vs human closed/open ----------------
SIG = [("lid_rel", True), ("open_rel", True), ("min_conf", True), ("centre_off", False), ("aspect", True)]
rec = []
for unit, lab, _, _ in VIDEOS[:5]:
    labs = HERE / unit / "training-data" / "labels"
    newest = whole_pred(unit)
    sets = [("test50", newest), ("val20", newest)] + [(b.name, whole_pred(unit, int(b.name[5:]) - 1)) for b in sorted(labs.glob("batch*")) if int(b.name[5:]) >= 2]
    cache = {}
    for s, ph in sets:
        folder = labs / s
        if not ph or not glob.glob(str(folder / "CollectedData_*.h5")):
            continue
        if ph not in cache:
            cache[ph] = signals(pt.load(ph))
        S = cache[ph]
        h = flat(human_table(folder))
        for name, r in h.iterrows():
            n_pupil = sum(np.isfinite(r[(b, "x")]) for b in P4)
            if n_pupil not in (0, 4):
                continue
            i = fidx(name)
            if i >= len(S):
                continue
            rec.append({"video": lab, "set": s, "frame": i, "closed": n_pupil == 0, **S.iloc[i].to_dict()})
Bf = pd.DataFrame(rec)
rowsB = []
for v, g in list(Bf.groupby("video", sort=False)) + [("all videos 0-4", Bf)]:
    c, o = g[g.closed], g[~g.closed]
    row = {"video": v, "closed_frames": len(c), "open_frames": len(o)}
    for s, low in SIG:
        row[f"AUC_{s}"] = round(auc(c[s], o[s], low), 2)
    row["production_rule_hit_rate_closed"] = round(float(c.untrusted.mean()), 2) if len(c) else np.nan
    row["production_rule_false_alarm_open"] = round(float(o.untrusted.mean()), 2)
    row["mincONF_lt_0.6_false_alarm_open".replace("mincONF", "minconf")] = round(float((o.min_conf < 0.6).mean()), 2)
    rowsB.append(row)
B = pd.DataFrame(rowsB)

# ---------------- C. area vs eye opening on the whole video ----------------
rowsC = []
for unit, lab, _, _ in VIDEOS:
    ph = whole_pred(unit) or (sorted(glob.glob(str(HERE / unit / "training-data" / "predictions_shuffle*" / "*.h5"))) or [None])[-1]
    if not ph:
        continue
    d = pt.load(ph); S = signals(d); _, a4 = areas(d)
    ok = (S.untrusted == 0).to_numpy() & np.isfinite(a4.to_numpy())
    rowsC.append({"video": lab, "frames": len(d), "trusted_frames": int(ok.sum()),
                  "corr_area4_opening_trusted": round(float(np.corrcoef(a4.to_numpy()[ok], S.opening.to_numpy()[ok])[0, 1]), 2),
                  "frac_untrusted_production": round(float(S.untrusted.mean()), 3),
                  "frac_minconf_lt_0.6": round(float((S.min_conf < 0.6).mean()), 3)})
C = pd.DataFrame(rowsC)

R = HERE / "results"
A.to_csv(R / "blink_area_consistency_area.csv", index=False)
B.to_csv(R / "blink_area_consistency_blink.csv", index=False)
C.to_csv(R / "blink_area_consistency_coupling.csv", index=False)
Bf.to_csv(R / "blink_area_consistency_blink_frames.csv", index=False)
pd.set_option("display.width", 250)
print(A.to_string(index=False)); print(); print(B.to_string(index=False)); print(); print(C.to_string(index=False))

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), constrained_layout=True)
ax = axes[0]
x = np.arange(len(A))
ax.bar(x - 0.2, A.err_3pt_median_abs_pct, 0.4, color="0.6", label="3-point (current)")
ax.bar(x + 0.2, A.err_4pt_median_abs_pct, 0.4, color="#2A9D8F", label="4-point")
ax.set_xticks(x); ax.set_xticklabels([f"{v}{' *' if a else ''}" for v, a in zip(A.video, A.anchored_prelabels)], rotation=30, ha="right", fontsize=8.5)
ax.set_ylabel("median |area error| vs human 4-point area (%)"); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
ax.set_title("A. Pupil area on each frozen test set (* = pre-labels from the scoring model's line)", fontsize=10.5)
ax = axes[1]
sigs = [s for s, _ in SIG]
per = B[B.video != "all videos 0-4"]
for j, (_, r) in enumerate(per.iterrows()):
    ax.plot(range(len(sigs)), [r[f"AUC_{s}"] for s in sigs], "o-", color=plt.cm.viridis(0.85 * j / max(1, len(per) - 1)), lw=1.2, ms=6,
            label=f"video {r.video} ({r.closed_frames} closed / {r.open_frames} open)")
allr = B[B.video == "all videos 0-4"].iloc[0]
ax.plot(range(len(sigs)), [allr[f"AUC_{s}"] for s in sigs], "k-", lw=2.5, marker="s", ms=8, label=f"pooled ({allr.closed_frames} closed / {allr.open_frames} open)")
ax.axhline(0.5, color="0.5", ls="--", lw=0.8)
ax.set_xticks(range(len(sigs))); ax.set_xticklabels(["eye opening /\n5-s median\n(production)", "eye opening /\nvideo median", "lowest pupil\nconfidence", "pupil axes\ncentre offset", "pupil\nheight / width"], fontsize=8.5)
ax.set_ylim(0.3, 1.02); ax.set_ylabel("AUC: human closed vs open frames"); ax.legend(fontsize=7.5, loc="lower left"); ax.grid(alpha=0.3)
ax.set_title("B. Which signal separates the labeler's closed-eye frames?", fontsize=10.5)
ax = axes[2]
xc = np.arange(len(C))
ax.bar(xc - 0.2, C.frac_untrusted_production * 100, 0.4, color="#E39B3C", label="untrusted by the production blink rule")
ax.bar(xc + 0.2, C["frac_minconf_lt_0.6"] * 100, 0.4, color="k", alpha=0.6, label="lowest pupil confidence < 0.6")
for xi, (fu, fm, r) in enumerate(zip(C.frac_untrusted_production, C["frac_minconf_lt_0.6"], C.corr_area4_opening_trusted)):
    ax.text(xi, max(fu, fm) * 100 + 1.5, f"r={r:.2f}", ha="center", fontsize=8, color="#7B4EA3")
ax.set_xticks(xc); ax.set_xticklabels(C.video, rotation=30, ha="right", fontsize=8.5); ax.set_ylabel("% of all frames")
ax.legend(fontsize=8, loc="upper left", title="purple r = corr(4-pt area, eye opening), trusted frames", title_fontsize=7.5); ax.grid(axis="y", alpha=0.3)
ax.set_title("C. Whole videos: flagged fractions and area-opening coupling", fontsize=10.5)
fig.suptitle("Blink and pupil-area rules across all videos: what is consistent?", fontsize=12)
fig.savefig(R / "blink_area_consistency.png", dpi=130)
print("saved", R / "blink_area_consistency.png")
