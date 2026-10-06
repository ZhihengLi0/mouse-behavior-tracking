#!/usr/bin/env python3
"""Final eye-closure rule (first round, 2026-10-05): every frame becomes OPEN or CLOSED; evaluation of the rule and of
its simpler variants against the labeler's spot-check verdicts; per-video closure counts. Read-only analysis.

    python blink_final_rule.py

Inputs: whole-video prediction of each finished video's plateau-point model (plateau at 0 labels: the 20-label model);
the spot-check sample with the labeler's verdicts (results/blink_eyelid_distance/blink_eyelid_spotcheck_all_videos.csv).

Per frame t (image y points down):
  opening   O_t = y(eyelid_bottom) - y(eyelid_top);   r_t = O_t / median_t(O)
  trusted   eyelid points plausible: O_t >= 0, |x(eyelid_top) - x(eyelid_bottom)| <= 0.3 x eye width, both confidences >= 0.3
  pupil_t   mean confidence of the four pupil points
Rule (variants A-E, E is the final one):
  A  r_t < 0.70                                              (first version: no trust check)
  B  trusted and r_t < 0.70                                  (second version)
  C  B, plus untrusted frames within GAP = 5 frames of a B-closed frame are closed; other untrusted frames open
  D  C, plus an isolated untrusted run is closed when the median of pupil_t over the run < 0.527
     (threshold of the earlier analysis on the 28 human empty-pupil frames; not tuned on the spot check)   <- FINAL
  E  C, plus an isolated untrusted run is closed when the minimum of pupil_t over the run < 0.30
     (fits all 18 isolated items of the spot check, but the 0.30 was chosen on them; shown for comparison only)
Events: runs of closed frames (merged across gaps < GAP) of at least MIN_LEN = 3 frames. An event is "full" when its
minimum r < 0.45 or it contains untrusted frames (the eye disappeared), otherwise "partial".
Evaluation on the spot check: an item counts as predicted closed when at least half of its frames are closed under the
variant; truth = the labeler's verdict ("eye closed" or "partly closed" = closure; "eye open" = not).
Outputs in results/blink_eyelid_distance/: blink_final_variants_on_spotcheck.csv, blink_final_events_all_videos.csv,
blink_final_summary_all_videos.csv, blink_final_summary.png. Nothing else is modified."""
import csv
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

FPS, THR, GAP, MIN_LEN, DX_FRAC, CONF_MIN, PUPIL_THR, FULL_R = 60.0, 0.70, 5, 3, 0.3, 0.3, 0.527, 0.45
OUT = HERE / "results" / "blink_eyelid_distance"
PL = pd.read_csv(HERE / "results" / "labels_to_plateau.csv")
PL = PL[PL.status == "final"]
P4 = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right"]


def runs(mask, gap=0):
    i = np.flatnonzero(mask)
    if not len(i):
        return []
    cut = np.flatnonzero(np.diff(i) > gap + 1 if gap else np.diff(i) > 1)
    return list(zip(i[np.r_[0, cut + 1]], i[np.r_[cut, len(i) - 1]]))


def near(mask, gap):
    """True where a True of mask is within gap frames."""
    k = np.ones(2 * gap + 1)
    return np.convolve(mask.astype(float), k, mode="same") > 0


def states(d):
    O = (d["eyelid_bottom"]["y"] - d["eyelid_top"]["y"]).to_numpy(float)
    r = O / np.nanmedian(O)
    w = float(np.nanmedian(np.hypot(d["eye_temporal_corner"]["x"] - d["eye_nasal_corner"]["x"], d["eye_temporal_corner"]["y"] - d["eye_nasal_corner"]["y"])))
    dx = np.abs((d["eyelid_top"]["x"] - d["eyelid_bottom"]["x"]).to_numpy(float)) / w
    conf = np.minimum(d["eyelid_top"]["likelihood"].to_numpy(float), d["eyelid_bottom"]["likelihood"].to_numpy(float))
    trusted = (O >= 0) & (dx <= DX_FRAC) & (conf >= CONF_MIN) & np.isfinite(r)
    pupil = np.stack([d[b]["likelihood"].to_numpy(float) for b in P4], 1).mean(1)
    A = np.nan_to_num(r, nan=np.inf) < THR
    B = trusted & A
    adj = ~trusted & near(B, GAP)
    C = B | adj
    D, E = C.copy(), C.copy()
    for a, b in runs(~trusted & ~adj):
        seg = pupil[a:b + 1]
        if np.median(seg) < PUPIL_THR:
            D[a:b + 1] = True
        if np.min(seg) < 0.30:
            E[a:b + 1] = True
    return {"A": A, "B": B, "C": C, "D": D, "E": E}, r, trusted


def events(closed, r, trusted):
    out = []
    for a, b in runs(closed, GAP):
        if b - a + 1 < MIN_LEN:
            continue
        seg_r = r[a:b + 1]
        full = (np.nanmin(seg_r) < FULL_R) or (~trusted[a:b + 1]).any()
        out.append((int(a), int(b), round((b - a + 1) / FPS * 1000), round(float(np.nanmin(seg_r)), 3), "full" if full else "partial"))
    return out


items = list(csv.DictReader(open(OUT / "blink_eyelid_spotcheck_all_videos.csv")))
truth = {it["item"]: it["human_verdict"] for it in items}
rows_ev, rows_sum, pred = [], [], {k: {} for k in "ABCDE"}
for rr in PL.itertuples():
    step = max(1, int(rr.labels_at_plateau) // 20)
    h5 = sorted(glob.glob(str(HERE / rr.unit / "training-data" / f"predictions_step{step:02d}" / "*snapshot*120*.h5")))
    d = pt.load(h5[-1])
    st, r, trusted = states(d)
    for it in items:
        if int(it["video"]) != int(rr.video):
            continue
        a, b = int(it["start"]), int(it["end"])
        for k in st:
            pred[k][it["item"]] = float(st[k][a:b + 1].mean()) >= 0.5
    ev = events(st["D"], r, trusted)
    rows_ev += [{"video": int(rr.video), "unit": rr.unit, "start": a, "end": b, "start_s": round(a / FPS, 2), "duration_ms": du, "min_rel_opening": mr, "depth": dep} for a, b, du, mr, dep in ev]
    dur = [e[2] for e in ev]
    rows_sum.append({"video": int(rr.video), "mouse": rr.mouse, "model_step": step, "minutes": round(len(d) / FPS / 60, 2),
                     "closure_events": len(ev), "per_minute": round(len(ev) / (len(d) / FPS / 60), 2), "full": sum(e[4] == "full" for e in ev),
                     "partial": sum(e[4] == "partial" for e in ev), "median_duration_ms": float(np.median(dur)) if dur else np.nan,
                     "closed_frames_pct": round(100 * float(st["D"].mean()), 2), "untrusted_frames_pct": round(100 * float((~trusted).mean()), 2),
                     "untrusted_frames_counted_closed_pct": round(100 * float((~trusted & st["D"]).mean()), 2)})
EV, SM = pd.DataFrame(rows_ev), pd.DataFrame(rows_sum)
EV.to_csv(OUT / "blink_final_events_all_videos.csv", index=False)
SM.to_csv(OUT / "blink_final_summary_all_videos.csv", index=False)

# ---- variants against the verdicts ----
closure_truth = {k: v.startswith(("eye closed", "partly")) for k, v in truth.items()}
full_truth = {k: v.startswith("eye closed") for k, v in truth.items()}
names = {"A": "A: r < 0.70 (first version)", "B": "B: trusted and r < 0.70", "C": "C: B + untrusted next to closed frames",
         "D": "D: C + isolated untrusted by pupil confidence (median < 0.527)  [final]", "E": "E: C + isolated untrusted by min pupil confidence < 0.30 (tuned on the sample)"}
rows = []
for k in "ABCDE":
    p = pred[k]
    flagged = [i for i in p if p[i]]
    tp = sum(closure_truth[i] for i in flagged)
    all_true = sum(closure_truth.values())
    rows.append({"variant": names[k], "items_predicted_closed": len(flagged), "of_which_real_closures": tp, "precision_pct": round(100 * tp / max(1, len(flagged)), 1),
                 "real_closures_in_sample": all_true, "recall_pct": round(100 * tp / all_true, 1),
                 "false_closures": len(flagged) - tp, "missed_closures": all_true - tp})
V = pd.DataFrame(rows)
V.to_csv(OUT / "blink_final_variants_on_spotcheck.csv", index=False)
pd.set_option("display.width", 250)
print(V.to_string(index=False)); print(); print(SM.to_string(index=False))

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(17, 5.5), constrained_layout=True)
x = np.arange(len(V))
ax[0].bar(x - 0.2, V.precision_pct, 0.4, color="#2A9D8F", label="precision: flagged items that are real closures")
ax[0].bar(x + 0.2, V.recall_pct, 0.4, color="#E9A03B", label="recall: real closures in the sample that are flagged")
for xi, (p_, r_) in enumerate(zip(V.precision_pct, V.recall_pct)):
    ax[0].text(xi - 0.2, p_ + 1, f"{p_:.0f}", ha="center", fontsize=9); ax[0].text(xi + 0.2, r_ + 1, f"{r_:.0f}", ha="center", fontsize=9)
ax[0].set_xticks(x); ax[0].set_xticklabels(["A\nr < 0.70", "B\ntrusted\n& r < 0.70", "C\nB + adjacent\nuntrusted", "D (final)\nC + pupil conf\nmedian < 0.527", "E\nC + pupil conf\nmin < 0.30"], fontsize=9)
ax[0].set_ylim(0, 110); ax[0].set_ylabel("%"); ax[0].grid(axis="y", alpha=0.3); ax[0].legend(fontsize=8.5, loc="lower right")
ax[0].set_title(f"Rule variants against the labeler's spot check ({len(items)} items, closure = fully or partly closed)", fontsize=10.5)
xv = np.arange(len(SM))
ax[1].bar(xv, SM.full, color="#1D3557", label="full closures")
ax[1].bar(xv, SM.partial, bottom=SM.full, color="#A8DADC", label="partial closures (lid covers part of the pupil)")
ax[1].set_xticks(xv); ax[1].set_xticklabels([f"v{v}" for v in SM.video]); ax[1].set_ylabel("closure events in the video"); ax[1].legend(fontsize=9); ax[1].grid(axis="y", alpha=0.3)
for xi, (n, pm) in enumerate(zip(SM.closure_events, SM.per_minute)):
    ax[1].text(xi, n + 1, f"{pm:.1f}/min", ha="center", fontsize=8)
ax[1].set_title("Final rule D on the whole videos: closure events per video", fontsize=10.5)
fig.savefig(OUT / "blink_final_summary.png", dpi=130)
