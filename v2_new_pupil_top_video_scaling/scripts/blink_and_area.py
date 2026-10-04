#!/usr/bin/env python3
"""Pupil area and blink detection on all finished videos (user request 2026-10-04), read-only analysis.

    python blink_and_area.py

Goal: the most accurate rule for the pupil area and the most accurate rule for detecting eye closure, checked on every
finished video of the v2 line (labels_to_plateau.csv, status = final) with one script and one set of formulas.

Model of a video = the model at its plateau point (step = labels_at_plateau / 20; video 7, plateau at 0 labels: step 1,
because the 0-label model has no whole-video prediction). Its whole-video prediction (every frame, final snapshot,
no confidence cut-off) is in <unit>/training-data/predictions_stepNN/.

A. Area. Frames: the 50 frozen test frames of each video whose four pupil points are all labeled.
   Truth A = pi/4 * |xR - xL| * |yB - yT| from the human points. Rules on the model's points (image y points down):
     four          pi/4 * w * h                      w = |xR - xL|, h = |yB - yT|
     no_top        pi/4 * w * 2 (yB - (yL + yR)/2)   (the rule of pupil_trace.py)
     no_bottom     pi/4 * w * 2 ((yL + yR)/2 - yT)
     no_left       pi/4 * 2 (xR - (xT + xB)/2) * h
     no_right      pi/4 * 2 ((xT + xB)/2 - xL) * h
     drop_lowconf  per frame, the no_* rule of the pupil point with the lowest confidence
     four_fallback four when all four confidences >= 0.6, otherwise drop_lowconf
     four_med5     centred 5-frame median of the `four` trace
     four_med31    centred 31-frame (0.5 s) median of the `four` trace
   Error of a frame = 100 * |A_hat - A| / A; per video and pooled: median, 90th percentile, median signed error.

B. Eye closure. Truth from the human labels: closed = all four pupil points left empty, open = all four labeled
   (frames with 1-3 pupil points are not used). Only frames the predicting model was not trained on: test50 and val20
   (plateau model) and batches N >= 2 (the model of step N-1, which selected them). Signals per frame:
     open_rel5   eye opening / its centred 5-s rolling median     (eye opening = yEB - yET)
     open_relv   eye opening / the median opening of the video
     min_conf    lowest confidence of the four pupil points
     mean_conf   mean confidence of the four pupil points
     axes_off    distance between the midpoints of the two pupil axes / w
     aspect      h / w
   AUC of each signal (closed vs open; ties count 1/2), per video and pooled. For each signal a threshold rule
   "closed if signal < t" (axes_off: > t); t maximizes hit rate - false-alarm rate on the pooled frames, and
   leave-one-video-out (t chosen without the video it is applied to) gives the reported hit and false-alarm rates.
   The production rule of pupil_trace.py is scored on the same frames.

C. Time series. Per video (plateau model, whole video): closure events by the best rule of B (flagged frames closer
   than 0.25 s merged), events per minute, median duration, share of frames flagged, share flagged by the production
   rule, relative spread of the `four` area on unflagged frames (1.4826 * MAD / median) and its correlation with the
   eye opening.

Outputs, all in results/blink_and_area/ (file -> content):
  blink_area_area_methods_all_videos.csv/.png      part A
  blink_area_closure_frames_all_videos.csv         part B, one row per human-labeled frame with its signals
  blink_area_closure_signals_all_videos.csv/.png   part B, AUC per signal and video / figure
  blink_area_closure_rules_all_videos.csv          part B, threshold rules
  blink_area_closed_frames_all_videos.jpg          part B, every human closed-eye frame
  blink_area_timeseries_videoNN.png                part C, one figure per video (NN = video index)
  blink_area_timeseries_all_videos.csv/.png        part C, summary table / all videos in one figure
No existing file is modified."""
import glob
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pupil_trace as pt  # noqa: E402
from scale_step import HERE, flat, human_table  # noqa: E402

FPS = 60.0
PCUT = 0.6
GAP = 15                                    # frames; flagged frames closer than this are one event
P4 = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right"]
DROP = {"pupil_top": "no_top", "pupil_bottom": "no_bottom", "pupil_left": "no_left", "pupil_right": "no_right"}
RULES = ["four", "no_top", "no_bottom", "no_left", "no_right", "drop_lowconf", "four_fallback", "four_med5", "four_med31"]
OUT = HERE / "results" / "blink_and_area"
OUT.mkdir(exist_ok=True)
PL = pd.read_csv(HERE / "results" / "labels_to_plateau.csv")
PL = PL[PL.status == "final"]


def fidx(name):
    return int(re.findall(r"\d+", Path(str(name)).stem)[0])


def whole_pred(unit, step):
    h = sorted(glob.glob(str(HERE / unit / "training-data" / f"predictions_step{step:02d}" / "*snapshot*120*.h5")))
    return h[-1] if h else None


def areas(d, conf=None):
    """Area rules on a flat (bodypart, coord) table; `conf` (frames x 4, order P4) adds the confidence-based rules."""
    T, B, L, R = (d[b] for b in P4)
    w, h = (R["x"] - L["x"]).abs(), (B["y"] - T["y"]).abs()
    cy, cx = (L["y"] + R["y"]) / 2, (T["x"] + B["x"]) / 2
    a = pd.DataFrame(index=d.index)
    a["four"] = np.pi / 4 * w * h
    a["no_top"] = np.pi / 4 * w * 2 * (B["y"] - cy)
    a["no_bottom"] = np.pi / 4 * w * 2 * (cy - T["y"])
    a["no_left"] = np.pi / 4 * 2 * (R["x"] - cx) * h
    a["no_right"] = np.pi / 4 * 2 * (cx - L["x"]) * h
    if conf is not None:
        low = np.array([DROP[P4[i]] for i in np.argmin(conf, axis=1)])
        a["drop_lowconf"] = [a[r].iloc[i] for i, r in enumerate(low)]
        a["four_fallback"] = np.where(conf.min(axis=1) >= PCUT, a["four"], a["drop_lowconf"])
        a["four_med5"] = a["four"].rolling(5, center=True, min_periods=1).median()
        a["four_med31"] = a["four"].rolling(31, center=True, min_periods=1).median()
    return a


def signals(d):
    T, B, L, R = (d[b] for b in P4)
    w = (R["x"] - L["x"]).abs().to_numpy(float)
    h = (B["y"] - T["y"]).abs().to_numpy(float)
    c1 = np.c_[(L["x"] + R["x"]) / 2, (L["y"] + R["y"]) / 2]
    c2 = np.c_[(T["x"] + B["x"]) / 2, (T["y"] + B["y"]) / 2]
    opening = (d["eyelid_bottom"]["y"] - d["eyelid_top"]["y"]).to_numpy(float)
    conf = np.stack([d[b]["likelihood"].to_numpy(float) for b in P4], axis=1)
    _, _, _, info = pt.pupil_trace(d, fps=FPS)
    wn = np.where(w > 1, w, np.nan)
    return pd.DataFrame({"open_rel5": info["rel_open"], "open_relv": opening / np.nanmedian(opening), "min_conf": conf.min(axis=1),
                         "mean_conf": conf.mean(axis=1), "axes_off": np.hypot(*(c1 - c2).T) / wn, "aspect": h / wn,
                         "production": info["bad"].astype(float), "opening": opening}), conf


def auc(c, o, low):
    c, o = np.asarray(c, float), np.asarray(o, float)
    c, o = c[np.isfinite(c)], o[np.isfinite(o)]
    if not len(c) or not len(o):
        return np.nan
    if low:
        c, o = -c, -o
    return float(((c[:, None] > o[None, :]).sum() + 0.5 * (c[:, None] == o[None, :]).sum()) / (len(c) * len(o)))


def best_threshold(c, o, low):
    """Threshold maximizing hit rate - false-alarm rate; midpoints between neighbouring observed values."""
    c, o = np.asarray(c, float), np.asarray(o, float)
    c, o = c[np.isfinite(c)], o[np.isfinite(o)]
    v = np.unique(np.r_[c, o])
    cand = (v[:-1] + v[1:]) / 2
    j = [((c < t).mean() - (o < t).mean()) if low else ((c > t).mean() - (o > t).mean()) for t in cand]
    return float(cand[int(np.argmax(j))])


cache = {}


def pred(unit, step):
    k = (unit, step)
    if k not in cache:
        d = pt.load(whole_pred(unit, step))
        cache[k] = (d, *signals(d))
    return cache[k]


# ---------------- A. area on every test set ----------------
rowsA, perframe = [], []
for r in PL.itertuples():
    step = max(1, int(r.labels_at_plateau) // 20)
    d, S, conf = pred(r.unit, step)
    gt = flat(human_table(HERE / r.unit / "training-data" / "labels" / "test50"))
    ok = pd.concat([gt[(b, c)].notna() for b in P4 for c in ("x", "y")], axis=1).all(axis=1)
    gt = gt[ok]
    idx = [fidx(n) for n in gt.index]
    truth = areas(gt)["four"].to_numpy(float)
    pa = areas(d, conf).iloc[idx]
    for rule in RULES:
        e = (pa[rule].to_numpy(float) - truth) / truth * 100
        rowsA.append({"video": r.video, "unit": r.unit, "model_step": step, "rule": rule, "n": len(e),
                      "median_abs_err_pct": round(float(np.median(np.abs(e))), 2), "p90_abs_err_pct": round(float(np.percentile(np.abs(e), 90)), 2),
                      "median_signed_err_pct": round(float(np.median(e)), 2)})
        perframe += [{"video": r.video, "rule": rule, "err": x} for x in e]
PF = pd.DataFrame(perframe)
for rule in RULES:
    e = PF[PF.rule == rule].err.to_numpy()
    rowsA.append({"video": "all", "unit": "all videos pooled", "model_step": "", "rule": rule, "n": len(e),
                  "median_abs_err_pct": round(float(np.median(np.abs(e))), 2), "p90_abs_err_pct": round(float(np.percentile(np.abs(e), 90)), 2),
                  "median_signed_err_pct": round(float(np.median(e)), 2)})
A = pd.DataFrame(rowsA)
A.to_csv(OUT / "blink_area_area_methods_all_videos.csv", index=False)

# ---------------- B. closure signals vs human closed / open ----------------
SIG = [("open_rel5", True), ("open_relv", True), ("min_conf", True), ("mean_conf", True), ("axes_off", False), ("aspect", True)]
rec = []
for r in PL.itertuples():
    step = max(1, int(r.labels_at_plateau) // 20)
    labs = HERE / r.unit / "training-data" / "labels"
    sets = [("test50", step), ("val20", step)] + [(b.name, int(b.name[5:]) - 1) for b in sorted(labs.glob("batch*")) if int(b.name[5:]) >= 2]
    for s, st in sets:
        if not glob.glob(str(labs / s / "CollectedData_*.h5")) or not whole_pred(r.unit, st):
            continue
        _, S, _ = pred(r.unit, st)
        for name, row in flat(human_table(labs / s)).iterrows():
            k = sum(np.isfinite(row[(b, "x")]) for b in P4)
            if k in (0, 4):
                rec.append({"video": r.video, "unit": r.unit, "set": s, "model_step": st, "frame": fidx(name), "closed": k == 0,
                            **S.iloc[fidx(name)].round(4).to_dict()})
F = pd.DataFrame(rec)
F.to_csv(OUT / "blink_area_closure_frames_all_videos.csv", index=False)
rowsB = []
for v, g in list(F.groupby("video", sort=False)) + [("all", F)]:
    c, o = g[g.closed], g[~g.closed]
    rowsB.append({"video": v, "closed_frames": len(c), "open_frames": len(o), **{f"AUC_{s}": round(auc(c[s], o[s], low), 3) for s, low in SIG}})
B = pd.DataFrame(rowsB)
B.to_csv(OUT / "blink_area_closure_signals_all_videos.csv", index=False)

rowsR = []
C_, O_ = F[F.closed], F[~F.closed]
for s, low in SIG:
    t_all = best_threshold(C_[s], O_[s], low)
    hit = fa = 0
    for v in F.video.unique():
        tr = F[F.video != v]
        t = best_threshold(tr[tr.closed][s], tr[~tr.closed][s], low)
        te = F[F.video == v]
        flag = (te[s] < t) if low else (te[s] > t)
        hit += int((flag & te.closed).sum()); fa += int((flag & ~te.closed).sum())
    rowsR.append({"rule": f"{s} {'<' if low else '>'} t", "threshold_all_frames": round(t_all, 3), "AUC": round(auc(C_[s], O_[s], low), 3),
                  "hits_of_closed": hit, "closed": len(C_), "false_alarms_of_open": fa, "open": len(O_),
                  "hit_rate": round(hit / len(C_), 3), "false_alarm_rate": round(fa / len(O_), 3), "thresholds": "leave one video out"})
for name, flag in [("production rule (pupil_trace.py)", F.production == 1), ("min_conf < 0.6", F.min_conf < PCUT)]:
    rowsR.append({"rule": name, "threshold_all_frames": "", "AUC": "", "hits_of_closed": int((flag & F.closed).sum()), "closed": len(C_),
                  "false_alarms_of_open": int((flag & ~F.closed).sum()), "open": len(O_), "hit_rate": round(float(flag[F.closed].mean()), 3),
                  "false_alarm_rate": round(float(flag[~F.closed].mean()), 3), "thresholds": "fixed"})
RU = pd.DataFrame(rowsR)
RU.to_csv(OUT / "blink_area_closure_rules_all_videos.csv", index=False)
best = max(SIG, key=lambda s: auc(C_[s[0]], O_[s[0]], s[1]))
BEST, BEST_LOW = best
THR = best_threshold(C_[BEST], O_[BEST], BEST_LOW)

# contact sheet of every human closed-eye frame used in B (to check what "all four pupil points empty" looks like)
tiles = []
for q in C_.itertuples():
    im = cv2.imread(str(HERE / q.unit / "training-data" / "labels" / q.set / f"img{q.frame:06d}.png"))
    if im is None:
        continue
    im = cv2.resize(im, (464, 368))
    cv2.putText(im, f"video {q.video} {q.set} frame {q.frame}  min_conf {q.min_conf:.2f}  open {q.open_relv:.2f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    tiles.append(im)
if tiles:
    cols = 6
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    cv2.imwrite(str(OUT / "blink_area_closed_frames_all_videos.jpg"), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 85])

# ---------------- C. time series per video ----------------
rowsC, traces = [], []
for r in PL.itertuples():
    step = max(1, int(r.labels_at_plateau) // 20)
    d, S, conf = pred(r.unit, step)
    a4 = areas(d)["four"].to_numpy(float)
    flag = (S[BEST].to_numpy() < THR) if BEST_LOW else (S[BEST].to_numpy() > THR)
    flag = flag | ~np.isfinite(S[BEST].to_numpy())
    i = np.flatnonzero(flag)
    if len(i):
        cut = np.flatnonzero(np.diff(i) > GAP)
        starts, ends = i[np.r_[0, cut + 1]], i[np.r_[cut, len(i) - 1]]
    else:
        starts = ends = np.array([], int)
    dur = (ends - starts + 1) / FPS * 1000
    ok = ~flag & np.isfinite(a4)
    med = np.median(a4[ok])
    t = np.arange(len(d)) / FPS
    rowsC.append({"video": r.video, "unit": r.unit, "mouse": r.mouse, "model_step": step, "frames": len(d), "minutes": round(len(d) / FPS / 60, 2),
                  "rule": f"{BEST} {'<' if BEST_LOW else '>'} {THR:.3f}", "flagged_frames_pct": round(100 * flag.mean(), 2), "events": len(starts),
                  "events_per_min": round(len(starts) / (len(d) / FPS / 60), 2), "median_event_ms": round(float(np.median(dur)), 0) if len(dur) else np.nan,
                  "production_flagged_pct": round(100 * float(S.production.mean()), 2), "min_conf_lt_0.6_pct": round(100 * float((S.min_conf < PCUT).mean()), 2),
                  "area_median_px2": round(float(med), 0), "area_rel_spread_pct": round(100 * 1.4826 * float(np.median(np.abs(a4[ok] - med))) / med, 1),
                  "corr_area_opening": round(float(np.corrcoef(a4[ok], S.opening.to_numpy()[ok])[0, 1]), 2)})
    traces.append((r, t, a4 / med, flag, S))

    fig, ax = plt.subplots(4, 1, figsize=(20, 10), sharex=True, constrained_layout=True)
    shade = lambda x: x.fill_between(t, 0, 1, where=flag, transform=x.get_xaxis_transform(), color="#E76F51", alpha=0.35, lw=0)
    ax[0].plot(t, np.where(flag, np.nan, a4), lw=0.5, color="#1D3557"); shade(ax[0])
    ax[0].set_ylim(0, np.nanpercentile(a4[ok], 99.9) * 1.15); ax[0].set_ylabel("pupil area, rule `four` (px$^2$)")
    ax[1].plot(t, S.opening, lw=0.5, color="#2A9D8F"); shade(ax[1]); ax[1].set_ylabel("eye opening (px)")
    ax[2].plot(t, S.min_conf, lw=0.5, color="0.3"); ax[2].axhline(THR if BEST == "min_conf" else PCUT, color="#E76F51", ls="--", lw=0.9)
    ax[2].set_ylim(0, 1.02); ax[2].set_ylabel("lowest pupil confidence")
    cxs = (d["pupil_left"]["x"] + d["pupil_right"]["x"]) / 2
    cys = (d["pupil_top"]["y"] + d["pupil_bottom"]["y"]) / 2
    ax[3].plot(t, np.where(flag, np.nan, cxs - np.nanmedian(cxs)), lw=0.5, color="#457B9D", label="x")
    ax[3].plot(t, np.where(flag, np.nan, cys - np.nanmedian(cys)), lw=0.5, color="#E9C46A", label="y")
    ax[3].set_ylim(-80, 80); ax[3].set_ylabel("pupil centre - median (px)"); ax[3].legend(loc="upper right", ncol=2, fontsize=9); ax[3].set_xlabel("time (s)")
    c = rowsC[-1]
    fig.suptitle(f"Video {r.video} ({r.unit}), model of step {step}: closure rule {c['rule']} flags {c['flagged_frames_pct']}% of frames in {c['events']} events "
                 f"({c['events_per_min']} per min, median {c['median_event_ms']:.0f} ms); red = flagged", fontsize=12)
    fig.savefig(OUT / f"blink_area_timeseries_video{int(r.video):02d}.png", dpi=110)
    plt.close(fig)
C = pd.DataFrame(rowsC)
C.to_csv(OUT / "blink_area_timeseries_all_videos.csv", index=False)

# ---------------- figures ----------------
NAMES = {"four": "four points", "no_top": "no top\n(production)", "no_bottom": "no bottom", "no_left": "no left", "no_right": "no right",
         "drop_lowconf": "drop lowest\nconfidence", "four_fallback": "four, fallback\nif conf < 0.6", "four_med5": "four, 5-frame\nmedian", "four_med31": "four, 0.5-s\nmedian"}
fig, axes = plt.subplots(1, 2, figsize=(19, 6), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.25]})
pool = A[A.video == "all"].set_index("rule").loc[RULES]
ax = axes[0]
ax.bar(range(len(RULES)), pool.median_abs_err_pct, color=["#2A9D8F" if r == "four" else "0.6" for r in RULES])
ax.plot(range(len(RULES)), pool.p90_abs_err_pct, "k_", ms=18, mew=1.5, label="90th percentile")
for i, v in enumerate(pool.median_abs_err_pct):
    ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=10)
ax.set_xticks(range(len(RULES))); ax.set_xticklabels([NAMES[r] for r in RULES], fontsize=9)
ax.set_ylabel("area error vs human four-point area (%)"); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
ax.set_title(f"A. All {PL.video.nunique()} videos pooled ({int(pool.n.iloc[0])} test frames): bar = median, dash = 90th percentile", fontsize=11)
ax = axes[1]
per = A[A.video != "all"]
x = np.arange(PL.video.nunique())
for rule, col, off in [("four", "#2A9D8F", -0.27), ("no_top", "0.6", 0), ("four_med31", "#264653", 0.27)]:
    ax.bar(x + off, per[per.rule == rule].median_abs_err_pct, 0.27, color=col, label=NAMES[rule].replace("\n", " "))
ax.set_xticks(x); ax.set_xticklabels([f"video {v}" for v in PL.video], fontsize=9); ax.set_ylabel("median area error (%)")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3); ax.set_title("B. Per video (50 test frames each, model at the plateau point)", fontsize=11)
fig.savefig(OUT / "blink_area_area_methods_all_videos.png", dpi=130)
plt.close(fig)

SN = {"open_rel5": "eye opening /\n5-s median", "open_relv": "eye opening /\nvideo median", "min_conf": "lowest pupil\nconfidence",
      "mean_conf": "mean pupil\nconfidence", "axes_off": "pupil axes\nmidpoint offset", "aspect": "pupil\nheight / width"}
fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
ax = axes[0]
allr = B[B.video == "all"].iloc[0]
vals = [allr[f"AUC_{s}"] for s, _ in SIG]
ax.bar(range(len(SIG)), vals, color=["#2A9D8F" if s == BEST else "0.6" for s, _ in SIG])
for i, v in enumerate(vals):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
ax.axhline(0.5, color="0.4", ls="--", lw=0.8)
ax.set_xticks(range(len(SIG))); ax.set_xticklabels([SN[s] for s, _ in SIG], fontsize=9); ax.set_ylim(0, 1.08)
ax.set_ylabel("AUC, human closed vs open frames"); ax.grid(axis="y", alpha=0.3)
ax.set_title(f"A. Signals, pooled: {allr.closed_frames} closed / {allr.open_frames} open frames", fontsize=11)
ax = axes[1]
lo, hi = np.log10(max(F[BEST].min(), 1e-4)), np.log10(F[BEST].max())
bins = np.logspace(lo, hi, 40) if BEST.endswith("conf") else np.linspace(F[BEST].min(), F[BEST].max(), 40)
ax.hist(O_[BEST], bins=bins, color="0.6", label=f"open ({len(O_)})")
ax.hist(C_[BEST], bins=bins, color="#E76F51", label=f"closed ({len(C_)})")
ax.axvline(THR, color="k", ls="--", lw=1, label=f"threshold {THR:.3f}")
if BEST.endswith("conf"):
    ax.set_xscale("log")
ax.set_yscale("log"); ax.set_xlabel(SN[BEST].replace("\n", " ")); ax.set_ylabel("frames (log scale)"); ax.legend(fontsize=9)
ax.set_title("B. Best signal: values on closed and open frames", fontsize=11)
ax = axes[2]
lab = [q.rule.replace(" t", "") + ("\n(leave one video out)" if q.thresholds != "fixed" else "") for q in RU.itertuples()]
y = np.arange(len(RU))[::-1]
ax.barh(y + 0.2, RU.hit_rate * 100, 0.4, color="#2A9D8F", label="closed frames detected (%)")
ax.barh(y - 0.2, RU.false_alarm_rate * 100, 0.4, color="#E76F51", label="open frames flagged (%)")
for yi, q in zip(y, RU.itertuples()):
    ax.text(q.hit_rate * 100 + 1, yi + 0.2, f"{q.hits_of_closed}/{q.closed}", va="center", fontsize=8.5)
    ax.text(q.false_alarm_rate * 100 + 1, yi - 0.2, f"{q.false_alarms_of_open}/{q.open}", va="center", fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=8.5); ax.set_xlim(0, 112); ax.legend(fontsize=9, loc="lower right"); ax.grid(axis="x", alpha=0.3)
ax.set_title("C. Threshold rules on the same frames", fontsize=11)
fig.savefig(OUT / "blink_area_closure_signals_all_videos.png", dpi=130)
plt.close(fig)

fig, axes = plt.subplots(len(traces), 1, figsize=(20, 1.5 * len(traces) + 1), sharex=True, constrained_layout=True)
for ax, (r, t, rel, flag, S), c in zip(axes, traces, rowsC):
    ax.plot(t / 60, np.where(flag, np.nan, rel), lw=0.4, color="#1D3557")
    ax.fill_between(t / 60, 0, 1, where=flag, transform=ax.get_xaxis_transform(), color="#E76F51", alpha=0.5, lw=0)
    ax.set_ylim(0, 2.5); ax.set_yticks([0, 1, 2])
    ax.set_ylabel(f"video {r.video}\n{r.mouse}", fontsize=9, rotation=0, ha="right", va="center")
    ax.text(0.995, 0.86, f"{c['events']} events, {c['flagged_frames_pct']}% of frames flagged; area spread {c['area_rel_spread_pct']}%", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, bbox=dict(fc="w", ec="none", alpha=0.8, pad=1))
axes[-1].set_xlabel("time (min)")
fig.suptitle(f"Pupil area (rule `four`) / its median in every video; red = frames flagged as eye closure ({rowsC[0]['rule']})", fontsize=12)
fig.savefig(OUT / "blink_area_timeseries_all_videos.png", dpi=110)
plt.close(fig)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(A[A.video == "all"].to_string(index=False)); print()
print(A[A.video != "all"].pivot(index="video", columns="rule", values="median_abs_err_pct")[RULES].to_string()); print()
print(B.to_string(index=False)); print(); print(RU.to_string(index=False)); print(); print(C.drop(columns=["unit"]).to_string(index=False))
print("best closure signal:", BEST, "threshold", round(THR, 4))
