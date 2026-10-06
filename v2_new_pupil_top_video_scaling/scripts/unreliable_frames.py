#!/usr/bin/env python3
"""Which frames of a video are predicted unreliably? Check of the jump rule and of the confidence on human-labeled frames
(direction set by Kaiwen 2026-09-23 and asked again 2026-10-04), read-only analysis.

    python unreliable_frames.py

Frames: the 50 test frames and the 20 validation frames of every finished video (results/labels_to_plateau.csv). They are
evenly spaced in time (not picked by any model), have human labels and were never trained on.
Model of a video = the model at its plateau point (step = labels_at_plateau / 20; plateau at 0 labels: step 1), its
whole-video prediction (final snapshot, no confidence cut-off).

Truth of a frame: frame RMSE = sqrt(mean of squared point errors over the human-labeled keypoints), as in scale_step.py.
  wrong50 = frame RMSE > 50 px (the "frames > 50 px" of every results table);  wrong20 = frame RMSE > 20 px.
Signals of a frame, all label-free:
  jump        largest displacement of any keypoint from the previous frame (px) / median eye width of the video;
              the jump rule of select_frames.py flags a frame when jump > 0.03
  min_conf    lowest confidence of the 8 keypoints;   mean_conf  mean confidence of the 8 keypoints
  fitting     DLC's "fitting" outlier detector re-implemented: per keypoint coordinate a SARIMAX(3,0,1) model is fitted to
              the whole video (points with confidence < 0.01 set to missing, as in DLC); deviation of a keypoint = distance
              between its one-step prediction and its predicted position; fitting = mean deviation over the 8 keypoints (px);
              DLC flags a frame when it exceeds epsilon = 20 px. Deviations are cached in fitting_dev_videoNN.npy.
For every signal: AUC (wrong vs not wrong; ties 1/2) and Spearman correlation with the frame RMSE.
For threshold rules (jump > 0.03; min_conf < 0.6; mean_conf < t; jump > 0.03 OR min_conf < 0.6; fitting > 20 px):
  flagged share, precision = wrong frames among the flagged, recall = flagged among the wrong frames.
  t of mean_conf is chosen leave-one-video-out (maximizing recall - false-alarm rate for wrong20).
Outputs in results/unreliable_frames/: unreliable_frames_all_videos.csv (one row per frame),
unreliable_rules_all_videos.csv, unreliable_signals_all_videos.csv, unreliable_frames_all_videos.png. Nothing else is modified."""
import glob
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pupil_trace as pt  # noqa: E402
from scale_step import HERE, flat, human_table  # noqa: E402

OUT = HERE / "results" / "unreliable_frames"
OUT.mkdir(exist_ok=True)
EPS_FRAC, PCUT, FIT_EPS, FIT_PBOUND = 0.03, 0.6, 20.0, 0.01
PL = pd.read_csv(HERE / "results" / "labels_to_plateau.csv")
PL = PL[PL.status == "final"]


def fidx(name):
    return int(re.findall(r"\d+", Path(str(name)).stem)[0])


def fitting_deviation(d, bps, cache):
    """Mean over keypoints of |one-step SARIMAX(3,0,1) prediction - position| per frame (DLC 'fitting', ARdegree 3, MAdegree 1)."""
    if cache.exists():
        return np.load(cache)
    import warnings
    dev = np.zeros((len(d), len(bps)))
    for k, b in enumerate(bps):
        pred = {}
        for c in ("x", "y"):
            y = d[b][c].to_numpy(float).copy()
            y[d[b]["likelihood"].to_numpy(float) < FIT_PBOUND] = np.nan
            m = np.nanmean(y)                       # fit the centred series: SARIMAX has no intercept, and on the raw series
            with warnings.catch_warnings():         # the optimizer failed on 3 videos (prediction 0 -> constant deviation)
                warnings.simplefilter("ignore")
                res = SARIMAX(y - m, order=(3, 0, 1), seasonal_order=(0, 0, 0, 0), simple_differencing=True).fit(disp=False)
                if not res.mle_retvals.get("converged", True):
                    res = SARIMAX(y - m, order=(3, 0, 1), seasonal_order=(0, 0, 0, 0), simple_differencing=True).fit(disp=False, method="powell", maxiter=500)
                pred[c] = np.asarray(res.get_prediction(end=res.model.nobs - 1).predicted_mean) + m
        n = min(len(pred["x"]), len(d))
        dev[-n:, k] = np.hypot(d[b]["x"].to_numpy(float)[-n:] - pred["x"][-n:], d[b]["y"].to_numpy(float)[-n:] - pred["y"][-n:])
    out = np.nanmean(dev, axis=1)
    np.save(cache, out)
    return out


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return np.nan
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


rows = []
for r in PL.itertuples():
    step = max(1, int(r.labels_at_plateau) // 20)
    h5 = sorted(glob.glob(str(HERE / r.unit / "training-data" / f"predictions_step{step:02d}" / "*snapshot*120*.h5")))
    if not h5:
        print("no whole-video prediction:", r.unit); continue
    d = pt.load(h5[-1])
    bps = list(d.columns.get_level_values(0).unique())
    width = float(np.nanmedian(np.hypot(d["eye_temporal_corner"]["x"] - d["eye_nasal_corner"]["x"], d["eye_temporal_corner"]["y"] - d["eye_nasal_corner"]["y"])))
    jump = np.zeros(len(d))
    for b in bps:
        jump = np.fmax(jump, np.nan_to_num(np.hypot(np.diff(d[b]["x"], prepend=np.nan), np.diff(d[b]["y"], prepend=np.nan))))
    conf = np.stack([d[b]["likelihood"].to_numpy(float) for b in bps], axis=1)
    fit = fitting_deviation(d, bps, OUT / f"fitting_dev_video{int(r.video):02d}.npy")
    for s in ("test50", "val20"):
        g = flat(human_table(HERE / r.unit / "training-data" / "labels" / s))
        for name, row in g.iterrows():
            i = fidx(name)
            e2 = [(d[b]["x"].iat[i] - row[(b, "x")]) ** 2 + (d[b]["y"].iat[i] - row[(b, "y")]) ** 2 for b in bps if np.isfinite(row[(b, "x")])]
            if not e2:
                continue
            rows.append({"video": int(r.video), "unit": r.unit, "set": s, "frame": i, "model_step": step, "labeled_points": len(e2),
                         "frame_rmse_px": round(float(np.sqrt(np.mean(e2))), 2), "jump": round(float(jump[i] / width), 4),
                         "min_conf": round(float(conf[i].min()), 4), "mean_conf": round(float(conf[i].mean()), 4), "fitting": round(float(fit[i]), 2)})
F = pd.DataFrame(rows)
F["wrong50"], F["wrong20"] = F.frame_rmse_px > 50, F.frame_rmse_px > 20
F["jump_rule"], F["minconf_rule"], F["fitting_rule"] = F.jump > EPS_FRAC, F.min_conf < PCUT, F.fitting > FIT_EPS
F.to_csv(OUT / "unreliable_frames_all_videos.csv", index=False)

SIG = [("jump", 1), ("min_conf", -1), ("mean_conf", -1), ("fitting", 1)]            # sign: +1 = larger means wrong
sig_rows = []
for v, g in list(F.groupby("video")) + [("all", F)]:
    row = {"video": v, "frames": len(g), "wrong50": int(g.wrong50.sum()), "wrong20": int(g.wrong20.sum())}
    for s, sg in SIG:
        for w in ("wrong50", "wrong20"):
            row[f"AUC_{s}_{w}"] = round(auc(sg * g[g[w]][s], sg * g[~g[w]][s]), 3)
        row[f"spearman_{s}"] = round(float(spearmanr(g[s], g.frame_rmse_px)[0]), 2)
    sig_rows.append(row)
S = pd.DataFrame(sig_rows)
S.to_csv(OUT / "unreliable_signals_all_videos.csv", index=False)


def best_t(g):
    v = np.unique(g.mean_conf); cand = (v[:-1] + v[1:]) / 2
    j = [(g[g.wrong20].mean_conf < t).mean() - (g[~g.wrong20].mean_conf < t).mean() for t in cand]
    return float(cand[int(np.argmax(j))])


F["meanconf_rule"] = False
for v in F.video.unique():
    F.loc[F.video == v, "meanconf_rule"] = F[F.video == v].mean_conf < best_t(F[F.video != v])
T_ALL = best_t(F)
F["either_rule"] = F.jump_rule | F.minconf_rule
rule_rows = []
for name, col in [(f"jump > {EPS_FRAC} of the eye width (rule of select_frames.py)", "jump_rule"), (f"lowest confidence < {PCUT}", "minconf_rule"),
                  (f"mean confidence < t (leave one video out; {T_ALL:.3f} on all frames)", "meanconf_rule"), ("jump rule OR lowest confidence < 0.6", "either_rule"),
                  (f"fitting: mean SARIMAX deviation > {FIT_EPS:.0f} px (DLC 'fitting' detector)", "fitting_rule")]:
    for w in ("wrong50", "wrong20"):
        fl = F[col]
        rule_rows.append({"rule": name, "wrong_means": "frame RMSE > " + w[-2:] + " px", "frames": len(F), "wrong": int(F[w].sum()), "flagged": int(fl.sum()),
                          "flagged_pct": round(100 * fl.mean(), 1), "flagged_and_wrong": int((fl & F[w]).sum()),
                          "precision_pct": round(100 * (fl & F[w]).sum() / max(1, fl.sum()), 1), "recall_pct": round(100 * (fl & F[w]).sum() / max(1, F[w].sum()), 1)})
R = pd.DataFrame(rule_rows)
R.to_csv(OUT / "unreliable_rules_all_videos.csv", index=False)
F.to_csv(OUT / "unreliable_frames_all_videos.csv", index=False)

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 4, figsize=(27, 6), constrained_layout=True)
for a, s, xl, thr in ((ax[0], "jump", "jump: largest keypoint displacement from the previous frame / eye width", EPS_FRAC), (ax[1], "min_conf", "lowest confidence of the 8 keypoints", PCUT),
                      (ax[2], "fitting", "fitting: mean deviation from the SARIMAX one-step prediction (px)", FIT_EPS)):
    ok = F[~F.wrong20]; mid = F[F.wrong20 & ~F.wrong50]; bad = F[F.wrong50]
    lg = s in ("jump", "fitting"); lo = 1e-3 if s == "jump" else 1e-1
    a.scatter(ok[s].clip(lower=lo) if lg else ok[s], ok.frame_rmse_px, s=12, color="0.6", alpha=0.6, lw=0, label=f"frame RMSE <= 20 px ({len(ok)})")
    a.scatter(mid[s].clip(lower=lo) if lg else mid[s], mid.frame_rmse_px, s=18, color="#E9A03B", alpha=0.9, lw=0, label=f"20-50 px ({len(mid)})")
    a.scatter(bad[s].clip(lower=lo) if lg else bad[s], bad.frame_rmse_px, s=22, color="#D1495B", alpha=0.95, lw=0, label=f"> 50 px ({len(bad)})")
    a.axvline(thr, color="k", ls="--", lw=1); a.axhline(20, color="0.5", ls=":", lw=0.8); a.axhline(50, color="0.5", ls=":", lw=0.8)
    a.set_yscale("log"); a.set_ylabel("frame RMSE against the human labels (px, log scale)"); a.set_xlabel(xl); a.grid(alpha=0.3); a.legend(fontsize=9)
    if s == "jump":
        a.set_xscale("log"); a.set_title(f"A. Jump of a frame against its error; dashed = rule threshold {thr} (flagged to the right)", fontsize=11)
    elif s == "fitting":
        a.set_xscale("log"); a.set_title(f"C. Fitting deviation against the error; dashed = DLC epsilon {thr:.0f} px (flagged to the right)", fontsize=11)
    else:
        a.set_title(f"B. Lowest confidence against the error; dashed = {thr} (flagged to the left)", fontsize=11)
a = ax[3]
names = ["jump rule", "lowest confidence\n< 0.6", "mean confidence\n< t", "jump rule OR\nlowest conf < 0.6", "fitting\n> 20 px"]
x = np.arange(5)
for k, (w, col, lab) in enumerate((("wrong50", "#D1495B", "frames > 50 px"), ("wrong20", "#E9A03B", "frames > 20 px"))):
    q = R[R.wrong_means.str.contains(w[-2:])]
    a.bar(x - 0.3 + 0.2 * k, q.recall_pct, 0.2, color=col, label=f"recall: share of the {lab} that are flagged")
    a.bar(x + 0.1 + 0.2 * k, q.precision_pct, 0.2, color=col, alpha=0.45, hatch="//", label=f"precision: share of flagged frames that are {lab[7:]}")
    for xi, (rc, pr) in enumerate(zip(q.recall_pct, q.precision_pct)):
        a.text(xi - 0.3 + 0.2 * k, rc + 1.5, f"{rc:.0f}", ha="center", fontsize=8); a.text(xi + 0.1 + 0.2 * k, pr + 1.5, f"{pr:.0f}", ha="center", fontsize=8)
fl = R[R.wrong_means.str.contains("50")].flagged_pct.to_numpy()
a.set_xticks(x); a.set_xticklabels([f"{n}\nflags {f:.0f}%" for n, f in zip(names, fl)], fontsize=8.5); a.set_ylim(0, 112); a.set_ylabel("%"); a.grid(axis="y", alpha=0.3)
a.legend(fontsize=8, loc="upper left"); a.set_title("D. Threshold rules: recall and precision", fontsize=11)
fig.suptitle(f"Can unreliable frames be found without labels? {len(F)} evenly spaced human-labeled frames of {F.video.nunique()} videos (test50 + val20), model at each video's plateau point", fontsize=12.5)
fig.savefig(OUT / "unreliable_frames_all_videos.png", dpi=125)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(S[S.video == "all"].T.to_string()); print(); print(R.to_string(index=False)); print()
print(S[["video", "frames", "wrong50", "wrong20", "AUC_jump_wrong20", "AUC_min_conf_wrong20", "AUC_mean_conf_wrong20", "AUC_fitting_wrong20"]].to_string(index=False))
