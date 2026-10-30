#!/usr/bin/env python3
"""Flag by the fitting deviation, repair by the heatmap's second peak, accept the repair by the same fitting deviation.

    python flag_and_repair.py --video 8 [--accept fit|geometry] [--thr 36]

Decisions of 2026-10-30 (user): (1) the flagging threshold is read off the > 50 px truth frames (results/fit_threshold/
chosen.csv, scripts/fit_threshold.py), not a quantile; (2) nothing is ever deleted: a flagged frame that cannot be
repaired stays flagged ("flagged_not_repaired"); whether such frames are removed later is a decision for whoever uses
the data; (3) the acceptance test of a repair should be the same criterion that flagged the frame (--accept fit, the
default), not a separate hand-made geometry rule; the geometry version (--accept geometry, the rule of
triggered_repair.py) is kept only as the comparison.

Finder (both modes): fit_o200_max = for each of the 8 key points, SARIMAX (2,0,0) fitted on the whole video (points with
likelihood < 0.01 treated as missing, as in unreliable_frames_sweep.py), one-step forward prediction, distance between
the predicted and the actual position; the frame's score is the largest of the 8; score > threshold flags the frame.
Closed-eye frames (eye opening below half the video's median) are left as they are in both modes and counted apart.

Repair, --accept fit: on a flagged open frame, every pupil point whose deviation is above the threshold is replaced by
its own heatmap second peak (cell * stride + the per-point median offset, as in second_peak_repair.py); the deviation
of the new position from the same one-step prediction is computed; if afterwards all 8 points are within the threshold
the frame is "repaired", otherwise it stays "flagged_not_repaired" and nothing is changed. If the point above the
threshold is not a pupil point (eyelid, corner), nothing is tried: "flagged_not_repaired". The prediction is the one the
finder used, i.e. computed from the model's original trajectory; a repaired frame does not change the prediction for
the frames after it (reported as a limitation).
Repair, --accept geometry: repair_triggered() of triggered_repair.py with the new threshold; its "flagged_ok" is
reported as "flagged_passes_check" and its "drop" as "flagged_not_repaired".
Repair, --accept conf | jump | labelgeo (added the same day, scripts/acceptance_thresholds.py): same candidate points and
replacement as --accept fit, accepted when the second peak's heatmap value is >= the conf cut (DeepLabCut's confidence
on the candidate point), or when the candidate's displacement to the previous and next frame, relative to the median eye
width, is <= the jump cut (DeepLabCut's jump criterion), or when the frame's twelve pupil-point quantities lie inside the
range spanned by the 1330 human labels (labelgeo, no constant). Cuts and region from results/acceptance_thresholds/.

Outputs in results/flag_and_repair/videoNN_<accept>/: summary.csv (truth frames), whole_video.csv, truth_frames.csv,
frames_all.csv.gz (every frame, local only); the one-step predictions are cached in results/flag_and_repair/cache/
videoNN_pred200.npy and checked against the sweep's cached deviations. Nothing existing is modified."""
import argparse
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from acceptance_thresholds import eye_quantities, inside
from second_peak_repair import BPS, P4, RES, Geometry, frame_rmse, load, peak2_pixels, pooled_bands, xy_frame
from triggered_repair import repair_triggered

OUT = RES / "flag_and_repair"
CACHE = OUT / "cache"
FIT_PBOUND, ORDER = 0.01, (2, 0, 0)
STATUSES = ("ok", "closed", "flagged_passes_check", "unflagged_violation", "repaired", "flagged_not_repaired")


def one_step_prediction(d):
    """(T, 8, 2) one-step forward SARIMAX (2,0,0) prediction of every key point, as in unreliable_frames_sweep.py."""
    T = len(d)
    pm = np.full((T, len(BPS), 2), np.nan)
    for k, b in enumerate(BPS):
        for j, c in enumerate(("x", "y")):
            y = d[b][c].to_numpy(float).copy()
            y[d[b]["likelihood"].to_numpy(float) < FIT_PBOUND] = np.nan
            m = np.nanmean(y)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(y - m, order=ORDER, seasonal_order=(0, 0, 0, 0), simple_differencing=True).fit(disp=False)
                if not res.mle_retvals.get("converged", True):
                    res = SARIMAX(y - m, order=ORDER, seasonal_order=(0, 0, 0, 0), simple_differencing=True).fit(disp=False, method="powell", maxiter=500)
                mean = np.asarray(res.get_prediction(end=res.model.nobs - 1).predicted_mean) + m
            pm[T - len(mean):, k, j] = mean
    return pm


def predictions_cached(video, d, dev_cached):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"video{video:02d}_pred200.npy"
    if f.exists():
        pm = np.load(f)
    else:
        pm = one_step_prediction(d)
        np.save(f, pm)
    xy = np.stack([np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS], axis=1)
    dev = np.hypot(*(xy - pm).transpose(2, 0, 1))
    diff = np.nanmax(np.abs(dev - dev_cached))
    print(f"one-step predictions: largest difference to the sweep's cached deviations {diff:.3f} px")
    assert diff < 1.0, "the recomputed fitting deviation differs from the finder's"
    return pm


def repair_fit(G, xy, p2, pm, dev, i, thr, flagged):
    if G.closed[i]:
        return "closed", (), xy_frame(xy, i)
    if not flagged:
        return "ok", (), xy_frame(xy, i)
    over = [b for b in BPS if dev[i, BPS.index(b)] > thr]
    cand = [b for b in sorted(over, key=lambda b: -dev[i, BPS.index(b)]) if b in P4]
    if not cand or len(cand) < len(over):
        return "flagged_not_repaired", (), xy_frame(xy, i)
    new = xy_frame(xy, i)
    for b in cand:
        q = p2[b][i]
        if not np.isfinite(q).all() or np.hypot(*(q - pm[i, BPS.index(b)])) > thr:
            return "flagged_not_repaired", (), xy_frame(xy, i)
        new[b] = q
    return "repaired", tuple(cand), new


def repair_scored(G, xy, p2, peak2, dev, i, thr, flagged, mode, cutv, R, n):
    """Same candidates and replacement as repair_fit; acceptance by conf / jump / labelgeo."""
    if G.closed[i]:
        return "closed", (), xy_frame(xy, i)
    if not flagged:
        return "ok", (), xy_frame(xy, i)
    over = [b for b in BPS if dev[i, BPS.index(b)] > thr]
    cand = [b for b in over if b in P4]
    if not cand or len(cand) < len(over) or any(not np.isfinite(p2[b][i]).all() for b in cand):
        return "flagged_not_repaired", (), xy_frame(xy, i)
    new = xy_frame(xy, i)
    for b in cand:
        new[b] = p2[b][i]
    if mode == "conf":
        ok = min(peak2[b][i] for b in cand) >= cutv
    elif mode == "jump":
        jumps = [np.hypot(*(new[b] - xy[b][g])) / G.Wm for b in cand for g in (i - 1, i + 1) if 0 <= g < n and np.isfinite(xy[b][g]).all()]
        ok = bool(jumps) and max(jumps) <= cutv
    else:
        ok = inside(eye_quantities(new, G.sign_lr, G.sign_tb), R)
    return ("repaired", tuple(cand), new) if ok else ("flagged_not_repaired", (), xy_frame(xy, i))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=int, required=True)
    ap.add_argument("--accept", choices=["fit", "geometry", "conf", "jump", "labelgeo"], default="fit")
    ap.add_argument("--thr", type=float, default=None, help="flagging threshold in px; default = results/fit_threshold/chosen.csv")
    ap.add_argument("--k-mad", type=float, default=5.0)
    a = ap.parse_args()
    thr = a.thr if a.thr is not None else float(pd.read_csv(RES / "fit_threshold" / "chosen.csv").iloc[0].threshold_px)
    out = OUT / f"video{a.video:02d}_{a.accept}"
    out.mkdir(parents=True, exist_ok=True)
    rr, d, P, H, truth, fwd, sig = load(a.video)
    xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
    p2, off = peak2_pixels(d, P)
    B, scale = pooled_bands(RES / "second_peak_repair")
    G = Geometry(xy, scale)
    n = len(d)
    dev = fwd["200"]
    assert dev.shape == (n, len(BPS))
    fit = np.nanmax(dev, axis=1)
    flag = fit > thr
    worst = np.array(BPS)[np.nanargmax(np.nan_to_num(dev, nan=-1), axis=1)]
    print(f"video {a.video}: {n} frames, fit_o200_max > {thr:.0f} px flags {100 * flag.mean():.2f} %, acceptance = {a.accept}")
    pm = predictions_cached(a.video, d, dev) if a.accept == "fit" else None
    cutv, R = None, None
    if a.accept in ("conf", "jump"):
        ch = pd.read_csv(RES / "acceptance_thresholds" / "chosen.csv").set_index("test")
        cutv = float(ch.loc[a.accept, "cut"])
        print(f"acceptance cut {a.accept} {ch.loc[a.accept, 'accept_if']} {cutv}")
    elif a.accept == "labelgeo":
        R = pd.read_csv(RES / "acceptance_thresholds" / "label_region.csv").set_index("quantity")
    peak2 = {b: H[f"{b}_peak2"].to_numpy(float) for b in BPS}

    status, reps, new_xy = [], [], {b: xy[b].copy() for b in BPS}
    for i in range(n):
        if a.accept == "fit":
            st, rep, new = repair_fit(G, xy, p2, pm, dev, i, thr, bool(flag[i]))
        elif a.accept in ("conf", "jump", "labelgeo"):
            st, rep, new = repair_scored(G, xy, p2, peak2, dev, i, thr, bool(flag[i]), a.accept, cutv, R, n)
        else:
            st, rep, new = repair_triggered(G, xy, p2, i, a.k_mad, bool(flag[i]), worst[i])
            st = {"flagged_ok": "flagged_passes_check", "drop": "flagged_not_repaired"}.get(st, st)
        status.append(st)
        reps.append("+".join(rep))
        for b in rep:
            new_xy[b][i] = new[b]
    status = np.array(status)
    F = pd.DataFrame({"frame": np.arange(n), "status": status, "points": reps, "fit_o200_max": fit, "flag": flag, "worst_point": worst})
    for b in P4:
        F[f"{b}_moved_px"] = np.hypot(*(new_xy[b] - xy[b]).T)
    if a.accept == "fit":
        fit_after = fit.copy()
        for i in np.where(status == "repaired")[0]:
            fit_after[i] = max(np.hypot(*(new_xy[b][i] - pm[i, k])) for k, b in enumerate(BPS))
        F["fit_o200_max_after"] = fit_after
    F.to_csv(out / "frames_all.csv.gz", index=False, float_format="%.3f")
    counts = {s: int((status == s).sum()) for s in STATUSES}
    W = pd.DataFrame([{"video": a.video, "frames": n, "accept": a.accept, "thr_px": thr, "flagged_pct": round(100 * flag.mean(), 2),
                       "flagged_closed": int((flag & G.closed).sum()), **counts,
                       "repaired_pct": round(100 * counts["repaired"] / n, 2), "flagged_not_repaired_pct": round(100 * counts["flagged_not_repaired"] / n, 2),
                       "worst_not_pupil_among_flagged": int((flag & ~np.isin(worst, P4) & ~G.closed).sum()),
                       **{f"repaired_{b}": int((F.points.str.contains(b).to_numpy() & (status == "repaired")).sum()) for b in P4}}])
    W.to_csv(out / "whole_video.csv", index=False)
    print(W.T.to_string())

    rows = []
    for f, hr in truth.iterrows():
        before = {b: xy[b][f] for b in BPS}
        after = {b: new_xy[b][f] for b in BPS}
        r = {"frame": f, "set": hr.set, "status": status[f], "points": reps[f], "rmse_before": frame_rmse(before, hr), "rmse_after": frame_rmse(after, hr),
             "rmse_file": hr.frame_rmse_px, "flag": bool(flag[f]), "fit_o200_max": fit[f], "worst_point": worst[f]}
        for b in P4:
            hx, hy = hr[f"human_{b}_x"], hr[f"human_{b}_y"]
            r[f"{b}_before"] = np.hypot(before[b][0] - hx, before[b][1] - hy)
            r[f"{b}_after"] = np.hypot(after[b][0] - hx, after[b][1] - hy)
        rows.append(r)
    T = pd.DataFrame(rows)
    assert np.allclose(T.rmse_before, T.rmse_file, atol=0.05), "frame RMSE formula differs from scores_lovo.csv"
    T.round(2).to_csv(out / "truth_frames.csv", index=False)
    big, clean, rep = T.rmse_before > 50, T.rmse_before <= 20, T.status == "repaired"
    nr = T.status == "flagged_not_repaired"
    untouched = T.status.isin(["ok", "closed", "flagged_passes_check", "unflagged_violation"])
    S = {"video": a.video, "accept": a.accept, "thr_px": thr, "truth_frames": len(T), "gt50_before": int(big.sum()), "gt20_before": int((T.rmse_before > 20).sum()),
         **{s: int((T.status == s).sum()) for s in STATUSES},
         "gt50_flagged": int((big & T.flag).sum()), "gt50_repaired_frames": int((big & rep).sum()),
         "gt50_repaired_to_lt50": int((big & rep & (T.rmse_after <= 50)).sum()), "gt50_repaired_to_lt20": int((big & rep & (T.rmse_after <= 20)).sum()),
         "gt50_flagged_not_repaired": int((big & nr).sum()), "gt50_left_untouched": int((big & untouched).sum()), "gt50_unflagged": int((big & ~T.flag).sum()),
         "gt50_after_all": int((T.rmse_after > 50).sum()), "gt50_after_unflagged": int(((T.rmse_after > 50) & ~nr).sum()),
         "gt20_after_all": int((T.rmse_after > 20).sum()),
         "repairs_lt20": int((clean & rep).sum()), "false_repairs_lt20_worse5px": int((clean & rep & (T.rmse_after > T.rmse_before + 5)).sum()),
         "lt20_flagged_not_repaired": int((clean & nr).sum()), "lt20_flagged": int((clean & T.flag).sum()),
         "mid_20to50_flagged_not_repaired": int(((T.rmse_before > 20) & (T.rmse_before <= 50) & nr).sum()),
         "median_rmse_before": round(float(T.rmse_before.median()), 2), "median_rmse_after": round(float(T.rmse_after.median()), 2),
         "repaired_median_before": round(float(T.rmse_before[rep].median()), 2) if rep.any() else np.nan,
         "repaired_median_after": round(float(T.rmse_after[rep].median()), 2) if rep.any() else np.nan}
    pd.DataFrame([S]).to_csv(out / "summary.csv", index=False)
    print(pd.Series(S).to_string())
    pd.set_option("display.width", 250)
    show = ["frame", "set", "status", "points", "worst_point", "fit_o200_max", "rmse_before", "rmse_after"] + [f"{b}_before" for b in P4] + [f"{b}_after" for b in P4]
    print(T[~T.status.isin(["ok", "closed"]) | big].sort_values("rmse_before", ascending=False)[show].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
