#!/usr/bin/env python3
"""Flag by the fitting deviation, repair by the heatmap's second peak, accept the repair by one of several tests.
Everything in units of the video's eye width, nothing is ever deleted.

    python flag_and_repair.py --video 8 [--accept geometry|fit|conf|jump|labelgeo|labelregion] [--thr 0.14] [--margin 0]
                              [--finder fit|pca|either] [--out DIR]

Finder option (user, 2026-10-31): --finder fit (default) is the fitting deviation below, unchanged; --finder pca flags by
the PCA reconstruction error of scripts/pca_threshold.py (score and per-point residuals read from
results/finder_comparison/pca_cache/videoNN_pca.npz, threshold = that file's per-video threshold, a multiple of the
video's median score); --finder either flags a frame when either finder does, candidate points = the union of the two
candidate sets. Candidate point of a PCA-flagged frame: the point with the largest residual, if it is a pupil point, none when
the largest residual sits on an eye-outline point (candidates_pca; the reconstruction residual spreads over all 8
points, so the strict fitting rule repairs nothing here, see the function for the rules tried). The repair
and the acceptance tests are the same for every finder.
Output folder videoNN_<accept> for fit (as before), videoNN_<finder>_<accept> for pca and either; --out overrides it
(verification). With pca or either the outputs carry the extra columns pca_rel, flag_fit, flag_pca, finder; with fit
they are unchanged. The acceptance test "fit" is only defined for the fitting finder.

Points option (user, 2026-10-31 night): --points pupil (default) repairs pupil points only, as before (a frame whose
over-threshold points include an eyelid or a corner is left flagged); --points all repairs every point: candidate
points = every point over the threshold (fitting), the point with the largest residual whichever it is (PCA), the
union (either); accepted by the two-step human-label range on the pupil quantities (unchanged) AND on the eight
outline quantities of outline_region.py (corner distance, angle and midpoint against the video's medians; eyelid
positions and opening in the frame's eye frame), each range with its own leave-one-video-out margin. A replaced corner
moves the eye frame, so the pupil quantities are re-checked in the new frame. Only with --accept labelregion.
Output folder gets the suffix _all; summary / whole_video / truth_frames carry a "points" column and per-point columns
for all eight points.

Decisions of 2026-10-30 (user): (1) errors and thresholds are measured in units of the eye width (median corner-to-corner
distance of the plateau model's predictions, label-free), so the numbers carry over to another camera; (2) the only
error line is results/error_tier/tier.csv (0.14 of the eye width): a truth frame above it is a wrong frame, every other
truth frame is a good frame; the 20-px level is label noise and is not reported; (3) the flagging threshold is the same
line (results/fit_threshold/chosen.csv); (4) nothing is deleted: a flagged frame that cannot be repaired stays flagged
("flagged_not_repaired"), whether such frames are removed later is a decision for whoever uses the data.

Finder: fit_o200_max = for each of the 8 key points, SARIMAX (2,0,0) fitted on the whole video (points with likelihood
< 0.01 treated as missing, as in unreliable_frames_sweep.py), one-step forward prediction, distance between predicted and
actual position divided by the eye width; the frame's score is the largest of the 8; score > threshold flags the frame.
Closed-eye frames (eye opening below half the video's median) are left as they are and counted apart.

Repair: on a flagged open frame the candidate pupil points are replaced by their own heatmap second peak (cell * stride
+ the per-point median offset, second_peak_repair.py). Acceptance tests:
  geometry    the rule of triggered_repair.py (ratio bands of 5 MAD around the video's median, inside the corners and the
              eyelid band, left/right and top/bottom order), hand-made; candidate points = the violating points plus the
              point with the largest deviation; a flagged frame that already passes is left alone
              ("flagged_passes_check"); an unflagged frame that violates it is counted ("unflagged_violation"), not touched
  fit         same criterion as the finder: every pupil point over the threshold is replaced; accepted if afterwards all 8
              deviations from the same one-step prediction are within the threshold (prediction cached in cache/)
  conf        DeepLabCut's confidence on the candidate point: the second peak's heatmap value >= cut
  jump        DeepLabCut's jump criterion on the candidate point: displacement to the previous and next frame, in eye
              widths, <= cut
  labelgeo    the frame's twelve pupil-point quantities inside the range spanned by the 1330 human labels
  labelregion labelgeo in the two steps of the geometry rule (a frame already inside the range is left alone), with the
              range widened by --margin (default: the leave-one-video-out margin of label_region_margin.csv)
Cuts and ranges: results/acceptance_thresholds/ (scripts/acceptance_thresholds.py).

Outputs in results/flag_and_repair/videoNN_<accept>[<margin>]/: summary.csv (truth frames), whole_video.csv,
truth_frames.csv, frames_all.csv.gz (every frame, local only). Nothing existing is modified."""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from acceptance_thresholds import eye_quantities, inside
from outline_region import VideoRef, inside_outline, outline_quantities
from second_peak_repair import BPS, P4, RES, Geometry, frame_rmse, load, peak2_pixels, pooled_bands, xy_frame
from triggered_repair import repair_triggered

OUT = RES / "flag_and_repair"
CACHE = OUT / "cache"
FIT_PBOUND, ORDER = 0.01, (2, 0, 0)
WORSE = 0.01  # a repair "made a good frame worse" when its error grew by more than this fraction of the eye width
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


def predictions_cached(video, d, dev_px):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"video{video:02d}_pred200.npy"
    if f.exists():
        pm = np.load(f)
    else:
        pm = one_step_prediction(d)
        np.save(f, pm)
    xy = np.stack([np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS], axis=1)
    dev = np.hypot(*(xy - pm).transpose(2, 0, 1))
    diff = np.nanmax(np.abs(dev - dev_px))
    print(f"one-step predictions: largest difference to the sweep's cached deviations {diff:.3f} px")
    assert diff < 1.0, "the recomputed fitting deviation differs from the finder's"
    return pm


def candidates(dev, i, thr):
    """Pupil points over the threshold; None if a non-pupil point is over it (nothing to repair there)."""
    over = [b for b in BPS if dev[i, BPS.index(b)] > thr]
    cand = [b for b in over if b in P4]
    return cand if cand and len(cand) == len(over) else None


def candidates_pca(res, i, thr):
    """PCA finder: the point with the largest reconstruction residual, if it is a pupil point; None when the largest
    residual sits on an eye-outline point (nothing to repair there). The residual spreads over all 8 points, so the
    strict fitting rule (every point over the threshold) repairs nothing here; of four rules tried on videos 1, 8, 12, 15
    (that strict rule plus the worst pupil point; the worst pupil point alone; the worst plus the pupil points with at
    least half its residual; all four pupil points) the worst point alone repaired by far the most frames (video 1:
    3665 against 146 / 282 / 66) and repaired the truth frames as well as the fitting finder does (6 of 8 in video 1,
    the one of video 8); the others repaired 1 of 8. 2026-10-31."""
    r = np.nan_to_num(res[i], nan=-1)
    worst = BPS[int(np.argmax(r))]
    return [worst] if worst in P4 else None


def candidates_either(dev, res, i, thr, thr_pca, flagged_fit, flagged_pca):
    """Union of the candidate sets of the finders that flagged the frame; None when none of them names a pupil point."""
    sets = [c for c in ((candidates(dev, i, thr) if flagged_fit else None), (candidates_pca(res, i, thr_pca) if flagged_pca else None)) if c]
    return [b for b in P4 if any(b in c for c in sets)] if sets else None


def candidates_all(dev, i, thr):
    """--points all, fitting finder: every point over the threshold, outline points included."""
    over = [b for b in BPS if dev[i, BPS.index(b)] > thr]
    return over or None


def candidates_pca_all(res, i):
    """--points all, PCA finder: the point with the largest residual, whichever it is."""
    r = np.nan_to_num(res[i], nan=-1)
    return [BPS[int(np.argmax(r))]]


def candidates_either_all(dev, res, i, thr, flagged_fit, flagged_pca):
    sets = [c for c in ((candidates_all(dev, i, thr) if flagged_fit else None), (candidates_pca_all(res, i) if flagged_pca else None)) if c]
    return [b for b in BPS if any(b in c for c in sets)] if sets else None


def plausible(pts, G, R, R_out=None, ref=None):
    """Two-step range test: pupil quantities inside the human-label range and, with --points all, the outline
    quantities inside theirs."""
    return inside(eye_quantities(pts, G.sign_lr, G.sign_tb), R) and (R_out is None or inside_outline(outline_quantities(pts, ref), R_out))


def repair_fit(G, xy, p2, pm, dev, i, thr, flagged, cand_fn=None):
    if G.closed[i]:
        return "closed", (), xy_frame(xy, i)
    if not flagged:
        return "ok", (), xy_frame(xy, i)
    cand = cand_fn(i) if cand_fn else candidates(dev, i, thr)
    if not cand:
        return "flagged_not_repaired", (), xy_frame(xy, i)
    new = xy_frame(xy, i)
    for b in sorted(cand, key=lambda b: -dev[i, BPS.index(b)]):
        q = p2[b][i]
        if not np.isfinite(q).all() or np.hypot(*(q - pm[i, BPS.index(b)])) / G.Wm > thr:
            return "flagged_not_repaired", (), xy_frame(xy, i)
        new[b] = q
    return "repaired", tuple(cand), new


def repair_scored(G, xy, p2, peak2, dev, i, thr, flagged, mode, cutv, R, n, cand_fn=None, R_out=None, ref=None):
    if G.closed[i]:
        return "closed", (), xy_frame(xy, i)
    if not flagged:
        return "ok", (), xy_frame(xy, i)
    cand = cand_fn(i) if cand_fn else candidates(dev, i, thr)
    if not cand or any(not np.isfinite(p2[b][i]).all() for b in cand):
        return "flagged_not_repaired", (), xy_frame(xy, i)
    new = xy_frame(xy, i)
    for b in cand:
        new[b] = p2[b][i]
    if mode == "labelregion":
        if plausible(xy_frame(xy, i), G, R, R_out, ref):
            return "flagged_passes_check", (), xy_frame(xy, i)
        ok = plausible(new, G, R, R_out, ref)
    elif mode == "conf":
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
    ap.add_argument("--accept", choices=["geometry", "fit", "conf", "jump", "labelgeo", "labelregion"], default="geometry")
    ap.add_argument("--thr", type=float, default=None, help="flagging threshold in eye widths; default = results/fit_threshold/chosen.csv, first row")
    ap.add_argument("--k-mad", type=float, default=5.0)
    ap.add_argument("--margin", type=float, default=None, help="labelregion only: widen the label range by this fraction; output folder videoNN_labelregion<margin*100>")
    ap.add_argument("--finder", choices=["fit", "pca", "either"], default="fit", help="which finder flags the frames (default fit, unchanged behaviour)")
    ap.add_argument("--points", choices=["pupil", "all"], default="pupil", help="which points may be repaired (default pupil, unchanged behaviour)")
    ap.add_argument("--out", default=None, help="output folder override (verification only)")
    a = ap.parse_args()
    assert a.finder == "fit" or a.accept != "fit", "the acceptance test 'fit' is only defined for the fitting finder"
    assert a.points == "pupil" or a.accept == "labelregion", "--points all is only defined with --accept labelregion"
    thr = a.thr if a.thr is not None else float(pd.read_csv(RES / "fit_threshold" / "chosen.csv").iloc[0].threshold_eye_widths)
    line = float(pd.read_csv(RES / "error_tier" / "tier.csv").iloc[0].line_eye_widths)
    tag = a.accept if a.margin is None else f"{a.accept}{int(round(100 * a.margin))}"
    out = Path(a.out) if a.out else OUT / ((f"video{a.video:02d}_{tag}" if a.finder == "fit" else f"video{a.video:02d}_{a.finder}_{tag}") + ("_all" if a.points == "all" else ""))
    out.mkdir(parents=True, exist_ok=True)
    rr, d, P, H, truth, fwd, sig = load(a.video)
    xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
    p2, off = peak2_pixels(d, P)
    B, scale = pooled_bands(RES / "second_peak_repair")
    G = Geometry(xy, scale)
    n = len(d)
    dev_px = fwd["200"]
    assert dev_px.shape == (n, len(BPS))
    dev = dev_px / G.Wm
    fit = np.nanmax(dev, axis=1)
    flag_fit = fit > thr
    cand_fn, cdev = None, dev                               # how the candidate points of a flagged frame are picked (None = the fitting rule)
    if a.finder == "fit":
        flag, pca_rel, flag_pca, thr_pca = flag_fit, None, None, np.nan
    else:
        z = np.load(RES / "finder_comparison" / "pca_cache" / f"video{a.video:02d}_pca.npz")
        pca_res, pca_rel, thr_pca = z["residuals"], z["score"], float(z["threshold"])
        assert pca_res.shape == (n, len(BPS)) and abs(float(z["eye_width_px"]) - G.Wm) < 0.5
        flag_pca = pca_rel > thr_pca
        if a.finder == "pca":
            flag, cdev = flag_pca, pca_res
            cand_fn = (lambda i: candidates_pca(pca_res, i, thr_pca)) if a.points == "pupil" else (lambda i: candidates_pca_all(pca_res, i))  # noqa: E731
        else:
            flag, cdev = flag_fit | flag_pca, np.fmax(dev / thr, pca_res / thr_pca)
            cand_fn = (lambda i: candidates_either(dev, pca_res, i, thr, thr_pca, bool(flag_fit[i]), bool(flag_pca[i]))) if a.points == "pupil" else (lambda i: candidates_either_all(dev, pca_res, i, thr, bool(flag_fit[i]), bool(flag_pca[i])))  # noqa: E731
    if a.finder == "fit" and a.points == "all":
        cand_fn = lambda i: candidates_all(dev, i, thr)  # noqa: E731
    worst = np.array(BPS)[np.nanargmax(np.nan_to_num(cdev, nan=-1), axis=1)]
    print(f"video {a.video}: {n} frames, eye width {G.Wm:.0f} px, finder {a.finder}: fit_o200_max > {thr:.3f} eye widths (≈ {thr * G.Wm:.0f} px)"
          + (f", pca > {thr_pca:.4f} eye widths (≈ {thr_pca * G.Wm:.0f} px)" if a.finder != "fit" else "") + f" flags {100 * flag.mean():.2f} %, acceptance = {tag}")
    pm = predictions_cached(a.video, d, dev_px) if a.accept == "fit" else None
    cutv, R = None, None
    if a.accept in ("conf", "jump"):
        ch = pd.read_csv(RES / "acceptance_thresholds" / "chosen.csv").set_index("test")
        cutv = float(ch.loc[a.accept, "cut"])
        print(f"acceptance cut {a.accept} {ch.loc[a.accept, 'accept_if']} {cutv}")
    elif a.accept == "labelgeo":
        R = pd.read_csv(RES / "acceptance_thresholds" / "label_region.csv").set_index("quantity")
    elif a.accept == "labelregion":
        R0 = pd.read_csv(RES / "acceptance_thresholds" / "label_region.csv").set_index("quantity")
        if a.margin is None:
            R = pd.read_csv(RES / "acceptance_thresholds" / "label_region_margin.csv").set_index("quantity")
        else:
            w = R0["max"] - R0["min"]
            R = pd.DataFrame({"min": R0["min"] - a.margin * w, "max": R0["max"] + a.margin * w, "margin": a.margin})
        print(f"label region widened by margin {R.margin.iloc[0]:.2f}")
    R_out, ref = None, None
    if a.points == "all":
        R_out = pd.read_csv(RES / "acceptance_thresholds" / "label_region_outline.csv").set_index("quantity")
        ref = VideoRef(xy)
        print(f"outline range (margin {R_out.margin.iloc[0]:.2f}) in use: all eight points may be repaired")
    peak2 = {b: H[f"{b}_peak2"].to_numpy(float) for b in BPS}

    status, reps, new_xy = [], [], {b: xy[b].copy() for b in BPS}
    for i in range(n):
        if a.accept == "fit":
            st, rep, new = repair_fit(G, xy, p2, pm, dev, i, thr, bool(flag[i]), cand_fn)
        elif a.accept == "geometry":
            st, rep, new = repair_triggered(G, xy, p2, i, a.k_mad, bool(flag[i]), worst[i])
            st = {"flagged_ok": "flagged_passes_check", "drop": "flagged_not_repaired"}.get(st, st)
        else:
            st, rep, new = repair_scored(G, xy, p2, peak2, dev, i, thr, bool(flag[i]), a.accept, cutv, R, n, cand_fn, R_out, ref)
        status.append(st)
        reps.append("+".join(rep))
        for b in rep:
            new_xy[b][i] = new[b]
    status = np.array(status)
    F = pd.DataFrame({"frame": np.arange(n), "status": status, "points": reps, "fit_rel": fit, "fit_px": fit * G.Wm, "flag": flag, "worst_point": worst})
    extra = {} if a.finder == "fit" else {"finder": a.finder, "thr_pca_eye_widths": thr_pca, "flagged_fit_pct": round(100 * flag_fit.mean(), 2), "flagged_pca_pct": round(100 * flag_pca.mean(), 2),
                                           "flagged_both_pct": round(100 * (flag_fit & flag_pca).mean(), 2)}
    if a.finder != "fit":
        F["pca_rel"], F["flag_fit"], F["flag_pca"] = pca_rel, flag_fit, flag_pca
    PTS = BPS if a.points == "all" else P4
    if a.points == "all":
        extra["points"] = "all"
    for b in PTS:
        F[f"{b}_moved_px"] = np.hypot(*(new_xy[b] - xy[b]).T)
    F.to_csv(out / "frames_all.csv.gz", index=False, float_format="%.4f")
    counts = {s: int((status == s).sum()) for s in STATUSES}
    W = pd.DataFrame([{"video": a.video, "frames": n, "accept": tag, "eye_width_px": round(G.Wm, 1), "thr_eye_widths": thr, "thr_px": round(thr * G.Wm, 1),
                       "flagged_pct": round(100 * flag.mean(), 2), "flagged_closed": int((flag & G.closed).sum()), **counts,
                       "repaired_pct": round(100 * counts["repaired"] / n, 2), "flagged_not_repaired_pct": round(100 * counts["flagged_not_repaired"] / n, 2),
                       "worst_not_pupil_among_flagged": int((flag & ~np.isin(worst, P4) & ~G.closed).sum()),
                       **{f"repaired_{b}": int((F.points.str.contains(b).to_numpy() & (status == "repaired")).sum()) for b in PTS}, **extra}])
    W.to_csv(out / "whole_video.csv", index=False)
    print(W.T.to_string())

    rows = []
    for f, hr in truth.iterrows():
        before = {b: xy[b][f] for b in BPS}
        after = {b: new_xy[b][f] for b in BPS}
        rb, ra = frame_rmse(before, hr), frame_rmse(after, hr)
        r = {"frame": f, "set": hr.set, "status": status[f], "points": reps[f], "error_before": rb / G.Wm, "error_after": ra / G.Wm,
             "error_before_px": rb, "error_after_px": ra, "rmse_file": hr.frame_rmse_px, "flag": bool(flag[f]), "fit_rel": fit[f], "worst_point": worst[f]}
        if a.finder != "fit":
            r.update({"pca_rel": pca_rel[f], "flag_fit": bool(flag_fit[f]), "flag_pca": bool(flag_pca[f]), "finder": a.finder})
        for b in PTS:
            hx, hy = hr[f"human_{b}_x"], hr[f"human_{b}_y"]
            r[f"{b}_before"] = np.hypot(before[b][0] - hx, before[b][1] - hy) / G.Wm
            r[f"{b}_after"] = np.hypot(after[b][0] - hx, after[b][1] - hy) / G.Wm
        rows.append(r)
    T = pd.DataFrame(rows)
    assert np.allclose(T.error_before_px, T.rmse_file, atol=0.05), "frame RMSE formula differs from scores_lovo.csv"
    T.round(4).to_csv(out / "truth_frames.csv", index=False)
    wrong, good, rep = T.error_before > line, T.error_before <= line, T.status == "repaired"
    nr = T.status == "flagged_not_repaired"
    untouched = T.status.isin(["ok", "closed", "flagged_passes_check", "unflagged_violation"])
    S = {"video": a.video, "accept": tag, "eye_width_px": round(G.Wm, 1), "thr_eye_widths": thr, "line_eye_widths": line, "truth_frames": len(T),
         "wrong_before": int(wrong.sum()), **{s: int((T.status == s).sum()) for s in STATUSES},
         "wrong_flagged": int((wrong & T.flag).sum()), "wrong_repaired_frames": int((wrong & rep).sum()),
         "wrong_repaired_to_good": int((wrong & rep & (T.error_after <= line)).sum()),
         "wrong_flagged_not_repaired": int((wrong & nr).sum()), "wrong_left_untouched": int((wrong & untouched).sum()), "wrong_unflagged": int((wrong & ~T.flag).sum()),
         "wrong_after_all": int((T.error_after > line).sum()), "wrong_after_unflagged": int(((T.error_after > line) & ~nr).sum()),
         "good_flagged": int((good & T.flag).sum()), "good_repaired": int((good & rep).sum()),
         "good_made_worse": int((good & rep & (T.error_after > T.error_before + WORSE)).sum()), "good_pushed_over_line": int((good & rep & (T.error_after > line)).sum()),
         "good_flagged_not_repaired": int((good & nr).sum()),
         "median_error_before_pct": round(100 * float(T.error_before.median()), 2), "median_error_after_pct": round(100 * float(T.error_after.median()), 2),
         "repaired_median_before_pct": round(100 * float(T.error_before[rep].median()), 2) if rep.any() else np.nan,
         "repaired_median_after_pct": round(100 * float(T.error_after[rep].median()), 2) if rep.any() else np.nan, **extra}
    pd.DataFrame([S]).to_csv(out / "summary.csv", index=False)
    print(pd.Series(S).to_string())
    pd.set_option("display.width", 250)
    show = ["frame", "set", "status", "points", "worst_point", "fit_rel", "error_before", "error_after"]
    print((T[~T.status.isin(["ok", "closed"]) | wrong].sort_values("error_before", ascending=False)[show]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
