#!/usr/bin/env python3
"""Second-peak repair triggered by the chosen time-series detector, geometry only as the acceptance test.

    python triggered_repair.py --video 8 [--k-mad 5]

Variant of second_peak_repair.py asked for on 2026-10-27: under the constraint "repair only the flagged frames",
the trigger must be the detector chosen at the 50-px tier (fit_o200_max = largest-of-8 one-step fitting deviation,
SARIMAX (2,0,0); pooled threshold = the 95 % quantile on the 1330 truth frames, the same number the repair script
reports as thr_o200_px). Per flagged, open frame: candidate points = the geometry-violating pupil points plus the
pupil point with the largest fitting deviation; the smallest replacement by peak 2 (one point, then pairs) after which
the geometry check passes is accepted ("repaired"); if the frame passed the geometry check already and only the
time series complains, it is left as it is ("flagged_ok": there is no frame-local evidence for a change); if no
replacement passes, "drop". Unflagged frames are never touched, even if they violate the geometry (counted as
"unflagged_violation" — the video-4 glint case). Same truth-frame summary columns as second_peak_repair.py, output in
results/second_peak_repair/videoNN_trig/ (summary.csv, whole_video.csv, truth_frames.csv; frames_all.csv.gz local)."""
import argparse

import numpy as np
import pandas as pd

from second_peak_repair import BPS, P4, RES, Geometry, frame_rmse, load, peak2_pixels, pooled_bands, xy_frame


def repair_triggered(G, xy, p2, i, k, flagged, worst):
    if G.closed[i]:
        return "closed", (), xy_frame(xy, i)
    E = {b: G.to_eye(xy[b][i], i) for b in BPS}
    bad = G.violations(E, i, k)
    if not flagged:
        return ("unflagged_violation" if bad else "ok"), (), xy_frame(xy, i)
    if not bad:
        return "flagged_ok", (), xy_frame(xy, i)
    cand_pts = set(bad) | ({worst} if worst in P4 else set())
    cands = [(b,) for b in sorted(cand_pts)]
    if len(cand_pts) >= 2:
        cands += [(a, b) for a in sorted(cand_pts) for b in sorted(cand_pts) if a < b]
    for rep in cands:
        E2 = dict(E)
        for b in rep:
            E2[b] = G.to_eye(p2[b][i], i)
        if not G.violations(E2, i, k):
            new = xy_frame(xy, i)
            for b in rep:
                new[b] = p2[b][i]
            return "repaired", rep, new
    return "drop", tuple(sorted(bad)), xy_frame(xy, i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=int, required=True)
    ap.add_argument("--k-mad", type=float, default=5.0)
    a = ap.parse_args()
    out = RES / "second_peak_repair" / f"video{a.video:02d}_trig"
    out.mkdir(parents=True, exist_ok=True)
    rr, d, P, H, truth, fwd, sig = load(a.video)
    xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
    p2, off = peak2_pixels(d, P)
    B, scale = pooled_bands(out.parent)
    G = Geometry(xy, scale)
    n = len(d)
    dev = fwd["200"]
    assert dev.shape == (n, len(BPS))
    thr200 = float(np.nanquantile(sig.fit_o200_max, 0.95))
    fit = np.nanmax(dev, axis=1)
    flag = fit > thr200
    worst = np.array(BPS)[np.nanargmax(np.nan_to_num(dev, nan=-1), axis=1)]
    print(f"video {a.video}: {n} frames, trigger fit_o200_max > {thr200:.1f} px flags {100 * flag.mean():.2f} %")

    status, reps, new_xy = [], [], {b: xy[b].copy() for b in BPS}
    for i in range(n):
        st, rep, new = repair_triggered(G, xy, p2, i, a.k_mad, bool(flag[i]), worst[i])
        status.append(st); reps.append("+".join(rep))
        for b in rep:
            new_xy[b][i] = new[b]
    status = np.array(status)
    F = pd.DataFrame({"frame": np.arange(n), "status": status, "points": reps, "fit_o200_max": fit, "flag_o200": flag, "worst_point": worst})
    for b in P4:
        F[f"{b}_moved_px"] = np.hypot(*(new_xy[b] - xy[b]).T)
    F.to_csv(out / "frames_all.csv.gz", index=False, float_format="%.3f")
    counts = {s: int((status == s).sum()) for s in ("ok", "closed", "flagged_ok", "unflagged_violation", "repaired", "drop")}
    W = pd.DataFrame([{"video": a.video, "frames": n, "k_mad": a.k_mad, "bands": "pooled", "trigger": "fit_o200_max", "thr_o200_px": round(thr200, 1), "flag_o200_pct": round(100 * flag.mean(), 2),
                       **counts, "repaired_pct": round(100 * counts["repaired"] / n, 2), "drop_pct": round(100 * counts["drop"] / n, 2),
                       "unflagged_violation_pct": round(100 * counts["unflagged_violation"] / n, 2),
                       **{f"repaired_{b}": int((F.points.str.contains(b).to_numpy() & (status == "repaired")).sum()) for b in P4}}])
    W.to_csv(out / "whole_video.csv", index=False)
    print(W.T.to_string())

    rows = []
    for f, hr in truth.iterrows():
        before = {b: xy[b][f] for b in BPS}
        after = {b: new_xy[b][f] for b in BPS}
        r = {"frame": f, "set": hr.set, "status": status[f], "points": reps[f], "rmse_before": frame_rmse(before, hr), "rmse_after": frame_rmse(after, hr),
             "rmse_file": hr.frame_rmse_px, "flag_o200": bool(flag[f]), "worst_point": worst[f]}
        for b in P4:
            hx, hy = hr[f"human_{b}_x"], hr[f"human_{b}_y"]
            r[f"{b}_before"] = np.hypot(before[b][0] - hx, before[b][1] - hy)
            r[f"{b}_after"] = np.hypot(after[b][0] - hx, after[b][1] - hy)
        rows.append(r)
    T = pd.DataFrame(rows)
    assert np.allclose(T.rmse_before, T.rmse_file, atol=0.05), "frame RMSE formula differs from scores_lovo.csv"
    T.round(2).to_csv(out / "truth_frames.csv", index=False)
    kept = T.status != "drop"
    untouched = T.status.isin(["ok", "closed", "flagged_ok", "unflagged_violation"])
    S = {"video": a.video, "k_mad": a.k_mad, "bands": "pooled", "trigger": "fit_o200_max", "truth_frames": len(T), "gt50_before": int((T.rmse_before > 50).sum()), "gt20_before": int((T.rmse_before > 20).sum()),
         **{s: int((T.status == s).sum()) for s in ("ok", "closed", "flagged_ok", "unflagged_violation", "repaired", "drop")},
         "gt50_repaired_frames": int(((T.rmse_before > 50) & (T.status == "repaired")).sum()),
         "gt50_repaired_to_lt50": int(((T.rmse_before > 50) & (T.status == "repaired") & (T.rmse_after <= 50)).sum()),
         "gt50_repaired_to_lt20": int(((T.rmse_before > 50) & (T.status == "repaired") & (T.rmse_after <= 20)).sum()),
         "gt50_dropped": int(((T.rmse_before > 50) & (T.status == "drop")).sum()),
         "gt50_left_untouched": int(((T.rmse_before > 50) & untouched).sum()),
         "gt50_unflagged": int(((T.rmse_before > 50) & ~T.flag_o200).sum()),
         "gt50_after_among_kept": int(((T.rmse_after > 50) & kept).sum()), "gt20_after_among_kept": int(((T.rmse_after > 20) & kept).sum()),
         "false_repairs_lt20_worse5px": int(((T.rmse_before <= 20) & (T.status == "repaired") & (T.rmse_after > T.rmse_before + 5)).sum()),
         "repairs_lt20": int(((T.rmse_before <= 20) & (T.status == "repaired")).sum()),
         "false_drops_lt20": int(((T.rmse_before <= 20) & (T.status == "drop")).sum()),
         "drops_20to50": int(((T.rmse_before > 20) & (T.rmse_before <= 50) & (T.status == "drop")).sum()),
         "median_rmse_before": round(float(T.rmse_before.median()), 2), "median_rmse_after_kept": round(float(T.rmse_after[kept].median()), 2),
         "repaired_median_before": round(float(T.rmse_before[T.status == "repaired"].median()), 2) if (T.status == "repaired").any() else np.nan,
         "repaired_median_after": round(float(T.rmse_after[T.status == "repaired"].median()), 2) if (T.status == "repaired").any() else np.nan}
    pd.DataFrame([S]).to_csv(out / "summary.csv", index=False)
    print(pd.Series(S).to_string())
    pd.set_option("display.width", 250)
    show = ["frame", "set", "status", "points", "worst_point", "rmse_before", "rmse_after", "flag_o200"] + [f"{b}_before" for b in P4] + [f"{b}_after" for b in P4]
    print(T[~T.status.isin(["ok", "closed"]) | (T.rmse_before > 50)].sort_values("rmse_before", ascending=False)[show].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
