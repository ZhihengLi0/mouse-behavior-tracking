#!/usr/bin/env python3
"""The per-frame geometry check of second_peak_repair.py evaluated as a detector, in the frame of tier50_reeval.py.

    python geometry_detector.py

Question (user, 2026-10-27 night): the geometry check that gates the second-peak repair is itself a "find the
suspicious frame" rule — how does it compare with the detectors already ranked at the 50-px tier, and what does
combining it with the chosen one (fit_o200_max, largest-of-8 fitting deviation) give?

Score per frame (label-free, no temporal term; pooled bands = median over the 19 videos of each video's MAD, as in
the repair script): geo_z = largest |z| of the four pupil-point ratios; a hard violation (point outside the corners
or the eyelid band, or wrong left/right / top/bottom ordering) adds 1000 so that it ranks above every soft z; closed
frames (opening < half the video's median) score 0 (the pupil is undefined, the repair leaves them alone). The fixed
rule of the repair script is geo_z >= 5 (k = 5 MADs) or a hard violation.

Evaluated on the same 1330 truth frames as tier50_reeval.py (truth = frame RMSE > 50 px, 32 frames; > 20 px shown):
AUC with a bootstrap interval, leave-one-video-out thresholds at 1 / 2 / 5 % flagged, the fixed rule's recall and
flagged share, and two combinations with fit_o200_max (rank average; OR of the two 5 % flags). Whole-video flagged
shares of the fixed rule per video next to fit_o200_max's. Outputs in results/geometry_detector/: signals.csv,
caught_per_video.csv, whole_video.csv, truth_frames.csv. Nothing existing is modified."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_peak_repair import BPS, P4, PL, RES, Geometry, pooled_bands, predictions  # noqa: E402
from tier50_reeval import NEG, SHARES, boot_auc, evaluate, load, lovo_flags  # noqa: E402

OUT = RES / "geometry_detector"
K = 5.0
HARD = 1000.0


def geometry_scores(video, scale):
    rr, d = predictions(video)
    xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
    G = Geometry(xy, scale)
    E = {b: G.to_eye(xy[b]) for b in BPS}
    z = np.stack([np.abs(G.ratio(E, b) - G.med[b]) / max(G.mad[b], 1e-6) for b in P4], axis=1)
    soft = np.nanmax(z, axis=1)
    soft[~np.isfinite(soft)] = HARD
    hard = np.zeros(len(d), bool)
    for b in P4:
        hard |= (np.abs(E[b][:, 0]) > 0.6) | ~((G.lid_lo <= E[b][:, 1]) & (E[b][:, 1] <= G.lid_hi))
    hard |= G.sign_lr * (E["pupil_right"][:, 0] - E["pupil_left"][:, 0]) <= 0
    hard |= G.sign_tb * (E["pupil_bottom"][:, 1] - E["pupil_top"][:, 1]) <= 0
    s = soft + HARD * hard
    s[G.closed] = 0.0
    return pd.DataFrame({"video": video, "frame": np.arange(len(d)), "geo_z": s, "geo_soft": soft, "geo_hard": hard, "closed": G.closed,
                         "geo_rule": (s >= K) & ~G.closed, "worst_point": np.array(P4)[np.nanargmax(np.nan_to_num(z, nan=-1), axis=1)]})


def main():
    OUT.mkdir(exist_ok=True)
    B, scale = pooled_bands(RES / "second_peak_repair")
    print("pooled scale:", {b: round(v, 4) for b, v in scale.items()})
    allv = []
    for v in PL.video:
        allv.append(geometry_scores(int(v), scale))
        g = allv[-1]
        print(f"video {v}: {len(g)} frames, rule flags {100 * g.geo_rule.mean():.2f} % (hard {100 * g.geo_hard.mean():.2f} %), closed {100 * g.closed.mean():.2f} %")
    A = pd.concat(allv, ignore_index=True)

    M, cols = load()
    M = M.merge(A, on=["video", "frame"], how="left")
    assert M.geo_z.notna().all(), "geometry score missing for some truth frames"
    M.to_csv(OUT / "truth_frames.csv", index=False, columns=["video", "frame", "set", "frame_rmse_px", "wrong50", "wrong20", "geo_z", "geo_soft", "geo_hard", "closed",
                                                                "geo_rule", "worst_point", "fit_o200_max", "fit_pupil"])

    rows = [evaluate(M, "geo_z (geometry check as a score)", M.geo_z.to_numpy(float))]
    for c in ("fit_o200_max", "fit_o311_max", "fit_pupil"):
        rows.append(evaluate(M, c, M[c].to_numpy(float)))
    comb = ((M.geo_z.rank(pct=True) + M.fit_o200_max.rank(pct=True)) / 2).to_numpy()
    rows.append(evaluate(M, "rank_avg(geo_z, fit_o200_max)", comb))
    T = pd.DataFrame(rows)

    # fixed rule (k = 5 or hard) and its OR with the 5 % fit_o200_max flag
    y50, y20 = M.wrong50.to_numpy(bool), M.wrong20.to_numpy(bool)
    rule = M.geo_rule.to_numpy(bool)
    f5 = lovo_flags(M, M.fit_o200_max.to_numpy(float), 0.05)
    fixed = []
    for name, f in (("geo rule (k=5 or hard violation)", rule), ("fit_o200_max 5 % flag", f5), ("OR(geo rule, fit_o200_max 5 %)", rule | f5), ("AND(geo rule, fit_o200_max 5 %)", rule & f5)):
        fixed.append({"signal": name, "n_scored": len(M), "flagged_pct": round(100 * f.mean(), 1), "caught50": int((f & y50).sum()), "recall50": round(100 * (f & y50).sum() / y50.sum(), 1),
                      "prec50": round(100 * (f & y50).sum() / max(1, f.sum()), 1), "caught20": int((f & y20).sum()), "recall20": round(100 * (f & y20).sum() / y20.sum(), 1)})
    X = pd.DataFrame(fixed)
    T.round(3).to_csv(OUT / "signals.csv", index=False)
    X.to_csv(OUT / "fixed_rules.csv", index=False)
    pd.set_option("display.width", 250)
    show = ["signal", "auc50", "auc50_lo", "auc50_hi", "auc20", "recall50_1pct", "recall50_2pct", "recall50_5pct", "prec50_5pct", "flagged_for_recall50_90"]
    print(T[show].round(3).to_string(index=False))
    print(X.to_string(index=False))

    rows = []
    for v in sorted(M.video.unique()):
        k = (M.video == v).to_numpy()
        g = A[A.video == v]
        rows.append({"video": v, "frames_gt50": int((y50 & k).sum()), "geo_rule_caught": int((rule & y50 & k).sum()), "fit_o200_5pct_caught": int((f5 & y50 & k).sum()),
                     "OR_caught": int(((rule | f5) & y50 & k).sum()), "geo_rule_flagged_truth": int((rule & k).sum()), "fit_o200_5pct_flagged_truth": int((f5 & k).sum()),
                     "geo_rule_whole_video_pct": round(100 * g.geo_rule.mean(), 2), "geo_hard_whole_video_pct": round(100 * g.geo_hard.mean(), 2), "closed_pct": round(100 * g.closed.mean(), 2)})
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "caught_per_video.csv", index=False)
    print(C.to_string(index=False))
    A[["video", "frame", "geo_z", "geo_rule", "closed", "worst_point"]].to_csv(OUT / "whole_video_scores.csv.gz", index=False, float_format="%.2f")


if __name__ == "__main__":
    main()
