#!/usr/bin/env python3
"""Choose the flagging threshold of the fitting deviation from the truth frames, in units of eye width.

    python fit_threshold.py

The finder is fixed (user, 2026-10-30): the fitting deviation of DeepLabCut's outlier detection, SARIMAX (2,0,0) fitted on
the whole video, one-step forward prediction, largest of the 8 key points (fit_o200_max in
results/unreliable_frames_sweep/signals_all_frames.csv), divided by the video's eye width (results/error_tier/
eye_width.csv) so that the threshold carries over to another camera or distance.

The threshold is read off the frames that must be found: the truth frames whose error is above the line of
results/error_tier/tier.csv (0.14 of the eye width, 23 of 1330 frames). For every candidate threshold t: recall = share
of the wrong frames with score > t; false share = share of all other frames (the good frames) with score > t.
With the line at 0.14 every wrong frame has a score of at least 0.257, so the curve is flat at full recall from 0.04 up
to 0.257 and the ROC optimum (max recall - false share) sits on the edge, at the smallest wrong frame's score: a
threshold with no margin, which leave-one-video-out shows (20 of 23 caught). The threshold used is therefore the line
itself: a frame is flagged when a point's fitting deviation exceeds the same amount of eye width that makes an error a
wrong frame. One number for the whole pipeline, no extra constant, and a margin of almost two to the smallest wrong
frame. The ROC optimum is kept in chosen.csv as the comparison; leave-one-video-out is reported for both rules.

Outputs in results/fit_threshold/: threshold_curve.csv (every candidate), chosen.csv (the threshold in eye widths and
its pixel equivalents, what it catches and flags), lovo.csv (per held-out video), fit_threshold_curve.png."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_step import HERE  # noqa: E402

RES = HERE / "results"
OUT = RES / "fit_threshold"
OUT.mkdir(parents=True, exist_ok=True)
SCORE = "fit_o200_max"


def table(M, grid, line):
    wrong, good = M[M.error_rel > line], M[M.error_rel <= line]
    rows = []
    for t in grid:
        rows.append({"threshold_eye_widths": t, "caught_wrong": int((wrong.score > t).sum()), "wrong_total": len(wrong), "recall": (wrong.score > t).mean(),
                     "flagged_good": int((good.score > t).sum()), "good_total": len(good), "false_share": (good.score > t).mean(),
                     "flagged_share_all": (M.score > t).mean()})
    T = pd.DataFrame(rows)
    T["recall_minus_false"] = T.recall - T.false_share
    return T


def choose(M, line):
    grid = np.unique(np.round(M.score.to_numpy(), 3))
    T = table(M, grid, line)
    best = T.loc[T.recall_minus_false.idxmax()]
    return float(best.threshold_eye_widths), T


def main():
    line = float(pd.read_csv(RES / "error_tier" / "tier.csv").iloc[0].line_eye_widths)
    E = pd.read_csv(RES / "error_tier" / "truth_errors.csv")[["video", "frame", "error_rel", "eye_width_px"]]
    S = pd.read_csv(RES / "unreliable_frames_sweep" / "signals_all_frames.csv")[["video", "frame", SCORE]]
    M = E.merge(S, on=["video", "frame"], how="inner")
    M["score"] = M[SCORE] / M.eye_width_px
    assert len(M) == 1330 and M.score.notna().all(), len(M)
    W = pd.read_csv(RES / "error_tier" / "eye_width.csv").eye_width_px
    opt, T = choose(M, line)
    T.round(4).to_csv(OUT / "threshold_curve.csv", index=False)
    rows = []
    for rule, thr in (("threshold = the error line", line), ("max(recall - false share), comparison", opt)):
        ch = table(M, [thr], line).iloc[0].to_dict()
        ch.update({"rule": rule, "line_eye_widths": line, "threshold_px_min": round(thr * W.min(), 1), "threshold_px_median": round(thr * W.median(), 1), "threshold_px_max": round(thr * W.max(), 1)})
        rows.append(ch)
        print(f"{rule}: threshold {thr:.3f} eye widths (≈ {ch['threshold_px_median']} px) catches {ch['caught_wrong']}/{ch['wrong_total']}, "
              f"flags {100 * ch['false_share']:.1f} % of the good frames, {100 * ch['flagged_share_all']:.1f} % of all truth frames")
    pd.DataFrame(rows).round(4).to_csv(OUT / "chosen.csv", index=False)
    thr = line
    print("neighbourhood:")
    print(T[(T.threshold_eye_widths >= 0.04) & (T.threshold_eye_widths <= 0.2)].iloc[::6][["threshold_eye_widths", "caught_wrong", "false_share", "flagged_share_all", "recall_minus_false"]].round(3).to_string(index=False))

    rows = []
    for v in sorted(M.video.unique()):
        topt, _ = choose(M[M.video != v], line)
        H = M[M.video == v]
        wrong, good = H[H.error_rel > line], H[H.error_rel <= line]
        for rule, tv in (("line", line), ("roc_optimum", topt)):
            rows.append({"rule": rule, "held_out_video": v, "threshold_from_other_18": tv, "wrong": len(wrong), "caught_wrong": int((wrong.score > tv).sum()),
                         "flagged_good": int((good.score > tv).sum()), "good": len(good), "flagged_share_all": round(float((H.score > tv).mean()), 4)})
    Lv = pd.DataFrame(rows)
    Lv.to_csv(OUT / "lovo.csv", index=False)
    for rule in ("line", "roc_optimum"):
        s = Lv[Lv.rule == rule]
        print(f"leave-one-video-out, {rule}: thresholds {sorted(s.threshold_from_other_18.unique())}, caught {int(s.caught_wrong.sum())}/{int(s.wrong.sum())}, "
              f"flagged {int(s.flagged_good.sum())}/{int(s.good.sum())} good frames")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    Tp = T[(T.threshold_eye_widths >= 0.01) & (T.threshold_eye_widths <= 0.3)]
    xp = 100 * Tp.threshold_eye_widths
    ax.plot(xp, 100 * Tp.recall, color="#2f6db3", lw=2, label="wrong frames caught (%)")
    ax.plot(xp, 100 * Tp.false_share, color="#d1495b", lw=2, label="good frames flagged (%)")
    ax.plot(xp, 100 * Tp.recall_minus_false, color="0.3", lw=1.2, ls="--", label="difference (%)")
    ax.axvline(100 * thr, color="0.6", lw=1)
    ax.text(100 * thr + 0.3, 50, f"used: {100 * thr:.0f} % of eye width\n(= the error line)", color="0.3", fontsize=9)
    ax.axvline(100 * opt, color="0.8", lw=1, ls=":")
    ax.text(100 * opt + 0.3, 30, f"ROC optimum {100 * opt:.1f} %\n(no margin)", color="0.5", fontsize=9)
    ax.set_xlabel("threshold on the fitting deviation, largest of 8 points, in % of the eye width")
    ax.set_ylabel("%")
    ax.set_title("Flagging threshold read off the 1330 truth frames")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="0.92")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.savefig(OUT / "fit_threshold_curve.png", dpi=150)


if __name__ == "__main__":
    main()
