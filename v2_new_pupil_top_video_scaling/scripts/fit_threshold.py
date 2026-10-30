#!/usr/bin/env python3
"""Choose the flagging threshold of the fitting deviation from the truth frames, the way the blink threshold was chosen.

    python fit_threshold.py

The finder is fixed (user, 2026-10-30): the fitting deviation of DeepLabCut's outlier detection, SARIMAX (2,0,0) fitted on
the whole video, one-step forward prediction, largest of the 8 key points (fit_o200_max in
results/unreliable_frames_sweep/signals_all_frames.csv). What was open is the threshold: until now it was the 95 %
quantile of the score on the truth frames (44 px), a budget of frames to flag with no basis in the errors themselves.

Here the threshold is read off the frames that must be found: the truth frames with frame RMSE > 50 px (32 of 1330).
For every candidate threshold t: recall = share of the > 50 px frames with score > t; false share = share of the frames
<= 20 px with score > t; the 20-50 px frames are neither target nor clean and are only reported. Chosen t = the one that
maximises recall - false share (the usual ROC optimum, no free constant). Leave-one-video-out: the same rule on the other
18 videos, applied to the held-out video, to see whether the pooled choice holds up.

Outputs in results/fit_threshold/: threshold_curve.csv (every candidate), chosen.csv (pooled threshold and what it
does), lovo.csv (per held-out video), fit_threshold_curve.png. Nothing existing is modified."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_step import HERE  # noqa: E402

RES = HERE / "results"
OUT = RES / "fit_threshold"
OUT.mkdir(parents=True, exist_ok=True)
SCORE, BIG, CLEAN = "fit_o200_max", 50.0, 20.0


def table(M, grid):
    big, clean, mid = M[M.rmse > BIG], M[M.rmse <= CLEAN], M[(M.rmse > CLEAN) & (M.rmse <= BIG)]
    rows = []
    for t in grid:
        rows.append({"threshold_px": t, "caught_gt50": int((big.score > t).sum()), "positives_gt50": len(big),
                     "recall_gt50": (big.score > t).mean(), "flagged_le20": int((clean.score > t).sum()), "n_le20": len(clean),
                     "false_share_le20": (clean.score > t).mean(), "flagged_20to50": int((mid.score > t).sum()), "n_20to50": len(mid),
                     "flagged_share_all": (M.score > t).mean()})
    T = pd.DataFrame(rows)
    T["recall_minus_false"] = T.recall_gt50 - T.false_share_le20
    return T


def choose(M):
    grid = np.unique(np.floor(M.score.to_numpy()))  # integer pixels: a threshold of t flags score > t
    T = table(M, grid)
    best = T.loc[T.recall_minus_false.idxmax()]
    return float(best.threshold_px), T


def main():
    S = pd.read_csv(RES / "unreliable_frames_sweep" / "signals_all_frames.csv")[["video", "frame", SCORE]]
    L = pd.read_csv(RES / "error_detector" / "scores_lovo.csv")
    L = L[L.plateau].drop_duplicates(["video", "frame"])[["video", "frame", "frame_rmse_px"]]
    M = L.merge(S, on=["video", "frame"], how="inner").rename(columns={SCORE: "score", "frame_rmse_px": "rmse"})
    assert len(M) == 1330 and M.score.notna().all(), len(M)
    thr, T = choose(M)
    T.round(4).to_csv(OUT / "threshold_curve.csv", index=False)
    chosen = T[T.threshold_px == thr].iloc[0].to_dict()
    old = T[T.threshold_px == 44].iloc[0].to_dict()
    pd.DataFrame([{"rule": "max(recall - false share)", **chosen}, {"rule": "95 % quantile (previous)", **old}]).round(4).to_csv(OUT / "chosen.csv", index=False)
    print(f"chosen threshold {thr:.0f} px: catches {chosen['caught_gt50']}/{chosen['positives_gt50']} > 50 px, flags {100 * chosen['false_share_le20']:.1f} % of the <= 20 px frames, {100 * chosen['flagged_share_all']:.1f} % of all truth frames")
    print(f"previous 44 px: catches {old['caught_gt50']}/{old['positives_gt50']}, flags {100 * old['false_share_le20']:.1f} % / {100 * old['flagged_share_all']:.1f} %")

    rows = []
    for v in sorted(M.video.unique()):
        tv, _ = choose(M[M.video != v])
        H = M[M.video == v]
        big, clean = H[H.rmse > BIG], H[H.rmse <= CLEAN]
        rows.append({"held_out_video": v, "threshold_from_other_18_px": tv, "positives_gt50": len(big), "caught_gt50": int((big.score > tv).sum()),
                     "flagged_le20": int((clean.score > tv).sum()), "n_le20": len(clean), "flagged_share_all": round(float((H.score > tv).mean()), 4)})
    Lv = pd.DataFrame(rows)
    Lv.to_csv(OUT / "lovo.csv", index=False)
    print(Lv.to_string(index=False))
    print(f"leave-one-video-out: thresholds {sorted(Lv.threshold_from_other_18_px.unique())}, caught {int(Lv.caught_gt50.sum())}/{int(Lv.positives_gt50.sum())}, "
          f"flagged {int(Lv.flagged_le20.sum())}/{int(Lv.n_le20.sum())} of the <= 20 px frames")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    Tp = T[(T.threshold_px >= 5) & (T.threshold_px <= 120)]
    ax.plot(Tp.threshold_px, 100 * Tp.recall_gt50, color="#2f6db3", lw=2, label="recall of the > 50 px frames (%)")
    ax.plot(Tp.threshold_px, 100 * Tp.false_share_le20, color="#d1495b", lw=2, label="share of the <= 20 px frames flagged (%)")
    ax.plot(Tp.threshold_px, 100 * Tp.recall_minus_false, color="0.3", lw=1.2, ls="--", label="difference (%)")
    ax.axvline(thr, color="0.6", lw=1)
    ax.text(thr + 1, 50, f"chosen {thr:.0f} px", color="0.3", fontsize=9)
    ax.axvline(44, color="0.8", lw=1, ls=":")
    ax.text(45, 30, "previous 44 px", color="0.5", fontsize=9)
    ax.set_xlabel("threshold on the fitting deviation, largest of 8 points (px)")
    ax.set_ylabel("%")
    ax.set_title("Flagging threshold read off the 1330 truth frames")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="0.92")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.savefig(OUT / "fit_threshold_curve.png", dpi=150)


if __name__ == "__main__":
    main()
