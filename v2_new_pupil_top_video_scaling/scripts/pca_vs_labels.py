#!/usr/bin/env python3
"""How the PCA finder depends on the number of labeled poses it is fitted on (report 9.2).

    python pca_vs_labels.py

For every video the PCA of the pose (8 points, 16 coordinates, 99 % of variance, at least 2 components) is fitted on
the first k labeled poses in active-learning order (batch01, batch02, ...), k = 5, 10, 15, 20, 30, 40, 60, 80 where
that many exist; poses that are truth frames are never used. For each k: on the 1330 truth frames, the share of right
frames flagged when the threshold is set to find every wrong frame of the videos available at that k (line =
results/error_tier/tier.csv); on every frame of the videos, the share flagged at the fixed threshold of
finder_comparison.py (3.0 % of eye width) and at the median-multiple rule of pca_threshold.py, and the median score of the
video (its background).
Outputs results/finder_comparison/pca_vs_labels.csv (per video and k), pca_vs_labels_summary.csv, pca_vs_labels.png.
Nothing existing is modified."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finder_comparison as F  # noqa: E402
import unreliable_frames_sweep as U  # noqa: E402
from scale_step import BPS, HERE, flat, human_table  # noqa: E402
from second_peak_repair import predictions  # noqa: E402

KS = [5, 10, 15, 20, 30, 40, 60, 80]
THR = float(pd.read_csv(F.OUT / "whole_video_share.csv").pca_threshold.iloc[0])
MULT = float(pd.read_csv(F.OUT / "pca_threshold_rule.csv").iloc[0].M)   # the median-multiple rule of pca_threshold.py
line = F.line


def poses_in_order(unit, exclude):
    parts = []
    for folder in sorted((HERE / unit / "training-data" / "labels").glob("batch*")):
        if list(folder.glob("CollectedData_*.h5")):
            parts.append(flat(human_table(folder)))
    L = pd.concat(parts)
    frames = np.array([int(re.search(r"(\d+)", i).group(1)) for i in L.index])
    X = np.stack([np.stack([L[(b, "x")].to_numpy(float), L[(b, "y")].to_numpy(float)], axis=1) for b in BPS], axis=1)
    ok = np.isfinite(X).all(axis=(1, 2)) & ~np.isin(frames, list(exclude))
    return X[ok], int((~ok & np.isfinite(X).all(axis=(1, 2))).sum())


def pca_error(train, P):
    Xt = train.reshape(len(train), -1)
    pca = PCA(n_components=0.99, svd_solver="full").fit(Xt)
    if pca.n_components_ < 2:
        pca = PCA(n_components=2, svd_solver="full").fit(Xt)
    R = pca.inverse_transform(pca.transform(P.reshape(len(P), -1))).reshape(P.shape)
    return np.sqrt(np.mean(np.sum((P - R) ** 2, axis=2), axis=1)), pca.n_components_


def main():
    E = F.E
    rows, truth = [], []
    for v in sorted(E.video.unique()):
        v = int(v)
        rr, d = predictions(v)
        tf = E[E.video == v]
        train, n_excl = poses_in_order(rr.unit, set(tf.frame))
        P = U.pose_matrix(d)
        idx = tf.frame.to_numpy()
        for k in KS:
            if k > len(train):
                continue
            s, nc = pca_error(train[:k], P)
            s = s / F.W[v]
            med = np.nanmedian(s)
            rows.append({"video": v, "k": k, "poses_available": len(train), "truth_frames_excluded": n_excl, "components": nc,
                         "video_median_pct": 100 * med, "video_p90_pct": 100 * np.nanquantile(s, 0.9), "flagged_at_fixed_thr_pct": 100 * np.nanmean(s > THR),
                         "flagged_at_median_rule_pct": 100 * np.nanmean(s > MULT * med), "wrong_caught_median_rule": int((s[idx][(tf.error_rel > line).to_numpy()] > MULT * med).sum())})
            truth.append(pd.DataFrame({"video": v, "k": k, "frame": idx, "score": s[idx], "wrong": (tf.error_rel > line).to_numpy()}))
        print("video", v, "poses", len(train), "excluded truth", n_excl)
    R = pd.DataFrame(rows)
    T = pd.concat(truth)
    R.round(3).to_csv(F.OUT / "pca_vs_labels.csv", index=False)
    summ = []
    for k in KS:
        Rk, Tk = R[R.k == k], T[T.k == k]
        if Rk.empty:
            continue
        vids = set(Rk.video)
        Tk = Tk[Tk.video.isin(vids)]
        tmin = Tk[Tk.wrong].score.min()
        good = Tk[~Tk.wrong]
        summ.append({"k": k, "videos": len(vids), "wrong_frames": int(Tk.wrong.sum()), "good_frames": len(good), "good_flagged_at_full_recall_pct": 100 * (good.score >= tmin).mean(),
                     "threshold_finds_all_pct": 100 * tmin, "wrong_caught_at_fixed_thr": int((Tk[Tk.wrong].score > THR).sum()), "good_flagged_at_fixed_thr_pct": 100 * (good.score > THR).mean(),
                     "whole_video_flagged_median_pct": Rk.flagged_at_fixed_thr_pct.median(), "whole_video_flagged_max_pct": Rk.flagged_at_fixed_thr_pct.max(),
                     "median_rule_flagged_median_pct": Rk.flagged_at_median_rule_pct.median(), "median_rule_flagged_max_pct": Rk.flagged_at_median_rule_pct.max(),
                     "median_rule_wrong_caught": int(Rk.wrong_caught_median_rule.sum()),
                     "video_median_score_median_pct": Rk.video_median_pct.median(), "components_median": Rk.components.median()})
    S = pd.DataFrame(summ)
    S.round(3).to_csv(F.OUT / "pca_vs_labels_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(S.round(2).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)
    ax.plot(S.k, S.whole_video_flagged_max_pct, "^-", color="#d1495b", lw=1.6, label=f"fixed threshold ({100 * THR:.1f} % of eye width): worst video")
    ax.plot(S.k, S.whole_video_flagged_median_pct, "o-", color="#e0a458", lw=1.6, label="fixed threshold: median video")
    ax.plot(S.k, S.median_rule_flagged_max_pct, "^-", color="#2f6db3", lw=1.6, label=f"{MULT:.1f} × the video's median (used): worst video")
    ax.plot(S.k, S.median_rule_flagged_median_pct, "o-", color="#7fa7d6", lw=1.6, label=f"{MULT:.1f} × the video's median (used): median video")
    for k, c, n in zip(S.k, S.median_rule_wrong_caught, S.wrong_frames):
        ax.annotate(f"{c}/{n}", (k, S.median_rule_flagged_max_pct[S.k == k].iloc[0]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="#2f6db3")
    ax.set_xlabel("labeled poses the PCA is fitted on")
    ax.set_ylabel("% of the video's frames flagged")
    ax.set_title("Whole videos: share flagged under the two threshold rules against the number of labels\n(blue numbers: wrong truth frames found by the median rule)", fontsize=10)
    ax.set_xticks(KS)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="0.92")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5)
    fig.savefig(F.OUT / "pca_vs_labels.png", dpi=150)


if __name__ == "__main__":
    main()
