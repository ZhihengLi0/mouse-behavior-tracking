#!/usr/bin/env python3
"""The PCA finder as it would be used: scores for every frame of every video, and its threshold rule (report 9.2).

    python pca_threshold.py

Finder: the pose of a frame (8 points, 16 coordinates, from the plateau model's predictions) is projected on the principal
components (99 % of the variance, at least 2) of the labeled poses the plateau model was trained on, the same ones
unreliable_frames_sweep.py used (labeled_poses(unit, step)); the point's residual = |pose - reconstruction| / eye width,
the frame's score = RMS of the 8 residuals. No truth frame is among the training poses.

Threshold rule (user, 2026-10-31): a threshold fixed in eye widths drifts with the number of labels the PCA saw and
leaves video 18 at 18 % flagged (results/finder_comparison/pca_vs_labels.csv), so the threshold is set per video from
the video's own frames, no labels: threshold = M × median score of the video. M is read off the 1330 truth frames the
way the fitting threshold was: the smallest multiple at which a wrong frame (error > the line of results/error_tier/
tier.csv) sits, divided by the margin the fitting line leaves to its smallest wrong frame (0.257 / 0.14). One constant
for every video; leave-one-video-out reported. The fixed rule of finder_comparison.py (3.0 % of eye width) is kept in
the table for comparison.

Outputs in results/finder_comparison/: pca_cache/videoNN_pca.npz (score, residuals (T, 8), median, threshold; the
repair reads these), pca_threshold.csv (per video), pca_threshold_rule.csv (M, margin, leave-one-video-out),
pca_threshold.png (flagged share per video, fixed rule against median-multiple rule). Nothing existing is modified."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unreliable_frames_sweep as U  # noqa: E402
from second_peak_repair import RES, predictions  # noqa: E402

OUT = RES / "finder_comparison"
CACHE = OUT / "pca_cache"
CACHE.mkdir(parents=True, exist_ok=True)
line = float(pd.read_csv(RES / "error_tier" / "tier.csv").iloc[0].line_eye_widths)
E = pd.read_csv(RES / "error_tier" / "truth_errors.csv")[["video", "frame", "error_rel", "eye_width_px"]]
W = dict(zip(E.video, E.eye_width_px))
FIXED = float(pd.read_csv(OUT / "whole_video_share.csv").pca_threshold.iloc[0])


def residuals(train, P):
    """(T, 8) per-point |pose - PCA reconstruction| in px; PCA keeps 99 % of the variance, at least 2 components."""
    Xt = train.reshape(len(train), -1)
    pca = PCA(n_components=0.99, svd_solver="full").fit(Xt)
    if pca.n_components_ < 2:
        pca = PCA(n_components=2, svd_solver="full").fit(Xt)
    R = pca.inverse_transform(pca.transform(P.reshape(len(P), -1))).reshape(P.shape)
    return np.sqrt(np.sum((P - R) ** 2, axis=2)), pca.n_components_


def main():
    S = pd.read_csv(RES / "unreliable_frames_sweep" / "signals_all_frames.csv")[["video", "frame", "pca_video"]]
    M = E.merge(S, on=["video", "frame"])
    rows, truth = [], []
    for v in sorted(E.video.unique()):
        v = int(v)
        rr, d = predictions(v)
        step = max(1, int(rr.labels_at_plateau) // 20)
        train = U.labeled_poses(rr.unit, step)
        P = U.pose_matrix(d)
        res_px, nc = residuals(train, P)
        res = res_px / W[v]
        score = np.sqrt(np.mean(res ** 2, axis=1))
        med = float(np.nanmedian(score))
        tf = M[M.video == v]
        chk = np.nanmax(np.abs(score[tf.frame.to_numpy()] - tf.pca_video.to_numpy() / W[v]))
        assert chk < 1e-9, (v, chk)
        rows.append({"video": v, "unit": rr.unit, "frames": len(P), "labels_seen": len(train), "components": nc, "eye_width_px": W[v], "median_score": med})
        truth.append(pd.DataFrame({"video": v, "frame": tf.frame.to_numpy(), "score": score[tf.frame.to_numpy()], "multiple": score[tf.frame.to_numpy()] / med, "wrong": (tf.error_rel > line).to_numpy()}))
        np.savez(CACHE / f"video{v:02d}_pca.npz", score=score, residuals=res, median=med, eye_width_px=W[v], labels_seen=len(train), components=nc)
        print(f"video {v}: {len(train)} poses, {nc} components, median score {100 * med:.2f} % of eye width")
    V = pd.DataFrame(rows)
    T = pd.concat(truth, ignore_index=True)

    fit_line, fit_min = line, None
    F = pd.read_csv(RES / "fit_threshold" / "chosen.csv")
    fit_min = float(F[F.rule.str.startswith("max(recall")].threshold_eye_widths.iloc[0])   # the smallest wrong frame's fitting score (0.257)
    margin = fit_min / fit_line
    m_full = float(T[T.wrong].multiple.min())
    Mult = m_full / margin
    good = T[~T.wrong]
    lovo = []
    for v in sorted(T.video.unique()):
        o = T[(T.video != v) & T.wrong].multiple.min() / margin
        h = T[T.video == v]
        lovo.append({"held_out_video": v, "multiple_from_other_18": o, "wrong": int(h.wrong.sum()), "caught": int((h[h.wrong].multiple > o).sum()), "good": int((~h.wrong).sum()), "good_flagged": int((h[~h.wrong].multiple > o).sum())})
    Lv = pd.DataFrame(lovo)
    rule = pd.DataFrame([{"rule": "threshold = M x video median", "M": Mult, "M_finds_all": m_full, "margin": margin, "caught": int((T[T.wrong].multiple > Mult).sum()), "wrong": int(T.wrong.sum()),
                          "good_flagged": int((good.multiple > Mult).sum()), "good": len(good), "good_above_smallest_wrong": int((good.multiple >= m_full).sum()),
                          "lovo_caught": int(Lv.caught.sum()), "lovo_good_flagged": int(Lv.good_flagged.sum()), "lovo_M_min": Lv.multiple_from_other_18.min(), "lovo_M_max": Lv.multiple_from_other_18.max()},
                         {"rule": "fixed, eye widths (comparison)", "M": FIXED, "M_finds_all": float(T[T.wrong].score.min()), "margin": margin, "caught": int((T[T.wrong].score > FIXED).sum()), "wrong": int(T.wrong.sum()),
                          "good_flagged": int((good.score > FIXED).sum()), "good": len(good), "good_above_smallest_wrong": int((good.score >= T[T.wrong].score.min()).sum()),
                          "lovo_caught": np.nan, "lovo_good_flagged": np.nan, "lovo_M_min": np.nan, "lovo_M_max": np.nan}])
    rule.round(4).to_csv(OUT / "pca_threshold_rule.csv", index=False)
    Lv.round(3).to_csv(OUT / "pca_threshold_lovo.csv", index=False)
    print(rule.T.to_string())

    V["threshold_eye_widths"] = Mult * V.median_score
    V["threshold_px"] = V.threshold_eye_widths * V.eye_width_px
    sh_m, sh_f = [], []
    for v in V.video:
        z = np.load(CACHE / f"video{v:02d}_pca.npz")
        thr = Mult * float(z["median"])
        np.savez(CACHE / f"video{v:02d}_pca.npz", score=z["score"], residuals=z["residuals"], median=z["median"], eye_width_px=z["eye_width_px"], labels_seen=z["labels_seen"], components=z["components"], threshold=thr, multiple=Mult)
        sh_m.append(100 * np.nanmean(z["score"] > thr))
        sh_f.append(100 * np.nanmean(z["score"] > FIXED))
    V["flagged_pct_median_rule"] = sh_m
    V["flagged_pct_fixed_rule"] = sh_f
    V.round(4).to_csv(OUT / "pca_threshold.csv", index=False)
    pd.set_option("display.width", 200)
    print(V[["video", "labels_seen", "components", "median_score", "threshold_eye_widths", "threshold_px", "flagged_pct_median_rule", "flagged_pct_fixed_rule"]].round(3).to_string(index=False))
    tot = V.frames.sum()
    print(f"all frames: median rule flags {(V.flagged_pct_median_rule * V.frames).sum() / tot:.2f} %, fixed rule {(V.flagged_pct_fixed_rule * V.frames).sum() / tot:.2f} %")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    xs = np.arange(len(V))
    ax.bar(xs - 0.2, V.flagged_pct_fixed_rule, 0.4, color="0.7", label=f"fixed threshold, {100 * FIXED:.1f} % of eye width for every video")
    ax.bar(xs + 0.2, V.flagged_pct_median_rule, 0.4, color="#d1495b", label=f"threshold = {Mult:.1f} × the video's own median score (used)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v}\n{n}" for v, n in zip(V.video, V.labels_seen)], fontsize=8.5)
    ax.set_xlabel("video (second line: labeled poses the video's PCA was fitted on)")
    ax.set_ylabel("% of the video's frames flagged")
    ax.set_title("PCA finder on every frame of the 19 videos: two threshold rules, both set to find all 23 wrong truth frames")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.92")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(OUT / "pca_threshold.png", dpi=150)


if __name__ == "__main__":
    main()
