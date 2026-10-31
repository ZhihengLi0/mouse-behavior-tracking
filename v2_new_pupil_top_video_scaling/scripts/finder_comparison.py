#!/usr/bin/env python3
"""Which finder of unreliable frames to use, compared at the error line (report 9.2).

    python finder_comparison.py

Part 1, every candidate signal on the 1330 truth frames against the line of results/error_tier/tier.csv (0.14 of the
eye width, 23 wrong frames): for each signal the threshold that just finds all 23 wrong frames (its smallest wrong-frame
score), the share of the 1307 good frames flagged at that threshold, and leave-one-video-out (threshold from the other
18 videos, applied to the held-out one). Signals: the sweep of results/unreliable_frames_sweep/signals_all_frames.csv
(DeepLabCut jump / uncertain / fitting and variants, two-seed ensemble, geometry, PCA reconstruction error, rank and
logistic combinations) and the trained detectors of results/error_detector/ (leave-one-video-out scores).
-> signals_at_line.csv, signals_at_line.png

Part 2, fitting (largest of 8, order (2,0,0), threshold = the line) against the PCA reconstruction error (PCA of the
labeled poses the plateau model was trained on, threshold = its just-finds-all score with the same margin as the line,
0.14 / 0.257) on every frame of the 19 videos: flagged share per video next to the number of labels the video's PCA
saw. -> whole_video_share.csv, whole_video_share.png; examples.jpg = frames flagged by the PCA but not by fitting
(random sample from the videos where the PCA flags most), model points drawn, to be judged by eye.
Nothing existing is modified."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unreliable_frames_sweep as U  # noqa: E402
from second_peak_repair import BPS, HERE, RES, predictions  # noqa: E402

OUT = RES / "finder_comparison"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = RES / "unreliable_frames_sweep" / "cache"
line = float(pd.read_csv(RES / "error_tier" / "tier.csv").iloc[0].line_eye_widths)
E = pd.read_csv(RES / "error_tier" / "truth_errors.csv")[["video", "frame", "error_rel", "eye_width_px"]]
W = dict(zip(E.video, E.eye_width_px))
PL = pd.read_csv(RES / "labels_to_plateau.csv")
PL = PL[PL.status == "final"].set_index("video")
rng = np.random.default_rng(0)


def truth_table():
    S = pd.read_csv(RES / "unreliable_frames_sweep" / "signals_all_frames.csv").drop(columns=["eye_width_px"], errors="ignore")
    L = pd.read_csv(RES / "error_detector" / "scores_lovo.csv")
    L = L[L.plateau].drop_duplicates(["video", "frame"])
    M = E.merge(S, on=["video", "frame"]).merge(L[["video", "frame", "d1_pairs_all_deploy", "d1_pairs_all", "d1_pairs_nocontext", "d1_plateau_all"]], on=["video", "frame"], how="left")
    assert len(M) == 1330
    ew = M.eye_width_px
    sig = {  # name: (values, high = suspicious; family, needs labels of this video)
        "jump (DLC): largest move since previous frame": (M.jump, "DeepLabCut", False),
        "uncertain (DLC): lowest confidence": (-M.min_conf, "DeepLabCut", False),
        "mean confidence": (-M.mean_conf, "DeepLabCut", False),
        "fitting (DLC default): mean of 8 points, order (3,0,1)": (M.fitting / ew, "DeepLabCut", False),
        "fitting, largest of 8, order (3,0,1)": (M.fit_max / ew, "fitting variant", False),
        "fitting, largest of 8, order (2,0,0)  [used]": (M.fit_o200_max / ew, "fitting variant", False),
        "fitting, largest of 8, order (1,0,0)": (M.fit_o100_max / ew, "fitting variant", False),
        "fitting, largest of 8, order (5,0,1)": (M.fit_o501_max / ew, "fitting variant", False),
        "fitting, mean of 4 pupil points": (M.fit_pupil / ew, "fitting variant", False),
        "fitting, forward and backward, largest": (M.fit_fb_max / ew, "fitting variant", False),
        "two models (seeds): largest disagreement": (M.ens_sd_max / ew, "ensemble", False),
        "geometry: pupil axes midpoint offset": (M.geo_axes_off, "geometry", False),
        "geometry: pupil width deviation": (M.geo_width_dev, "geometry", False),
        "PCA of this video's training poses: reconstruction error": (M.pca_video / ew, "PCA", True),
        "PCA of the mouse's training poses: reconstruction error": (M.pca_mouse / ew, "PCA", True),
        "rank combination (fitting, PCA, ensemble, confidence)": (M.combo_rank, "combination", True),
        "logistic combination of the signals (leave-one-video-out)": (M.combo_lr, "combination", True),
        "trained detector D1 (leave-one-video-out)": (M.d1_pairs_all_deploy, "trained detector", True),
        "trained detector D1, no context": (M.d1_pairs_nocontext, "trained detector", True),
    }
    w = (M.error_rel > line).to_numpy()
    rows = []
    for name, (v, fam, lab) in sig.items():
        v = v.to_numpy(float)
        ok = np.isfinite(v)
        vv, ww = v[ok], w[ok]
        t = vv[ww].min()
        fs = (vv[~ww] >= t).mean()
        c = g = 0
        for vid in sorted(M.video.unique()):
            held = (M.video == vid).to_numpy()[ok]
            tv = vv[~held & ww].min()
            c += (vv[held & ww] >= tv).sum()
            g += (vv[held & ~ww] >= tv).sum()
        from sklearn.metrics import roc_auc_score
        rows.append({"signal": name, "family": fam, "needs_labels_of_video": lab, "n": int(ok.sum()), "auc": roc_auc_score(ww, vv),
                     "threshold_finds_all": t, "good_flagged_at_full_recall": int((vv[~ww] >= t).sum()), "good_flagged_share": fs,
                     "lovo_caught": int(c), "lovo_good_flagged": int(g)})
    T = pd.DataFrame(rows).sort_values("good_flagged_share")
    T.round(4).to_csv(OUT / "signals_at_line.csv", index=False)
    return T, M


def fig_signals(T):
    """One row per family of signals; the form used on the page gives the bar, the other forms of the family the whisker."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fam = [("PCA of this video's training poses  [used]", "PCA", lambda n: n.startswith("PCA of this video"), "#d1495b"),
           ("fitting: largest of 8, order (2,0,0)  [used]\n(whisker: the other 5 forms of fitting over 8 points)", "fitting variant", lambda n: "[used]" in n, "#2f6db3"),
           ("trained detector (leave-one-video-out), 2 forms", "trained detector", lambda n: n.startswith("trained detector D1 (leave"), "0.55"),
           ("combinations of signals, 2 forms", "combination", lambda n: n.startswith("rank combination"), "0.55"),
           ("two models trained with different seeds: disagreement", "ensemble", lambda n: n.startswith("two models"), "0.55"),
           ("PCA of the mouse's training poses (other videos)", "PCA", lambda n: n.startswith("PCA of the mouse"), "0.55"),
           ("confidence (DLC uncertain), 2 forms", "DeepLabCut", lambda n: n.startswith("uncertain"), "0.55"),
           ("jump (DLC): largest move since the previous frame", "DeepLabCut", lambda n: n.startswith("jump"), "0.55"),
           ("geometry of the pupil points, 2 forms", "geometry", lambda n: n.startswith("geometry"), "0.55")]
    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    labels = []
    for i, (label, family, pick, col) in enumerate(fam):
        main = T[T.signal.apply(pick)]
        assert len(main) >= 1, label
        group = T[(T.family == family)] if family in ("fitting variant", "trained detector", "combination", "geometry") else main
        if family == "fitting variant":
            group = T[T.signal.str.startswith("fitting") & ~T.signal.str.contains("4 pupil")]
        if "confidence" in label:
            group = T[T.signal.str.startswith("uncertain") | T.signal.str.startswith("mean confidence")]
        r = main.iloc[0]
        x = max(100 * r.good_flagged_share, 0.045)
        ax.barh(i, x, color=col, height=0.6)
        if r.good_flagged_share == 0:
            ax.plot(0.045, i, marker=">", color=col, ms=11, clip_on=False)
            txt = f"0 of 1307 right frames flagged"
        else:
            txt = f"{100 * r.good_flagged_share:.1f} %"
        if len(group) > 1:
            lo, hi = 100 * group.good_flagged_share.min(), 100 * group.good_flagged_share.max()
            ax.plot([max(lo, 0.045), hi], [i, i], color="0.2", lw=1.4)
            ax.plot([max(lo, 0.045), hi], [i, i], "|", color="0.2", ms=9)
            txt += f"   (forms: {lo:.1f}–{hi:.1f} %)"
            xt = hi
        else:
            xt = x if r.good_flagged_share > 0 else 0.056
        txt += f"   held-out videos: {int(r.lovo_caught)} of 23 found"
        ax.text(xt * 1.12, i, txt, va="center", fontsize=8.5, color=col if col != "0.55" else "0.25")
        labels.append(label)
    ax.set_yticks(range(len(fam)))
    ax.set_yticklabels(labels, fontsize=9)
    for lab in ax.get_yticklabels():
        if "[used]" in lab.get_text():
            lab.set_color("#d1495b" if lab.get_text().startswith("PCA") else "#2f6db3")
            lab.set_fontweight("bold")
    ax.invert_yaxis()
    ax.axhline(3.5, color="0.4", lw=1, ls="--")
    ax.text(380, 3.38, "first tier", ha="right", va="bottom", fontsize=9, color="0.35", style="italic")
    ax.text(380, 3.62, "second tier", ha="right", va="top", fontsize=9, color="0.35", style="italic")
    ax.set_xscale("log")
    ax.set_xlim(0.04, 400)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(["0.1 %", "1 %", "10 %", "100 %"])
    ax.set_xlabel("share of the 1307 right truth frames flagged when the threshold is set to find all 23 wrong frames (log scale)")
    ax.set_title("Candidate finders at the error line (14 % of eye width), 1330 truth frames: further left is better", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="0.92")
    ax.set_axisbelow(True)
    fig.savefig(OUT / "signals_at_line.png", dpi=150)


def whole_video(M):
    w = (M.error_rel > line).to_numpy()
    pca_truth = (M.pca_video / M.eye_width_px).to_numpy()
    fit_truth = (M.fit_o200_max / M.eye_width_px).to_numpy()
    margin = fit_truth[w].min() / line                      # the margin the line leaves to the smallest wrong frame (≈ 1.84)
    thr_pca = pca_truth[w].min() / margin
    rows, scores = [], {}
    for v in sorted(E.video.unique()):
        v = int(v)
        rr, d = predictions(v)
        step = max(1, int(rr.labels_at_plateau) // 20)
        P = U.pose_matrix(d)
        lab = U.labeled_poses(rr.unit, step)
        pca = U.pca_error(lab, P) / W[v]
        fit = np.nanmax(np.load(CACHE / f"video{v:02d}_fwd200.npy"), axis=1) / W[v]
        scores[v] = (pca > thr_pca, fit > line)
        rows.append({"video": v, "unit": rr.unit, "frames": len(P), "labels_seen_by_pca": len(lab), "labels_at_plateau": int(rr.labels_at_plateau),
                     "pca_threshold": thr_pca, "fit_threshold": line, "pca_flagged_pct": 100 * np.nanmean(pca > thr_pca), "fit_flagged_pct": 100 * np.nanmean(fit > line),
                     "pca_only_pct": 100 * np.nanmean((pca > thr_pca) & ~(fit > line)), "fit_only_pct": 100 * np.nanmean(~(pca > thr_pca) & (fit > line))})
    S = pd.DataFrame(rows)
    S.round(3).to_csv(OUT / "whole_video_share.csv", index=False)
    print(S[["video", "labels_seen_by_pca", "pca_flagged_pct", "fit_flagged_pct", "pca_only_pct"]].round(2).to_string(index=False))
    print(f"margin {margin:.2f}, pca threshold {thr_pca:.3f}; all frames: pca {100 * sum(a.sum() for a, _ in scores.values()) / S.frames.sum():.2f} %, fit {100 * sum(b.sum() for _, b in scores.values()) / S.frames.sum():.2f} %")
    return S, scores


def fig_share(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.4), constrained_layout=True)
    xs = np.arange(len(S))
    ax.bar(xs - 0.2, S.fit_flagged_pct, 0.4, color="#2f6db3", label="fitting, threshold = the line (14 % of eye width)")
    ax.bar(xs + 0.2, S.pca_flagged_pct, 0.4, color="#d1495b", label=f"PCA reconstruction error, threshold {100 * S.pca_threshold.iloc[0]:.1f} % of eye width (same margin to the smallest wrong frame)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v}\n{n} labels" for v, n in zip(S.video, S.labels_seen_by_pca)], fontsize=8.5)
    ax.set_xlabel("video, and the number of labeled poses the video's PCA was fitted on")
    ax.set_ylabel("% of the video's frames flagged")
    ax.set_title("Flagged share of every frame of the 19 videos: fitting against the PCA, both set to find all 23 wrong truth frames")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.92")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(OUT / "whole_video_share.png", dpi=150)


def examples(S, scores, n_per=4, videos=None):
    TW, TH = 300, 225
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    fb = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    if videos is None:
        videos = list(S.sort_values("pca_only_pct", ascending=False).video.iloc[:4])
    tiles = []
    for v in videos:
        pca_f, fit_f = scores[v]
        cand = np.flatnonzero(pca_f & ~fit_f)
        pick = np.sort(rng.choice(cand, size=min(n_per, len(cand)), replace=False))
        rr, d = predictions(v)
        cap = cv2.VideoCapture(str(HERE / rr.unit / f"{rr.unit}.mp4"))
        xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
        for f in pick:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, img = cap.read()
            assert ok, (v, f)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pts = np.array([xy[b][f] for b in BPS])
            pts = pts[np.isfinite(pts).all(1)]
            cx, cy = pts.mean(0)
            wdt = max(1.6 * W[v], np.ptp(pts[:, 0]) + 60)
            hgt = wdt * TH / TW
            box = (int(cx - wdt / 2), int(cy - hgt / 2), int(cx + wdt / 2), int(cy + hgt / 2))
            crop = img.crop(box).resize((TW, TH))
            sc = TW / wdt
            cd = ImageDraw.Draw(crop)
            for b in BPS:
                px, py = (xy[b][f] - box[:2]) * sc
                if np.isfinite(px):
                    cd.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(255, 60, 60))
            cd.rectangle([0, 0, TW, 18], fill=(0, 0, 0))
            cd.text((3, 2), f"video {v}  frame {f}  flagged by the PCA, not by fitting", fill=(255, 255, 255), font=font)
            tiles.append(crop)
    cols = n_per
    rows_n = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * TW, 30 + rows_n * TH), (255, 255, 255))
    ImageDraw.Draw(sheet).text((6, 7), f"Frames flagged by the PCA reconstruction error but not by fitting, random sample from the videos where the PCA flags most (videos {', '.join(map(str, videos))}); red = the model's 8 points", fill=(0, 0, 0), font=fb)
    for j, t in enumerate(tiles):
        sheet.paste(t, ((j % cols) * TW, 30 + (j // cols) * TH))
    sheet.save(OUT / "examples.jpg", quality=88)
    print("examples", len(tiles))


if __name__ == "__main__":
    T, M = truth_table()
    print(T[["signal", "good_flagged_share", "lovo_caught", "lovo_good_flagged"]].round(4).to_string(index=False))
    fig_signals(T)
    S, scores = whole_video(M)
    fig_share(S)
    examples(S, scores)
