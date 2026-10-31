#!/usr/bin/env python3
"""The repair with each finder: fitting, PCA, either (report 9.3).

    python finder_repair_comparison.py

Reads results/flag_and_repair/videoNN_labelregion (fitting finder), videoNN_pca_labelregion and videoNN_either_labelregion
(flag_and_repair.py --finder pca | either), acceptance = human-label range in every case. Per finder: on the 1330 truth
frames, wrong frames before, flagged, repaired to good, wrong after, good frames repaired / made worse; on every frame of
the 19 videos, shares flagged, repaired, flagged and not repaired, flagged but passing the check. Totals and per video.
Outputs results/flag_and_repair/all_videos/finders.csv, finders_by_video.csv, finders.png, finders_share_by_video.png
(flagged share per video, fitting against PCA). Nothing existing is modified."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_peak_repair import RES  # noqa: E402

FAR = RES / "flag_and_repair"
OUT = FAR / "all_videos"
FINDERS = [("fit", "fitting", "videoNN_labelregion"), ("pca", "PCA", "videoNN_pca_labelregion"), ("either", "either finder", "videoNN_either_labelregion")]
line = float(pd.read_csv(RES / "error_tier" / "tier.csv").iloc[0].line_eye_widths)
WORSE = 0.01


def main():
    rows, per = [], []
    for key, name, pat in FINDERS:
        S, Wv, T = [], [], []
        for v in range(19):
            d = FAR / pat.replace("NN", f"{v:02d}")
            if not (d / "summary.csv").exists():
                print("missing", d)
                continue
            S.append(pd.read_csv(d / "summary.csv"))
            Wv.append(pd.read_csv(d / "whole_video.csv"))
            T.append(pd.read_csv(d / "truth_frames.csv").assign(video=v))
        S, Wv, T = pd.concat(S), pd.concat(Wv), pd.concat(T)
        assert len(T) == 1330, len(T)
        wrong, good, rep = T.error_before > line, T.error_before <= line, T.status == "repaired"
        n = Wv.frames.sum()
        tot = {"finder": key, "name": name, "videos": len(S), "truth_frames": len(T), "wrong_before": int(wrong.sum()), "wrong_flagged": int((wrong & T.flag).sum()),
               "wrong_repaired": int((wrong & rep).sum()), "wrong_repaired_to_good": int((wrong & rep & (T.error_after <= line)).sum()),
               "wrong_after": int((T.error_after > line).sum()), "good_flagged": int((good & T.flag).sum()), "good_repaired": int((good & rep).sum()),
               "good_made_worse": int((good & rep & (T.error_after > T.error_before + WORSE)).sum()), "good_pushed_over_line": int((good & rep & (T.error_after > line)).sum()),
               "mean_error_before_pct": 100 * T.error_before.mean(), "mean_error_after_pct": 100 * T.error_after.mean(),
               "median_error_before_pct": 100 * T.error_before.median(), "median_error_after_pct": 100 * T.error_after.median(),
               "frames": int(n), "flagged_pct": 100 * (Wv.flagged_pct / 100 * Wv.frames).sum() / n, "repaired_pct": 100 * Wv.repaired.sum() / n,
               "flagged_not_repaired_pct": 100 * Wv.flagged_not_repaired.sum() / n, "flagged_passes_check_pct": 100 * Wv.flagged_passes_check.sum() / n,
               "flagged_closed_pct": 100 * Wv.flagged_closed.sum() / n}
        rows.append(tot)
        for (_, s), (_, w) in zip(S.iterrows(), Wv.iterrows()):
            per.append({"finder": key, "video": int(s.video), "wrong_before": s.wrong_before, "wrong_flagged": s.wrong_flagged, "wrong_repaired_to_good": s.wrong_repaired_to_good,
                        "wrong_after": s.wrong_after_all, "good_repaired": s.good_repaired, "good_made_worse": s.good_made_worse,
                        "flagged_pct": w.flagged_pct, "repaired_pct": w.repaired_pct, "flagged_not_repaired_pct": w.flagged_not_repaired_pct,
                        "flagged_passes_check_pct": round(100 * w.flagged_passes_check / w.frames, 2)})
    R = pd.DataFrame(rows)
    R.round(3).to_csv(OUT / "finders.csv", index=False)
    pd.DataFrame(per).to_csv(OUT / "finders_by_video.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.T.to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    x = np.arange(len(R))
    ax = axs[0]
    wdt = 0.2
    ax.bar(x - 1.5 * wdt, R.wrong_before, wdt, color="0.75", label="wrong before")
    ax.bar(x - 0.5 * wdt, R.wrong_flagged, wdt, color="#e0a458", label="wrong frames flagged")
    ax.bar(x + 0.5 * wdt, R.wrong_repaired_to_good, wdt, color="#2f6db3", label="wrong frames repaired to good")
    ax.bar(x + 1.5 * wdt, R.wrong_after, wdt, color="#d1495b", label="wrong after")
    for i, r in R.iterrows():
        ax.text(i + 1.5 * wdt, r.wrong_after + 0.4, f"{int(r.wrong_after)}", ha="center", fontsize=9, color="#d1495b")
        ax.text(i + 0.5 * wdt, r.wrong_repaired_to_good + 0.4, f"{int(r.wrong_repaired_to_good)}", ha="center", fontsize=9, color="#2f6db3")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['name']}\n(good frames made worse: {int(r.good_made_worse)})" for _, r in R.iterrows()], fontsize=9)
    ax.set_ylabel("truth frames (of 1330)")
    ax.set_title("Truth frames: the 23 wrong frames with each finder", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    ax = axs[1]
    ax.bar(x - wdt, R.repaired_pct, wdt, color="#2f6db3", label="repaired")
    ax.bar(x, R.flagged_not_repaired_pct, wdt, color="#d1495b", label="flagged, not repaired")
    ax.bar(x + wdt, R.flagged_passes_check_pct, wdt, color="0.6", label="flagged, passes the check, left alone")
    for i, r in R.iterrows():
        for dx, val in ((-wdt, r.repaired_pct), (0, r.flagged_not_repaired_pct), (wdt, r.flagged_passes_check_pct)):
            ax.text(i + dx, val + 0.05, f"{val:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['name']}\n(flags {r.flagged_pct:.1f} %)" for _, r in R.iterrows()], fontsize=9)
    ax.set_ylabel("% of all 1.31 M frames")
    ax.set_title("Every frame of the 19 videos: what happens to the flagged frames", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.92")
        ax.set_axisbelow(True)
    fig.savefig(OUT / "finders.png", dpi=150)

    # flagged share per video, fitting against PCA
    Pv = pd.DataFrame(per)
    fit_s, pca_s = Pv[Pv.finder == "fit"].set_index("video").flagged_pct, Pv[Pv.finder == "pca"].set_index("video").flagged_pct
    lab = pd.read_csv(RES / "finder_comparison" / "pca_threshold.csv").set_index("video")
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    xs = np.arange(19)
    ax.bar(xs - 0.2, fit_s.reindex(range(19)), 0.4, color="#2f6db3", label="fitting, threshold = the line (14 % of eye width)")
    ax.bar(xs + 0.2, pca_s.reindex(range(19)), 0.4, color="#d1495b", label="PCA, threshold = 3.9 × the video's median score")
    # wrong truth frames of each video and how many each finder found
    wf = Pv[Pv.finder == "fit"].set_index("video"); wp = Pv[Pv.finder == "pca"].set_index("video")
    top = max(fit_s.max(), pca_s.max())
    for v in range(19):
        n = int(wf.wrong_before[v])
        if n == 0:
            continue
        a_, b_ = int(wf.wrong_flagged[v]), int(wp.wrong_flagged[v])
        txt = f"{a_}/{n}" if a_ == b_ else f"{a_}/{n} | {b_}/{n}"
        ax.text(v, max(fit_s[v], pca_s[v]) + 0.25, txt, ha="center", va="bottom", fontsize=8.5, color="0.2", fontweight="bold")
    ax.text(0.99, 0.72, "n/n above a video = its wrong truth frames found / its wrong truth frames\n(same for both finders; videos without a number have no wrong truth frame)", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="0.3")
    ax.set_ylim(0, top * 1.12)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v}\n{int(lab.labels_seen[v])}" for v in range(19)], fontsize=8.5)
    ax.set_xlabel("video (second line: labeled poses the video's PCA was fitted on)")
    ax.set_ylabel("% of the video's frames flagged")
    ax.set_title("Share of every frame flagged by each finder, per video, and the wrong truth frames found")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.92")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(OUT / "finders_share_by_video.png", dpi=150)


if __name__ == "__main__":
    main()
