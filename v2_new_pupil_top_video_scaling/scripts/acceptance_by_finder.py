#!/usr/bin/env python3
"""The acceptance tests under each finder (report 9.3): the repair is always the heatmap's second peak; what is compared is
the check that accepts or refuses it, once with the frames fitting flags and once with the frames the PCA flags.

    python acceptance_by_finder.py

Reads results/flag_and_repair/videoNN_<accept>/ (fitting finder) and videoNN_pca_<accept>/ (PCA finder,
flag_and_repair.py --finder pca) for the acceptance tests labelregion (two steps, with margin), geometry, labelregion0
(two steps, no margin), conf, labelgeo (one step, no margin), jump, and fit (fitting finder only: the test is the finder's
own score). Sums the per-video summary.csv and whole_video.csv over the 19 videos.
Outputs results/flag_and_repair/all_videos/acceptance_by_finder.csv and acceptance_by_finder.png. Nothing existing is
modified."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_peak_repair import RES  # noqa: E402

FAR = RES / "flag_and_repair"
OUT = FAR / "all_videos"
ORDER = ["labelregion", "geometry", "labelregion0", "conf", "labelgeo", "jump", "fit"]
NAMES = {"geometry": "(b) geometry rule\n(hand-made)", "fit": "(c) same fitting\nscore", "conf": "(d) second-peak\nconfidence", "jump": "(e) jump", "labelgeo": "(f) label range,\none step",
         "labelregion0": "(g) label range,\ntwo steps", "labelregion": "(a) label range,\ntwo steps,\nwith margin"}
FINDERS = [("fit", "fitting finder", "videoNN_{a}"), ("pca", "PCA finder", "videoNN_pca_{a}")]


def totals(pat, accept):
    S, Wv = [], []
    for v in range(19):
        d = FAR / pat.format(a=accept).replace("NN", f"{v:02d}")
        if not (d / "summary.csv").exists():
            return None
        S.append(pd.read_csv(d / "summary.csv"))
        Wv.append(pd.read_csv(d / "whole_video.csv"))
    S, Wv = pd.concat(S), pd.concat(Wv)
    n = Wv.frames.sum()
    cols = ["wrong_before", "wrong_flagged", "wrong_repaired_to_good", "wrong_flagged_not_repaired", "wrong_left_untouched", "wrong_after_all",
            "good_flagged", "good_repaired", "good_made_worse", "good_pushed_over_line", "good_flagged_not_repaired"]
    r = {c: int(S[c].sum()) for c in cols}
    r.update({"videos": len(S), "flagged_pct": round(100 * (Wv.flagged_pct / 100 * Wv.frames).sum() / n, 2), "repaired_pct": round(100 * Wv.repaired.sum() / n, 3),
              "flagged_not_repaired_pct": round(100 * Wv.flagged_not_repaired.sum() / n, 3), "flagged_passes_check_pct": round(100 * Wv.flagged_passes_check.sum() / n, 3)})
    return r


def main():
    rows = []
    for key, fname, pat in FINDERS:
        for a in ORDER:
            r = totals(pat, a)
            if r is None:
                print("missing", fname, a)
                continue
            rows.append({"finder": key, "finder_name": fname, "accept": a, **r})
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "acceptance_by_finder.csv", index=False)
    pd.set_option("display.width", 250)
    print(T[["finder", "accept", "wrong_flagged", "wrong_repaired_to_good", "wrong_flagged_not_repaired", "wrong_left_untouched", "good_flagged", "good_repaired", "good_made_worse",
             "good_flagged_not_repaired", "flagged_pct", "repaired_pct"]].to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6), constrained_layout=True)
    for row, (key, fname, _) in enumerate(FINDERS):
        Tf = T[T.finder == key].set_index("accept")
        Tf = Tf.loc[[a for a in ORDER if a in Tf.index]]
        x = np.arange(len(Tf))
        ax = axes[row][0]
        ax.bar(x, Tf.wrong_repaired_to_good, color="#2f6db3", label="repaired")
        ax.bar(x, Tf.wrong_flagged_not_repaired, bottom=Tf.wrong_repaired_to_good, color="#e0a458", label="still flagged")
        ax.bar(x, Tf.wrong_left_untouched, bottom=Tf.wrong_repaired_to_good + Tf.wrong_flagged_not_repaired, color="#cfcfcf", label="left alone (eye judged closed)")
        for i, r in enumerate(Tf.itertuples()):
            ax.text(i, r.wrong_repaired_to_good / 2, f"{int(r.wrong_repaired_to_good)}", ha="center", va="center", color="white", fontsize=10)
        ax.set_title(f"{fname}: the 23 wrong frames", fontsize=10)
        ax.set_ylabel("frames")
        ax = axes[row][1]
        bw = 0.28
        ax.bar(x - bw, Tf.good_repaired, width=bw - 0.03, color="#2f6db3", label="repaired")
        ax.bar(x, Tf.good_made_worse, width=bw - 0.03, color="#d1495b", label="made worse")
        ax.bar(x + bw, Tf.good_flagged_not_repaired, width=bw - 0.03, color="#e0a458", label="still flagged")
        ax.set_title(f"{fname}: the {int(Tf.good_flagged.iloc[0])} right frames it flagged", fontsize=10)
        for ax in axes[row]:
            ax.set_xticks(x)
            ax.set_xticklabels([NAMES[a] for a in Tf.index], fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="0.92")
            ax.set_axisbelow(True)
            ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle("Acceptance tests for the second-peak repair under each finder, same 19 videos", fontsize=11)
    fig.savefig(OUT / "acceptance_by_finder.png", dpi=150)
    print("saved", OUT / "acceptance_by_finder.png")


if __name__ == "__main__":
    main()
