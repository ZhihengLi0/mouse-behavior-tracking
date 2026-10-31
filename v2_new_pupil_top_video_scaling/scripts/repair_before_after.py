#!/usr/bin/env python3
"""Old error against new error, all 19 videos: what the repair does to the error of the 1330 truth frames.

    python repair_before_after.py

Reads only the existing per-video outputs of flag_and_repair.py (results/flag_and_repair/videoNN_<test>/
truth_frames.csv, whole_video.csv) for the seven acceptance tests; nothing is rerun, nothing existing is modified.
The comparison is only possible on the truth frames, the only frames with human labels; for the whole videos only the
share of frames the repair changed can be reported.

Outputs in results/flag_and_repair/all_videos/: error_before_after.csv (one row per video and test plus an "all" row
per test: mean and median error before and after in % of eye width and in px, wrong frames before and after, frames
changed, frames made worse by more than 1 % of eye width, whole-video share of frames changed) and
error_before_after.png (left: per video the mean error before and after for the adopted test, the geometry rule as a
marker; right: every truth frame before against after, log axes, the line on both axes)."""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from second_peak_repair import RES

FR = RES / "flag_and_repair"
OUT = FR / "all_videos"
TESTS = ("geometry", "fit", "conf", "jump", "labelgeo", "labelregion0", "labelregion")
USED, COMP = "labelregion", "geometry"
WORSE = 0.01
line = float(pd.read_csv(RES / "error_tier" / "tier.csv").iloc[0].line_eye_widths)


def load(test):
    parts = []
    for d in sorted(FR.glob(f"video??_{test}")):
        T = pd.read_csv(d / "truth_frames.csv")
        W = pd.read_csv(d / "whole_video.csv").iloc[0]
        T["video"], T["whole_frames"], T["whole_repaired"] = int(d.name[5:7]), int(W.frames), int(W.repaired)
        parts.append(T)
    D = pd.concat(parts, ignore_index=True)
    assert len(D) == 1330, (test, len(D))
    return D


def row(D, test, video):
    b, a = D.error_before, D.error_after
    return {"video": video, "accept": test, "truth_frames": len(D),
            "mean_before_pct": 100 * b.mean(), "mean_after_pct": 100 * a.mean(), "median_before_pct": 100 * b.median(), "median_after_pct": 100 * a.median(),
            "mean_before_px": D.error_before_px.mean(), "mean_after_px": D.error_after_px.mean(),
            "wrong_before": int((b > line).sum()), "wrong_after": int((a > line).sum()), "changed": int((a != b).sum()),
            "improved": int((a < b - WORSE).sum()), "worse": int((a > b + WORSE).sum()),
            "whole_frames": int(D.drop_duplicates("video").whole_frames.sum()), "whole_changed_pct": 100 * D.drop_duplicates("video").whole_repaired.sum() / D.drop_duplicates("video").whole_frames.sum()}


def main():
    data, rows = {}, []
    for test in TESTS:
        D = load(test)
        data[test] = D
        rows += [row(g, test, int(v)) for v, g in D.groupby("video")]
        rows.append(row(D, test, "all"))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "error_before_after.csv", index=False, float_format="%.3f")

    totals = pd.read_csv(OUT / "totals.csv").set_index("accept")
    A = R[R.video == "all"].set_index("accept")
    for test in TESTS:
        assert A.loc[test, "wrong_before"] == totals.loc[test, "wrong_before"] == 23, test
        assert A.loc[test, "wrong_after"] == totals.loc[test, "wrong_after_all"], test
        assert A.loc[test, "worse"] == totals.loc[test, "good_made_worse"] + (data[test].query("error_before > @line").eval("error_after > error_before + @WORSE")).sum(), test
    S = pd.read_csv(OUT / "truth_summary_all.csv")
    chk = R[R.video != "all"].assign(video=lambda x: x.video.astype(int)).merge(S, on=["video", "accept"])
    assert np.allclose(chk.median_before_pct, chk.median_error_before_pct, atol=0.006) and np.allclose(chk.median_after_pct, chk.median_error_after_pct, atol=0.006)
    print(A[["mean_before_pct", "mean_after_pct", "median_before_pct", "median_after_pct", "wrong_before", "wrong_after", "changed", "improved", "worse", "whole_changed_pct"]].round(2).to_string())
    U = R[(R.accept == USED) & (R.video != "all")]
    print(f"\n{USED}: videos whose truth-frame error changes:", " ".join(str(v) for v in U[U.changed > 0].video))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    G = R[(R.accept == COMP) & (R.video != "all")]
    x, bw = np.arange(len(U)), 0.38
    ax.bar(x - bw / 2, U.mean_before_pct, width=bw - 0.04, color="#cfcfcf", label="before repair")
    ax.bar(x + bw / 2, U.mean_after_pct, width=bw - 0.04, color="#2f6db3", label="after repair, human-label range (used)")
    ax.plot(x + bw / 2, G.mean_after_pct, "D", color="0.25", ms=4, label="after repair, geometry rule (comparison)")
    for i, (wb, wa) in enumerate(zip(U.wrong_before, U.wrong_after)):
        if wb:
            ax.text(i, max(U.mean_before_pct.iloc[i], U.mean_after_pct.iloc[i]) + 0.15, f"{wb}→{wa}", ha="center", fontsize=8.5, color="0.25")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in U.video])
    ax.set_xlabel("video")
    ax.set_ylabel("mean frame error of the 70 truth frames, % of eye width")
    ax.set_title("Mean error before and after the repair (numbers: wrong frames before→after)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.92")
    ax.set_axisbelow(True)

    ax = axes[1]
    D = data[USED]
    b, a = 100 * D.error_before.clip(lower=0.003), 100 * D.error_after.clip(lower=0.003)
    same, better, worse = D.error_after == D.error_before, D.error_after < D.error_before - WORSE, D.error_after > D.error_before + WORSE
    other = ~(same | better | worse)
    ax.plot(b[same], a[same], "o", color="#cfcfcf", ms=3.5, label=f"unchanged ({int(same.sum())})")
    if other.any():
        ax.plot(b[other], a[other], "o", color="#9bb7d4", ms=5, label=f"changed by less than 1 % ({int(other.sum())})")
    ax.plot(b[better], a[better], "o", color="#2f6db3", ms=6, label=f"improved ({int(better.sum())})")
    ax.plot(b[worse], a[worse], "o", color="#d1495b", ms=6, label=f"made worse ({int(worse.sum())})")
    lim = (0.25, 200)
    ax.plot(lim, lim, color="0.6", lw=0.8)
    ax.axvline(100 * line, color="0.25", lw=1, ls="--")
    ax.axhline(100 * line, color="0.25", lw=1, ls="--")
    ax.text(100 * line * 1.08, 0.3, "line 14 %", fontsize=8.5, color="0.25")
    nb, na = int((D.error_before > line).sum()), int((D.error_after > line).sum())
    ax.text(0.3, 100 * line * 1.15, f"wrong frames: {nb} before, {na} after", fontsize=9, color="0.25")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ticks = [0.5, 1, 2, 5, 10, 20, 50, 100]
    for setter in (ax.set_xticks, ax.set_yticks):
        setter(ticks)
    ax.set_xticklabels([f"{t:g} %" for t in ticks])
    ax.set_yticklabels([f"{t:g} %" for t in ticks])
    ax.set_xlabel("error before repair, % of eye width")
    ax.set_ylabel("error after repair, % of eye width")
    ax.set_title(f"The 1330 truth frames, human-label range")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="0.92", which="major")
    ax.set_axisbelow(True)
    fig.savefig(OUT / "error_before_after.png", dpi=150)
    print("figure", OUT / "error_before_after.png")


if __name__ == "__main__":
    main()
