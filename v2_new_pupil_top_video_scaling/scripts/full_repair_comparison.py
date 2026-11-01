#!/usr/bin/env python3
"""Pupil-only repair against the repair of all eight points, same finder (either) and acceptance (human-label range).

    python full_repair_comparison.py

Reads results/flag_and_repair/videoNN_either_labelregion (pupil points only, as before) and videoNN_either_labelregion_all
(--points all) for the 19 videos. Writes to results/flag_and_repair/all_videos/: full_repair.csv (totals),
full_repair_by_video.csv, full_repair.png. Truth frames: the 23 wrong frames by the kind of their worst point (pupil /
eyelid / corner) and what happened to them; good frames repaired and made worse. Whole videos: status shares and the
repaired points by kind. Nothing existing is modified."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_peak_repair import BPS, P4, PL, RES  # noqa: E402

FR = RES / "flag_and_repair"
OUT = FR / "all_videos"
LINE, WORSE = 0.14, 0.01
RUNS = {"pupil": "video{v:02d}_either_labelregion", "all": "video{v:02d}_either_labelregion_all", "smallest": "video{v:02d}_either_labelregion_all_smallest"}
NAMES = {"pupil": "pupil points only", "all": "all eight points,\nevery candidate at once", "smallest": "all eight points,\nsmallest subset that passes"}
KIND = {b: ("pupil" if b.startswith("pupil") else "eyelid" if b.startswith("eyelid") else "corner") for b in BPS}


def main():
    rows, truth = [], []
    for v in sorted(PL[PL.status == "final"].video.astype(int)):
        for run, pat in RUNS.items():
            W = pd.read_csv(FR / pat.format(v=v) / "whole_video.csv").iloc[0]
            T = pd.read_csv(FR / pat.format(v=v) / "truth_frames.csv")
            T["video"], T["run"] = v, run
            truth.append(T)
            rep_pts = {k: sum(int(W.get(f"repaired_{b}", 0)) for b in BPS if KIND[b] == k) for k in ("pupil", "eyelid", "corner")}
            rows.append({"video": v, "run": run, "frames": int(W.frames), "flagged_pct": W.flagged_pct, "repaired": int(W.repaired), "flagged_not_repaired": int(W.flagged_not_repaired),
                         "flagged_passes_check": int(W.flagged_passes_check), "closed": int(W.closed), **{f"repaired_points_{k}": n for k, n in rep_pts.items()}})
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "full_repair_by_video.csv", index=False)
    A = pd.concat(truth, ignore_index=True)
    A["kind"] = A.worst_point.map(KIND)
    wrong, good, rep, nr = A.error_before > LINE, A.error_before <= LINE, A.status == "repaired", A.status == "flagged_not_repaired"
    tot = []
    for run in RUNS:
        m = A.run == run
        r = {"run": run, "wrong_before": int((wrong & m).sum()), "wrong_flagged": int((wrong & m & A.flag).sum()), "wrong_repaired": int((wrong & m & rep).sum()),
             "wrong_repaired_to_good": int((wrong & m & rep & (A.error_after <= LINE)).sum()), "wrong_still_flagged": int((wrong & m & nr).sum()),
             "wrong_closed": int((wrong & m & (A.status == "closed")).sum()), "wrong_after": int((m & (A.error_after > LINE)).sum()),
             "good_flagged": int((good & m & A.flag).sum()), "good_repaired": int((good & m & rep).sum()), "good_made_worse": int((good & m & rep & (A.error_after > A.error_before + WORSE)).sum()),
             "good_pushed_over_line": int((good & m & rep & (A.error_after > LINE)).sum()), "good_passes_check": int((good & m & (A.status == "flagged_passes_check")).sum())}
        for k in ("pupil", "eyelid", "corner"):
            km = A.kind == k
            r[f"wrong_{k}_worst"] = int((wrong & m & km).sum())
            r[f"wrong_{k}_repaired_to_good"] = int((wrong & m & km & rep & (A.error_after <= LINE)).sum())
        g = B[B.run == run]
        for c in ("repaired", "flagged_not_repaired", "flagged_passes_check"):
            r[f"{c}_pct"] = round(100 * g[c].sum() / g.frames.sum(), 3)
        r["flagged_pct"] = round(float(np.average(g.flagged_pct, weights=g.frames)), 2)
        for k in ("pupil", "eyelid", "corner"):
            r[f"repaired_points_{k}"] = int(g[f"repaired_points_{k}"].sum())
        tot.append(r)
    Tt = pd.DataFrame(tot).set_index("run")
    Tt.to_csv(OUT / "full_repair.csv")
    print(Tt.T.to_string())
    A.round(4).to_csv(OUT / "full_repair_truth_frames.csv", index=False)
    base = A[A.run == "pupil"].set_index(["video", "frame"])
    chg = []
    for run in [r for r in RUNS if r != "pupil"]:
        o = A[A.run == run].set_index(["video", "frame"]).join(base[["status", "error_after"]], rsuffix="_pupil")
        o = o[(o.status != o.status_pupil) | ((o.error_after - o.error_after_pupil).abs() > 1e-6)]
        chg.append(o.reset_index())
    chg = pd.concat(chg, ignore_index=True)
    print("\ntruth frames whose outcome differs from the pupil-only run:")
    print(chg[["run", "video", "frame", "worst_point", "error_before", "status_pupil", "error_after_pupil", "status", "points", "error_after"]].round(3).to_string(index=False))
    chg.round(4).to_csv(OUT / "full_repair_changed_truth_frames.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(RUNS))
    bw = 0.38
    labels = [NAMES[r] for r in RUNS]
    cols = {"pupil": "#9bb7d4", "eyelid": "#7fb069", "corner": "#e0a458"}
    for j, run in enumerate(RUNS):
        bottom = 0
        for k in ("pupil", "eyelid", "corner"):
            n = Tt.loc[run, f"wrong_{k}_repaired_to_good"]
            ax.bar(j, n, bottom=bottom, color=cols[k], label=f"worst point on a {k}" if j == 0 else None, width=0.6)
            bottom += n
        ax.text(j, bottom / 2, f"{bottom} of {Tt.loc[run, 'wrong_before']}", ha="center", va="center", fontsize=10, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("wrong truth frames repaired to good")
    ax.set_title("The 23 wrong truth frames", pad=10)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for j, run in enumerate(RUNS):
        ax.bar(j - bw / 2, Tt.loc[run, "good_repaired"], width=bw, color="#2f6db3", label="good frames repaired" if j == 0 else None)
        ax.bar(j + bw / 2, Tt.loc[run, "good_made_worse"], width=bw, color="#d1495b", label="of which made worse" if j == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(f"The {Tt.loc['pupil', 'good_flagged']} / {Tt.loc['all', 'good_flagged']} right truth frames the finders flagged")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    for j, run in enumerate(RUNS):
        bottom = 0
        for c, col, lab in (("repaired_pct", "#2f6db3", "repaired"), ("flagged_not_repaired_pct", "#e0a458", "still flagged"), ("flagged_passes_check_pct", "0.75", "judged plausible as it is, left alone")):
            ax.bar(j, Tt.loc[run, c], bottom=bottom, color=col, width=0.6, label=lab if j == 0 else None)
            bottom += Tt.loc[run, c]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("% of all frames, 19 videos")
    ax.set_title("Whole videos: what happens to a flagged frame")
    ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.92")
        ax.set_axisbelow(True)
    fig.suptitle("Repairing the pupil points only against repairing all eight points (finder: fitting or PCA; acceptance: human-label range)", fontsize=10)
    fig.savefig(OUT / "full_repair.png", dpi=150)
    print("figure", OUT / "full_repair.png")


if __name__ == "__main__":
    main()
