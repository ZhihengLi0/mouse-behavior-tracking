#!/usr/bin/env python3
"""Tabulate the by-eye verdicts of finder_disagreement_review.py (report 9.2).

    python finder_disagreement_eval.py

Reads results/finder_comparison/disagreement_review/{items.csv, finder_disagreement_verdicts.csv}. Per group (flagged by
the PCA only / by fitting only): frames judged, really wrong, fine, cannot tell.

The frames were drawn with the PCA at its first, fixed threshold (3.0 % of eye width, finder_comparison.py). The rule in
use is 3.93 × the video's median (pca_threshold.py, pca_cache/). The main outputs therefore keep only the judged frames
that belong to the same group under the rule in use (PCA-only: flagged by the median-rule PCA and not by fitting;
fitting-only: flagged by fitting and not by the median-rule PCA); the frames the median rule drops are left out
(user, 2026-10-31). Because 4 frames per video and group were judged, the whole-video estimate weights each video's
wrong share by its number of such frames under the rule in use.
Outputs: result.csv, per_video.csv, result.png (rule in use); result_fixed_threshold.csv, result_fixed_threshold.png
(every judged frame, the threshold the frames were drawn with); items_rule_in_use.csv (which frames were kept).
Nothing existing is modified."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_peak_repair import RES  # noqa: E402

D = RES / "finder_comparison" / "disagreement_review"
CACHE = RES / "finder_comparison" / "pca_cache"
FWD = RES / "unreliable_frames_sweep" / "cache"
line = float(pd.read_csv(RES / "error_tier" / "tier.csv").iloc[0].line_eye_widths)
GROUPS = (("pca_only", "flagged by the PCA only"), ("fit_only", "flagged by fitting only"))


def tabulate(M):
    rows, per = [], []
    for g, lab in GROUPS:
        s = M[M.group == g]
        judged = s[s.verdict != "not judged"]
        est_wrong = est_all = 0.0
        for v, sv in judged.groupby("video"):
            n_cand = int(s[s.video == v].candidates.iloc[0])
            share = (sv.verdict == "wrong").mean()
            est_wrong += n_cand * share
            est_all += n_cand
            per.append({"group": g, "video": v, "frames_in_group": n_cand, "judged": len(sv), "wrong": int((sv.verdict == "wrong").sum()), "fine": int((sv.verdict == "fine").sum()),
                        "cannot_tell": int((sv.verdict == "cannot tell").sum()), "est_wrong_frames": round(n_cand * share)})
        rows.append({"group": lab, "sampled": len(s), "judged": len(judged), "wrong": int((judged.verdict == "wrong").sum()), "fine": int((judged.verdict == "fine").sum()),
                     "cannot_tell": int((judged.verdict == "cannot tell").sum()), "wrong_share_of_judged": round((judged.verdict == "wrong").mean(), 3),
                     "frames_in_group_all_videos": int(est_all), "est_wrong_frames_all_videos": int(round(est_wrong))})
    return pd.DataFrame(rows), pd.DataFrame(per)


def figure(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    cols = {"wrong": "#d1495b", "fine": "#9bb7d4", "cannot_tell": "0.75"}
    for i, r in R.iterrows():
        left = 0
        for k, lab in (("wrong", "really wrong"), ("fine", "all points fine"), ("cannot_tell", "cannot tell")):
            ax.barh(i, r[k], left=left, color=cols[k], label=lab if i == 0 else None, height=0.55)
            if r[k] > 3:
                ax.text(left + r[k] / 2, i, str(r[k]), ha="center", va="center", fontsize=10, color="white" if k == "wrong" else "0.2")
            left += r[k]
    ax.set_yticks(range(len(R)))
    ax.set_yticklabels([f"{r.group}\n({r.judged} frames judged)" for r in R.itertuples()], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("frames judged by eye (4 per video and group, drawn at random)")
    ax.set_title("Where the two finders disagree: what the eye says")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    fig.savefig(path, dpi=150)


def main():
    I = pd.read_csv(D / "items.csv")
    V = pd.read_csv(D / "finder_disagreement_verdicts.csv")
    M = I.merge(V, on="item", how="left")
    M["verdict"] = M.verdict.fillna("not judged").str.split(",").str[0].replace({"yes": "wrong", "no": "fine"})
    same, n_group = [], {}
    for v, g in M.groupby("video"):
        z = np.load(CACHE / f"video{v:02d}_pca.npz")
        pca = z["score"] > float(z["threshold"])
        fit = np.nanmax(np.load(FWD / f"video{v:02d}_fwd200.npy"), axis=1) / float(z["eye_width_px"]) > line
        n_group[(v, "pca_only")], n_group[(v, "fit_only")] = int((pca & ~fit).sum()), int((fit & ~pca).sum())
        for r in g.itertuples():
            same.append(bool(pca[r.frame] and not fit[r.frame]) if r.group == "pca_only" else bool(fit[r.frame] and not pca[r.frame]))
    M["same_group_under_rule_in_use"] = same
    M["candidates_rule_in_use"] = [n_group[(v, g)] for v, g in zip(M.video, M.group)]
    M.to_csv(D / "items_rule_in_use.csv", index=False)

    K = M[M.same_group_under_rule_in_use].copy()
    K["candidates"] = K.candidates_rule_in_use
    R, P = tabulate(K)
    R.to_csv(D / "result.csv", index=False)
    P.to_csv(D / "per_video.csv", index=False)
    figure(R, D / "result.png")
    Ra, _ = tabulate(M)
    Ra.to_csv(D / "result_fixed_threshold.csv", index=False)
    figure(Ra, D / "result_fixed_threshold.png")
    pd.set_option("display.width", 250)
    print("rule in use (judged frames in the same group under 3.93 x median):")
    print(R.T.to_string())
    print("all judged frames (the fixed threshold the frames were drawn with):")
    print(Ra[["group", "judged", "wrong", "fine", "cannot_tell"]].to_string(index=False))


if __name__ == "__main__":
    main()
