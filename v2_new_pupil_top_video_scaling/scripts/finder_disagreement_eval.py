#!/usr/bin/env python3
"""Tabulate the by-eye verdicts of finder_disagreement_review.py (report 9.2).

    python finder_disagreement_eval.py

Reads results/finder_comparison/disagreement_review/{items.csv, finder_disagreement_verdicts.csv} and
whole_video_share.csv. Per group (flagged by the PCA only / by fitting only): frames judged, really wrong, fine,
cannot tell. Because 4 frames per video and group were judged, the whole-video estimate weights each video's wrong
share by its number of such frames. Outputs result.csv, per_video.csv, result.png. Nothing existing is modified."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_peak_repair import RES  # noqa: E402

D = RES / "finder_comparison" / "disagreement_review"
I = pd.read_csv(D / "items.csv")
V = pd.read_csv(D / "finder_disagreement_verdicts.csv")
S = pd.read_csv(RES / "finder_comparison" / "whole_video_share.csv")
M = I.merge(V, on="item", how="left")
M["verdict"] = M.verdict.fillna("not judged").str.split(",").str[0].replace({"yes": "wrong", "no": "fine"})
rows, per = [], []
for g, lab in (("pca_only", "flagged by the PCA only"), ("fit_only", "flagged by fitting only")):
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
                 "frames_in_group_all_videos": int(est_all), "est_wrong_frames_all_videos": int(round(est_wrong)),
                 "frames_in_group_without_video18": int(s[s.video != 18].candidates.drop_duplicates().sum()) if False else int(sum(s[s.video == v].candidates.iloc[0] for v in s.video.unique() if v != 18)),
                 "est_wrong_frames_without_video18": int(round(sum(r["est_wrong_frames"] for r in per if r["group"] == g and r["video"] != 18)))})
R = pd.DataFrame(rows)
R.to_csv(D / "result.csv", index=False)
pd.DataFrame(per).to_csv(D / "per_video.csv", index=False)
print(R.T.to_string())
total_frames = int(S.frames.sum())
fit_flagged = int(round((S.fit_flagged_pct / 100 * S.frames).sum()))
print(f"all frames {total_frames}; fitting flags {fit_flagged} ({100 * fit_flagged / total_frames:.2f} %), of which 59 % really wrong by the earlier spot check ≈ {round(0.59 * fit_flagged)}")

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
fig.savefig(D / "result.png", dpi=150)
