#!/usr/bin/env python3
"""Result of the spot check by eye of the adopted repair (acceptance = labelregion) on the whole videos.

    python spotcheck_review_eval.py

Reads results/flag_and_repair/spotcheck_review/{items,spotcheck_review_verdicts}.csv (the frames of
spotcheck_review.py and the labeler's verdicts, 2026-10-30) and results/flag_and_repair/all_videos/whole_video_all.csv.
The frames were drawn one per episode of touched frames, so the shares are shares of episodes, not of frames; none of
these frames took part in setting the line, the threshold or the label range.
  repaired frame      verdict yes = the point put in is on the pupil where it belongs; "better, but still not right";
                      no; cannot tell
  still-flagged frame verdict yes = a model point is really wrong; no = false alarm; cannot tell
Outputs in results/flag_and_repair/spotcheck_review/: spotcheck_review_result.csv (per video and over all videos),
spotcheck_review_result.png. Nothing existing is modified."""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_peak_repair import RES  # noqa: E402

FAR = RES / "flag_and_repair"
D = FAR / "spotcheck_review"
TEST = "labelregion"


def main():
    I = pd.read_csv(D / "items.csv")
    V = pd.read_csv(D / "spotcheck_review_verdicts.csv")
    M = I.merge(V, on="item", how="left")
    M["verdict"] = M.verdict.fillna("not judged")
    W = pd.read_csv(FAR / "all_videos" / "whole_video_all.csv")
    W = W[W.accept == TEST].set_index("video")
    rows = []
    for v, g in list(M.groupby("video")) + [("all", M)]:
        r = g[g.status == "repaired"].verdict.value_counts()
        s = g[g.status == "flagged_not_repaired"].verdict.value_counts()
        rj = r.get("yes", 0) + r.get("better, but still not right", 0) + r.get("no", 0)
        sj = s.get("yes", 0) + s.get("no", 0)
        row = {"video": v, "repaired_shown": int((g.status == "repaired").sum()), "repaired_right": int(r.get("yes", 0)),
               "repaired_better_not_right": int(r.get("better, but still not right", 0)), "repaired_no": int(r.get("no", 0)),
               "repaired_cannot_tell": int(r.get("cannot tell", 0) + r.get("not judged", 0)),
               "repaired_right_share": r.get("yes", 0) / rj if rj else np.nan,
               "flagged_shown": int((g.status == "flagged_not_repaired").sum()), "flagged_really_wrong": int(s.get("yes", 0)), "flagged_false_alarm": int(s.get("no", 0)),
               "flagged_cannot_tell": int(s.get("cannot tell", 0) + s.get("not judged", 0)),
               "flagged_really_wrong_share": s.get("yes", 0) / sj if sj else np.nan}
        if v != "all":
            row.update({"whole_repaired_pct": float(W.loc[v, "repaired_pct"]), "whole_flagged_not_repaired_pct": float(W.loc[v, "flagged_not_repaired_pct"])})
        rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(D / "spotcheck_review_result.csv", index=False, float_format="%.3f")
    a = R[R.video == "all"].iloc[0]
    print(f"repaired frames shown {a.repaired_shown}: right {a.repaired_right}, better but not right {a.repaired_better_not_right}, no {a.repaired_no}, "
          f"cannot tell {a.repaired_cannot_tell}; right share {100 * a.repaired_right_share:.0f} %")
    print(f"still-flagged frames shown {a.flagged_shown}: really wrong {a.flagged_really_wrong}, false alarm {a.flagged_false_alarm}, "
          f"cannot tell {a.flagged_cannot_tell}; really-wrong share {100 * a.flagged_really_wrong_share:.0f} %")
    Pv = R[R.video != "all"].copy()
    Pv["video"] = Pv.video.astype(int)
    print(Pv[["video", "repaired_shown", "repaired_right", "repaired_better_not_right", "repaired_no", "flagged_shown", "flagged_really_wrong", "flagged_false_alarm", "flagged_cannot_tell", "whole_repaired_pct", "whole_flagged_not_repaired_pct"]].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), constrained_layout=True)
    x = np.arange(len(Pv))
    ax = axes[0]
    b0 = Pv.repaired_right
    ax.bar(x, b0, color="#2f6db3", label="on the pupil where it belongs")
    ax.bar(x, Pv.repaired_better_not_right, bottom=b0, color="#e0a458", label="closer, but still not right")
    ax.bar(x, Pv.repaired_no, bottom=b0 + Pv.repaired_better_not_right, color="#d1495b", label="not better")
    ax.bar(x, Pv.repaired_cannot_tell, bottom=b0 + Pv.repaired_better_not_right + Pv.repaired_no, color="#cfcfcf", label="cannot tell")
    ax.set_title(f"Repaired frames, judged by eye: {int(a.repaired_right)} of {int(a.repaired_shown)} right")
    ax.set_ylabel("frames shown (one per episode, at most 10 per video)")
    ax = axes[1]
    b0 = Pv.flagged_really_wrong
    ax.bar(x, b0, color="#2f6db3", label="a point is really wrong")
    ax.bar(x, Pv.flagged_false_alarm, bottom=b0, color="#e0a458", label="false alarm: the points are right")
    ax.bar(x, Pv.flagged_cannot_tell, bottom=b0 + Pv.flagged_false_alarm, color="#cfcfcf", label="cannot tell")
    ax.set_title(f"Still-flagged frames, judged by eye: {int(a.flagged_really_wrong)} of {int(a.flagged_shown)} really wrong")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in Pv.video])
        ax.set_xlabel("video")
        ax.set_ylim(0, 13.5)
        ax.set_yticks(range(0, 11, 2))
        ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.92")
        ax.set_axisbelow(True)
    fig.savefig(D / "spotcheck_review_result.png", dpi=150)
    print("figure", D / "spotcheck_review_result.png")


if __name__ == "__main__":
    main()
