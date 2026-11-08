#!/usr/bin/env python3
"""By-eye check of the frames the confidence closed-eye rule hands back to the repair (experiment 4's training frames):
status closed in the pipeline run (opening rule), repaired in the --closed confidence run (2008 frames on the 19 videos).

    python reopened_review.py            # 20 frames, at most 5 per video, one per episode, seed 0 -> subset_review/reopened_review.html
    python reopened_review.py --eval verdicts.csv

Panels: the frame, the model's points, "opening: not repaired" (the rule in use calls the eye closed), "conf: replaced ..."
(blue, arrows). Question: which is right: conf (the repair is right) / opening (closed or no pupil to see, nothing
should be repaired) / model as it was / none / cannot tell. Reuses render and page of subset_order_comparison.py."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import subset_order_comparison as M  # noqa: E402
from second_peak_repair import PL  # noqa: E402

N, PER, SEED = 20, 5, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=None)
    ap.add_argument("--html-only", action="store_true")
    a = ap.parse_args()
    if a.eval:
        C = pd.read_csv(M.OUT / "reopened_sample.csv")
        J = C.merge(pd.read_csv(a.eval), on="item")
        J = J[J.verdict != "cannot tell"]
        S = pd.DataFrame([{"judged": len(J), "conf_repair_right": int(J.verdict.str.split("+").apply(lambda g: "conf" in g).sum()),
                           "opening_right_nothing_to_repair": int(J.verdict.str.split("+").apply(lambda g: "opening" in g).sum()),
                           "model_was_right": int((J.verdict == "model as it was (no repair needed)").sum()),
                           "another_point_not_caught": int((J.verdict == "moved point right, another point still wrong (not caught)").sum()),
                           "none": int((J.verdict == "none of them").sum())}])
        S.to_csv(M.OUT / "reopened_verdicts.csv", index=False)
        print(S.to_string(index=False))
        return
    rows = []
    for v in sorted(PL[PL.status == "final"].video.astype(int)):
        A = pd.read_csv(M.FAR / f"video{v:02d}_{M.BASE}" / "frames_all.csv.gz", dtype={"points": str})
        Nn = pd.read_csv(M.FAR / f"video{v:02d}_{M.BASE}_confclosed" / "frames_all.csv.gz", dtype={"points": str})
        for i in np.flatnonzero(((A.status == "closed") & (Nn.status == "repaired")).to_numpy()):
            rows.append({"video": v, "frame": int(i), "status_A": "closed", "points_A": "", "status_B": "repaired", "points_B": Nn.points[i], "status_C": "repaired", "points_C": Nn.points[i],
                         "worst_point": Nn.worst_point[i], "fit_rel": Nn.fit_rel[i]})
    D = pd.DataFrame(rows)
    D.to_csv(M.OUT / "reopened_frames.csv", index=False)
    print("opening rule closed, confidence rule repaired:", len(D), "frames;", D.groupby("video").size().to_dict())
    D["run_id"] = D.groupby("video").frame.transform(lambda f: np.cumsum(np.r_[True, np.diff(f.to_numpy()) > 5]))
    one = D.groupby(["video", "run_id"]).sample(1, random_state=SEED)
    S = one.groupby("video").sample(frac=1, random_state=SEED).groupby("video").head(PER)
    if len(S) > N:   # keep the videos with most reopened frames, then thin evenly
        S = S.sample(N, random_state=SEED, weights=S.groupby("video").frame.transform("size"))
    S = S.head(N).sort_values(["video", "frame"]).reset_index(drop=True)
    S["item"] = [f"video{v:02d}_f{f:06d}_r" for v, f in zip(S.video, S.frame)]
    S.to_csv(M.OUT / "reopened_sample.csv", index=False)
    M.ITEMS.mkdir(parents=True, exist_ok=True)
    if not a.html_only:
        for v, g in S.groupby("video"):
            M.render(int(v), list(g.itertuples()), names=("opening", "conf", "conf"))
    M.page(S, None, names=("opening", "conf", "conf"), out_html="reopened_review.html", key="reopened_review")


if __name__ == "__main__":
    main()
