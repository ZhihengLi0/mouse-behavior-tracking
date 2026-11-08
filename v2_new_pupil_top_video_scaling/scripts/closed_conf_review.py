#!/usr/bin/env python3
"""By-eye check of the frames the confidence closed-eye rule takes away from the repair: frames the opening rule's run
(A) repaired but the confidence rule (mean pupil confidence < 0.40) calls closed (525 frames on the 19 videos).

    python closed_conf_review.py            # 20 frames, at most 2 per video, one per episode, seed 0 -> subset_review/closed_conf_review.html
    python closed_conf_review.py --eval verdicts.csv

Panels: the frame, the model's points, A's repair (blue, arrows), "conf: not repaired". Question: which is right:
A / conf (the eye is closed or the pupil cannot be seen, nothing should be repaired) / model as it was / none / cannot tell.
Reuses render and page of subset_order_comparison.py. Nothing existing is modified."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import subset_order_comparison as M  # noqa: E402
from second_peak_repair import PL  # noqa: E402

N, PER, SEED = 20, 2, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=None)
    ap.add_argument("--html-only", action="store_true")
    a = ap.parse_args()
    if a.eval:
        C = pd.read_csv(M.OUT / "closed_conf_sample.csv")
        J = C.merge(pd.read_csv(a.eval), on="item")
        J = J[J.verdict != "cannot tell"]
        S = pd.DataFrame([{"judged": len(J), "A_repair_right": int(J.verdict.str.split("+").apply(lambda g: "A" in g).sum()), "conf_right_nothing_to_repair": int(J.verdict.str.split("+").apply(lambda g: "conf" in g).sum()),
                           "model_was_right": int((J.verdict == "model as it was (no repair needed)").sum()), "none": int((J.verdict == "none of them").sum())}])
        S.to_csv(M.OUT / "closed_conf_verdicts.csv", index=False)
        print(S.to_string(index=False))
        return
    rows = []
    for v in sorted(PL[PL.status == "final"].video.astype(int)):
        A = pd.read_csv(M.FAR / f"video{v:02d}_{M.BASE}" / "frames_all.csv.gz", dtype={"points": str})
        Nn = pd.read_csv(M.FAR / f"video{v:02d}_{M.BASE}_devorder_margin_confclosed" / "frames_all.csv.gz", dtype={"points": str})
        for i in np.flatnonzero(((A.status == "repaired") & (Nn.status == "closed")).to_numpy()):
            rows.append({"video": v, "frame": int(i), "status_A": "repaired", "points_A": A.points[i], "status_B": "closed", "points_B": "", "status_C": "closed", "points_C": "",
                         "worst_point": A.worst_point[i], "fit_rel": A.fit_rel[i]})
    D = pd.DataFrame(rows)
    D.to_csv(M.OUT / "closed_conf_taken_from_repair.csv", index=False)
    print("A repaired, confidence rule calls closed:", len(D), "frames")
    D["run_id"] = D.groupby("video").frame.transform(lambda f: np.cumsum(np.r_[True, np.diff(f.to_numpy()) > 5]))
    one = D.groupby(["video", "run_id"]).sample(1, random_state=SEED)
    S = one.groupby("video").sample(frac=1, random_state=SEED).groupby("video").head(PER)
    pool = one[~one.index.isin(S.index)]
    if len(S) < N and len(pool):
        S = pd.concat([S, pool.sample(min(N - len(S), len(pool)), random_state=SEED)])
    S = S.head(N).sort_values(["video", "frame"]).reset_index(drop=True)
    S["item"] = [f"video{v:02d}_f{f:06d}_c" for v, f in zip(S.video, S.frame)]
    S.to_csv(M.OUT / "closed_conf_sample.csv", index=False)
    M.ITEMS.mkdir(parents=True, exist_ok=True)
    if not a.html_only:
        for v, g in S.groupby("video"):
            M.render(int(v), list(g.itertuples()), names=("A", "conf", "conf"))
    M.page(S, None, names=("A", "conf", "conf"), out_html="closed_conf_review.html", key="closed_conf_review")


if __name__ == "__main__":
    main()
