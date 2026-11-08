#!/usr/bin/env python3
"""Variant D against C: by-eye sample of the frames where D (largest range margin among the passing subsets of any
size, `--tiebreak margin-all`) replaces more points than C (largest margin among the smallest passing subsets).

    python subset_d_review.py            # sample + page results/flag_and_repair/subset_review/subset_d_review.html
    python subset_d_review.py --eval verdicts.csv

20 frames, at most 2 per video, one per episode, seed 0. Panels: the frame, the model's points, C's result, D's result.
Question: which has every point right: C / D / both wrong / model as it was / cannot tell. Reuses render and page of
subset_order_comparison.py (names C, D). Nothing existing is modified."""
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
        C = pd.read_csv(M.OUT / "subset_d_sample.csv")
        J = C.merge(pd.read_csv(a.eval), on="item")
        J = J[J.verdict != "cannot tell"]
        S = pd.DataFrame([{"judged": len(J), "C_right": int(J.verdict.str.split("+").apply(lambda g: "C" in g).sum()), "D_right": int(J.verdict.str.split("+").apply(lambda g: "D" in g).sum()),
                           "model_was_right": int((J.verdict == "model as it was (no repair needed)").sum()), "both_changes_needed": int(J.verdict.str.startswith("both").sum()),
                           "another_not_caught": int(J.verdict.str.startswith("moved").sum()), "none": int((J.verdict == "none of them").sum())}])
        S.to_csv(M.OUT / "subset_d_verdicts.csv", index=False)
        print(S.to_string(index=False))
        return
    videos = sorted(PL[PL.status == "final"].video.astype(int))
    rows = []
    for v in videos:
        C = pd.read_csv(M.FAR / f"video{v:02d}_{M.BASE}_devorder_margin" / "frames_all.csv.gz", dtype={"points": str})
        D = pd.read_csv(M.FAR / f"video{v:02d}_{M.BASE}_devorder_marginall" / "frames_all.csv.gz", dtype={"points": str})
        pc = C.points.fillna("").map(lambda x: frozenset(x.split("+")) - {""}); pdd = D.points.fillna("").map(lambda x: frozenset(x.split("+")) - {""})
        for i in np.flatnonzero((pc != pdd).to_numpy()):
            rows.append({"video": v, "frame": int(i), "status_A": C.status[i], "points_A": C.points[i], "status_B": D.status[i], "points_B": D.points[i],
                         "status_C": D.status[i], "points_C": D.points[i], "worst_point": C.worst_point[i], "fit_rel": C.fit_rel[i]})
    Dd = pd.DataFrame(rows)
    Dd.to_csv(M.OUT / "subset_d_disagreements.csv", index=False)
    print("C vs D differ in", len(Dd), "frames")
    Dd["run_id"] = Dd.groupby("video").frame.transform(lambda f: np.cumsum(np.r_[True, np.diff(f.to_numpy()) > 5]))
    one = Dd.groupby(["video", "run_id"]).sample(1, random_state=SEED)
    S = one.groupby("video").sample(frac=1, random_state=SEED).groupby("video").head(PER)
    pool = one[~one.index.isin(S.index)]
    if len(S) < N and len(pool):
        S = pd.concat([S, pool.sample(min(N - len(S), len(pool)), random_state=SEED)])
    S = S.head(N).sort_values(["video", "frame"]).reset_index(drop=True)
    S["item"] = [f"video{v:02d}_f{f:06d}_d" for v, f in zip(S.video, S.frame)]
    S.to_csv(M.OUT / "subset_d_sample.csv", index=False)
    M.ITEMS.mkdir(parents=True, exist_ok=True)
    if not a.html_only:
        for v, g in S.groupby("video"):
            M.render(int(v), list(g.itertuples()), names=("C", "D", "D"))
    M.page(S, None, names=("C", "D", "D"), out_html="subset_d_review.html", key="subset_d_review")


if __name__ == "__main__":
    main()
