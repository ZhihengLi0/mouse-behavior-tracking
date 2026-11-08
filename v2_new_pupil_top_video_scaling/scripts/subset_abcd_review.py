#!/usr/bin/env python3
"""The four subset-search variants side by side, judged by eye on one page (user's request 2026-11-08: compare A, B, C
and D together instead of C-vs-D on its own): yesterday's 35 frames where A, B, C disagree (subset_order_sample.csv)
plus the 20 frames where C and D disagree (subset_d_sample.csv). Variants that give the same result share a panel.

    python subset_abcd_review.py                 -> subset_review/subset_abcd_review.html (items video.._abcd.jpg)
    python subset_abcd_review.py --eval verdicts.csv   -> all_videos/subset_abcd_verdicts.csv"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import subset_order_comparison as M  # noqa: E402

RUNS = {"A": "", "B": "_devorder", "C": "_devorder_margin", "D": "_devorder_marginall"}
NAMES = tuple(RUNS)
HEADER = ("<b>Subset search: A, B, C and D side by side</b> ({n} frames: the 35 where A, B, C disagree and the 20 where C and D disagree). "
          "Panels: the frame; the model's points (red = pupil, yellow = eyelids and corners); then one panel per distinct result: blue = the points that "
          "variant replaced, arrow from where the model had the point (small filled dot) to where the repair moved it, with the point's name "
          "(p = pupil, lid = eyelid, c = corner). Variants with the same result share a panel.<br>"
          "每帧问：哪个变体的 8 个点全都在正确位置？A = 现在的顺序（先轮廓后瞳孔、先单点后两点、第一个过验收的就用）；B = 偏差最大的点先试；"
          "C = B 加「同一大小的子集里取验收余量最大的」；D = 所有大小的子集里取余量最大的（可能多换一个点）。结果相同的变体合成一个面板。"
          "模型原来的点就全对、不该修的选 model as it was；换上去的点对了、但还有一个没被找出来的点仍然错的选 moved point right, another point still wrong；"
          "几个面板换的点都对、要合起来这帧才全对（没有一个面板全换了）选 both changes needed；都不对选 none of them；看不清选 cannot tell。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=None)
    ap.add_argument("--html-only", action="store_true")
    a = ap.parse_args()
    if a.eval:
        C = pd.read_csv(M.OUT / "subset_abcd_sample.csv")
        J = C.merge(pd.read_csv(a.eval), on="item")
        J = J[J.verdict != "cannot tell"]
        rows = []
        for k in NAMES:
            right = J.verdict.str.split("+").apply(lambda g, k=k: k in g)
            rows.append({"variant": k, "judged": len(J), "right": int(right.sum()), "right_pct": round(100 * float(right.mean()), 1),
                         "right_on_the_35": int(right[J.source == "abc"].sum()), "right_on_the_20": int(right[J.source == "cd"].sum())})
        S = pd.DataFrame(rows)
        S["model_was_right"] = int((J.verdict == "model as it was (no repair needed)").sum())
        S["both_changes_needed"] = int(J.verdict.str.startswith("both").sum())
        S["another_point_not_caught"] = int(J.verdict.str.startswith("moved").sum())
        S["none_right"] = int((J.verdict == "none of them").sum())
        S.to_csv(M.OUT / "subset_abcd_verdicts.csv", index=False)
        print(S.to_string(index=False))
        return
    a35 = pd.read_csv(M.OUT / "subset_order_sample.csv")[["video", "frame"]].assign(source="abc")
    d20 = pd.read_csv(M.OUT / "subset_d_sample.csv")[["video", "frame"]].assign(source="cd")
    S = pd.concat([a35, d20]).drop_duplicates(["video", "frame"]).sort_values(["video", "frame"]).reset_index(drop=True)
    for v in sorted(S.video.unique()):
        F = {k: pd.read_csv(M.FAR / f"video{v:02d}_{M.BASE}{suf}" / "frames_all.csv.gz", dtype={"points": str}) for k, suf in RUNS.items()}
        for i in S.index[S.video == v]:
            f = int(S.frame[i])
            for k in NAMES:
                S.loc[i, f"status_{k}"] = F[k].status[f]
                S.loc[i, f"points_{k}"] = F[k].points[f] if isinstance(F[k].points[f], str) else ""
            S.loc[i, "worst_point"] = F["A"].worst_point[f]
    S["item"] = [f"video{v:02d}_f{f:06d}_abcd" for v, f in zip(S.video, S.frame)]
    # yesterday's verdicts on the 35 frames (A, B, C only) carried over as the pre-selected answer: a variant verdict is
    # mapped to the panel that now holds that variant (D may share it); the other answers keep their wording
    Y = pd.read_csv(M.OUT / "subset_review_verdicts.csv")
    Y = dict(zip(Y.item, Y.verdict))
    pre, notes = {}, []
    for r in S.itertuples():
        y = Y.get(f"video{r.video:02d}_f{r.frame:06d}")
        if y is None:
            notes.append("")
            continue
        groups = {}
        for k in NAMES:
            groups.setdefault((getattr(r, f"status_{k}"), "+".join(sorted(str(getattr(r, f"points_{k}")).split("+")))), []).append(k)
        letters = [k for k in y.split("+") if k in NAMES]
        if letters:
            g = next(("+".join(g) for g in groups.values() if letters[0] in g), None)
            pre[r.item] = g
        else:
            pre[r.item] = y
        notes.append(f"yesterday: {y}" + (f" (pre-selected as {pre[r.item]})" if pre[r.item] != y else ""))
    S["note"] = notes
    S.to_csv(M.OUT / "subset_abcd_sample.csv", index=False)
    print(len(S), "frames:", S.source.value_counts().to_dict())
    M.ITEMS.mkdir(parents=True, exist_ok=True)
    if not a.html_only:
        for v, g in S.groupby("video"):
            M.render(int(v), list(g.itertuples()), names=NAMES)
    M.page(S, None, names=NAMES, out_html="subset_abcd_review.html", key="subset_abcd_review", header=HEADER.format(n=len(S)),
           title="Subset search: A, B, C, D", prefill=pre)


if __name__ == "__main__":
    main()
