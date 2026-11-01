#!/usr/bin/env python3
"""Count the by-eye verdicts of the all-point repair review (outline_review.py), per point kind and per video.

    python outline_review_eval.py

Reads results/flag_and_repair/outline_review/items.csv and outline_review_verdicts.csv (one verdict per replaced point,
key item__point: yes / better, but still not right / no / already right before, should not have moved / cannot tell).
Writes outline_review_result.csv (by point kind), outline_review_by_point.csv, outline_review_per_video.csv,
outline_review_result.png. Nothing existing is modified."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_peak_repair import BPS, RES  # noqa: E402

OUT = RES / "flag_and_repair" / "outline_review"
KIND = {b: ("pupil" if b.startswith("pupil") else "eyelid" if b.startswith("eyelid") else "corner") for b in BPS}
ORDER = ["yes", "better, but still not right", "no", "already right before, should not have moved", "cannot tell", "not judged"]
SHORT = {"yes": "right", "better, but still not right": "closer, not right", "no": "not better", "already right before, should not have moved": "was right, moved for nothing", "cannot tell": "cannot tell", "not judged": "not judged"}


def main():
    I = pd.read_csv(OUT / "items.csv", dtype={"points": str})
    V = pd.read_csv(OUT / "outline_review_verdicts.csv")
    V[["item", "point"]] = V.item.str.split("__", expand=True)
    rows = []
    for r in I.itertuples():
        for b in str(r.points).split("+"):
            rows.append({"item": r.item, "video": r.video, "frame": r.frame, "point": b, "kind": KIND[b], "episode_len": r.episode_len})
    P = pd.DataFrame(rows).merge(V[["item", "point", "verdict"]], on=["item", "point"], how="left")
    P["verdict"] = P.verdict.fillna("not judged")
    P.to_csv(OUT / "outline_review_by_point.csv", index=False)
    T = pd.crosstab(P.kind, P.verdict).reindex(columns=[c for c in ORDER if c in set(P.verdict)], fill_value=0)
    T.loc["all"] = T.sum()
    T["points"] = T.sum(axis=1)
    T.to_csv(OUT / "outline_review_result.csv")
    print(T.to_string())
    by_pt = pd.crosstab(P.point, P.verdict).reindex(columns=[c for c in ORDER if c in set(P.verdict)], fill_value=0)
    print(by_pt.to_string())
    pv = pd.crosstab([P.video, P.kind], P.verdict).reindex(columns=[c for c in ORDER if c in set(P.verdict)], fill_value=0)
    pv.to_csv(OUT / "outline_review_per_video.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"yes": "#2f6db3", "better, but still not right": "#e0a458", "no": "#d1495b", "already right before, should not have moved": "#8e6c8a", "cannot tell": "0.75", "not judged": "0.9"}
    kinds = [k for k in ("pupil", "eyelid", "corner") if k in T.index]
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    bottom = [0] * len(kinds)
    for c in [c for c in ORDER if c in T.columns]:
        vals = [T.loc[k, c] for k in kinds]
        ax.bar(kinds, vals, bottom=bottom, color=cols[c], label=SHORT[c])
        for j, (b0, v) in enumerate(zip(bottom, vals)):
            if v:
                ax.text(j, b0 + v / 2, str(v), ha="center", va="center", fontsize=9, color="white" if c in ("yes", "no", "already right before, should not have moved") else "black")
        bottom = [b0 + v for b0, v in zip(bottom, vals)]
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels([f"{k} points\n({T.loc[k, 'points']})" for k in kinds])
    ax.set_ylabel("points put in by the repair, judged by eye")
    ax.set_title(f"Repair of all eight points: {int(T.loc['all', 'points'])} replaced points in {I.item.nunique()} frames, by verdict")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.92")
    ax.set_axisbelow(True)
    fig.savefig(OUT / "outline_review_result.png", dpi=150)
    print("figure", OUT / "outline_review_result.png")


if __name__ == "__main__":
    main()
