#!/usr/bin/env python3
"""Figures and two small tables for the final loop (experiment 5b) on all 19 videos, three turns (2026-11-28).

Reads only result tables, nothing is recomputed from predictions:
    results/self_training/videoNN/{teacher,round1_formal,round2_formal,round3_formal}/eval.csv   truth frames, flagged share
    results/self_training/videoNN/roundN_formal/train_info.csv                                     frames fed back, minutes
    results/self_training/final_loop_summary.csv        written by final_loop_summary.py, which holds the decision rule
    results/blind_test/videoNN/final/per_frame.csv      hand-labeled check frames of the 15 videos without a blind set

    python final_loop_figures.py  ->  results/self_training/figures/loop_final_all_videos.png    truth frames, 19 videos
                                      results/self_training/figures/loop_final_check_frames.png  hand-labeled check frames
                                      results/self_training/final_loop_by_stratum.csv
                                      results/self_training/final_loop_cost.csv

Colours as in the other loop figures: grey = starting model, pink = turn 1, indigo = turn 2, brown = turn 3.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from self_training_figures import COL, PINK, plt  # noqa: E402  (same style settings as the other loop figures)

RES = Path(__file__).resolve().parents[1] / "results"
ST, BT, OUT = RES / "self_training", RES / "blind_test", RES / "self_training" / "figures"
MODELS = {"teacher": ("starting model", COL["teacher"]), "round1_formal": ("turn 1", PINK[0]),
          "round2_formal": ("turn 2", PINK[1]), "round3_formal": ("turn 3", PINK[2])}
LINE = 0.14
BLIND4 = (1, 6, 8, 15)


def truth():
    rows = []
    for v in range(19):
        for m in MODELS:
            e = pd.read_csv(ST / f"video{v:02d}" / m / "eval.csv").iloc[0]
            rows.append({"video": v, "model": m, "wrong": int(e.wrong_frames), "median": float(e.median_error_pct),
                         "flagged": float(e.flagged_either_pct)})
    return pd.DataFrame(rows)


def fig_truth(T):
    """Truth frames (70 per video) and the flagged share of the whole video, four bars per video."""
    panels = [("wrong", "wrong truth frames (of 70)", "{:.0f}", None),
              ("flagged", "flagged (fitting + PCA), % of the video", "{:.1f}", 9.5),
              ("median", "median truth error, % of eye width", "{:.1f}", None)]
    fig, axes = plt.subplots(3, 1, figsize=(15.5, 10.2), sharex=True)
    w = 0.8 / len(MODELS)
    for ax, (col, ylabel, fmt, clip) in zip(axes, panels):
        for j, (m, (lab, c)) in enumerate(MODELS.items()):
            g = T[T.model == m].sort_values("video")
            xs = g.video.to_numpy() + (j - (len(MODELS) - 1) / 2) * w
            ys = g[col].to_numpy()
            ax.bar(xs, np.minimum(ys, clip) if clip else ys, w * 0.9, color=c, label=lab, edgecolor="white", linewidth=0.5)
            for x, y in zip(xs, ys):
                if col == "wrong" and y > 0:
                    ax.text(x, y, fmt.format(y), ha="center", va="bottom", fontsize=7.5, color="#222")
                if clip and y > clip:            # the bar is cut at the axis limit: its value is written above it
                    ax.text(x, clip, fmt.format(y) + " ↑", ha="center", va="bottom", fontsize=8, color="#222")
        if clip:
            ax.set_ylim(0, clip * 1.08)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
        ax.set_axisbelow(True)
    tot = T.groupby("model").wrong.sum()
    axes[0].set_title("Final loop on all 19 videos, truth frames: wrong frames in total "
                      + " → ".join(str(int(tot[m])) for m in MODELS) + " of 1,330", fontsize=11, loc="left")
    axes[0].legend(ncol=4, frameon=False, loc="upper right")
    axes[0].set_ylim(0, T.wrong.max() + 1.6)
    axes[-1].set_xticks(range(19))
    axes[-1].set_xlim(-0.6, 18.6)
    axes[-1].set_xlabel("video")
    fig.tight_layout()
    fig.savefig(OUT / "loop_final_all_videos.png", dpi=170)
    plt.close(fig)


def fig_check(S):
    """Hand-labeled check frames: per video the wrong frames before and after each turn, as compared in final_loop_summary.csv."""
    comps = [("start -> turn 1", "teacher", "round1_formal", "Starting model → turn 1"),
             ("turn 1 -> turn 2", "round1_formal", "round2_formal", "Turn 1 → turn 2"),
             ("turn 2 -> turn 3 (earlier check frames)", "round2_formal", "round3_formal", "Turn 2 → turn 3 (check frames drawn after turn 2)")]
    fig, axes = plt.subplots(3, 1, figsize=(15.5, 10.6))
    w = 0.38
    for ax, (name, a, b, title) in zip(axes, comps):
        G = S[S.comparison == name]
        A = G[G.video.astype(str) == "all"].iloc[0]
        g = G[G.video.astype(str) != "all"].copy()
        g["video"] = g.video.astype(int)
        g = g.sort_values("video")
        x = np.arange(len(g))
        for k, (col, m) in enumerate((("wrong_a", a), ("wrong_b", b))):
            ys = g[col].to_numpy()
            ax.bar(x + (k - 0.5) * w, ys, w * 0.92, color=MODELS[m][1], label=MODELS[m][0], edgecolor="white", linewidth=0.5)
            for xi, y in zip(x + (k - 0.5) * w, ys):
                ax.text(xi, y, f"{int(y)}", ha="center", va="bottom", fontsize=8, color="#222")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v}{'*' if v in BLIND4 else ''}\n{int(n)}" for v, n in zip(g.video, g.frames)])
        ax.set_xlim(-0.6, len(g) - 0.4)
        ax.set_ylim(0, max(g.wrong_a.max(), g.wrong_b.max()) * 1.2 + 1)
        ax.set_ylabel("wrong frames")
        ax.set_title(f"{title}: {int(A.wrong_a)} → {int(A.wrong_b)} wrong of {int(A.frames)} hand-labeled frames; "
                     f"seed band {A.seed_band:.1f} frames; videos better / worse / same {int(A.videos_better)} / {int(A.videos_worse)} / {int(A.videos_same)}",
                     fontsize=10.5, loc="left")
        ax.legend(ncol=2, frameon=False, loc="upper right")
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("video (below: frames compared).  * videos with a blind set: their 34 blind frames in the top panel, their turn-2 check frames in the other two;"
                        " the other videos: the frames drawn after turn 2")
    fig.tight_layout()
    fig.savefig(OUT / "loop_final_check_frames.png", dpi=170)
    plt.close(fig)


def by_stratum():
    """Wrong frames by stratum and model on the check frames of the 15 videos without a blind set (final/)."""
    names = {"S": "flagged by the starting model only", "T1": "flagged by turn 1, not by turn 2", "TB": "flagged by turn 1 and turn 2",
             "T2": "flagged by turn 2, not by turn 1", "N": "flagged by none of the three"}
    E = pd.concat([pd.read_csv(p).assign(video=int(p.parts[-3][5:])) for p in sorted(BT.glob("video*/final/per_frame.csv"))])
    E = E[E.model.isin(MODELS)]
    E["wrong"] = E.error > LINE
    P = E.pivot_table(index="stratum", columns="model", values="wrong", aggfunc="sum").reindex(list(names))[list(MODELS)].astype(int)
    P.insert(0, "frames", E[E.model == "teacher"].groupby("stratum").size().reindex(list(names)))
    P.insert(0, "meaning", [names[s] for s in P.index])
    P.loc["all"] = ["all strata", P.frames.sum()] + [int(P[m].sum()) for m in MODELS]
    P.insert(0, "videos", E.video.nunique())
    P.reset_index().to_csv(ST / "final_loop_by_stratum.csv", index=False)
    return P


def cost():
    """Frames in the training set and the minutes of each turn, per video (train_info.csv)."""
    rows = []
    for v in range(19):
        for t in (1, 2, 3):
            p = ST / f"video{v:02d}" / f"round{t}_formal" / "train_info.csv"
            if p.exists():
                r = pd.read_csv(p).iloc[0]
                rows.append({"video": v, "turn": t, "hand_labeled_frames": int(r.train_human), "frames_from_the_model": int(r.train_pseudo),
                             "iterations": int(r.iterations), "train_minutes": float(r.train_minutes), "predict_minutes": float(r.predict_minutes)})
    C = pd.DataFrame(rows)
    C.to_csv(ST / "final_loop_cost.csv", index=False)
    return C


def main():
    T = truth()
    fig_truth(T)
    S = pd.read_csv(ST / "final_loop_summary.csv")
    fig_check(S)
    P = by_stratum()
    C = cost()
    pd.set_option("display.width", 220)
    print("truth frames, totals and averages over the 19 videos")
    print(T.groupby("model").agg(wrong=("wrong", "sum"), median_of_medians=("median", "median"), flagged_mean=("flagged", "mean"),
                                 flagged_median=("flagged", "median")).reindex(list(MODELS)).round(2).to_string())
    print("\nwrong frames by stratum, check frames of the 15 videos without a blind set")
    print(P.to_string())
    print("\ncost per turn (all videos that ran the turn)")
    print(C.groupby("turn").agg(videos=("video", "size"), frames_from_the_model_median=("frames_from_the_model", "median"),
                                train_minutes_median=("train_minutes", "median"), predict_minutes_median=("predict_minutes", "median"),
                                hours_total=("train_minutes", lambda x: 0)).assign(
        hours_total=lambda d: [round(float((C[C.turn == t].train_minutes + C[C.turn == t].predict_minutes).sum()) / 60, 1) for t in d.index]).round(1).to_string())
    print(f"\nfigures in {OUT}")


if __name__ == "__main__":
    main()
