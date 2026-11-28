#!/usr/bin/env python3
"""Figures and two small tables for the final loop (experiment 5b) on all 19 videos, three turns (2026-11-28).

Layout (user 2026-11-28): the measures every model shares get one figure each, and every turn gets its own figure on
the frames hand-labeled for that turn:
    loop_final_truth_frames.png   the 70 truth frames of every video: wrong frames and median error, all four models
    loop_final_flagged.png        flagged share of the whole video (fitting + PCA), all four models
    loop_final_check_turn1.png    turn 1 against the starting model on the check frames of turn 1
    loop_final_check_turn2.png    turn 2 against turn 1 on the check frames of turn 2
    loop_final_check_turn3.png    turn 3 against turn 2 on the check frames of turn 3 (only the videos labeled so far)

Reads only result tables, nothing is recomputed from predictions:
    results/self_training/videoNN/{teacher,round1_formal,round2_formal,round3_formal}/eval.csv   truth frames, flagged share
    results/self_training/videoNN/roundN_formal/train_info.csv                                     frames fed back, minutes
    results/self_training/final_loop_summary.csv        written by final_loop_summary.py, which holds the decision rule
    results/blind_test/videoNN/final/per_frame.csv      hand-labeled check frames of the 15 videos without a blind set

    python final_loop_figures.py  ->  the five figures in results/self_training/figures/,
                                      results/self_training/final_loop_by_stratum.csv, final_loop_cost.csv

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
                         "flagged": float(e.flagged_either_pct), "jump": float(e.get("indep_jump_pct", np.nan))})
    return pd.DataFrame(rows)


def grouped(ax, T, col, ylabel, labels=False, clip=None, fmt="{:.0f}", models=None):
    """One bar per model and video (by default the starting model and the three turns)."""
    models = {m: MODELS[m] for m in (models or MODELS)}
    w = 0.8 / len(MODELS)
    for j, (m, (lab, c)) in enumerate(models.items()):
        g = T[T.model == m].sort_values("video")
        xs = g.video.to_numpy() + (j - (len(models) - 1) / 2) * w
        ys = g[col].to_numpy()
        ax.bar(xs, np.minimum(ys, clip) if clip else ys, w * 0.9, color=c, label=lab, edgecolor="white", linewidth=0.5)
        for x, y in zip(xs, ys):
            if labels and y > 0:
                ax.text(x, y, fmt.format(y), ha="center", va="bottom", fontsize=7.5, color="#222")
            if clip and y > clip:                # the bar is cut at the axis limit: its value is written above it
                ax.text(x, clip, f"{y:.1f} ↑", ha="center", va="bottom", fontsize=8, color="#222")
    if clip:
        ax.set_ylim(0, clip * 1.08)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(19))
    ax.set_xlim(-0.6, 18.6)
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
    ax.set_axisbelow(True)


def fig_truth(T):
    """The 70 truth frames of every video: wrong frames and median error for the four models."""
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 7.2), sharex=True)
    grouped(axes[0], T, "wrong", "wrong truth frames (of 70)", labels=True)
    grouped(axes[1], T, "median", "median truth error, % of eye width")
    tot = T.groupby("model").wrong.sum()
    axes[0].set_title("Truth frames (70 per video, the same frames for every model): wrong frames in total "
                      + " → ".join(str(int(tot[m])) for m in MODELS) + " of 1,330", fontsize=11, loc="left")
    axes[0].legend(ncol=4, frameon=False, loc="upper right")
    axes[0].set_ylim(0, T.wrong.max() + 1.6)
    axes[-1].set_xlabel("video")
    fig.tight_layout()
    fig.savefig(OUT / "loop_final_truth_frames.png", dpi=170)
    plt.close(fig)


def fig_flagged(T):
    """Flagged share of the whole video, no labels involved. Top: the two finders the loop uses (fitting + PCA).
    Bottom, as a control: the jump detector, which the loop never uses to pick or repair frames (largest frame-to-frame
    move of any point, threshold where the starting model flags 5 % of the video, self_training.independent_detectors)."""
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 8.2), sharex=True)
    ax = axes[0]
    grouped(ax, T, "flagged", "flagged (fitting + PCA), % of the video", clip=9.5)
    mean = T.groupby("model").flagged.mean()
    ax.set_title("Flagged by the loop's own finders (fitting + PCA): mean over the 19 videos "
                 + " → ".join(f"{mean[m]:.2f}" for m in MODELS) + " % of the video", fontsize=11, loc="left")
    ax.legend(ncol=4, frameon=False, loc="upper right")
    ax = axes[1]
    turns = [m for m in MODELS if m != "teacher"]
    start = T[T.model == "teacher"].jump.dropna()
    assert (start == 5.0).all(), "the starting model flags 5 % by construction"
    J = T.copy()                                 # the threshold is the starting model's own 95 % quantile, so its share is 5 %
    J.loc[J.model == "teacher", "jump"] = 5.0    # on every video (older eval.csv files of the starting model do not store it)
    grouped(ax, J, "jump", "flagged (jump), % of the video")
    ax.set_ylim(0, 6.3)
    mj = T.groupby("model").jump.mean()
    ax.set_title("Control, a detector the loop never uses (jump): mean over the 19 videos 5.00 → "
                 + " → ".join(f"{mj[m]:.2f}" for m in turns) + " % of the video  (starting model: 5 % by construction)", fontsize=11, loc="left")
    ax.legend(ncol=4, frameon=False, loc="upper right")
    ax.set_xlabel("video")
    fig.tight_layout()
    fig.savefig(OUT / "loop_final_flagged.png", dpi=170)
    plt.close(fig)


CHECKS = {1: ("start -> turn 1", "teacher", "round1_formal", "Turn 1 against the starting model, on the frames labeled to check turn 1",
              "* videos with a blind set: their 34 blind frames, drawn before the loop;  the other videos: the frames drawn after turn 2"),
          2: ("turn 1 -> turn 2", "round1_formal", "round2_formal", "Turn 2 against turn 1, on the frames labeled to check turn 2",
              "all videos: frames drawn after turn 2  (* by the flags of turn 1 and turn 2; the other videos by the flags of the starting model, turn 1 and turn 2)"),
          3: ("turn 2 -> turn 3", "round2_formal", "round3_formal", "Turn 3 against turn 2, on the frames labeled to check turn 3",
              "all videos: frames drawn after turn 3 by the flags of turn 2 and turn 3;  videos without bars are not labeled yet")}


def fig_check(S, turn):
    """One turn against the one before, on the check frames hand-labeled for that turn (rows of final_loop_summary.csv)."""
    name, a, b, title, note = CHECKS[turn]
    G = S[S.comparison == name]
    A = G[G.video.astype(str) == "all"].iloc[0]
    g = G[G.video.astype(str) != "all"].copy()
    g["video"] = g.video.astype(int)
    g = g.set_index("video")
    fig, ax = plt.subplots(figsize=(15.5, 4.6))
    w = 0.38
    for k, (col, m) in enumerate((("wrong_a", a), ("wrong_b", b))):
        xs = np.array(g.index) + (k - 0.5) * w
        ys = g[col].to_numpy()
        ax.bar(xs, ys, w * 0.92, color=MODELS[m][1], label=MODELS[m][0], edgecolor="white", linewidth=0.5)
        for xi, y in zip(xs, ys):
            ax.text(xi, y, f"{int(y)}", ha="center", va="bottom", fontsize=8, color="#222")
    ax.set_xticks(range(19))
    ax.set_xticklabels([f"{v}{'*' if v in BLIND4 else ''}\n{int(g.frames[v]) if v in g.index else '–'}" for v in range(19)])
    ax.set_xlim(-0.6, 18.6)
    top = max(g.wrong_a.max(), g.wrong_b.max()) * 1.2 + 1
    ax.set_ylim(0, top)
    missing = [v for v in range(19) if v not in g.index]
    if missing:
        ax.text((min(missing) + max(missing)) / 2, top * 0.5, f"videos {min(missing)} to {max(missing)}: not labeled yet", ha="center", va="center", fontsize=11, color="#888")
    ax.set_ylabel("wrong frames")
    part = f" ({len(g)} of 19 videos labeled)" if missing else ""
    ax.set_title(f"{title}{part}: {int(A.wrong_a)} → {int(A.wrong_b)} wrong of {int(A.frames)} hand-labeled frames; "
                 f"seed band {A.seed_band:.1f} frames; videos better / worse / same {int(A.videos_better)} / {int(A.videos_worse)} / {int(A.videos_same)}",
                 fontsize=10.5, loc="left")
    ax.legend(ncol=2, frameon=False, loc="upper right")
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("video (below: frames compared).  " + note)
    fig.tight_layout()
    fig.savefig(OUT / f"loop_final_check_turn{turn}.png", dpi=170)
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
    fig_flagged(T)
    S = pd.read_csv(ST / "final_loop_summary.csv")
    for turn in (1, 2, 3):
        fig_check(S, turn)
    P = by_stratum()
    C = cost()
    pd.set_option("display.width", 220)
    print("truth frames, totals and averages over the 19 videos")
    print(T.groupby("model").agg(wrong=("wrong", "sum"), median_of_medians=("median", "median"), flagged_mean=("flagged", "mean"),
                                 flagged_median=("flagged", "median")).reindex(list(MODELS)).round(2).to_string())
    print("\nwrong frames by stratum, check frames of the 15 videos without a blind set")
    print(P.to_string())
    print("\ncost per turn, medians over the videos")
    print(C.groupby("turn")[["frames_from_the_model", "train_minutes", "predict_minutes"]].median().round(1).to_string())
    print(f"\nfigures in {OUT}")


if __name__ == "__main__":
    main()
