"""Figures of the self-training loop for the summary page (step 9.4).

Reads only committed csv tables (results/self_training, results/blind_test) plus the whole-video predictions for
the example frames; writes results/self_training/figures/loop_*.png|jpg.   python scripts/self_training_figures.py
"""
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
OUT = RES / "self_training" / "figures"
sys.path.insert(0, str(HERE / "scripts"))
import blind_test_set as B  # noqa: E402

VIDEOS = (1, 8, 15, 6)
ALL = (0, 1, 2, 4, 6, 7, 8, 12, 15, 18)
MODELS = {"teacher": "starting model", "teacher + repair": "starting model + repair", "round1_every30": "experiment 1",
          "round1_repaired": "experiment 2", "round2_every30": "experiment 1, second turn",
          "round1_both": "experiment 3, first turn", "round2_both": "experiment 3, second turn",
          "round3_both": "experiment 3, third turn"}
COL = {"teacher": "#6f6f6f", "teacher + repair": "#b9b9b9", "round1_every30": "#e69f00", "round1_repaired": "#d55e00",
       "round2_every30": "#f0c35c", "round1_both": "#0072b2", "round2_both": "#5aa5d6", "round3_both": "#b3d6ee"}   # 2026-11-14: one colour per experiment in every figure (1 orange, 2 vermillion, 3 blue, 4c green, 5b pink), later turns lighter
LOOP = ("round1_every30", "round1_repaired", "round1_both")       # the three first-turn loop models shown side by side
BARS = [m for m in MODELS if m not in ("round2_both", "round3_both")]   # later turns of experiment 3: figure loop_turns
LINE = 14.0
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888",
                     "axes.labelcolor": "#222", "xtick.color": "#444", "ytick.color": "#444", "axes.titlesize": 11})


def comparison():
    C = pd.read_csv(RES / "self_training" / "comparison.csv")
    return C[C.model.isin(MODELS)]


def bars(ax, C, col, models, ylabel, fmt, vids):
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        xs, ys = [], []
        for j, v in enumerate(vids):
            r = C[(C.video == v) & (C.model == m)]
            if len(r) and np.isfinite(r.iloc[0][col]):
                xs.append(j + (i - (len(models) - 1) / 2) * w)
                ys.append(float(r.iloc[0][col]))
        b = ax.bar(xs, ys, w * 0.92, color=COL[m], label=MODELS[m], hatch="///" if m == "teacher + repair" else None,
                   edgecolor="white", linewidth=0.5)
        for x, y in zip(xs, ys):
            ax.text(x, y, fmt(y), ha="center", va="bottom", fontsize=6 if len(models) > 2 else 7, color="#222",
                    rotation=90 if len(models) > 2 else 0)
    ax.set_xticks(range(len(vids)))
    ax.set_xticklabels([str(v) for v in vids])
    ax.set_xlabel("video")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
    ax.set_axisbelow(True)


def fig_truth():
    C = comparison()
    models = BARS
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    bars(axes[0], C, "median_error_pct", models, "median error, 70 truth frames, % of eye width", lambda y: f"{y:.1f}", VIDEOS)
    bars(axes[1], C, "wrong_frames", models, "wrong truth frames (error over 14 %), of 70", lambda y: f"{y:.0f}", VIDEOS)
    bars(axes[2], C, "flagged_either_pct", [m for m in models if m != "teacher + repair"],
         "frames flagged (fitting + PCA), % of the video", lambda y: f"{y:.1f}", VIDEOS)
    axes[1].set_ylim(0, 9.5)
    axes[0].set_ylim(0, 6.2)
    axes[2].set_ylim(0, 17.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "loop_truth.png", dpi=170)
    plt.close(fig)


def fig_all_videos():
    C = comparison()
    models = ["teacher", "round1_every30"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))
    bars(axes[0], C, "median_error_pct", models, "median error, 70 truth frames, % of eye width", lambda y: f"{y:.1f}", ALL)
    bars(axes[1], C, "wrong_frames", models, "wrong truth frames of 70", lambda y: f"{y:.0f}", ALL)
    bars(axes[2], C, "flagged_either_pct", models, "frames flagged (fitting + PCA), % of the video", lambda y: f"{y:.1f}", ALL)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "loop_all_videos.png", dpi=170)
    plt.close(fig)


def fig_blind():
    mk = {"A": "o", "B": "s", "C": "^", "D": "D"}
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.7))
    for ax, v in zip(axes, VIDEOS):
        P = pd.read_csv(RES / "blind_test" / f"video{v:02d}" / "per_frame.csv")
        T = P[P.model == "teacher"].set_index("frame").error * 100
        for m in LOOP:
            M = P[P.model == m].set_index("frame")
            for s, marker in mk.items():
                idx = M.index[M.stratum == s]
                if len(idx):
                    ax.scatter(T.loc[idx], M.loc[idx].error * 100, s=26, marker=marker, color=COL[m], alpha=0.85,
                               edgecolor="white", linewidth=0.5)
        lim = max(45, float(T.max()) + 3)
        ax.plot([0, lim], [0, lim], color="#bbb", linewidth=0.8, zorder=0)
        ax.axhline(LINE, color="#999", linestyle="--", linewidth=0.7)
        ax.axvline(LINE, color="#999", linestyle="--", linewidth=0.7)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(f"video {v}")
        ax.set_xlabel("starting model error, % of eye width")
        ax.grid(color="#eee", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("loop model error, % of eye width")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=COL[m], marker="o", linestyle="", label=MODELS[m]) for m in LOOP]
    hs += [Line2D([], [], color="#555", marker=mk[s], linestyle="", label=f"stratum {s}") for s in mk]
    fig.legend(handles=hs, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "loop_blind.png", dpi=170)
    plt.close(fig)


def fig_examples():
    """Three stratum-A blind frames: human labels and the points of the starting model and both loop models."""
    picks = []
    for v, k in ((1, 2), (8, 1)):
        P = pd.read_csv(RES / "blind_test" / f"video{v:02d}" / "per_frame.csv")
        A = P[(P.model == "teacher") & (P.stratum == "A")].sort_values("error", ascending=False)
        picks += [(v, int(f), nm) for f, nm in zip(A.frame[:k], A.blind_name[:k])]
    panels = []
    for v, f, nm in picks:
        img = cv2.imread(str(RES / "blind_test" / f"video{v:02d}" / "labeling" / "blind" / nm))
        L = pd.read_csv(RES / "blind_test" / f"video{v:02d}" / "labeling" / "blind" / "CollectedData_Zhiheng.csv",
                        header=[0, 1, 2], index_col=[0, 1, 2])
        sc = L.columns.get_level_values(0)[0]
        row = L.loc[[i for i in L.index if i[2] == nm][0]]
        human = {b: (row[(sc, b, "x")], row[(sc, b, "y")]) for b in B.BPS}
        models = B.models_of(v, None)
        pts = {}
        for m in ("teacher",) + LOOP:
            xy = B.coords_of(models[m])
            pts[m] = {b: xy[b][f] for b in B.BPS}
        panels.append((v, f, img, human, pts))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    for ax, (v, f, img, human, pts) in zip(axes, panels):
        xs = [p[0] for p in human.values() if np.isfinite(p[0])] + [p[0] for d in pts.values() for p in d.values()]
        ys = [p[1] for p in human.values() if np.isfinite(p[1])] + [p[1] for d in pts.values() for p in d.values()]
        x0, x1 = max(0, int(min(xs)) - 40), min(img.shape[1], int(max(xs)) + 40)
        y0, y1 = max(0, int(min(ys)) - 40), min(img.shape[0], int(max(ys)) + 40)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1], extent=(x0, x1, y1, y0))
        for m, d in pts.items():
            ax.scatter([p[0] for p in d.values()], [p[1] for p in d.values()], s=34, color=COL[m], edgecolor="black",
                       linewidth=0.4, label=MODELS[m], zorder=3)
        ax.scatter([p[0] for p in human.values()], [p[1] for p in human.values()], s=60, facecolor="none",
                   edgecolor="white", linewidth=1.4, label="human label", zorder=4)
        ax.set_title(f"video {v}, frame {f}")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=COL[m], marker="o", markeredgecolor="black", markeredgewidth=0.4, linestyle="", label=MODELS[m])
          for m in ("teacher",) + LOOP]
    hs.append(Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="#333", markeredgewidth=1.4, markersize=9,
                     linestyle="", label="human label (white ring in the image)"))
    fig.legend(handles=hs, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "loop_examples.jpg", dpi=150, pil_kwargs={"quality": 88})
    plt.close(fig)
    return picks


def blind_wrong(v, model):
    """Wrong frames of a model on the 24 blind frames plus the 10 supplementary frames (error over the 14 % line)."""
    n = 0
    for f in ("per_frame.csv", "supplement/per_frame.csv"):
        P = pd.read_csv(RES / "blind_test" / f"video{v:02d}" / f)
        n += int((P[P.model == model].error > LINE / 100).sum())
    return n


def fig_turns():
    """Experiment 3 turn by turn: flagged share of the video, wrong blind frames (of 34) and wrong truth frames (of 70),
    turn 0 = the starting model; a dotted line marks the turn at which the stop rule stopped the loop."""
    C = comparison()
    series = [("flagged_either_pct", "frames flagged (fitting + PCA), % of the video", "o", "#0072b2", (0, 5), "center"),   # experiment 3's blue for all three; the measure is told by marker and line style
              ("blind_wrong_pct", "wrong blind frames, % of 34", "s", "#0072b2", (0, 5), "center"),
              ("truth_wrong_pct", "wrong truth frames, % of 70", "^", "#0072b2", (7, -3), "left")]
    LS = {"flagged_either_pct": "-", "blind_wrong_pct": "--", "truth_wrong_pct": ":"}
    FACE = {"flagged_either_pct": "#0072b2", "blind_wrong_pct": "white", "truth_wrong_pct": "#b3d6ee"}
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), sharey=True)
    for ax, v in zip(axes, VIDEOS):
        rows = []
        for r in range(4):
            m = "teacher" if r == 0 else f"round{r}_both"
            c = C[(C.video == v) & (C.model == m)]
            if not len(c):
                break
            rows.append({"turn": r, "flagged_either_pct": float(c.iloc[0].flagged_either_pct),
                         "blind_wrong_pct": 100 * blind_wrong(v, m) / 34, "truth_wrong_pct": 100 * float(c.iloc[0].wrong_frames) / 70})
        T = pd.DataFrame(rows)
        for col, label, marker, colour, off, ha in series:
            ax.plot(T.turn, T[col], marker=marker, color=colour, linewidth=1.6, markersize=6, label=label, linestyle=LS[col], markerfacecolor=FACE[col], markeredgewidth=1.3)
            for x, y in zip(T.turn, T[col]):
                ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=off, ha=ha, fontsize=6.5, color="#333")
        status = RES / "self_training" / f"video{v:02d}" / "loop_status.csv"
        if status.exists():
            S = pd.read_csv(status)
            stop = S[S.stop.astype(str) == "True"]
            if len(stop):
                ax.axvline(int(stop.iloc[0]["round"]), color="#999", linestyle=":", linewidth=0.9)
                ax.text(int(stop.iloc[0]["round"]) + 0.05, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 1, "stopped",
                        fontsize=7, color="#666", va="top")
        ax.set_title(f"video {v}")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["start", "turn 1", "turn 2", "turn 3"], fontsize=8)
        ax.set_xlim(-0.4, 3.4)
        ax.grid(color="#eee", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("percent")
    axes[0].set_ylim(0, 62)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "loop_turns.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_truth()
    fig_all_videos()
    fig_blind()
    fig_turns()
    print("examples:", fig_examples())
    print("written to", OUT)


# ---- one figure per experiment: starting model against the experiment's model, video by video (page 9.4, 2026-11-07) ----
EXPS = {"round1_every30": ("experiment 1", "#e69f00", 24), "round1_repaired": ("experiment 2", "#d55e00", 24),
        "round1_both": ("experiment 3, turn 1", "#56b4e9", 34)}


def blind_wrong_n(v, model, n):
    """Wrong frames of a model on the 24 stratified blind frames (n = 24) or on all 34 (n = 34)."""
    files = ["per_frame.csv"] + (["supplement/per_frame.csv"] if n == 34 else [])
    tot = 0
    for f in files:
        P = pd.read_csv(RES / "blind_test" / f"video{v:02d}" / f)
        if model not in set(P.model):
            return np.nan
        tot += int((P[P.model == model].error > LINE / 100).sum())
    return tot


def fig_experiment(model, second_label="experiment 1, second turn (videos 1, 8)"):
    label, colour, nblind = EXPS[model]
    C = pd.read_csv(RES / "self_training" / "comparison.csv")
    vids = [int(v) for v in sorted(C[C.model == model].video.unique())]
    series = [("teacher", COL["teacher"], "starting model"), (model, colour, label)]
    if model == "round1_every30" and "round2_every30" in set(C.model):
        series.append(("round2_every30", COL["round2_every30"], second_label))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.2))
    axes = axes.ravel()
    cols = [("wrong_frames", "wrong truth frames (of 70)", "{:.0f}"), ("median_error_pct", "median truth error, % of eye width", "{:.1f}"),
            ("flagged_either_pct", "frames flagged, % of the video", "{:.1f}")]
    k = len(series)
    w = 0.8 / k
    for ax, (col, ylabel, fmt) in zip(axes[:3], cols):
        for j, (m, c, lab) in enumerate(series):
            pts = [(i, float(C[(C.video == v) & (C.model == m)].iloc[0][col])) for i, v in enumerate(vids) if len(C[(C.video == v) & (C.model == m)])]
            xs = [i + (j - (k - 1) / 2) * w for i, _ in pts]
            ys = [y for _, y in pts]
            ax.bar(xs, ys, w * 0.92, color=c, label=lab, edgecolor="white", linewidth=0.5)
            for x, y in zip(xs, ys):
                ax.text(x, y, fmt.format(y), ha="center", va="bottom", fontsize=8, color="#222")
        ax.set_xticks(range(len(vids)))
        ax.set_xticklabels([str(v) for v in vids])
        ax.set_xlabel("video")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    bv = [v for v in vids if (RES / "blind_test" / f"video{v:02d}" / "per_frame.csv").exists()]
    ax = axes[3]
    for j, (m, c, lab) in enumerate(series):
        pts = [(i, blind_wrong_n(v, m, nblind)) for i, v in enumerate(bv)]
        pts = [(i, y) for i, y in pts if np.isfinite(y)]
        xs = [i + (j - (k - 1) / 2) * w for i, _ in pts]
        ys = [y for _, y in pts]
        ax.bar(xs, ys, w * 0.92, color=c, edgecolor="white", linewidth=0.5)
        for x, y in zip(xs, ys):
            ax.text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=8, color="#222")
    ax.set_xticks(range(len(bv)))
    ax.set_xticklabels([str(v) for v in bv])
    ax.set_xlabel("video")
    ax.set_ylabel(f"wrong blind frames (of {nblind})")
    ax.set_ylim(0, 20)
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
    ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    name = {"round1_every30": "loop_exp1", "round1_repaired": "loop_exp2", "round1_both": "loop_exp3"}[model]
    fig.savefig(OUT / f"{name}.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__" and "--experiments" in sys.argv:
    for m in EXPS:
        fig_experiment(m)
    print("experiment figures written")


# ------------------------------------------------------------------ audit figures (experiments 4 and 5a, 2026-11-08)
FAR = RES / "flag_and_repair"
BASE = "either_labelregion_all_smallest"
VARIANTS = {"A": "", "B": "_devorder", "C": "_devorder_margin", "D": "_devorder_marginall"}
VCOL = {"A": "#6f6f6f", "B": "#e69f00", "C": "#56b4e9", "D": "#009e73"}
COL["round1_reopened"] = "#009e73"
MODELS["round1_reopened"] = "experiment 4c (reopened frames only)"
FATE = [("closed", "still closed", "#444444"), ("repaired", "repaired", "#009e73"), ("ok", "passes, not flagged", "#56b4e9"),
        ("flagged_passes_check", "flagged, plausible as it is", "#cc79a7"), ("flagged_not_repaired", "still flagged", "#e69f00"),
        ("unflagged_violation", "other", "#b9b9b9")]


def _frames(v, suffix):
    return pd.read_csv(FAR / f"video{v:02d}_{BASE}{suffix}" / "frames_all.csv.gz", dtype={"points": str}, low_memory=False)


def _closed_counts(v, model):
    """Frames of a whole-video prediction that each closed-eye rule calls closed."""
    import self_training as ST
    from second_peak_repair import BPS, P4, Geometry
    line = float(pd.read_csv(FAR / "closed_frames" / "closed_line_chosen.csv").iloc[0].pupil_confidence_line)
    h5 = ST.teacher_h5(v) if model == "teacher" else sorted((ST.LOCAL / f"video{v:02d}" / model / "predictions").glob("*.h5"))[-1]
    d = pd.read_hdf(h5)
    d.columns = d.columns.droplevel(0)
    xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
    conf = np.mean([d[b]["likelihood"].to_numpy(float) for b in P4], axis=0)
    return int(Geometry(xy).closed.sum()), int((conf < line).sum())


def fig_exp4a():
    """Experiment 4a, two questions: why the pupil confidence (the opening ratio does not separate the judged closed
    frames from the open ones, the confidence does) and why 0.40 (accuracy of every candidate line, with the truth
    frames each line would remove)."""
    S = pd.read_csv(FAR / "closed_frames" / "sample.csv")
    V = pd.read_csv(FAR / "closed_frames" / "closed_review_verdicts.csv")
    V = V[~V.item.str.contains(":")]
    J = S.merge(V, on="item")
    J = J[J.verdict != "cannot tell"]
    chosen = float(pd.read_csv(FAR / "closed_frames" / "closed_line_chosen.csv").iloc[0].pupil_confidence_line)
    C = pd.read_csv(FAR / "closed_frames" / "confidence_line_candidates.csv")
    O = pd.read_csv(FAR / "closed_frames" / "line_candidates.csv")
    closed = J.verdict == "fully closed"
    cats = [("fully closed", "#444444"), ("half open, pupil can be labeled", "#56b4e9"), ("open, eyelid points wrong", "#e69f00")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    rng = np.random.default_rng(0)

    def strip(ax, col, ylabel, title, line=None, line_label=None):
        for k, (cat, c) in enumerate(cats):
            y = J[col][J.verdict == cat].to_numpy()
            ax.scatter(k + rng.uniform(-0.18, 0.18, len(y)), y, s=26, color=c, edgecolor="white", linewidth=0.5, zorder=3)
        if line is not None:
            ax.axhline(line, color="#d55e00", lw=1.2, ls="--")
            ax.text(2.45, line, line_label, color="#d55e00", fontsize=9, ha="right", va="bottom")
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{lab}\n({int((J.verdict == cat).sum())} frames)" for (cat, _), lab in zip(cats, ("fully closed", "half open,\npupil can be labeled", "open, eyelid\npoints wrong"))], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
        ax.set_axisbelow(True)

    ob = O.loc[O.accuracy_pct.idxmax()]
    strip(axes[0], "r", "eyelid opening, ratio to the video's median", f"(a) why not the opening: {len(J)} judged frames",
          line=float(ob.line), line_label=f"best line {ob.line:.2f}: {ob.accuracy_pct:.0f} % right")
    axes[0].set_ylim(-0.02, 0.75)
    r = C[np.isclose(C.line, chosen)].iloc[0]
    strip(axes[1], "pupil_lk_mean", "mean heatmap confidence of the 4 pupil points", "(b) why the pupil confidence: same frames",
          line=chosen, line_label=f"line {chosen:.2f}: {r.accuracy_pct:.0f} % right")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].text(0.02, 0.97, f"below {chosen:.2f}: {int((closed & (J.pupil_lk_mean < chosen)).sum())} of {int(closed.sum())} closed frames,\n{int((~closed & (J.pupil_lk_mean < chosen)).sum())} of {int((~closed).sum())} open frames",
                 transform=axes[1].transAxes, va="top", fontsize=8.5, color="#222")
    ax = axes[2]
    ax.plot(C.line, C.accuracy_pct, "-o", color="#0072b2", ms=5, lw=1.6)
    ax.axvline(chosen, color="#d55e00", lw=1.2, ls="--")
    for _, q in C.iterrows():
        ax.text(q.line, q.accuracy_pct - 2.2, f"{int(q.labeled_truth_frames_called_closed)}", ha="center", va="top", fontsize=7.5, color="#cc79a7")
    ax.text(0.03, 0.06, "pink numbers: truth frames with a human pupil label the line would call closed (of 1318)", transform=ax.transAxes, fontsize=8, color="#cc79a7")
    ax.annotate(f"{chosen:.2f}: {r.accuracy_pct:.0f} % right, {int(r.labeled_truth_frames_called_closed)} truth frames removed, all already wrong",
                (chosen, r.accuracy_pct), (0.35, 0.93), textcoords="axes fraction", fontsize=8.5, color="#222", arrowprops=dict(arrowstyle="-", color="#999", lw=0.8))
    ax.set_xlabel("candidate line on the mean pupil confidence")
    ax.set_ylabel("judged frames classified right, %")
    ax.set_ylim(50, 100)
    ax.set_title("(c) why 0.40: every candidate line", loc="left")
    ax.grid(color="#e4e4e4", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "loop_exp4a.png", dpi=170)
    plt.close(fig)


def fig_exp4c():
    """Experiment 4c: the student trained on the reopened frames only against its teacher, per video: flagged share
    (with wrong truth and blind frames), and frames the confidence rule calls closed on the prediction."""
    vids = [v for v in range(19) if (RES / "self_training" / f"video{v:02d}" / "round1_reopened" / "eval.csv").exists()]
    series = [("teacher", COL["teacher"], "starting model (teacher)"), ("round1_reopened", COL["round1_reopened"], "student: human labels + reopened frames only")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    w = 0.36
    for j, (m, c, lab) in enumerate(series):
        for i, v in enumerate(vids):
            e = pd.read_csv(RES / "self_training" / f"video{v:02d}" / m / "eval.csv").iloc[0]
            x = i + (j - 0.5) * w
            axes[0].bar([x], [e.flagged_either_pct], w * 0.92, color=c, label=lab if i == 0 else None, edgecolor="white", linewidth=0.5)
            bl = ""
            bs = RES / "self_training" / f"video{v:02d}" / "round1_reopened" / "blind_scores.csv"
            if bs.exists():
                B_ = pd.read_csv(bs)
                n = int(B_[B_.model == m].wrong_frames.sum()); tot = int(B_[B_.model == m].frames.sum())
                bl = f"\nblind {n} of {tot}"
            axes[0].text(x, e.flagged_either_pct, f"{e.flagged_either_pct:.1f} %\ntruth {int(e.wrong_frames)} of 70{bl}", ha="center", va="bottom", fontsize=7.5, color="#222")
    # (b) the reopened frames the student was NOT trained on (within 60 frames of a truth or blind frame): the teacher
    # flagged every one of them (a repaired frame is a flagged frame); how many does the student still flag?
    for i, v in enumerate(vids):
        F = pd.read_csv(RES / "self_training" / f"video{v:02d}" / "pseudo_reopened" / "pseudo_frames.csv")
        held = F.frame[~F.kept].to_numpy()
        Sf = pd.read_csv(RES / "self_training" / f"video{v:02d}" / "round1_reopened" / "flags.csv.gz").set_index("frame").loc[held]
        st = int((Sf.fit_flag | Sf.pca_flag).sum())
        for j, (m, c, lab) in enumerate(series):
            y = len(held) if m == "teacher" else st
            x = i + (j - 0.5) * w
            axes[1].bar([x], [y], w * 0.92, color=c, label=lab if i == 0 else None, edgecolor="white", linewidth=0.5)
            axes[1].text(x, y, f"{y} of {len(held)}", ha="center", va="bottom", fontsize=8, color="#222")
    for ax in axes:
        ax.set_xticks(range(len(vids)))
        ax.set_xticklabels([str(v) for v in vids])
        ax.set_xlabel("video")
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    axes[0].set_ylabel("frames flagged, % of the video")
    axes[0].set_title("(a) flagged share; labels: wrong truth frames, wrong blind frames", loc="left")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1].set_ylabel("frames flagged")
    axes[1].set_title("(b) reopened frames left out of the training set:\nhow many are still flagged", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "loop_exp4c.png", dpi=170)
    plt.close(fig)


def fig_exp5a():
    """Experiment 5a, choice of the repair's subset rule: (a) whole videos, points replaced and frames where the variants
    differ; (b) 55 frames judged by eye; and the example sheet loop_exp5a_examples.jpg from the review items."""
    pts_total, frames_rep, sets = {k: 0 for k in VARIANTS}, {k: 0 for k in VARIANTS}, {k: [] for k in VARIANTS}
    for v in range(19):
        F = {k: _frames(v, s) for k, s in VARIANTS.items()}
        for k in VARIANTS:
            rep = F[k].status == "repaired"
            frames_rep[k] += int(rep.sum())
            p = F[k].points.where(rep, "")
            pts_total[k] += int(p.fillna("").map(lambda x: len([q for q in str(x).split("+") if q and q != "nan"])).sum())
            sets[k].append(p.fillna("").map(lambda x: frozenset(q for q in str(x).split("+") if q and q != "nan")).to_numpy())
    diff = {}
    for a in VARIANTS:
        for b in VARIANTS:
            if a < b:
                diff[(a, b)] = int(sum((sa != sb).sum() for sa, sb in zip(sets[a], sets[b])))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    xs = range(len(VARIANTS))
    ax.bar(xs, [pts_total[k] for k in VARIANTS], 0.6, color=[VCOL[k] for k in VARIANTS], edgecolor="white", linewidth=0.5)
    for i, k in enumerate(VARIANTS):
        ax.text(i, pts_total[k], f"{pts_total[k]:,} points\nin {frames_rep[k]:,} frames", ha="center", va="bottom", fontsize=8, color="#222")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["A: outline first,\nsingles first", "B: largest\ndeviation first", "C: B + largest margin\nwithin the size", "D: largest margin\nover all sizes"], fontsize=8.5)
    ax.set_ylabel("points replaced on the 19 videos")
    ax.set_ylim(0, max(pts_total.values()) * 1.3)
    ax.set_title("(a) whole videos: the same frames repaired, different points replaced", loc="left")
    ax.text(0.02, 0.97, "frames whose replaced points differ: " + ", ".join(f"{a} vs {b} {n}" for (a, b), n in diff.items() if n and (a, b) in (("A", "B"), ("B", "C"), ("C", "D"), ("A", "D"))),
            transform=ax.transAxes, va="top", fontsize=8.5, color="#222")
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
    ax.set_axisbelow(True)
    # (b) judged
    S = pd.read_csv(FAR / "all_videos" / "subset_abcd_sample.csv").merge(pd.read_csv(FAR / "all_videos" / "subset_abcd_review_verdicts.csv"), on="item")
    S = S[S.verdict != "cannot tell"]
    n35, n20 = int((S.source == "abc").sum()), int((S.source == "cd").sum())
    ax = axes[1]
    for i, k in enumerate(VARIANTS):
        right = S.verdict.str.split("+").apply(lambda g, k=k: k in g)
        a, b = int(right[S.source == "abc"].sum()), int(right[S.source == "cd"].sum())
        ax.bar([i], [a], 0.6, color=VCOL[k], edgecolor="white", linewidth=0.5)
        ax.bar([i], [b], 0.6, bottom=a, color=VCOL[k], alpha=0.45, edgecolor="white", linewidth=0.5)
        ax.text(i, a + b, f"{a + b} of {len(S)}\n({100 * (a + b) / len(S):.0f} %)", ha="center", va="bottom", fontsize=8.5, color="#222")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(list(VARIANTS))
    ax.set_ylabel("frames with every point right")
    ax.set_ylim(0, len(S) * 1.15)
    ax.set_title(f"(b) {len(S)} frames where the variants differ, judged by eye", loc="left")
    others = {"model as it was": int((S.verdict == "model as it was (no repair needed)").sum()), "more than one change needed": int(S.verdict.str.startswith("both").sum()),
              "another point not caught": int(S.verdict.str.startswith("moved").sum()), "none right": int((S.verdict == "none of them").sum())}
    ax.text(0.02, 0.97, f"dark: the {n35} frames where A, B, C differ; light: the {n20} frames where C and D differ\n" + "; ".join(f"{k} {n}" for k, n in others.items()),
            transform=ax.transAxes, va="top", fontsize=8.5, color="#222")
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "loop_exp5a.png", dpi=170)
    plt.close(fig)
    # example sheet: two review items stacked
    items = FAR / "subset_review" / "items"
    ims = [cv2.imread(str(items / f"{n}_abcd.jpg")) for n in ("video00_f009620", "video10_f042787")]
    ims = [im for im in ims if im is not None]
    if ims:
        wmax = max(im.shape[1] for im in ims)
        rows = [cv2.copyMakeBorder(im, 0, 12, 0, wmax - im.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for im in ims]
        cv2.imwrite(str(OUT / "loop_exp5a_examples.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return pts_total, frames_rep, diff


if __name__ == "__main__" and "--audit" in sys.argv:
    fig_exp4a()
    fig_exp4c()
    print("exp5a:", fig_exp5a())
    print("audit figures written")


# ---- second turn of every experiment (page 9.4, 2026-11-12): reads results/self_training/round2_analysis/*.csv ----
R2 = RES / "self_training" / "round2_analysis"
R2EXP = [("every30", "experiment 1 (grid frames only)", "#e69f00", "o"), ("repaired", "experiment 2 (repaired frames only)", "#d55e00", "s"),
         ("both", "experiment 3 (both kinds)", "#0072b2", "^"), ("reopened", "experiment 4c (reopened frames only)", "#009e73", "D"),
         ("formal", "experiment 5b (final version)", "#cc79a7", "v")]


SEED_PAIRS = [(1, "formal"), (15, "formal"), (6, "formal"), (8, "both")]   # turn-1 models trained again with seed 43 (round1_<source>_seed43)


def seed43_point(v, src):
    """Blind wrong count (of 34) and flagged share of the seed-43 replicate of a turn-1 model (its own eval.csv and blind_scores.csv)."""
    d = RES / "self_training" / f"video{v:02d}" / f"round1_{src}_seed43"
    if not (d / "eval.csv").exists() or not (d / "blind_scores.csv").exists():
        return None
    Bs = pd.read_csv(d / "blind_scores.csv")
    return {"blind_wrong_34": int(Bs[Bs.model == f"round1_{src}_seed43"].wrong_frames.sum()),
            "flagged_either_pct": float(pd.read_csv(d / "eval.csv").iloc[0].flagged_either_pct)}


def fig_round2(max_turn=2, seeds=False, detectors=False):
    """Every experiment turn by turn on the four videos: wrong blind frames of 34 (top) and flagged share (bottom, log
    scale); turn 0 = the starting model. max_turn = 3 adds the third turns (written to loop_all_turns.png).
    detectors = True adds two rows with the independent detectors (round2_analysis/independent_detectors.csv: lowest
    point confidence, largest frame-to-frame move; the starting model is 5 % by construction)."""
    S = pd.read_csv(R2 / "summary.csv")
    metrics = [("blind_wrong_34", "wrong blind frames (of 34)"), ("flagged_either_pct", "flagged (fitting + PCA), % of the video")]
    if detectors:
        Dd = pd.read_csv(R2 / "independent_detectors.csv")
        S = S.merge(Dd[Dd.seed == 42][["video", "model", "indep_lowconf_pct", "indep_jump_pct"]], on=["video", "model"], how="left")
        metrics += [("indep_lowconf_pct", "flagged (low confidence), % of the video"), ("indep_jump_pct", "flagged (jump), % of the video")]
    fig, axes = plt.subplots(len(metrics), 4, figsize=(13.5, 3.05 * len(metrics) + 0.5), sharex=True)
    for j, v in enumerate(VIDEOS):
        t = S[(S.video == v) & (S.experiment == "teacher")].iloc[0]
        for i, (col, ylabel) in enumerate(metrics):
            ax = axes[i, j]
            for k, (src, lab, c, mk) in enumerate(R2EXP):
                g = S[(S.video == v) & (S.experiment == src) & (S.turn <= max_turn)].sort_values("turn")
                if not len(g):
                    continue
                xs = [0] + [int(x) for x in g.turn]
                ys = [float(t[col])] + [float(y) for y in g[col]]
                ax.plot(xs, ys, marker=mk, color=c, linewidth=1.5, markersize=5, label=lab, alpha=0.95)
                if seeds and (v, src) in SEED_PAIRS and seed43_point(v, src):   # the same turn-1 model with seed 43: hollow marker beside the seed-42 point
                    if col.startswith("indep"):
                        y43 = float(Dd[(Dd.video == v) & (Dd.model == f"round1_{src}_seed43")][col].iloc[0])
                    else:
                        y43 = seed43_point(v, src)[col]
                    ax.plot([1.16], [y43], marker=mk, color=c, markerfacecolor="white", markeredgewidth=1.4, markersize=6.5, linestyle="", zorder=5)
            ax.set_xticks(range(max_turn + 1))
            ax.set_xticklabels(["start", "turn 1", "turn 2", "turn 3"][:max_turn + 1])
            ax.set_xlim(-0.25, max_turn + 0.25)
            ax.grid(color="#eee", linewidth=0.5)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(f"video {v}")
                ax.set_ylim(0, 20)
            else:
                ax.set_yscale("log")
                ax.set_ylim(0.25, 22)
                ax.set_yticks([0.3, 1, 3, 10])
                ax.set_yticklabels(["0.3", "1", "3", "10"])
            if j == 0:
                ax.set_ylabel(ylabel)
    if not seeds:
        axes[0, 3].errorbar([max_turn + 0.12], [12], yerr=[1], color="#555", capsize=3, linewidth=1)
        axes[0, 3].text(max_turn + 0.05, 12, "seed alone:\n\u00b11 frame", ha="right", va="center", fontsize=7.5, color="#555")
    h, l = axes[0, 0].get_legend_handles_labels()
    if seeds:
        from matplotlib.lines import Line2D
        h.append(Line2D([], [], marker="o", color="#555", markerfacecolor="white", markeredgewidth=1.4, markersize=6.5, linestyle=""))
        l.append("hollow: the same turn-1 model trained with another seed")
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955 if detectors else 0.92))
    fig.savefig(OUT / ("loop_all_turns_detectors.png" if detectors else "loop_all_turns_seeds.png" if seeds else "loop_round2.png" if max_turn == 2 else "loop_all_turns.png"), dpi=170)
    plt.close(fig)


def fig_round2_frames():
    """Turn 1 against turn 2 on the blind frames, frame by frame, summed over the videos: frames wrong in both turns,
    wrong -> right and right -> wrong, split by how the frame was drawn."""
    C = pd.read_csv(R2 / "blind_changes.csv")
    grp = {"A": "A: flagged by the starting model only", "B": "B: flagged by both models", "C": "C, D: other strata", "D": "C, D: other strata"}
    C["group"] = [grp.get(s, "supplement: frames two models disagree on") for s in C.stratum]
    gcol = {"A: flagged by the starting model only": "#56b4e9", "B: flagged by both models": "#444444", "C, D: other strata": "#b9b9b9",
            "supplement: frames two models disagree on": "#e69f00"}
    kinds = [("wrong_both", "wrong\nin both"), ("wrong_to_right", "wrong\n\u2192 right"), ("right_to_wrong", "right\n\u2192 wrong")]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    w = 0.26
    ticks, names = [], []
    for i, (src, lab, _, _) in enumerate(R2EXP):
        E = C[C.experiment == src]
        if not len(E):
            continue
        nv = E.video.nunique()
        for k, (col, klab) in enumerate(kinds):
            x = i + (k - 1) * w
            bottom = 0
            for gname, c in gcol.items():
                y = int(E[E.group == gname][col].sum())
                ax.bar([x], [y], w * 0.9, bottom=bottom, color=c, edgecolor="white", linewidth=0.6, label=gname if (i == 0 and k == 0) else None)
                bottom += y
            ax.text(x, bottom + 0.25, f"{bottom}", ha="center", va="bottom", fontsize=8.5, color="#222")
            ax.text(x, -0.4, klab, ha="center", va="top", fontsize=7.5, color="#444")
        ticks.append(i)
        names.append(lab.split(" (")[0] + f"\n{nv} video{'s' if nv > 1 else ''}, {int(E.frames.sum())} frames")
    ax.set_xticks(ticks)
    ax.set_xticklabels(names)
    ax.tick_params(axis="x", pad=30, length=0)
    ax.set_ylabel("blind frames")
    ax.set_ylim(0, 14)
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "loop_round2_frames.png", dpi=170)
    plt.close(fig)


def fig_exp4c_turns():
    """Experiment 4c with its second turn: (a) flagged share per video for the starting model, turn 1 and turn 2, with
    the wrong truth and blind frames; (b) the turn-1 reopened frames left out of the training set: how many each model flags."""
    ST_ = RES / "self_training"
    vids = [v for v in range(19) if (ST_ / f"video{v:02d}" / "round1_reopened" / "eval.csv").exists()]
    series = [("teacher", COL["teacher"], "starting model"), ("round1_reopened", "#009e73", "turn 1: human labels + reopened frames"),
              ("round2_reopened", "#7fd1b9", "turn 2: human labels + the turn's own reopened frames")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    w = 0.27
    for i, v in enumerate(vids):
        F = pd.read_csv(ST_ / f"video{v:02d}" / "pseudo_reopened" / "pseudo_frames.csv")
        held = F.frame[~F.kept].to_numpy()
        for j, (m, c, lab) in enumerate(series):
            ef = ST_ / f"video{v:02d}" / m / "eval.csv"
            x = i + (j - 1) * w
            if not ef.exists():
                if m == "round2_reopened":
                    for ax in axes:
                        ax.text(x, 0.3, "not run", ha="center", va="bottom", fontsize=7, color="#888", rotation=90)
                continue
            e = pd.read_csv(ef).iloc[0]
            axes[0].bar([x], [e.flagged_either_pct], w * 0.92, color=c, label=lab if i == 0 else None, edgecolor="white", linewidth=0.5)
            bl = ""
            bs = ST_ / f"video{v:02d}" / ("round1_reopened" if m == "teacher" else m) / "blind_scores.csv"
            if bs.exists():
                B_ = pd.read_csv(bs)
                bl = f"\nblind {int(B_[B_.model == m].wrong_frames.sum())}/{int(B_[B_.model == m].frames.sum())}"
            axes[0].text(x, e.flagged_either_pct, f"{e.flagged_either_pct:.1f} %\ntruth {int(e.wrong_frames)}/70{bl}", ha="center", va="bottom", fontsize=7, color="#222")
            if m == "teacher":
                y = len(held)
            else:
                ff = ST_ / f"video{v:02d}" / m / "flags.csv.gz"
                if not ff.exists():
                    continue
                Sf = pd.read_csv(ff).set_index("frame").loc[held]
                y = int((Sf.fit_flag | Sf.pca_flag).sum())
            axes[1].bar([x], [y], w * 0.92, color=c, label=lab if i == 0 else None, edgecolor="white", linewidth=0.5)
            axes[1].text(x, y, f"{y} of {len(held)}", ha="center", va="bottom", fontsize=7.5, color="#222")
    for ax in axes:
        ax.set_xticks(range(len(vids)))
        ax.set_xticklabels([str(v) for v in vids])
        ax.set_xlabel("video")
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.4)
    axes[0].set_ylabel("frames flagged, % of the video")
    axes[0].set_title("(a) flagged share; labels: wrong truth frames, wrong blind frames", loc="left")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].set_ylabel("frames flagged")
    axes[1].set_title("(b) turn-1 reopened frames left out of the training set:\nhow many are still flagged", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "loop_exp4c_turns.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__" and "--round2" in sys.argv:
    fig_round2()
    fig_round2_frames()
    fig_exp4c_turns()
    fig_experiment("round1_every30", second_label="experiment 1, second turn")
    fig_experiment("round1_both")
    print("round-2 figures written")


# ---- experiments 1 and 2 with every turn run (page Figures 9-15, 9-16; 2026-11-13) ----
TURN_COL = {"every30": ["#e69f00", "#f0c35c", "#f7e0a8"], "repaired": ["#d55e00", "#e89a5c", "#f5cfae"], "both": ["#0072b2", "#5aa5d6", "#b3d6ee"]}
TURN_OUT = {"every30": ("loop_exp1_turns.png", "experiment 1"), "repaired": ("loop_exp2_turns.png", "experiment 2"), "both": ("loop_exp3_turns.png", "experiment 3")}
TURN_BLIND = {"both": 34}   # experiment 3 is scored on all 34 blind frames (its training frames avoid them); experiments 1 and 2 on the 24 stratified ones, as in their tables


def fig_experiment_turns(src, seeds=False):
    """One experiment, every video and every turn it ran: wrong truth frames, median truth error, flagged share (each
    model's eval.csv) and wrong blind frames of the 24 stratified frames (round2_analysis/summary.csv); the starting
    model in grey, later turns in lighter tints of the experiment's colour."""
    ST_ = RES / "self_training"
    name = lambda r: "teacher" if r == 0 else f"round{r}_{src}"
    vids = [v for v in range(19) if (ST_ / f"video{v:02d}" / name(1) / "eval.csv").exists()]
    fname, label = TURN_OUT[src]
    series = [(0, COL["teacher"], "starting model")] + [(r, TURN_COL[src][r - 1], f"{label}, turn {r}") for r in (1, 2, 3)]
    S = pd.read_csv(R2 / "summary.csv")
    E = {(v, r): pd.read_csv(ST_ / f"video{v:02d}" / name(r) / "eval.csv").iloc[0] for v in vids for r in range(4)
         if (ST_ / f"video{v:02d}" / name(r) / "eval.csv").exists()}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.2))
    axes = axes.ravel()
    cols = [("wrong_frames", "wrong truth frames (of 70)", "{:.0f}"), ("median_error_pct", "median truth error, % of eye width", "{:.1f}"),
            ("flagged_either_pct", "frames flagged, % of the video", "{:.1f}")]
    w = 0.8 / 4

    def draw(ax, xs_vids, value, fmt, fs):
        for j, (r, c, lab) in enumerate(series):
            pts = [(i, value(v, r)) for i, v in enumerate(xs_vids)]
            pts = [(i, y) for i, y in pts if y is not None and np.isfinite(y)]
            xs = [i + (j - 1.5) * w for i, _ in pts]
            ys = [y for _, y in pts]
            ax.bar(xs, ys, w * 0.9, color=c, label=lab, edgecolor=TURN_COL[src][0] if r > 1 else "white", linewidth=0.5)
            for x, y in zip(xs, ys):
                ax.text(x, y, fmt.format(y), ha="center", va="bottom", fontsize=fs, color="#222",
                        rotation=90 if (len(xs_vids) > 6 and "." in fmt.format(y)) else 0)
        ax.set_xticks(range(len(xs_vids)))
        ax.set_xticklabels([str(v) for v in xs_vids])
        ax.set_xlabel("video")
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
        ax.set_axisbelow(True)

    fs = 6.5 if len(vids) > 6 else 8
    for ax, (col, ylabel, fmt) in zip(axes[:3], cols):
        draw(ax, vids, lambda v, r: float(E[(v, r)][col]) if (v, r) in E else None, fmt, fs)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    if seeds:   # turn-1 replicates with seed 43 (videos without a second turn: the bar takes the empty turn-2 slot, hatched)
        for ax, (col, ylabel, fmt) in zip(axes[:3], cols):
            first = True
            for i, v in enumerate(vids):
                f = ST_ / f"video{v:02d}" / f"round1_{src}_seed43" / "eval.csv"
                if not f.exists() or (v, 2) in E:
                    continue
                y = float(pd.read_csv(f).iloc[0][col])
                x = i + (2 - 1.5) * w
                ax.bar([x], [y], w * 0.9, color=TURN_COL[src][0], hatch="////", edgecolor="white", linewidth=0.5, label=f"{label}, turn 1 with another seed" if first else None)
                ax.text(x, y, fmt.format(y), ha="center", va="bottom", fontsize=fs, color="#222", rotation=90 if (len(vids) > 6 and "." in fmt.format(y)) else 0)
                first = False
    bv = [v for v in vids if len(S[(S.video == v) & (S.experiment == "teacher")])]

    def blind(v, r):
        g = S[(S.video == v) & (S.experiment == ("teacher" if r == 0 else src)) & (S.turn == r)]
        return float(g[f"blind_wrong_{nb}"].iloc[0]) if len(g) else None
    nb = TURN_BLIND.get(src, 24)
    draw(axes[3], bv, blind, "{:.0f}", 8)
    axes[3].set_ylabel(f"wrong blind frames (of {nb})")
    axes[3].set_ylim(0, 20 if nb == 34 else 14)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5 if seeds else 4, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=9 if seeds else None)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / (fname.replace(".png", "_seeds.png") if seeds else fname), dpi=170)
    plt.close(fig)


if __name__ == "__main__" and "--turns" in sys.argv:
    fig_experiment_turns("every30")
    fig_experiment_turns("repaired")
    fig_round2(max_turn=3)
    fig_experiment_turns("every30", seeds=True)
    fig_experiment_turns("both")
    fig_round2(max_turn=3, seeds=True)
    fig_round2(max_turn=3, seeds=True, detectors=True)
    print("turn figures written")


# ---- experiment 4b: two judged frames before and after the repair (page Figure 9-23, 2026-11-14) ----
def fig_exp4b_examples(items=(("video15_f050265_r", "video 15, frame 50265", "left pupil point replaced"),
                              ("video01_f059597_r", "video 1, frame 59597", "temporal corner replaced"))):
    """Two of the 20 frames judged in experiment 4b (both judged right), cut from the review items the user saw
    (flag_and_repair/subset_review/items/*_r.jpg: frame | model's points | opening rule | confidence rule): the model's
    points before the repair and the points after it, side by side."""
    src = FAR / "subset_review" / "items"
    crops = []
    for name, where, what in items:
        im = cv2.cvtColor(cv2.imread(str(src / f"{name}.jpg")), cv2.COLOR_BGR2RGB)
        pw = (im.shape[1] - 30) // 4                         # four panels, 10 px apart
        x0 = [i * (pw + 10) for i in range(4)]
        plain = im[24:, x0[0]:x0[0] + pw].mean(axis=2)       # the bare frame: its non-black part is the picture
        rows, cols = np.where(plain.mean(axis=1) > 20)[0], np.where(plain.mean(axis=0) > 20)[0]
        r0, r1, c0, c1 = rows.min() + 24, rows.max() + 24, cols.min(), cols.max()
        crops.append([(im[r0:r1, x0[1] + c0:x0[1] + c1], f"{where}: before the repair"), (im[r0:r1, x0[3] + c0:x0[3] + c1], f"after: {what}")])
    hh = min(c[0][0].shape[0] for c in crops)                # same height for both frames (cut at the bottom), widths in proportion
    ws = [c[0][0].shape[1] for c in crops]
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 13.2 * hh / (2 * sum(ws) + 0.04 * sum(ws)) + 0.42),
                             gridspec_kw={"wspace": 0.03, "width_ratios": [ws[0], ws[0], 0.04 * sum(ws), ws[1], ws[1]]})
    axes[2].axis("off")
    for k, pair in enumerate(crops):
        for j, (img, title) in enumerate(pair):
            ax = axes[3 * k + j]
            ax.imshow(img[:hh], aspect="auto")
            ax.set_title(title, fontsize=9.5, loc="left")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.01)
    fig.savefig(OUT / "loop_exp4b_examples.jpg", dpi=170, pil_kwargs={"quality": 90})
    plt.close(fig)


if __name__ == "__main__" and "--exp4b" in sys.argv:
    fig_exp4b_examples()
    print("4b example figure written")


# ---- per-experiment figures with the two independent detectors as well (page Figures 9-15 to 9-17, 9-24; 2026-11-14) ----
def fig_experiment_full(src, seeds=False):
    """One experiment, every video and every turn it ran, six panels: wrong truth frames, median truth error, wrong blind
    frames (top); flagged share, low-confidence detector, jump detector (bottom; the two detectors from
    round2_analysis/independent_detectors.csv, the starting model is 5 % by construction). Same bars and colours as
    fig_experiment_turns; seeds = True adds the turn-1 replicate with another seed as a hatched bar (videos without a
    second turn)."""
    ST_ = RES / "self_training"
    name = lambda r: "teacher" if r == 0 else f"round{r}_{src}"
    vids = [v for v in range(19) if (ST_ / f"video{v:02d}" / name(1) / "eval.csv").exists()]
    fname, label = TURN_OUT[src]
    nb = TURN_BLIND.get(src, 24)
    turns = (1, 2, 3) + tuple(r for r in (4,) if any((ST_ / f"video{v:02d}" / name(r) / "eval.csv").exists() for v in vids))   # a fourth turn exists for experiment 2 on video 6 only (2026-11-15)
    cols = TURN_COL[src] + ["#fcefe3"]
    series = [(0, COL["teacher"], "starting model")] + [(r, cols[r - 1], f"{label}, turn {r}") for r in turns]
    S = pd.read_csv(R2 / "summary.csv")
    Dd = pd.read_csv(R2 / "independent_detectors.csv")
    E = {(v, r): pd.read_csv(ST_ / f"video{v:02d}" / name(r) / "eval.csv").iloc[0] for v in vids for r in range(5)
         if (ST_ / f"video{v:02d}" / name(r) / "eval.csv").exists()}
    bv = [v for v in vids if len(S[(S.video == v) & (S.experiment == "teacher")])]

    def ev(col):
        return lambda v, r, m=None: (float(pd.read_csv(ST_ / f"video{v:02d}" / m / "eval.csv").iloc[0][col]) if m else float(E[(v, r)][col]) if (v, r) in E else None)

    def det(col):
        def f(v, r, m=None):
            g = Dd[(Dd.video == v) & (Dd.model == (m or name(r)))]
            return float(g[col].iloc[0]) if len(g) else None
        return f

    def blind(v, r, m=None):
        if m:
            return None
        g = S[(S.video == v) & (S.experiment == ("teacher" if r == 0 else src)) & (S.turn == r)]
        return float(g[f"blind_wrong_{nb}"].iloc[0]) if len(g) else None

    panels = [(ev("wrong_frames"), "wrong truth frames (of 70)", "{:.0f}", vids), (ev("median_error_pct"), "median truth error, % of eye width", "{:.1f}", vids),
              (blind, f"wrong blind frames (of {nb})", "{:.0f}", bv), (ev("flagged_either_pct"), "flagged (fitting + PCA), % of the video", "{:.1f}", vids),
              (det("indep_lowconf_pct"), "flagged (low confidence), % of the video", "{:.1f}", vids), (det("indep_jump_pct"), "flagged (jump), % of the video", "{:.1f}", vids)]
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 7.6))
    w = 0.8 / (len(turns) + 1)
    for ax, (value, ylabel, fmt, xv) in zip(axes.ravel(), panels):
        many = len(xv) > 6
        fs = 6.5 if many else 8
        rot = lambda t: 90 if (many and "." in t) else 0
        for j, (r, c, lab) in enumerate(series):
            pts = [(i, value(v, r)) for i, v in enumerate(xv)]
            pts = [(i, y) for i, y in pts if y is not None and np.isfinite(y)]
            xs = [i + (j - len(turns) / 2) * w for i, _ in pts]
            ax.bar(xs, [y for _, y in pts], w * 0.9, color=c, label=lab, edgecolor=TURN_COL[src][0] if r > 1 else "white", linewidth=0.5)
            for x, (_, y) in zip(xs, pts):
                ax.text(x, y, fmt.format(y), ha="center", va="bottom", fontsize=fs, color="#222", rotation=rot(fmt.format(y)))
        if seeds:
            first = True
            for i, v in enumerate(xv):
                m = f"round1_{src}_seed43"
                if not (ST_ / f"video{v:02d}" / m / "eval.csv").exists() or (v, 2) in E:
                    continue
                y = value(v, 1, m)
                if y is None:
                    continue
                x = i + (2 - len(turns) / 2) * w
                ax.bar([x], [y], w * 0.9, color=TURN_COL[src][0], hatch="////", edgecolor="white", linewidth=0.5, label=f"{label}, turn 1 with another seed" if first else None)
                ax.text(x, y, fmt.format(y), ha="center", va="bottom", fontsize=fs, color="#222", rotation=rot(fmt.format(y)))
                first = False
        ax.set_xticks(range(len(xv)))
        ax.set_xticklabels([str(v) for v in xv])
        ax.set_xlabel("video")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(0, (20 if nb == 34 else 14) if value is blind else ax.get_ylim()[1] * 1.18)
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / fname.replace("_turns.png", "_full.png"), dpi=170)
    plt.close(fig)


def fig_exp4c_full():
    """Experiment 4c with its second turn, the same measures as the figures of experiments 1 to 3 plus the reopened
    frames left out of training. Top: wrong truth frames, median truth error, wrong blind frames (of 34; video 11 has no
    blind set), turn-1 reopened frames left out of the training set that are still flagged. Bottom: flagged share by
    the two finders, by the low-confidence detector, by the jump detector."""
    ST_ = RES / "self_training"
    Dd = pd.read_csv(R2 / "independent_detectors.csv")
    vids = [v for v in range(19) if (ST_ / f"video{v:02d}" / "round1_reopened" / "eval.csv").exists()]
    series = [("teacher", COL["teacher"], "starting model"), ("round1_reopened", "#009e73", "experiment 4c, turn 1"), ("round2_reopened", "#7fd1b9", "experiment 4c, turn 2")]
    fig, axes = plt.subplots(2, 4, figsize=(17.5, 7.6))
    ax_truth, ax_med, ax_blind, ax_held, ax_flag, ax_low, ax_jump, ax_off = axes.ravel()
    ax_off.axis("off")
    w = 0.27

    def put(ax, x, y, txt, **kw):
        ax.bar([x], [y], w * 0.92, **kw)
        ax.text(x, y, txt, ha="center", va="bottom", fontsize=8, color="#222")

    for i, v in enumerate(vids):
        F = pd.read_csv(ST_ / f"video{v:02d}" / "pseudo_reopened" / "pseudo_frames.csv")
        held = F.frame[~F.kept].to_numpy()
        bsf = ST_ / f"video{v:02d}" / "round1_reopened" / "blind_scores.csv"
        if not bsf.exists():
            ax_blind.text(i, 0.2, "no blind set", ha="center", va="bottom", fontsize=8, color="#888")
        for j, (m, c, lab) in enumerate(series):
            x = i + (j - 1) * w
            ef = ST_ / f"video{v:02d}" / m / "eval.csv"
            if not ef.exists():
                for ax in (ax_truth, ax_med, ax_held, ax_flag, ax_low, ax_jump):
                    ax.text(x, 0, "not run", ha="center", va="bottom", fontsize=7, color="#888", rotation=90)
                continue
            e = pd.read_csv(ef).iloc[0]
            kw = dict(color=c, edgecolor="#009e73" if m == "round2_reopened" else "white", linewidth=0.5)
            ax_truth.bar([x], [e.wrong_frames], w * 0.92, label=lab if i == 0 else None, **kw)
            ax_truth.text(x, e.wrong_frames, f"{int(e.wrong_frames)}", ha="center", va="bottom", fontsize=8, color="#222")
            put(ax_med, x, e.median_error_pct, f"{e.median_error_pct:.1f}", **kw)
            bs = ST_ / f"video{v:02d}" / ("round1_reopened" if m == "teacher" else m) / "blind_scores.csv"
            if bs.exists():
                B_ = pd.read_csv(bs)
                nbw = int(B_[B_.model == m].wrong_frames.sum())
                put(ax_blind, x, nbw, f"{nbw}", **kw)
            if m == "teacher":
                y = len(held)
            else:
                Sf = pd.read_csv(ST_ / f"video{v:02d}" / m / "flags.csv.gz").set_index("frame").loc[held]
                y = int((Sf.fit_flag | Sf.pca_flag).sum())
            put(ax_held, x, y, f"{y}/{len(held)}", **kw)
            put(ax_flag, x, e.flagged_either_pct, f"{e.flagged_either_pct:.1f}", **kw)
            g = Dd[(Dd.video == v) & (Dd.model == m)].iloc[0]
            put(ax_low, x, g.indep_lowconf_pct, f"{g.indep_lowconf_pct:.1f}", **kw)
            put(ax_jump, x, g.indep_jump_pct, f"{g.indep_jump_pct:.1f}", **kw)
    labels = [(ax_truth, "wrong truth frames (of 70)"), (ax_med, "median truth error, % of eye width"), (ax_blind, "wrong blind frames (of 34)"),
              (ax_held, "reopened frames left out of training:\nstill flagged (fitting + PCA)"), (ax_flag, "flagged (fitting + PCA), % of the video"),
              (ax_low, "flagged (low confidence), % of the video"), (ax_jump, "flagged (jump), % of the video")]
    for ax, yl in labels:
        ax.set_xticks(range(len(vids)))
        ax.set_xticklabels([str(v) for v in vids])
        ax.set_xlabel("video")
        ax.set_ylabel(yl)
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    h, l = ax_truth.get_legend_handles_labels()
    ax_off.legend(h, l, loc="center", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "loop_exp4c_full.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__" and "--full" in sys.argv:
    fig_experiment_full("every30", seeds=True)
    fig_experiment_full("repaired")
    fig_experiment_full("both")
    fig_exp4c_full()
    print("full per-experiment figures written")


# ---- experiment 5b with every turn, and 5c: the blind check of turn 2 (page 9.4, 2026-11-14) ----
PINK = ["#cc79a7", "#5b4a9e", "#a8672a"]   # experiment 5b: turn 1 pink (its colour in every figure), turn 2 indigo, turn 3 brown (user 2026-11-14: later turns need not be pink)


def fig_exp5b_full():
    """The final loop (5b), every video and turn, six panels as in fig_experiment_full: wrong truth frames, median truth
    error, wrong blind frames of 34 (top); flagged share by the two finders, by the low-confidence detector, by the
    jump detector (bottom). Bars: starting model, turn 1, turn 1 trained again with another seed (hatched), turn 2, turn 3."""
    ST_ = RES / "self_training"
    S = pd.read_csv(R2 / "summary.csv")
    Dd = pd.read_csv(R2 / "independent_detectors.csv")
    vids = [v for v in range(19) if (ST_ / f"video{v:02d}" / "round1_formal" / "eval.csv").exists()]
    slots = [("teacher", COL["teacher"], None, "starting model"), ("round1_formal", PINK[0], None, "experiment 5b, turn 1"),
             ("round1_formal_seed43", PINK[0], "////", "experiment 5b, turn 1 with another seed"),
             ("round2_formal", PINK[1], None, "experiment 5b, turn 2"), ("round3_formal", PINK[2], None, "experiment 5b, turn 3")]

    def ev(col):
        def f(v, m):
            p = ST_ / f"video{v:02d}" / m / "eval.csv"
            return float(pd.read_csv(p).iloc[0][col]) if p.exists() else None
        return f

    def det(col):
        def f(v, m):
            g = Dd[(Dd.video == v) & (Dd.model == m)]
            return float(g[col].iloc[0]) if len(g) else None
        return f

    def blind(v, m):
        g = S[(S.video == v) & (S.model == m)]
        if len(g):
            return float(g.blind_wrong_34.iloc[0])
        p = ST_ / f"video{v:02d}" / m / "blind_scores.csv"
        if p.exists():
            Bs = pd.read_csv(p)
            return float(Bs[Bs.model == m].wrong_frames.sum())
        return None

    panels = [(ev("wrong_frames"), "wrong truth frames (of 70)", "{:.0f}"), (ev("median_error_pct"), "median truth error, % of eye width", "{:.1f}"),
              (blind, "wrong blind frames (of 34)", "{:.0f}"), (ev("flagged_either_pct"), "flagged (fitting + PCA), % of the video", "{:.1f}"),
              (det("indep_lowconf_pct"), "flagged (low confidence), % of the video", "{:.1f}"), (det("indep_jump_pct"), "flagged (jump), % of the video", "{:.1f}")]
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 7.6))
    w = 0.8 / len(slots)
    for ax, (value, ylabel, fmt) in zip(axes.ravel(), panels):
        for j, (m, c, hatch, lab) in enumerate(slots):
            pts = [(i, value(v, m)) for i, v in enumerate(vids)]
            pts = [(i, y) for i, y in pts if y is not None and np.isfinite(y)]
            xs = [i + (j - (len(slots) - 1) / 2) * w for i, _ in pts]
            ax.bar(xs, [y for _, y in pts], w * 0.9, color=c, hatch=hatch, label=lab, edgecolor="white", linewidth=0.5)
            for x, (_, y) in zip(xs, pts):
                ax.text(x, y, fmt.format(y), ha="center", va="bottom", fontsize=7.5, color="#222")
        ax.set_xticks(range(len(vids)))
        ax.set_xticklabels([str(v) for v in vids])
        ax.set_xlabel("video")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 20 if value is blind else ax.get_ylim()[1] * 1.18)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "loop_exp5b_full.png", dpi=170)
    plt.close(fig)


def fig_exp5c(suffix=""):
    """Experiment 5c: the frames labeled blind for turn 2 against turn 1 of the final loop (results/blind_test/videoNN/
    turn2/per_frame.csv; suffix "_before_review" = the labels before the user's review). (a) wrong frames per video and
    in total; (b) by stratum, the four videos summed; (c) turn 1 against turn 2 frame by frame."""
    E = pd.concat([pd.read_csv(RES / "blind_test" / f"video{v:02d}" / "turn2" / f"per_frame{suffix}.csv").assign(video=v) for v in VIDEOS], ignore_index=True)
    E["wrong"] = E.error > LINE / 100
    slots = [("teacher", COL["teacher"], None, "starting model"), ("round1_formal", PINK[0], None, "experiment 5b, turn 1"),
             ("round1_formal_seed43", PINK[0], "////", "turn 1 with another seed"), ("round2_formal", PINK[1], None, "experiment 5b, turn 2"),
             ("round3_formal", PINK[2], None, "experiment 5b, turn 3")]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.9), gridspec_kw={"width_ratios": [2.5, 1.9, 1.0]})
    # (a) per video and total
    ax = axes[0]
    groups = [(f"video {v}\n{int((E[(E.video == v) & (E.model == 'teacher')]).shape[0])} frames", E[E.video == v]) for v in VIDEOS] + \
             [(f"all four\n{int((E.model == 'teacher').sum())} frames", E)]
    w = 0.8 / len(slots)
    for j, (m, c, hatch, lab) in enumerate(slots):
        first = True
        for i, (name, g) in enumerate(groups):
            gm = g[g.model == m]
            if not len(gm) or gm.video.nunique() < g.video.nunique():     # a model missing on some video is not summed into the total
                continue
            y = int(gm.wrong.sum())
            x = i + (j - (len(slots) - 1) / 2) * w
            ax.bar([x], [y], w * 0.9, color=c, hatch=hatch, edgecolor="white", linewidth=0.5, label=lab if first else None)
            ax.text(x, y, str(y), ha="center", va="bottom", fontsize=8.5, color="#222")
            first = False
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([n for n, _ in groups])
    ax.set_ylabel("wrong frames (error over 14 % of the eye width)")
    ax.set_title("(a) per video and in total", loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    # (b) by stratum
    ax = axes[1]
    names = {"T1": "turn 1 only", "TB": "both turns", "T2": "turn 2 only", "TN": "neither"}
    w = 0.8 / 3
    for j, (m, c, hatch, lab) in enumerate([slots[0], slots[1], slots[3]]):
        for i, s in enumerate(names):
            g = E[(E.stratum == s) & (E.model == m)]
            y = int(g.wrong.sum())
            x = i + (j - 1) * w
            ax.bar([x], [y], w * 0.9, color=c, edgecolor="white", linewidth=0.5)
            ax.text(x, y, str(y), ha="center", va="bottom", fontsize=8.5, color="#222")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{names[s]}\n{int(((E.stratum == s) & (E.model == 'teacher')).sum())} frames" for s in names])
    ax.set_xlabel("frame flagged (fitting + PCA) by the model of")
    ax.set_title("(b) by how the frame was drawn, four videos summed", loc="left", fontsize=10.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    # (c) frame by frame
    ax = axes[2]
    a = E[E.model == "round1_formal"].set_index(["video", "frame"]).wrong
    b = E[E.model == "round2_formal"].set_index(["video", "frame"]).wrong.reindex(a.index)
    vals = [("wrong\n→ right", int((a & ~b).sum())), ("right\n→ wrong", int((~a & b).sum())), ("wrong\nin both", int((a & b).sum()))]
    ax.bar(range(3), [y for _, y in vals], 0.6, color="#8a8a8a", edgecolor="white")
    for i, (_, y) in enumerate(vals):
        ax.text(i, y, str(y), ha="center", va="bottom", fontsize=9, color="#222")
    ax.set_xticks(range(3))
    ax.set_xticklabels([n for n, _ in vals])
    ax.set_title("(c) turn 1 against turn 2,\nframe by frame", loc="left", fontsize=10.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    for ax in axes:
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / f"loop_exp5c{suffix}.png", dpi=170)
    plt.close(fig)
    return {"per video (teacher, turn 1, turn 2)": {int(v): [int(E[(E.video == v) & (E.model == m)].wrong.sum()) for m in ("teacher", "round1_formal", "round2_formal")] for v in VIDEOS},
            "by stratum": {s: [int(E[(E.stratum == s) & (E.model == m)].wrong.sum()) for m in ("teacher", "round1_formal", "round2_formal")] for s in names}, "frame by frame": dict(vals)}


if __name__ == "__main__" and "--exp5" in sys.argv:
    fig_exp5b_full()
    print("5c:", fig_exp5c())
    print("5c before review:", fig_exp5c("_before_review"))


def fig_exp5c_videos(suffix=""):
    """Experiment 5c in one panel (user 2026-11-14: per video, the bars of how many frames are wrong, nothing else):
    wrong frames among each video's hand-labeled frames for the starting model, turn 1 and turn 2 of the final loop."""
    E = pd.concat([pd.read_csv(RES / "blind_test" / f"video{v:02d}" / "turn2" / f"per_frame{suffix}.csv").assign(video=v) for v in VIDEOS], ignore_index=True)
    E["wrong"] = E.error > LINE / 100
    slots = [("teacher", COL["teacher"], "starting model"), ("round1_formal", PINK[0], "experiment 5b, turn 1"), ("round2_formal", PINK[1], "experiment 5b, turn 2")]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    w = 0.8 / 3
    for j, (m, c, lab) in enumerate(slots):
        ys = [int(E[(E.video == v) & (E.model == m)].wrong.sum()) for v in VIDEOS]
        xs = [i + (j - 1) * w for i in range(len(VIDEOS))]
        ax.bar(xs, ys, w * 0.9, color=c, edgecolor="white", linewidth=0.5, label=lab)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.2, str(y), ha="center", va="bottom", fontsize=14, color="#222")
    ax.set_xticks(range(len(VIDEOS)))
    ax.set_xticklabels([f"video {v}\n{int(((E.video == v) & (E.model == 'teacher')).sum())} frames" for v in VIDEOS])
    ax.set_ylabel("wrong frames", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.set_ylim(0, 28)
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=13, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / f"loop_exp5c_videos{suffix}.png", dpi=170)
    plt.close(fig)
