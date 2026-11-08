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
       "round2_every30": "#cc79a7", "round1_both": "#56b4e9", "round2_both": "#1f78b4", "round3_both": "#0b3d91"}
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
         "frames flagged by either finder, % of the video", lambda y: f"{y:.1f}", VIDEOS)
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
    bars(axes[2], C, "flagged_either_pct", models, "frames flagged by either finder, % of the video", lambda y: f"{y:.1f}", ALL)
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
    series = [("flagged_either_pct", "frames flagged by either finder, % of the video", "o", "#0072b2", (0, 5), "center"),
              ("blind_wrong_pct", "wrong blind frames, % of 34", "s", "#d55e00", (0, 5), "center"),
              ("truth_wrong_pct", "wrong truth frames, % of 70", "^", "#009e73", (7, -3), "left")]
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
            ax.plot(T.turn, T[col], marker=marker, color=colour, linewidth=1.6, markersize=5.5, label=label)
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


def fig_experiment(model):
    label, colour, nblind = EXPS[model]
    C = pd.read_csv(RES / "self_training" / "comparison.csv")
    vids = [int(v) for v in sorted(C[C.model == model].video.unique())]
    series = [("teacher", COL["teacher"], "starting model"), (model, colour, label)]
    if model == "round1_every30" and "round2_every30" in set(C.model):
        series.append(("round2_every30", COL["round2_every30"], "experiment 1, second turn (videos 1, 8)"))
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
            op, cf = _closed_counts(v, m)
            axes[1].bar([x], [cf], w * 0.92, color=c, label=lab if i == 0 else None, edgecolor="white", linewidth=0.5)
            axes[1].text(x, cf, f"{cf:,}", ha="center", va="bottom", fontsize=8, color="#222")
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
    axes[1].set_ylabel("frames the confidence rule calls closed")
    axes[1].set_title("(b) frames called closed on the whole-video prediction", loc="left")
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
