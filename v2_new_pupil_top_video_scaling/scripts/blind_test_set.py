#!/usr/bin/env python3
"""Blind, stratified test set for judging the self-trained students against their teachers (2026-11-03).

The 70 truth frames of a video are the plateau model's validation and test frames; the self-training students were
judged on them and on the share of the video the two finders flag. The finders are self-consistency measures, so a
lower flagged share does not prove that the unflagged frames are right. This script draws new frames, stratified by
what the teacher and the student say about them, has them labeled blind, and scores every model on them with
continuous paired errors.

    python blind_test_set.py select --video 1          -> results/blind_test/video01/selection_key.csv (do not open before labeling)
                                                          results/blind_test/video01/labeling/blind/img001..050.png
    python blind_test_set.py prelabel --video 1 --points P.csv
                                                       -> machinelabels.h5/.csv in the labeling folder from a csv
                                                          (blind_name, bodypart, x, y) written by the pre-labeler
    bash  label_folder.sh results/blind_test/video01/labeling/blind         (napari; the human corrects, saves CollectedData_Zhiheng)
    python blind_test_set.py score --video 1           -> results/blind_test/video01/{scores.csv, per_frame.csv, paired.csv, paired.png}
    python blind_test_set.py check --video 1           -> the scorer applied to the 70 truth frames must give the
                                                          numbers of results/self_training/video01/teacher/truth_errors.csv

Strata (per-frame finder state, fitting or PCA):
    A  teacher flagged, student (round 1, every 30) not      the frames the whole question is about
    B  both flagged
    C  neither flagged
    D  student flagged, teacher not
Excluded from the draw: the 70 truth frames and 60 frames on each side; every frame used as a human label (batch
folders) and 5 frames on each side; every pseudo-label frame of both students and 5 frames on each side; frames where
the teacher has a missing coordinate. Inside a stratum the frames are spread over the pose space (k-means on the
teacher's eight points divided by the eye width, one frame per cluster), so pupil sizes and eye states are covered
rather than moments in time. Weights = stratum share of the eligible frames / frames drawn in the stratum, so the
weighted overall numbers refer to the whole video.

Error per frame = RMSE over the labeled points / the teacher's median eye width (frame_rmse of second_peak_repair.py,
the same as self_training.py evaluate); a frame is wrong over the error-tier line.
"""
import argparse
import glob
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pupil_trace as pt  # noqa: E402
import unreliable_frames_sweep as U  # noqa: E402
from scale_step import SCORER  # noqa: E402
from second_peak_repair import BPS, RES, frame_rmse, load  # noqa: E402
from self_training import FINAL, LINE, LOCAL, mkdir, teacher_eye_width, teacher_h5, unit_of  # noqa: E402

HERE = RES.parent
OUT = RES / "blind_test"
STRATA = {"A": "teacher flagged, student not", "B": "both flagged", "C": "neither flagged", "D": "student flagged, teacher not"}
GUARD_TRUTH, GUARD_LABEL, GUARD_PSEUDO = 60, 5, 5
STUDENT = "round1_every30"
PUPIL = BPS[:4]


def folder(v):
    return OUT / f"video{v:02d}"


def label_dir(v):
    return folder(v) / "labeling" / "blind"


def dummy_name(n):
    return f"img{n + 1:03d}.png"


# ---------------------------------------------------------------------------------------------------------- selection
def flags_of(v):
    """(teacher flag, student flag) per frame, both 'either finder'."""
    T = pd.read_csv(RES / "flag_and_repair" / FINAL.format(v=v) / "frames_all.csv.gz")
    S = pd.read_csv(RES / "self_training" / f"video{v:02d}" / STUDENT / "flags.csv.gz")
    assert len(T) == len(S) and (T.frame.to_numpy() == S.frame.to_numpy()).all()
    return T.flag.to_numpy(bool), (S.fit_flag | S.pca_flag).to_numpy(bool)


def band(n, frames, guard):
    m = np.zeros(n, bool)
    for f in frames:
        m[max(0, f - guard): f + guard + 1] = True
    return m


def label_frames(v):
    unit = unit_of(v)
    fr = []
    for p in sorted((HERE / unit / "training-data" / "labels").glob("batch*/img*.png")):
        fr.append(int(p.stem[3:]))
    return sorted(set(fr))


def pseudo_frames(v):
    fr = []
    for f in (RES / "self_training" / f"video{v:02d}").glob("pseudo_every*/pseudo_frames.csv"):
        K = pd.read_csv(f)
        fr += K[K.kept].frame.astype(int).tolist()
    return sorted(set(fr))


def select(v, n_total, seed):
    rr, dt, P, H, truth, fwd, sig = load(v)
    n = len(dt)
    Wm = teacher_eye_width(dt)
    XY = U.pose_matrix(dt)                                   # (n, 8, 2)
    tflag, sflag = flags_of(v)
    excl = band(n, truth.index.astype(int), GUARD_TRUTH) | band(n, label_frames(v), GUARD_LABEL) | band(n, pseudo_frames(v), GUARD_PSEUDO)
    excl |= ~np.isfinite(XY).all(axis=(1, 2))
    elig = ~excl
    strat = np.where(tflag & ~sflag, "A", np.where(tflag & sflag, "B", np.where(~tflag & ~sflag, "C", "D")))
    share = {s: float(((strat == s) & elig).sum()) / elig.sum() for s in STRATA}
    base, extra = divmod(n_total, 4)
    quota = {"A": base + (extra > 0), "B": base, "C": base + (extra > 1), "D": base + (extra > 2)}
    rng = np.random.default_rng(seed)
    from sklearn.cluster import KMeans
    rows, short = [], 0
    for s in ["A", "B", "D", "C"]:                               # C last, it absorbs the shortfall of the others
        pool = np.flatnonzero((strat == s) & elig)
        k = quota[s] + (short if s == "C" else 0)
        if len(pool) <= k:
            chosen = pool
            short += k - len(pool)
            lab = np.arange(len(pool))
        else:
            feat = (XY[pool] / Wm).reshape(len(pool), -1)
            lab = KMeans(n_clusters=k, random_state=seed, n_init=4).fit_predict(feat)
            chosen = np.array([rng.choice(pool[lab == c]) for c in range(k)])
            lab = np.arange(k)
        for f, c in zip(chosen, lab):
            rows.append({"frame": int(f), "stratum": s, "cluster": int(c)})
    sel = pd.DataFrame(rows)
    assert len(sel) == n_total, (len(sel), n_total)
    assert not excl[sel.frame].any()
    for s in STRATA:
        cnt = int((sel.stratum == s).sum())
        sel.loc[sel.stratum == s, "stratum_frames"] = int(((strat == s) & elig).sum())
        sel.loc[sel.stratum == s, "stratum_share"] = share[s]
        sel.loc[sel.stratum == s, "weight"] = share[s] / cnt if cnt else np.nan
    sel["teacher_flag"] = tflag[sel.frame]
    sel["student_flag"] = sflag[sel.frame]
    sel = sel.sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    sel.insert(0, "blind_name", [f"img{i + 1:03d}.png" for i in range(len(sel))])
    sel["video"] = v
    ld = label_dir(v)
    if ld.exists():
        raise SystemExit(f"{ld} exists; remove it by hand for a fresh set")
    ld.mkdir(parents=True)
    out = mkdir(folder(v))
    sel.to_csv(out / "selection_key.csv", index=False)
    pd.DataFrame([{"video": v, "unit": unit_of(v), "frames": n, "eligible": int(elig.sum()), "excluded_truth_band": int(band(n, truth.index.astype(int), GUARD_TRUTH).sum()),
                   "excluded_label_band": int(band(n, label_frames(v), GUARD_LABEL).sum()), "excluded_pseudo_band": int(band(n, pseudo_frames(v), GUARD_PSEUDO).sum()),
                   **{f"share_{s}": round(share[s], 5) for s in STRATA}, **{f"drawn_{s}": int((sel.stratum == s).sum()) for s in STRATA},
                   "eye_width_px": round(Wm, 1), "seed": seed}]).to_csv(out / "selection_summary.csv", index=False)
    # PNGs: one sequential pass, frames written exactly as read (as select_frames.py and self_training.py pseudo do)
    import cv2
    want = dict(zip(sel.frame.astype(int), sel.blind_name))
    cap = cv2.VideoCapture(str(HERE / unit_of(v) / f"{unit_of(v)}.mp4"))
    i, last = 0, max(want)
    while i <= last:
        if not cap.grab():
            break
        if i in want:
            ok, im = cap.retrieve()
            assert ok
            cv2.imwrite(str(ld / want[i]), im)
        i += 1
    cap.release()
    missing = [nm for nm in sel.blind_name if not (ld / nm).exists()]
    assert not missing, missing
    print(f"video {v}: {len(sel)} frames drawn from {int(elig.sum())} eligible; strata " + ", ".join(f"{s} {int((sel.stratum == s).sum())} (share {100 * share[s]:.1f} %)" for s in STRATA))
    print(f"PNGs in {ld}; key in {out / 'selection_key.csv'} (do not open it before labeling)")


# --------------------------------------------------------------------------------------------------------- pre-labels
def prelabel(v, points_csv):
    """machinelabels.h5/.csv for napari from a csv (blind_name, bodypart, x, y) made without any model's output."""
    ld = label_dir(v)
    key = pd.read_csv(folder(v) / "selection_key.csv")
    P = pd.read_csv(points_csv)
    P["blind_name"] = [nm if nm.endswith(".png") else nm + ".png" for nm in P.blind_name]
    assert set(P.bodypart) <= set(BPS), set(P.bodypart) - set(BPS)
    assert set(P.blind_name) <= set(key.blind_name), set(P.blind_name) - set(key.blind_name)
    template = HERE / "8_20251024_Pluto_spont1" / "training-data" / "labels" / "test50" / "CollectedData_Zhiheng.h5"
    cols = pd.read_hdf(template).columns
    idx = pd.MultiIndex.from_tuples([(ld.parent.name, ld.name, nm) for nm in key.blind_name])
    df = pd.DataFrame(np.nan, index=idx, columns=cols)
    sc = cols.get_level_values(0)[0]
    for r in P.itertuples():
        df.loc[(ld.parent.name, ld.name, r.blind_name), [(sc, r.bodypart, "x"), (sc, r.bodypart, "y")]] = [float(r.x), float(r.y)]
    for old in ("machinelabels.h5", "machinelabels.csv"):
        if (ld / old).exists():
            bk = mkdir(ld / "_old_prelabels")
            shutil.copy2(ld / old, bk / old)
    df.to_hdf(ld / "machinelabels.h5", key="df_with_missing", mode="w")
    df.to_csv(ld / "machinelabels.csv")
    print(f"{df.notna().sum().sum() // 2} points on {len(df)} frames written to {ld / 'machinelabels.h5'}")


def subset(v, counts):
    """Keep a stratified subset of the drawn frames for labeling (user decision 2026-11-04: fewer frames, weighted to
    the strata's shares). counts = {stratum: frames to keep}; within a stratum the lowest cluster ids are kept.
    The other PNGs move to labeling/blind_unused, the pre-labels are cut to the kept frames (the full set is backed
    up to prelabels_full/), and the key's weight becomes stratum share / kept count (NaN for unused frames)."""
    ld = label_dir(v)
    kp = folder(v) / "selection_key.csv"
    key = pd.read_csv(kp)
    keep = []
    for s, n in counts.items():
        g = key[key.stratum == s].sort_values("cluster")
        assert n <= len(g), f"stratum {s}: {n} asked, {len(g)} drawn"
        keep += list(g.blind_name.iloc[:n])
    key["subset"] = key.blind_name.isin(keep)
    for s in STRATA:
        cnt = int((key.subset & (key.stratum == s)).sum())
        share = float(key.loc[key.stratum == s, "stratum_share"].iloc[0])
        key.loc[key.stratum == s, "weight"] = share / cnt if cnt else np.nan
    key.loc[~key.subset, "weight"] = np.nan
    unused = mkdir(ld.parent / "blind_unused")
    for nm in key.blind_name[~key.subset]:
        if (ld / nm).exists():
            shutil.move(str(ld / nm), str(unused / nm))
    bk = mkdir(folder(v) / "prelabels_full")
    for f in ("machinelabels.h5", "machinelabels.csv"):
        if (ld / f).exists() and not (bk / f).exists():
            shutil.copy2(ld / f, bk / f)
    ml = pd.read_hdf(bk / "machinelabels.h5")
    ml = ml[[Path(i[-1] if isinstance(i, tuple) else i).name in keep for i in ml.index]]
    ml.to_hdf(ld / "machinelabels.h5", key="df_with_missing", mode="w")
    ml.to_csv(ld / "machinelabels.csv")
    key.to_csv(kp, index=False)
    sp = folder(v) / "selection_summary.csv"
    S = pd.read_csv(sp)
    for s in STRATA:
        S[f"kept_{s}"] = int((key.subset & (key.stratum == s)).sum())
    S.to_csv(sp, index=False)
    assert len(list(ld.glob("img*.png"))) == len(keep) == len(ml)
    print(f"video {v}: {len(keep)} frames kept for labeling (" + ", ".join(f"{s} {counts.get(s, 0)}" for s in STRATA)
          + f"), {int((~key.subset).sum())} moved to {unused.name}; pre-labels cut to {len(ml)} rows (full set in {bk.name}/)")


# ------------------------------------------------------------------------------------------------------------ scoring
def read_labels(v):
    ld = label_dir(v)
    files = sorted(glob.glob(str(ld / "CollectedData_*.h5")))
    if not files:
        raise SystemExit(f"no CollectedData file in {ld}: label first")
    df = pd.read_hdf(files[-1])
    if df.columns.nlevels == 4:
        df.columns = df.columns.droplevel("individuals")
    sc = df.columns.get_level_values(0)[0]
    key = pd.read_csv(folder(v) / "selection_key.csv").set_index("blind_name")
    rows = []
    for i, row in df.iterrows():
        nm = Path(i[-1] if isinstance(i, tuple) else i).name
        if nm not in key.index:
            continue                                           # the dummy frame, if any
        r = {"blind_name": nm, "frame": int(key.loc[nm, "frame"]), "stratum": key.loc[nm, "stratum"], "weight": float(key.loc[nm, "weight"])}
        for b in BPS:
            r[f"human_{b}_x"], r[f"human_{b}_y"] = float(row[(sc, b, "x")]), float(row[(sc, b, "y")])
        rows.append(r)
    L = pd.DataFrame(rows).set_index("frame")
    L["labeled_points"] = sum(np.isfinite(L[f"human_{b}_x"]).astype(int) for b in BPS)
    return L


def coords_of(path):
    """{bodypart: (n, 2)} from a DLC h5 or from a coords_after.csv.gz of the repair."""
    if str(path).endswith(".h5"):
        d = pt.load(str(path))
        return {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
    C = pd.read_csv(path)
    return {b: np.c_[C[f"{b}_x"].to_numpy(float), C[f"{b}_y"].to_numpy(float)] for b in BPS}


def models_of(v, extra):
    unit = unit_of(v)
    m = {"teacher": teacher_h5(v), "teacher + repair": RES / "self_training" / f"video{v:02d}" / "repair" / "coords_after.csv.gz"}
    for name in ("round1_every30", "round1_every100", "human40"):
        h5 = sorted((LOCAL / f"video{v:02d}" / name / "predictions").glob("*.h5"))
        if h5:
            m[name] = h5[-1]
    if "human40" not in m:                                         # the 40-label model of the scaling curve, when its prediction is on disk
        h5 = sorted((HERE / unit / "training-data" / "predictions_step02").glob("*snapshot_120.h5"))
        if h5:
            m["human40"] = h5[-1]
    for e in extra or []:
        name, path = e.split("=", 1)
        m[name] = Path(path)
    return {k: p for k, p in m.items() if Path(p).exists()}


def errors(v, L, models):
    _, dt, *_ = load(v)
    Wm = teacher_eye_width(dt)
    rows = []
    for name, path in models.items():
        xy = coords_of(path)
        for f, hr in L.iterrows():
            pts = {b: xy[b][f] for b in BPS}
            r = {"model": name, "frame": f, "blind_name": hr.blind_name, "stratum": hr.stratum, "weight": hr.weight, "labeled_points": hr.labeled_points,
                 "error": frame_rmse(pts, hr) / Wm, "error_px": frame_rmse(pts, hr)}
            for b in BPS:
                r[b] = np.hypot(pts[b][0] - hr[f"human_{b}_x"], pts[b][1] - hr[f"human_{b}_y"]) / Wm
            rows.append(r)
    return pd.DataFrame(rows), Wm


def wquantile(x, w, q):
    o = np.argsort(x)
    x, w = np.asarray(x)[o], np.asarray(w)[o]
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), c, x))


def summarise(E):
    rows = []
    for name, g in E.groupby("model", sort=False):
        for s, gs in list(g.groupby("stratum")) + [("all (weighted)", g)]:
            gs = gs[np.isfinite(gs.error)]
            w = gs.weight.to_numpy() if s.startswith("all") else np.ones(len(gs))
            rows.append({"model": name, "stratum": s, "frames": len(gs), "median_error_pct": round(100 * wquantile(gs.error, w, 0.5), 2),
                         "p90_error_pct": round(100 * wquantile(gs.error, w, 0.9), 2), "wrong_frames": int((gs.error > LINE).sum()),
                         "wrong_pct": round(100 * float(np.average(gs.error > LINE, weights=w)), 2),
                         **{f"median_{b}_pct": round(100 * wquantile(gs[b][np.isfinite(gs[b])], w[np.isfinite(gs[b])], 0.5), 2) if np.isfinite(gs[b]).any() else np.nan for b in BPS}})
    return pd.DataFrame(rows)


def paired(E, ref="teacher"):
    from scipy.stats import binomtest, wilcoxon
    R = E[E.model == ref].set_index("frame")
    rows = []
    for name, g in E.groupby("model", sort=False):
        if name == ref:
            continue
        g = g.set_index("frame")
        for s, idx in list(g.groupby("stratum").groups.items()) + [("all", g.index)]:
            d = (g.loc[idx, "error"] - R.loc[idx, "error"]).dropna()
            better, worse = int((d < -0.01).sum()), int((d > 0.01).sum())
            same = len(d) - better - worse
            p_sign = binomtest(better, better + worse).pvalue if better + worse else np.nan
            try:
                p_w = wilcoxon(d).pvalue if (d != 0).any() else np.nan
            except ValueError:
                p_w = np.nan
            rows.append({"model": name, "against": ref, "stratum": s, "frames": len(d), "better": better, "worse": worse, "same_within_1pct": same,
                         "median_diff_pct": round(100 * float(d.median()), 2), "p_sign": round(p_sign, 4) if np.isfinite(p_sign) else np.nan,
                         "p_wilcoxon": round(p_w, 4) if np.isfinite(p_w) else np.nan})
    return pd.DataFrame(rows)


def figure(E, v, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    students = [m for m in E.model.unique() if m not in ("teacher",)]
    R = E[E.model == "teacher"].set_index("frame").error * 100
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), sharex=True, sharey=True)
    colors = {"teacher + repair": "#9e9e9e", "round1_every30": "#e69f00", "round1_every100": "#0072b2", "human40": "#009e73"}
    lim = max(float(np.nanmax(E.error)) * 100, LINE * 100) * 1.05
    for ax, s in zip(axes, STRATA):
        for m in students:
            g = E[(E.model == m) & (E.stratum == s)].set_index("frame").error * 100
            ax.scatter(R.loc[g.index], g, s=22, color=colors.get(m, "k"), label=m, alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.plot([0, lim], [0, lim], color="0.75", lw=0.8, zorder=0)
        ax.axhline(LINE * 100, color="0.6", lw=0.8, ls="--"); ax.axvline(LINE * 100, color="0.6", lw=0.8, ls="--")
        ax.set_title(f"{s}: {STRATA[s]}", fontsize=10)
        ax.set_xlabel("teacher error, % of eye width")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    axes[0].set_ylabel("model error, % of eye width")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Video {v}, blind test set: each point is one frame, below the diagonal = better than the teacher", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def score(v, extra):
    L = read_labels(v)
    models = models_of(v, extra)
    E, Wm = errors(v, L, models)
    out = folder(v)
    E.round(5).to_csv(out / "per_frame.csv", index=False)
    S = summarise(E)
    S.to_csv(out / "scores.csv", index=False)
    Pd = paired(E)
    Pd.to_csv(out / "paired.csv", index=False)
    figure(E, v, out / "paired.png")
    print(f"video {v}: {len(L)} labeled frames ({int((L.labeled_points == 0).sum())} with no point), eye width {Wm:.1f} px, models: {', '.join(models)}")
    print(S[S.stratum.str.startswith("all")][["model", "frames", "median_error_pct", "p90_error_pct", "wrong_frames", "wrong_pct"]].to_string(index=False))
    print(Pd[Pd.stratum == "all"][["model", "frames", "better", "worse", "same_within_1pct", "median_diff_pct", "p_sign", "p_wilcoxon"]].to_string(index=False))


def check(v):
    """The scorer on the 70 truth frames of the teacher must reproduce self_training's truth_errors.csv."""
    _, dt, P, H, truth, *_ = load(v)
    L = truth.copy()
    L["blind_name"] = [f"truth{int(f)}" for f in L.index]
    L["stratum"] = L["set"]
    L["weight"] = 1.0
    L["labeled_points"] = sum(np.isfinite(L[f"human_{b}_x"]).astype(int) for b in BPS)
    E, Wm = errors(v, L, {"teacher": teacher_h5(v)})
    T = pd.read_csv(RES / "self_training" / f"video{v:02d}" / "teacher" / "truth_errors.csv").set_index("frame")
    E = E.set_index("frame")
    diff = float(np.nanmax(np.abs(E.error - T.error)))
    assert diff < 1e-3, diff
    print(f"video {v}: scorer reproduces truth_errors.csv on {len(E)} truth frames (max difference {diff:.2e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "subset", "prelabel", "score", "check"])
    ap.add_argument("--video", type=int, required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--points", default=None, help="prelabel: csv (blind_name, bodypart, x, y)")
    ap.add_argument("--counts", default="A=10,B=3,C=8,D=3", help="subset: frames to keep per stratum")
    ap.add_argument("--extra", action="append", help="score: extra model as name=path (h5 or coords_after.csv.gz)")
    a = ap.parse_args()
    if a.cmd == "select":
        select(a.video, a.n, a.seed)
    elif a.cmd == "subset":
        subset(a.video, {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.counts.split(",")})
    elif a.cmd == "prelabel":
        prelabel(a.video, a.points)
    elif a.cmd == "score":
        score(a.video, a.extra)
    else:
        check(a.video)


if __name__ == "__main__":
    main()
