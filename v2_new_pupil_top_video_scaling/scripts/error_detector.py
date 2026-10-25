#!/usr/bin/env python3
"""Stage 2 of P1 (docs/next_phase_plan_2026-10-24.md): a detector FITTED to our truth instead of a hand-made signal.
Read-only with respect to every earlier result; everything is written to results/error_detector/.

    python error_detector.py --cache        # SARIMAX deviations for every whole-video prediction file (4 processes, ~40 min)
    python error_detector.py                # truth pairs, features, D1 (gradient boosting) under both protocols, tables, figure

Question: given only what the model produced on a frame (8 predicted points, 8 confidences, how the prediction moves
in time), can a classifier trained on frames with known error find the frames whose prediction is > 20 px wrong,
better than the best hand-made signal (fit_pupil, stage 1)?

Truth pairs (no new labels). A pair = (one whole-video prediction file, one human-labeled frame of that video that
the model never trained on). Files: every training-data/predictions_stepNN/*.h5 (1–7 per video, 65 in all; step 00 of
video 4 is the 0-label model). Frames: labels/batchNN for NN > step (the model of step s trained on batches 1..s),
and test50 + val20 always. Per pair: the error of each keypoint (NaN where the labeler left it empty), the frame RMSE
over the labeled keypoints (same formula as step 8), wrong20 / wrong50, the predicted coordinates (pred_<bp>_x/y) and the human ones (human_<bp>_x/y). The pairs of the plateau model on test50 +
val20 are exactly the 1330 frames of step 8 (checked against results/unreliable_frames/unreliable_frames_all_videos.csv).

Features per (file, frame), all label-free, all from that file (W = the video's median corner distance):
  conf_<bp> (8), min_conf, mean_conf, n_conf_lt_0.6
  pose: x and y of the 8 points minus the corner midpoint, / W (16)
  jump_<bp> forward and backward (displacement from the previous / to the next frame / W), jump_max, jump_bwd_max
  rough_<bp>: distance to the centred 31-frame median of the point / W (a feature only; nothing is smoothed), rough_max
  fit_<bp>: deviation from the SARIMAX(3,0,1) one-step forward prediction (px, as DLC's fitting); fit_mean (= step-8
  fitting), fit_pupil, fit_eye, fit_max, fit_ci (points outside the 99% interval)
  geometry / W: pupil width, height, height/width, axis-midpoint offset, pupil centre − eye centre (x, y), eye opening,
  pupil centre − lid midpoint (y), corner distance deviation, how far the pupil sticks out of the lids
  context: min_conf, fit_pupil, fit_max, jump_max, rough_max at t−1 and t+1 and their mean over t−5..t+5
  eye_width_px

Detector D1: sklearn HistGradientBoostingClassifier (class_weight balanced), target wrong20 of the pair.
Protocols (the held-out video's plateau pairs on test50 + val20 are the only frames ever scored, so the 1330 frames and
their 155 / 32 positives are those of step 8 and the sweep):
  lovo    train on all pairs of the other 18 videos
  deploy  lovo + the held-out video's pairs on its own batch frames (labels that exist anyway when the detector is used)
Variants (ablation): plateau pairs only × sweep-like scalar features (must land near the sweep's combo_lr, AUC 0.815);
plateau pairs only × all features; all pairs × all features; all pairs × features without temporal context.
Evaluation as in the sweep: AUC at 20 / 50 px (pooled, per video), recall and precision at 2 / 5 / 10 % flagged, flagged
share at recall 50 / 70 / 90 %; fit_pupil on the same frames as the reference row.
Outputs: truth_pairs.csv, features_pairs.csv, scores_lovo.csv, auc_table.csv, operating_points.csv, roc_curves.csv,
detector_overview.png, cache/ (local only)."""
import argparse
import glob
import re
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pupil_trace as pt  # noqa: E402
from scale_step import BPS, HERE, flat, human_table  # noqa: E402
from unreliable_frames_sweep import PL, STEP8, auc, fidx, operating_points, plateau_step, roc, sarimax_dev  # noqa: E402
from unreliable_frames_sweep import CACHE as SWEEP_CACHE  # noqa: E402

OUT = HERE / "results" / "error_detector"
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
PUPIL, EYE = BPS[:4], BPS[4:]
ORDER = (3, 0, 1)
SETS_HELD_OUT = ("test50", "val20")


# ---------------------------------------------------------------- files
def prediction_files():
    """One row per whole-video prediction file: video, unit, step, path, plateau (bool)."""
    rows = []
    for r in PL.itertuples():
        ps = plateau_step(r.labels_at_plateau)
        for folder in sorted(glob.glob(str(HERE / r.unit / "training-data" / "predictions_step*"))):
            h5 = sorted(glob.glob(str(Path(folder) / "*snapshot*120*.h5")))
            if not h5:
                continue
            step = int(re.search(r"predictions_step(\d+)", folder).group(1))
            rows.append(dict(video=int(r.video), unit=r.unit, step=step, path=h5[-1], plateau=(step == ps)))
    return pd.DataFrame(rows)


def label_sets(unit):
    """{set name: flat label table (index = png name)} for every set with human labels."""
    out = {}
    for folder in sorted(glob.glob(str(HERE / unit / "training-data" / "labels" / "*"))):
        name = Path(folder).name
        if not (name.startswith("batch") or name in SETS_HELD_OUT):
            continue
        if not glob.glob(str(Path(folder) / "CollectedData_*.h5")):
            continue
        out[name] = flat(human_table(Path(folder)))
    return out


def cache_path(video, step, tag):
    return CACHE / f"video{video:02d}_step{step:02d}_{tag}.npy"


def file_cache(args):
    """SARIMAX(3,0,1) forward deviations + 99%-interval flags of one file. The plateau file was already fitted by the sweep."""
    video, step, path, plateau = args
    pdev, pci = cache_path(video, step, "fwd301"), cache_path(video, step, "ci301")
    if pdev.exists() and pci.exists():
        return video, step, "cached"
    sdev, sci = SWEEP_CACHE / f"video{video:02d}_fwd301.npy", SWEEP_CACHE / f"video{video:02d}_ci301.npy"
    if plateau and sdev.exists() and sci.exists():
        np.save(pdev, np.load(sdev)); np.save(pci, np.load(sci))
        return video, step, "copied from the sweep"
    dev, out = sarimax_dev(pt.load(path), ORDER, backward=False)
    np.save(pdev, dev); np.save(pci, out)
    return video, step, "fitted"


# ---------------------------------------------------------------- features for every frame of one file
def xy(d, b):
    return d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)


def rolling_median(a, n):
    return pd.Series(a).rolling(n, center=True, min_periods=1).median().to_numpy()


def file_features(video, step, path):
    d = pt.load(path)
    T = len(d)
    P = np.stack([np.stack(xy(d, b), axis=1) for b in BPS], axis=1)          # (T, 8, 2)
    C = np.stack([d[b]["likelihood"].to_numpy(float) for b in BPS], axis=1)  # (T, 8)
    nas, tem = P[:, BPS.index("eye_nasal_corner")], P[:, BPS.index("eye_temporal_corner")]
    width = np.hypot(*(tem - nas).T)
    W = float(np.nanmedian(width))
    mid = (nas + tem) / 2
    F = {}
    for k, b in enumerate(BPS):
        F[f"conf_{b}"] = C[:, k]
    F["min_conf"], F["mean_conf"], F["n_conf_lt_0.6"] = C.min(1), C.mean(1), (C < 0.6).sum(1)
    rel = (P - mid[:, None, :]) / W
    for k, b in enumerate(BPS):
        F[f"pose_{b}_x"], F[f"pose_{b}_y"] = rel[:, k, 0], rel[:, k, 1]
    fwd = np.full((T, 8), np.nan); fwd[1:] = np.hypot(*(P[1:] - P[:-1]).transpose(2, 0, 1)) / W
    bwd = np.full((T, 8), np.nan); bwd[:-1] = fwd[1:]
    for k, b in enumerate(BPS):
        F[f"jump_{b}"], F[f"jumpb_{b}"] = fwd[:, k], bwd[:, k]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        F["jump_max"], F["jump_bwd_max"] = np.nanmax(fwd, 1), np.nanmax(bwd, 1)
    rough = np.zeros((T, 8))
    for k in range(8):
        mx, my = rolling_median(P[:, k, 0], 31), rolling_median(P[:, k, 1], 31)
        rough[:, k] = np.hypot(P[:, k, 0] - mx, P[:, k, 1] - my) / W
    for k, b in enumerate(BPS):
        F[f"rough_{b}"] = rough[:, k]
    F["rough_max"] = rough.max(1)
    dev, ci = np.load(cache_path(video, step, "fwd301")), np.load(cache_path(video, step, "ci301"))
    for k, b in enumerate(BPS):
        F[f"fit_{b}"] = dev[:, k]
    F["fit_mean"], F["fit_pupil"], F["fit_eye"] = np.nanmean(dev, 1), np.nanmean(dev[:, :4], 1), np.nanmean(dev[:, 4:], 1)
    F["fit_max"], F["fit_ci"] = np.nanmax(dev, 1), ci.sum(1)
    pt_, pb, pl, pr = (P[:, BPS.index(b)] for b in PUPIL)
    lt, lb = P[:, BPS.index("eyelid_top")], P[:, BPS.index("eyelid_bottom")]
    pw, ph = np.hypot(*(pr - pl).T), np.hypot(*(pb - pt_).T)
    F["geo_pupil_w"], F["geo_pupil_h"] = pw / W, ph / W
    F["geo_pupil_hw"] = ph / np.maximum(pw, 1e-6)
    F["geo_axes_off"] = np.hypot(*((pl + pr) / 2 - (pt_ + pb) / 2).T) / W
    pc = (pl + pr + pt_ + pb) / 4
    F["geo_pc_dx"], F["geo_pc_dy"] = (pc - mid)[:, 0] / W, (pc - mid)[:, 1] / W
    F["geo_opening"] = (lb[:, 1] - lt[:, 1]) / W
    F["geo_pc_lid_dy"] = (pc[:, 1] - (lt[:, 1] + lb[:, 1]) / 2) / W
    F["geo_width_dev"] = np.abs(width - W) / W
    F["geo_outside"] = (np.maximum(0, lt[:, 1] - pt_[:, 1]) + np.maximum(0, pb[:, 1] - lb[:, 1])) / W
    F["eye_width_px"] = np.full(T, W)
    F = pd.DataFrame(F)
    for c in ("min_conf", "fit_pupil", "fit_max", "jump_max", "rough_max"):
        s = F[c]
        F[f"{c}_prev"], F[f"{c}_next"] = s.shift(1), s.shift(-1)
        F[f"{c}_win11"] = s.rolling(11, center=True, min_periods=1).mean()
    return F


# ---------------------------------------------------------------- truth pairs
def truth_pairs(files):
    rows = []
    for r in files.itertuples():
        d = pt.load(r.path)
        P = np.stack([np.stack(xy(d, b), axis=1) for b in BPS], axis=1)
        sets = label_sets(r.unit)
        for name, L in sets.items():
            if name.startswith("batch") and int(name[5:]) <= r.step:
                continue
            for png in L.index:
                i = fidx(png)
                errs, hum = [], {}
                for b in BPS:
                    hx, hy = float(L[b]["x"].loc[png]), float(L[b]["y"].loc[png])
                    hum[f"human_{b}_x"], hum[f"human_{b}_y"] = hx, hy
                    errs.append(np.hypot(P[i, BPS.index(b), 0] - hx, P[i, BPS.index(b), 1] - hy) if np.isfinite(hx) and np.isfinite(hy) else np.nan)
                errs = np.array(errs)
                ok = np.isfinite(errs)
                if not ok.any():
                    continue
                row = dict(video=r.video, unit=r.unit, model_step=r.step, plateau=r.plateau, set=name, frame=i,
                           labeled_points=int(ok.sum()), frame_rmse_px=float(np.sqrt(np.mean(errs[ok] ** 2))))
                row.update({f"err_{b}": errs[k] for k, b in enumerate(BPS)})
                row.update({f"pred_{b}_{c}": float(P[i, k, j]) for k, b in enumerate(BPS) for j, c in enumerate("xy")})
                row.update(hum)
                rows.append(row)
    T = pd.DataFrame(rows)
    T["wrong20"], T["wrong50"] = T.frame_rmse_px > 20, T.frame_rmse_px > 50
    T["held_out"] = T.set.isin(SETS_HELD_OUT)
    T["eval"] = T.plateau & T.held_out
    return T


# ---------------------------------------------------------------- D1
SWEEP_LIKE = ["fit_mean", "fit_max", "fit_ci", "fit_pupil", "jump_max", "min_conf", "mean_conf", "geo_axes_off", "geo_outside", "geo_width_dev"]


def feature_groups(cols):
    ctx = [c for c in cols if c.endswith(("_prev", "_next", "_win11"))]
    return {"sweep_like": SWEEP_LIKE, "all": list(cols), "no_context": [c for c in cols if c not in ctx]}


def fit_predict(Xtr, ytr, Xte, seed=0):
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=25,
                                         l2_regularization=1.0, class_weight="balanced", random_state=seed)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def run_variants(D, feat_cols):
    """D: pairs with features. Returns the eval frames (1330) with one score column per variant."""
    E = D[D["eval"]].copy().reset_index(drop=True)
    groups = feature_groups(feat_cols)
    variants = [("d1_plateau_sweeplike", "plateau", "sweep_like", "lovo"),
                ("d1_plateau_all", "plateau", "all", "lovo"),
                ("d1_pairs_all", "pairs", "all", "lovo"),
                ("d1_pairs_nocontext", "pairs", "no_context", "lovo"),
                ("d1_pairs_all_deploy", "pairs", "all", "deploy")]
    for name, pool, grp, proto in variants:
        cols = groups[grp]
        E[name] = np.nan
        for v in sorted(E.video.unique()):
            if pool == "plateau":
                tr = D["eval"] & (D.video != v)
            else:
                tr = (D.video != v)
                if proto == "deploy":
                    tr |= (D.video == v) & ~D.held_out
            te = E.video == v
            E.loc[te, name] = fit_predict(D.loc[tr, cols], D.loc[tr, "wrong20"], E.loc[te, cols])
        print(f"  {name}: AUC20 {auc(E[name][E.wrong20], E[name][~E.wrong20]):.3f}  AUC50 {auc(E[name][E.wrong50], E[name][~E.wrong50]):.3f}", flush=True)
    return E, [v[0] for v in variants]


def evaluate(E, names):
    rows, ops, curves = [], [], []
    for name in names:
        s = E[name]
        per = [auc(s[(E.video == v) & E.wrong20], s[(E.video == v) & ~E.wrong20]) for v in sorted(E.video.unique())]
        rows.append({"signal": name, "AUC_wrong20": round(auc(s[E.wrong20], s[~E.wrong20]), 3), "AUC_wrong50": round(auc(s[E.wrong50], s[~E.wrong50]), 3),
                     "AUC_wrong20_videos_median": round(float(np.nanmedian(per)), 3), "AUC_wrong20_videos_min": round(float(np.nanmin(per)), 3)})
        ops += operating_points(name, s, E)
        for w in ("wrong20", "wrong50"):
            R = roc(s, E[w]); R.insert(0, "truth", w); R.insert(0, "signal", name); curves.append(R)
    return pd.DataFrame(rows), pd.DataFrame(ops).drop_duplicates(), pd.concat(curves, ignore_index=True)


def figure(E, names, A, O):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    for j, w in enumerate(("wrong20", "wrong50")):
        a = ax[j]
        for i, n in enumerate(names):
            R = roc(E[n], E[w]); lab = n.replace("d1_", "D1 ").replace("_", " ")
            a.plot(R.flagged * 100, R.recall * 100, color=cols[i], lw=1.6, label=f"{lab} (AUC {float(A.loc[A.signal == n, 'AUC_' + w].iloc[0]):.3f})")
        for x in (5, 10):
            a.axvline(x, color="0.6", ls=":", lw=1)
        a.set_xlim(0, 60); a.set_ylim(0, 101); a.set_xlabel("frames flagged (%)"); a.set_ylabel("recall (%)")
        a.set_title(f"{'> 20 px' if w == 'wrong20' else '> 50 px'} ({int(E[w].sum())} of {len(E)} frames)", fontsize=11)
        a.legend(fontsize=7.5, loc="lower right"); a.grid(alpha=0.3)
    a = ax[2]; a.axis("off")
    rows = []
    for n in names:
        q = O[(O.signal == n) & (O.truth == "wrong20")]
        g = lambda t: q[q.target == t]
        r5, r10, f50 = g("flag 5%"), g("flag 10%"), g("recall 50%")
        rows.append([n.replace("d1_", "").replace("_", " "),
                     f"{float(r5.recall_pct.iloc[0]):.0f} / {float(r5.precision_pct.iloc[0]):.0f}" if len(r5) else "–",
                     f"{float(r10.recall_pct.iloc[0]):.0f} / {float(r10.precision_pct.iloc[0]):.0f}" if len(r10) else "–",
                     f"{float(f50.flagged_pct.iloc[0]):.1f}%" if len(f50) else "–"])
    t = a.table(cellText=rows, colLabels=["detector", "5% flagged:\nrecall / prec", "10% flagged:\nrecall / prec", "flagged for\nrecall 50%"],
                loc="center", cellLoc="center", colWidths=[0.42, 0.2, 0.2, 0.18])
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.7)
    a.set_title("> 20 px truth, leave-one-video-out", fontsize=11)
    fig.suptitle(f"Error detector fitted to our truth, scored on the {len(E)} step-8 frames (plateau model, test50 + val20) of {E.video.nunique()} videos; "
                 "training pairs = other videos' models on frames they never trained on", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "detector_overview.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true", help="only the SARIMAX cache")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    files = prediction_files()
    print(f"{len(files)} whole-video prediction files, {files.video.nunique()} videos", flush=True)
    todo = [(r.video, r.step, r.path, r.plateau) for r in files.itertuples() if not cache_path(r.video, r.step, "ci301").exists()]
    if todo:
        print(f"SARIMAX fits for {len(todo)} files ({a.procs} processes) ...", flush=True)
        with Pool(a.procs) as pool:
            for v, s, how in pool.imap_unordered(file_cache, todo):
                print(f"  video {v} step {s}: {how}", flush=True)
    if a.cache:
        return
    T = truth_pairs(files)
    T.to_csv(OUT / "truth_pairs.csv", index=False)
    S8 = pd.read_csv(STEP8)
    chk = T[T["eval"]].merge(S8[["video", "frame", "frame_rmse_px"]], on=["video", "frame"], suffixes=("", "_step8"))
    assert len(chk) == len(S8) == T["eval"].sum(), (len(chk), len(S8), T["eval"].sum())
    assert np.abs(chk.frame_rmse_px - chk.frame_rmse_px_step8).max() < 0.01
    print(f"{len(T)} truth pairs ({T.wrong20.sum()} wrong20, {T.wrong50.sum()} wrong50); eval frames {T['eval'].sum()} "
          f"({T[T['eval']].wrong20.sum()} / {T[T['eval']].wrong50.sum()}) reproduce step 8", flush=True)
    parts = []
    for r in files.itertuples():
        Fv = file_features(r.video, r.step, r.path)
        g = T[(T.video == r.video) & (T.model_step == r.step)]
        parts.append(pd.concat([g.reset_index(drop=True), Fv.iloc[g.frame.to_numpy()].reset_index(drop=True)], axis=1))
    D = pd.concat(parts, ignore_index=True)
    feat_cols = [c for c in D.columns if c not in T.columns]
    D.to_csv(OUT / "features_pairs.csv", index=False)
    print(f"{len(feat_cols)} features; training D1 ...", flush=True)
    E, names = run_variants(D, feat_cols)
    E["fit_pupil_ref"] = E["fit_pupil"]
    names = names + ["fit_pupil_ref"]
    E.to_csv(OUT / "scores_lovo.csv", index=False)
    A, O, R = evaluate(E, names)
    A.to_csv(OUT / "auc_table.csv", index=False); O.to_csv(OUT / "operating_points.csv", index=False); R.to_csv(OUT / "roc_curves.csv", index=False)
    print(A.to_string(index=False)); print(O[O.truth == "wrong20"].to_string(index=False))
    figure(E, names, A, O)


if __name__ == "__main__":
    main()
