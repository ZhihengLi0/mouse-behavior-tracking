#!/usr/bin/env python3
"""Stage 1 of the next phase (docs/next_phase_plan_2026-10-24.md, experiments S1-1 … S1-8): can label-free signals find
the unreliable frames better than the step-8 rules? Read-only analysis; nothing of step 8 is modified.

    python unreliable_frames_sweep.py            # all videos, 4 processes for the SARIMAX fits

Frames, models and truth are exactly those of step 8 (results/unreliable_frames/unreliable_frames_all_videos.csv):
the 50 test + 20 validation frames of every video, the plateau-point model, frame RMSE against the human labels,
wrong20 = RMSE > 20 px, wrong50 = RMSE > 50 px. Signals computed here for every frame of every video and then read
at the 1330 evaluated frames (all label-free, from the plateau model's whole-video prediction unless stated):

  fitting (step 8)   mean over the 8 keypoints of the deviation from the SARIMAX(3,0,1) one-step forward prediction (DLC)
  fit_max            the largest of the 8 deviations                                                  (S1-2)
  fit_pupil, fit_eye mean over the 4 pupil / 4 eye points                                             (S1-2)
  fit_ci             number of keypoints whose x or y lies outside the 99% prediction interval          (S1-2, DLC computes it, does not use it)
  fit_bwd_max        the same with the series reversed (one-step backward prediction), largest of 8    (S1-3)
  fit_fb_max         max(fit_max, fit_bwd_max)                                                        (S1-3)
  fit_o100/o200/o501/o311_max   forward, largest of 8, with orders (1,0,0), (2,0,0), (5,0,1), (3,1,1)   (S1-3)
  *_rel              a px signal divided by the eye width of the video (median corner distance)       (S1-4)
  pca_video          RMS over the 8 keypoints of the reconstruction error of the predicted pose by the PCA (99% of
                     variance) fitted on the labeled poses the plateau model was trained on (this video's batches) (S1-7)
  pca_mouse          the same with the PCA fitted on all labeled batches of the same mouse, poses centred on the
                     corner midpoint before the PCA                                                    (S1-7)
  geo_axes_off       distance between the midpoints of the two pupil axes / eye width                 (S1-7b)
  geo_outside        how far the pupil top lies above the eyelid top or the pupil bottom below the eyelid bottom (px, 0 if inside)
  geo_width_dev      |corner distance − video median| / video median
  ens_sd_max/mean    standard deviation of the keypoint position across all models of the video
                     (every predictions_stepNN), sqrt(var_x + var_y); largest / mean over keypoints    (S1-8)
  ens_sdw_max        the same with the variance divided by the mean confidence (EKS weighting)
  ens_med_max        distance between the plateau model and the ensemble median, largest of 8
  prev_max           distance between the plateau model and the previous step's model, largest of 8
  combo_lr           leave-one-video-out logistic regression on the signals above (target wrong20)    (S1-5)
  combo_rank         mean rank of fit_max, pca_mouse, ens_sd_max and −min_conf                        (S1-5)

Evaluation per signal: AUC (wrong20 and wrong50, pooled and per video), ROC curve, and operating points: the threshold
that reaches recall 50 / 70 / 90% on wrong20 and 90% on wrong50 with its flagged share and precision, and recall /
precision at flagged shares 2 / 5 / 10%.                                                                 (S1-1)
Outputs in results/unreliable_frames_sweep/: signals_all_frames.csv, auc_per_signal.csv, operating_points.csv,
roc_curves.csv, sweep_overview.png, cache/ (per-video SARIMAX deviations). Rolling-window SARIMAX fits (S1-3) were
not run: one whole-video fit per series already costs 6–13 s and the windows would multiply it."""
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pupil_trace as pt  # noqa: E402
from scale_step import BPS, HERE, flat, human_table  # noqa: E402

OUT = HERE / "results" / "unreliable_frames_sweep"
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
STEP8 = HERE / "results" / "unreliable_frames" / "unreliable_frames_all_videos.csv"
FIT_PBOUND, ALPHA = 0.01, 0.01
PUPIL, EYE = BPS[:4], BPS[4:]
ORDERS = {"301": (3, 0, 1), "100": (1, 0, 0), "200": (2, 0, 0), "501": (5, 0, 1), "311": (3, 1, 1)}
PL = pd.read_csv(HERE / "results" / "labels_to_plateau.csv")
PL = PL[PL.status == "final"].reset_index(drop=True)


def fidx(name):
    return int(re.findall(r"\d+", Path(str(name)).stem)[0])


def plateau_step(labels):
    return max(1, int(labels) // 20)


def pred_file(unit, step):
    h5 = sorted(glob.glob(str(HERE / unit / "training-data" / f"predictions_step{step:02d}" / "*snapshot*120*.h5")))
    return h5[-1] if h5 else None


# ---------------------------------------------------------------- SARIMAX deviations, per keypoint, cached per video
def _fit_series(y, order):
    m = np.nanmean(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(y - m, order=order, seasonal_order=(0, 0, 0, 0), simple_differencing=True).fit(disp=False)
        if not res.mle_retvals.get("converged", True):
            res = SARIMAX(y - m, order=order, seasonal_order=(0, 0, 0, 0), simple_differencing=True).fit(disp=False, method="powell", maxiter=500)
        pr = res.get_prediction(end=res.model.nobs - 1)
        mean = np.asarray(pr.predicted_mean) + m
        ci = pr.conf_int(alpha=ALPHA)
        lo, hi = np.asarray(ci)[:, 0] + m, np.asarray(ci)[:, 1] + m
    return mean, lo, hi


def sarimax_dev(d, order, backward):
    """(T, 8) deviation of each keypoint from its one-step prediction and (T, 8) bool: outside the 99% interval."""
    T = len(d)
    dev, out = np.full((T, len(BPS)), np.nan), np.zeros((T, len(BPS)), bool)
    for k, b in enumerate(BPS):
        pm, lo, hi = {}, {}, {}
        for c in ("x", "y"):
            y = d[b][c].to_numpy(float).copy()
            y[d[b]["likelihood"].to_numpy(float) < FIT_PBOUND] = np.nan
            if backward:
                y = y[::-1]
            mean, l, h = _fit_series(y, order)
            n = len(mean)
            full = lambda a: np.concatenate([np.full(T - n, np.nan), a])   # simple differencing drops the first samples
            mean, l, h = full(mean), full(l), full(h)
            if backward:
                mean, l, h = mean[::-1], l[::-1], h[::-1]
            pm[c], lo[c], hi[c] = mean, l, h
        x, y = d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)
        dev[:, k] = np.hypot(x - pm["x"], y - pm["y"])
        out[:, k] = (x < lo["x"]) | (x > hi["x"]) | (y < lo["y"]) | (y > hi["y"])
    return dev, out


def video_cache(args):
    """All SARIMAX variants of one video, written to cache/videoNN_*.npy. Run in a worker process."""
    video, unit, step = args
    f = pred_file(unit, step)
    d = pt.load(f)
    jobs = [("fwd301", ORDERS["301"], False), ("bwd301", ORDERS["301"], True)] + [(f"fwd{k}", o, False) for k, o in ORDERS.items() if k != "301"]
    for tag, order, back in jobs:
        p = CACHE / f"video{video:02d}_{tag}.npy"
        if p.exists():
            continue
        dev, out = sarimax_dev(d, order, back)
        np.save(p, dev)
        if tag == "fwd301":
            np.save(CACHE / f"video{video:02d}_ci301.npy", out)
    return video


# ---------------------------------------------------------------- the other signals (fast)
def xy(d, b):
    return d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)


def eye_width(d):
    (nx, ny), (tx, ty) = xy(d, "eye_nasal_corner"), xy(d, "eye_temporal_corner")
    return np.hypot(tx - nx, ty - ny)


def pose_matrix(d):
    return np.stack([np.stack(xy(d, b), axis=1) for b in BPS], axis=1)   # (T, 8, 2)


def labeled_poses(unit, step):
    parts = []
    for b in range(1, step + 1):
        folder = HERE / unit / "training-data" / "labels" / f"batch{b:02d}"
        if folder.exists():
            parts.append(flat(human_table(folder)))
    if not parts:
        return np.zeros((0, 8, 2))
    L = pd.concat(parts)
    X = np.stack([np.stack([L[(b, "x")].to_numpy(float), L[(b, "y")].to_numpy(float)], axis=1) for b in BPS], axis=1)
    return X[np.isfinite(X).all(axis=(1, 2))]


def centre(P):
    c = (P[:, 6, :] + P[:, 7, :]) / 2
    return P - c[:, None, :]


def pca_error(P_train, P_eval):
    """RMS over the 8 keypoints of |pose − PCA reconstruction| (px); PCA keeps 99% of the variance, at least 2 components."""
    if len(P_train) < 17:
        return np.full(len(P_eval), np.nan)
    Xt = P_train.reshape(len(P_train), -1)
    pca = PCA(n_components=0.99, svd_solver="full").fit(Xt)
    if pca.n_components_ < 2:
        pca = PCA(n_components=2, svd_solver="full").fit(Xt)
    Xe = P_eval.reshape(len(P_eval), -1)
    R = pca.inverse_transform(pca.transform(Xe)).reshape(P_eval.shape)
    return np.sqrt(np.mean(np.sum((P_eval - R) ** 2, axis=2), axis=1))


def ensemble_signals(unit, step, d_plateau):
    files = sorted(glob.glob(str(HERE / unit / "training-data" / "predictions_step*" / "*snapshot*120*.h5")))
    by_step = {}                                          # one file per step folder (a 0-label model from another video counts as step 0)
    for f in files:
        by_step.setdefault(int(re.findall(r"predictions_step(\d+)", f)[0]), f)
    steps = sorted(by_step)
    P, C = [], []
    for s in steps:
        d = pt.load(by_step[s])
        P.append(pose_matrix(d)); C.append(np.stack([d[b]["likelihood"].to_numpy(float) for b in BPS], axis=1))
    P, C = np.stack(P), np.stack(C)                      # (M, T, 8, 2), (M, T, 8)
    var = P.var(axis=0).sum(axis=2)                      # (T, 8): var_x + var_y
    sd = np.sqrt(var)
    sdw = np.sqrt(var / np.clip(C.mean(axis=0), 1e-3, None))
    med = np.median(P, axis=0)
    Pp = pose_matrix(d_plateau)
    med_dist = np.hypot(*(Pp - med).transpose(2, 0, 1))  # (T, 8)
    prev = step - 1 if step - 1 in steps else (step + 1 if step + 1 in steps else None)
    prev_dist = np.hypot(*(Pp - P[steps.index(prev)]).transpose(2, 0, 1)) if prev is not None else np.full(Pp.shape[:2], np.nan)
    return dict(ens_models=len(steps), ens_sd_max=sd.max(axis=1), ens_sd_mean=sd.mean(axis=1), ens_sdw_max=sdw.max(axis=1),
                ens_med_max=med_dist.max(axis=1), prev_max=prev_dist.max(axis=1))


def video_signals(r, mouse_train):
    step = plateau_step(r.labels_at_plateau)
    d = pt.load(pred_file(r.unit, step))
    T = len(d)
    W = eye_width(d); w = float(np.nanmedian(W))
    S = {}
    fwd = np.load(CACHE / f"video{r.video:02d}_fwd301.npy"); bwd = np.load(CACHE / f"video{r.video:02d}_bwd301.npy")
    ci = np.load(CACHE / f"video{r.video:02d}_ci301.npy")
    S["fit_max"], S["fit_pupil"], S["fit_eye"] = np.nanmax(fwd, axis=1), np.nanmean(fwd[:, :4], axis=1), np.nanmean(fwd[:, 4:], axis=1)
    S["fit_ci"] = ci.sum(axis=1).astype(float)
    S["fit_bwd_max"] = np.nanmax(bwd, axis=1); S["fit_fb_max"] = np.fmax(S["fit_max"], S["fit_bwd_max"])
    for k in ORDERS:
        if k != "301":
            S[f"fit_o{k}_max"] = np.nanmax(np.load(CACHE / f"video{r.video:02d}_fwd{k}.npy"), axis=1)
    P = pose_matrix(d)
    S["pca_video"] = pca_error(labeled_poses(r.unit, step), P)
    S["pca_mouse"] = pca_error(centre(mouse_train[r.mouse]), centre(P))
    (tx, ty), (bx, by), (lx, ly), (rx, ry) = [xy(d, b) for b in PUPIL]
    S["geo_axes_off"] = np.hypot((tx + bx) / 2 - (lx + rx) / 2, (ty + by) / 2 - (ly + ry) / 2) / w
    (_, ety), (_, eby) = xy(d, "eyelid_top"), xy(d, "eyelid_bottom")
    S["geo_outside"] = np.fmax(0, ety - ty) + np.fmax(0, by - eby)          # image y points down
    S["geo_width_dev"] = np.abs(W - w) / w
    S.update(ensemble_signals(r.unit, step, d))
    for k in ["fit_max", "fit_fb_max", "pca_video", "pca_mouse", "ens_sd_max", "ens_med_max", "prev_max", "geo_outside"]:
        S[k + "_rel"] = S[k] / w
    S["eye_width_px"] = np.full(T, w)
    return pd.DataFrame(S)


# ---------------------------------------------------------------- evaluation
def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def roc(score, truth):
    """Points (flagged share, recall, precision, threshold) for 'flag if score >= t' over every distinct threshold."""
    s, y = np.asarray(score, float), np.asarray(truth, bool)
    ok = np.isfinite(s); s, y = s[ok], y[ok]
    o = np.argsort(-s); s, y = s[o], y[o]
    tp = np.cumsum(y); n = np.arange(1, len(s) + 1)
    keep = np.r_[s[1:] != s[:-1], True]
    return pd.DataFrame({"threshold": s[keep], "flagged": n[keep] / len(s), "recall": tp[keep] / max(1, y.sum()),
                         "precision": tp[keep] / n[keep], "false_alarm": (n[keep] - tp[keep]) / max(1, (~y).sum())})


def operating_points(name, score, F):
    rows = []
    for w in ("wrong20", "wrong50"):
        R = roc(score, F[w])
        for target in ([0.5, 0.7, 0.9] if w == "wrong20" else [0.9]):
            q = R[R.recall >= target]
            if len(q):
                p = q.iloc[0]
                rows.append(dict(signal=name, truth=w, target=f"recall {int(target * 100)}%", threshold=round(p.threshold, 4),
                                 flagged_pct=round(100 * p.flagged, 1), recall_pct=round(100 * p.recall, 1), precision_pct=round(100 * p.precision, 1)))
        for share in (0.02, 0.05, 0.10):
            q = R[R.flagged <= share]
            if len(q):
                p = q.iloc[-1]
                rows.append(dict(signal=name, truth=w, target=f"flag {int(share * 100)}%", threshold=round(p.threshold, 4),
                                 flagged_pct=round(100 * p.flagged, 1), recall_pct=round(100 * p.recall, 1), precision_pct=round(100 * p.precision, 1)))
    return rows


def main():
    F = pd.read_csv(STEP8)
    jobs = [(int(r.video), r.unit, plateau_step(r.labels_at_plateau)) for r in PL.itertuples()]
    todo = [j for j in jobs if not (CACHE / f"video{j[0]:02d}_fwd311.npy").exists()]
    if todo:
        print(f"SARIMAX fits for {len(todo)} videos (4 processes) ...", flush=True)
        with Pool(4) as pool:
            for v in pool.imap_unordered(video_cache, todo):
                print(f"  video {v} cached", flush=True)
    mouse_train = {}
    for m, g in PL.groupby("mouse"):
        mouse_train[m] = np.concatenate([labeled_poses(r.unit, plateau_step(r.labels_at_plateau)) for r in g.itertuples()])
    parts = []
    for r in PL.itertuples():
        S = video_signals(r, mouse_train)
        g = F[F.video == int(r.video)]
        T = S.iloc[g.frame.to_numpy()].reset_index(drop=True)
        parts.append(pd.concat([g.reset_index(drop=True), T], axis=1))
        print(f"video {r.video}: {S['ens_models'].iat[0]:.0f} models in the ensemble, {len(labeled_poses(r.unit, plateau_step(r.labels_at_plateau)))} labeled poses for the video PCA", flush=True)
    G = pd.concat(parts, ignore_index=True)

    # combinations (S1-5)
    feats = ["fitting", "fit_max", "fit_bwd_max", "fit_ci", "jump", "min_conf", "mean_conf", "pca_mouse", "ens_sd_max", "ens_med_max",
             "geo_axes_off", "geo_outside", "geo_width_dev"]
    X = G[feats].copy()
    X["pca_video_f"] = G.pca_video.fillna(G.pca_mouse)
    feats = feats + ["pca_video_f"]
    for c in feats:
        if c not in ("min_conf", "mean_conf", "fit_ci", "geo_axes_off", "geo_width_dev"):
            X[c] = np.log1p(X[c].clip(lower=0))
    X = X.fillna(X.median())
    G["combo_lr"] = np.nan
    for v in G.video.unique():
        tr, te = G.video != v, G.video == v
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(sc.transform(X[tr]), G.wrong20[tr])
        G.loc[te, "combo_lr"] = clf.decision_function(sc.transform(X[te]))
    G["combo_rank"] = np.mean([G[c].rank(pct=True) for c in ("fit_max", "pca_mouse", "ens_sd_max")] + [(-G.min_conf).rank(pct=True)], axis=0)
    G.to_csv(OUT / "signals_all_frames.csv", index=False)

    SIG = [("jump", 1), ("min_conf", -1), ("mean_conf", -1), ("fitting", 1), ("fit_max", 1), ("fit_pupil", 1), ("fit_eye", 1), ("fit_ci", 1),
           ("fit_bwd_max", 1), ("fit_fb_max", 1)] + [(f"fit_o{k}_max", 1) for k in ORDERS if k != "301"] + \
          [("pca_video", 1), ("pca_mouse", 1), ("geo_axes_off", 1), ("geo_outside", 1), ("geo_width_dev", 1),
           ("ens_sd_max", 1), ("ens_sd_mean", 1), ("ens_sdw_max", 1), ("ens_med_max", 1), ("prev_max", 1),
           ("fit_max_rel", 1), ("fit_fb_max_rel", 1), ("pca_mouse_rel", 1), ("ens_sd_max_rel", 1), ("combo_lr", 1), ("combo_rank", 1)]
    rows, ops, curves = [], [], []
    for name, sg in SIG:
        score = sg * G[name]
        row = {"signal": name, "AUC_wrong20": round(auc(score[G.wrong20], score[~G.wrong20]), 3), "AUC_wrong50": round(auc(score[G.wrong50], score[~G.wrong50]), 3)}
        per = [auc(score[(G.video == v) & G.wrong20], score[(G.video == v) & ~G.wrong20]) for v in sorted(G.video.unique())]
        row["AUC_wrong20_videos_median"] = round(float(np.nanmedian(per)), 3)
        row["AUC_wrong20_videos_min"] = round(float(np.nanmin(per)), 3)
        rows.append(row)
        ops += operating_points(name, score, G)
        for w in ("wrong20", "wrong50"):
            R = roc(score, G[w]); R["signal"], R["truth"] = name, w
            curves.append(R)
    A = pd.DataFrame(rows).sort_values("AUC_wrong20", ascending=False)
    A.to_csv(OUT / "auc_per_signal.csv", index=False)
    O = pd.DataFrame(ops); O.to_csv(OUT / "operating_points.csv", index=False)
    pd.concat(curves).to_csv(OUT / "roc_curves.csv", index=False)
    print(A.to_string(index=False))
    print(O[O.target.isin(["recall 50%", "recall 90%", "flag 5%"])].to_string(index=False))

    # figure: ROC of the main signals at both truth levels + the operating-point table at 5% flagged
    show = ["fitting", "fit_max", "fit_fb_max", "fit_ci", "pca_video", "pca_mouse", "ens_sd_max", "ens_med_max", "geo_axes_off", "min_conf", "jump", "combo_lr"]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    for ax, w in zip(axes[:2], ("wrong20", "wrong50")):
        for name in show:
            sg = dict(SIG)[name]; R = roc(sg * G[name], G[w])
            a = A.set_index("signal").loc[name, f"AUC_{w}"]
            ax.plot(R.false_alarm, R.recall, lw=1.6 if name in ("fitting", "combo_lr") else 1.1, label=f"{name} ({a:.2f})")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("false-alarm rate (share of reliable frames flagged)"); ax.set_ylabel("recall (share of unreliable frames flagged)")
        ax.set_title(f"ROC, truth = frame RMSE > {w[5:]} px ({int(G[w].sum())} of {len(G)} frames); legend = AUC", fontsize=11)
        ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    ax = axes[2]; ax.axis("off")
    t = O[(O.target == "flag 5%") & (O.truth == "wrong20") & O.signal.isin(show)].set_index("signal").loc[show]
    t2 = O[(O.target == "flag 5%") & (O.truth == "wrong50") & O.signal.isin(show)].set_index("signal").loc[show]
    cell = [[n, f"{a:.0f}", f"{b:.0f}", f"{c:.0f}", f"{e:.0f}"] for n, a, b, c, e in zip(show, t.recall_pct, t.precision_pct, t2.recall_pct, t2.precision_pct)]
    tb = ax.table(cellText=cell, colLabels=["signal", "recall >20", "precision >20", "recall >50", "precision >50"], loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.4)
    ax.set_title("Each signal at a flagged share of 5% of the 1330 frames (%)", fontsize=11)
    fig.suptitle("Label-free signals for unreliable frames: 1330 test + validation frames of 19 videos, plateau models", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "sweep_overview.png", dpi=130); plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
