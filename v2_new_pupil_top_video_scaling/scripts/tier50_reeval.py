#!/usr/bin/env python3
"""Re-evaluation of every unreliable-frame signal and detector at the 50-px truth tier (decision of 2026-10-26).

    python tier50_reeval.py

Inputs (nothing recomputed, nothing modified): the per-frame scores of the sweep (results/unreliable_frames_sweep/
signals_all_frames.csv, 1330 frames), of the tabular detector D1 (results/error_detector/scores_lovo.csv, leave-one-
video-out scores) and of the image detector D2 (results/error_detector/scores_image.csv). Truth: frame RMSE > 50 px
(32 frames), > 20 px shown next to it for reference.

Per score: AUC at 50 px with a 95 % bootstrap interval over the 1330 frames (2000 resamples; 32 positives make
the AUC noisy), recall / precision at 1, 2, 3, 5 and 9.2 % flagged with leave-one-video-out thresholds (the threshold
for a video is the quantile of the score on the other 18 videos), the pooled flagged share needed for 50 / 75 / 90 %
recall, and per-video counts of caught > 50 px frames for the best scores. Rank averages of the best score with
fit_pupil are tried as combinations.

Outputs in results/tier50_reeval/: signals_50px.csv (one row per score), caught_per_video.csv, combos.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unreliable_frames_sweep import auc, roc  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
OUT = RES / "tier50_reeval"
SHARES = [0.01, 0.02, 0.03, 0.05, 0.092]
NEG = {"min_conf", "mean_conf"}          # lower = more suspicious
SEED = 20261026


def load():
    S = pd.read_csv(RES / "unreliable_frames_sweep" / "signals_all_frames.csv")
    sig_cols = [c for c in S.columns if c not in {"video", "unit", "set", "frame", "model_step", "labeled_points", "frame_rmse_px",
                                                   "wrong50", "wrong20", "eye_width_px"} and not c.endswith("_rule")]
    L = pd.read_csv(RES / "error_detector" / "scores_lovo.csv")
    L = L[L.plateau].drop_duplicates(["video", "frame"])
    d1_cols = [c for c in L.columns if c.startswith("d1_")]
    I = pd.read_csv(RES / "error_detector" / "scores_image.csv")
    I = I[I.plateau].drop_duplicates(["video", "frame"])
    d2_cols = [c for c in I.columns if c.startswith("d2_") or c == "d1_pairs_all_5fold"]
    M = S.merge(L[["video", "frame"] + d1_cols], on=["video", "frame"], how="left").merge(I[["video", "frame"] + d2_cols], on=["video", "frame"], how="left")
    assert len(M) == 1330 and M.wrong50.sum() == 32 and M.wrong20.sum() == 155
    return M, sig_cols + d1_cols + d2_cols


def lovo_flags(M, s, share):
    flag = np.zeros(len(M), bool)
    for v in M.video.unique():
        k = (M.video == v).to_numpy()
        thr = np.nanquantile(s[~k], 1 - share)
        flag[k] = s[k] > thr
    return flag


def boot_auc(s, y, n=2000):
    rng = np.random.default_rng(SEED)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    idx_p, idx_n = np.flatnonzero(y), np.flatnonzero(~y)
    vals = []
    for _ in range(n):
        p = s[rng.choice(idx_p, len(idx_p))]
        q = s[rng.choice(idx_n, len(idx_n))]
        vals.append(auc(p, q))
    return np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5)


def evaluate(M, name, s):
    y50, y20 = M.wrong50.to_numpy(bool), M.wrong20.to_numpy(bool)
    r = {"signal": name, "n_scored": int(np.isfinite(s).sum()), "auc50": auc(s[y50], s[~y50]), "auc20": auc(s[y20], s[~y20])}
    r["auc50_lo"], r["auc50_hi"] = boot_auc(s, y50)
    for sh in SHARES:
        f = lovo_flags(M, s, sh)
        tag = f"{sh * 100:g}pct"
        r[f"flagged_{tag}"] = round(100 * f.mean(), 1)
        r[f"caught50_{tag}"] = int((f & y50).sum())
        r[f"recall50_{tag}"] = round(100 * (f & y50).sum() / y50.sum(), 1)
        r[f"prec50_{tag}"] = round(100 * (f & y50).sum() / max(1, f.sum()), 1)
        r[f"recall20_{tag}"] = round(100 * (f & y20).sum() / y20.sum(), 1)
    R = roc(s, y50)
    for rc in (0.5, 0.75, 0.9):
        ok = R[R.recall >= rc]
        r[f"flagged_for_recall50_{int(rc * 100)}"] = round(100 * ok.flagged.min(), 1) if len(ok) else np.nan
    return r


def main():
    OUT.mkdir(exist_ok=True)
    M, cols = load()
    rows = []
    for c in cols:
        s = M[c].to_numpy(float)
        if c in NEG:
            s = -s
        if np.isfinite(s).sum() < 1000:
            continue
        rows.append(evaluate(M, c, s))
    T = pd.DataFrame(rows).sort_values(["recall50_5pct", "auc50"], ascending=False)
    T.round(3).to_csv(OUT / "signals_50px.csv", index=False)
    pd.set_option("display.width", 250)
    show = ["signal", "auc50", "auc50_lo", "auc50_hi", "auc20", "recall50_1pct", "recall50_2pct", "recall50_5pct", "prec50_5pct", "recall50_9.2pct",
            "flagged_for_recall50_75", "flagged_for_recall50_90"]
    print(T[show].round(3).to_string(index=False))

    # per-video caught at 5 % for the top 6 signals
    top = T.signal.head(6).tolist()
    y50 = M.wrong50.to_numpy(bool)
    rows = []
    for v in sorted(M.video.unique()):
        k = (M.video == v).to_numpy()
        row = {"video": v, "frames_gt50": int((y50 & k).sum())}
        for c in top:
            s = M[c].to_numpy(float)
            if c in NEG:
                s = -s
            f = lovo_flags(M, s, 0.05)
            row[c] = int((f & y50 & k).sum())
        rows.append(row)
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "caught_per_video.csv", index=False)
    print(C.to_string(index=False))

    # combinations: rank average of fit_pupil with each of the other top signals
    rows = []
    base = M["fit_pupil"].rank(pct=True)
    for c in [t for t in T.signal.head(10) if t != "fit_pupil"]:
        s = M[c]
        if c in NEG:
            s = -s
        comb = ((base + s.rank(pct=True)) / 2).to_numpy()
        r = evaluate(M, f"rank_avg(fit_pupil,{c})", comb)
        rows.append(r)
    K = pd.DataFrame(rows).sort_values(["recall50_5pct", "auc50"], ascending=False)
    K.round(3).to_csv(OUT / "combos.csv", index=False)
    print(K[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
