#!/usr/bin/env python3
"""Seed-disagreement signal for finding unreliable frames (pilot on one video).

Two models trained on the identical training set, differing only in the random seed, are run on the video's truth
frames (test50 + val20 PNGs). The distance between their predictions is the per-frame "disagreement" score; it is
compared with fit_pupil, the confidence and the heatmap entropy on the same frames. The main-line coordinates stay
those of the plateau model; the twin is a second opinion only. Nothing existing is modified; results go to
results/seed_disagreement/.

usage: seed_disagreement.py --video 8 --twin 3081 [--main 911]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_step import BPS, HERE, PROJECT, human_table  # noqa: E402
from heatmap_signals import plateau_shuffle, unit_of  # noqa: E402

OUT = HERE / "results" / "seed_disagreement"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = PROJECT / "dlc-models-pytorch" / "iteration-0"
PUPIL = BPS[:4]
DIS = ("dis_max_all", "dis_mean_all", "dis_max_pupil", "dis_mean_pupil")


def fidx(name):
    return int(re.findall(r"\d+", Path(str(name)).stem)[0])


def truth_images(unit):
    """The truth frames of a video: frame index, png path, human x/y per point (test50 + val20)."""
    rows = []
    for s in ("test50", "val20"):
        folder = HERE / unit / "training-data" / "labels" / s
        t = human_table(folder)
        t.columns = ["_".join(c[-2:]) for c in t.columns]
        for i, r in t.iterrows():
            name = i[-1] if isinstance(i, tuple) else Path(str(i)).name
            row = {"frame": fidx(name), "set": s, "png": str(folder / name)}
            for b in BPS:
                row[f"human_{b}_x"], row[f"human_{b}_y"] = r.get(f"{b}_x", np.nan), r.get(f"{b}_y", np.nan)
            rows.append(row)
    return pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)


def predict_images(shuffle, paths, device="mps", batch=16):
    from deeplabcut.pose_estimation_pytorch.apis.utils import get_pose_inference_runner
    folder = next(MODELS.glob(f"*shuffle{shuffle}"))
    snap = folder / "train" / "snapshot-120.pt"
    if not snap.exists():
        snap = next((folder / "train").glob("snapshot-best-120.pt"))
    runner = get_pose_inference_runner(model_config=folder / "train" / "pytorch_config.yaml", snapshot_path=snap,
                                       batch_size=batch, device=device,
                                       inference_cfg={"multithreading": {"enabled": False}})
    preds = runner.inference(images=list(paths))
    return np.stack([np.asarray(p["bodyparts"])[0, :, :3] for p in preds]), snap  # (N, K, 3)


def add_model(D, tag, P):
    for k, b in enumerate(BPS):
        D[f"{tag}_{b}_x"], D[f"{tag}_{b}_y"], D[f"{tag}_{b}_conf"] = P[:, k, 0], P[:, k, 1], P[:, k, 2]
    err = np.stack([np.hypot(D[f"{tag}_{b}_x"] - D[f"human_{b}_x"], D[f"{tag}_{b}_y"] - D[f"human_{b}_y"]) for b in BPS], 1)
    D[f"{tag}_rmse"] = np.sqrt(np.nanmean(err ** 2, axis=1))
    return D


def rates(score, truth, auc, roc):
    r = {}
    for w, y in truth.items():
        r[f"auc_{w}"] = auc(score[y], score[~y])
        R = roc(score, y)
        for f in (10, 20, 30):
            ok = R[R.flagged <= f / 100 + 1e-9]
            r[f"recall_{w}_flag{f}"] = ok.recall.max() if len(ok) else 0.0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=int, required=True)
    ap.add_argument("--twin", type=int, required=True, help="shuffle of the model trained with the other seed")
    ap.add_argument("--main", type=int, help="shuffle of the plateau model (default: the one step 8 used)")
    a = ap.parse_args()
    from unreliable_frames_sweep import auc, roc
    unit = unit_of(a.video)
    main_sh, step, h5 = plateau_shuffle(a.video)
    if a.main:
        main_sh = a.main
    D = truth_images(unit)
    print(f"video {a.video} {unit}: {len(D)} truth frames; main model shuffle {main_sh} (plateau step {step}), twin {a.twin}")

    Pm, snap_m = predict_images(main_sh, D.png)
    Pt, snap_t = predict_images(a.twin, D.png)
    print("main", snap_m, "\ntwin", snap_t)
    D = add_model(D, "main", Pm)
    D = add_model(D, "twin", Pt)

    # sanity: the main model's png predictions against its whole-video prediction file at the same frames
    H = pd.read_hdf(h5)
    H.columns = [f"{b}_{c}" for _, b, c in H.columns]
    H = H.iloc[D.frame.values]
    dxy = max(np.abs(H[f"{b}_{c}"].values - D[f"main_{b}_{c}"].values).max() for b in BPS for c in ("x", "y"))
    print(f"main model, png vs whole-video file at the {len(D)} frames: max |dx,dy| = {dxy:.2f} px")

    dist = np.stack([np.hypot(D[f"main_{b}_x"] - D[f"twin_{b}_x"], D[f"main_{b}_y"] - D[f"twin_{b}_y"]) for b in BPS], 1)
    pup = [BPS.index(b) for b in PUPIL]
    D["dis_max_all"], D["dis_mean_all"] = dist.max(1), dist.mean(1)
    D["dis_max_pupil"], D["dis_mean_pupil"] = dist[:, pup].max(1), dist[:, pup].mean(1)
    for k, b in enumerate(BPS):
        D[f"dis_{b}"] = dist[:, k]

    T = pd.read_csv(HERE / "results" / "unreliable_frames_sweep" / "signals_all_frames.csv")
    T = T[T.video == a.video][["frame", "frame_rmse_px", "wrong20", "wrong50", "fit_pupil", "fitting", "jump", "min_conf"]]
    E = D.merge(T, on="frame", how="inner")
    assert len(E) == len(T) == len(D), (len(E), len(T), len(D))
    hm = OUT.parent / "heatmap_signals" / f"pilot_video{a.video:02d}_frames.csv"
    if hm.exists():
        E = E.merge(pd.read_csv(hm)[["frame", "hm_entropy_mean_pupil"]], on="frame", how="left")
    print(f"RMSE vs human, median over the frames: main (step-8 table) {E.frame_rmse_px.median():.1f} px, main from png "
          f"{E.main_rmse.median():.1f}, twin {E.twin_rmse.median():.1f}; frames > 20 px: twin {(E.twin_rmse > 20).sum()}, "
          f"main {int(E.wrong20.sum())}")

    truth = {"wrong20": E.wrong20.astype(bool), "wrong50": E.wrong50.astype(bool)}
    sig = {"fit_pupil": E.fit_pupil, "fitting": E.fitting, "jump": E.jump, "min_conf": -E.min_conf}
    if "hm_entropy_mean_pupil" in E:
        sig["hm_entropy_mean_pupil"] = E.hm_entropy_mean_pupil
    for c in DIS:
        sig[c] = E[c]
    best = max(DIS, key=lambda c: auc(E[c][truth["wrong20"]], E[c][~truth["wrong20"]]))
    rk = lambda s: pd.Series(np.asarray(s, float)).rank(pct=True).values
    sig[f"rank_avg(fit_pupil,{best})"] = (rk(E.fit_pupil) + rk(E[best])) / 2
    sig[f"rank_avg(min_conf,{best})"] = (rk(-E.min_conf) + rk(E[best])) / 2
    if "hm_entropy_mean_pupil" in E:
        sig[f"rank_avg(fit_pupil,{best},hm_entropy)"] = (rk(E.fit_pupil) + rk(E[best]) + rk(E.hm_entropy_mean_pupil)) / 3
    A = pd.DataFrame([{"signal": n, **rates(pd.Series(np.asarray(s, float)), truth, auc, roc)} for n, s in sig.items()])

    rng = np.random.default_rng(0)
    comb = np.asarray(sig[f"rank_avg(fit_pupil,{best})"])
    fp = E.fit_pupil.values
    for w in ("wrong20", "wrong50"):
        y = truth[w].values
        g = []
        for _ in range(2000):
            i = rng.integers(0, len(E), len(E))
            yi = y[i]
            if yi.sum() == 0 or (~yi).sum() == 0:
                continue
            g.append(auc(comb[i][yi], comb[i][~yi]) - auc(fp[i][yi], fp[i][~yi]))
        g = np.array(g)
        print(f"{w}: AUC gain of rank_avg(fit_pupil,{best}) over fit_pupil = {g.mean():+.3f}, "
              f"95% interval [{np.percentile(g, 2.5):+.3f}, {np.percentile(g, 97.5):+.3f}], gain > 0 in {(g > 0).mean():.0%}")
    print("spearman", best, "vs fit_pupil", round(E[best].corr(E.fit_pupil, method="spearman"), 2),
          "vs -min_conf", round(E[best].corr(-E.min_conf, method="spearman"), 2))

    A.to_csv(OUT / f"pilot_video{a.video:02d}.csv", index=False, float_format="%.3f")
    E.to_csv(OUT / f"pilot_video{a.video:02d}_frames.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 220)
    print(f"truth: wrong20 {int(truth['wrong20'].sum())}, wrong50 {int(truth['wrong50'].sum())} of {len(E)}")
    print(A.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
