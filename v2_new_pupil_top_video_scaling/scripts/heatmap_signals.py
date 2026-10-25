#!/usr/bin/env python3
"""Heatmap-shape signals for finding unreliable frames (pilot, section 8.2 follow-up).

The saved prediction files keep one coordinate and one confidence per key point; the shape of the network's heatmap
(a second peak, a flat map) is lost. This script re-runs the plateau model of one video with the standard DLC
inference runner and, inside the predictor, records a few numbers per frame and key point before the coordinates
are regressed. Heatmaps themselves are not stored. Nothing existing is modified: results go to
results/heatmap_signals/.

Per key point and frame (heatmap as the predictor sees it, negative values set to 0; stride s cells -> pixels):
  x, y, conf      the regressed prediction (must equal the saved .h5 of the same model)
  peak1           highest heatmap value (= conf before clipping to [0, 1])
  peak2           highest local maximum at least 3 cells away from peak1
  ratio           peak2 / peak1
  dist_px         distance between the two peaks in pixels
  n_peaks         local maxima above 0.3 * peak1
  entropy         entropy of the normalised heatmap divided by log(cells) (0 = one cell, 1 = flat)
  mass_near       share of the heatmap mass within 12 px of peak1

usage: heatmap_signals.py --video 8 [--shuffle 911] [--limit 300] [--batch 32] [--eval]
"""
import argparse
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_step import BPS, HERE, PROJECT  # noqa: E402

OUT = HERE / "results" / "heatmap_signals"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = PROJECT / "dlc-models-pytorch" / "iteration-0"
STATS = ["peak1", "peak2", "ratio", "dist_px", "n_peaks", "entropy", "mass_near"]


def unit_of(video):
    return next(p.name for p in HERE.iterdir() if p.is_dir() and re.match(rf"{video}_", p.name))


def plateau_shuffle(video):
    """Shuffle of the model whose whole-video prediction step 8 used (results/labels_to_plateau.csv + the
    unreliable-frames table)."""
    T = pd.read_csv(HERE / "results" / "unreliable_frames" / "unreliable_frames_all_videos.csv")
    row = T[T.video == video].iloc[0]
    step = int(row.model_step)
    unit = unit_of(video)
    h5 = sorted((HERE / unit / "training-data" / f"predictions_step{step:02d}").glob("*snapshot*120*.h5"))
    return int(re.search(r"shuffle(\d+)_", h5[-1].name).group(1)), step, h5[-1]


def heatmap_stats(H, stride):
    """H: (B, K, h, w) heatmaps after sigmoid. Returns (B, K, len(STATS)) float32 on CPU."""
    B, K, h, w = H.shape
    flat = H.reshape(B, K, -1)
    peak1, i1 = flat.max(dim=2)
    y1, x1 = (i1 // w).float(), (i1 % w).float()
    yy = torch.arange(h, device=H.device).view(1, 1, h, 1).float()
    xx = torch.arange(w, device=H.device).view(1, 1, 1, w).float()
    d = torch.sqrt((yy - y1.view(B, K, 1, 1)) ** 2 + (xx - x1.view(B, K, 1, 1)) ** 2)  # cells from peak1
    lm = H == Fn.max_pool2d(H, 3, stride=1, padding=1)  # local maxima
    far = d >= 3
    cand = torch.where(lm & far, H, torch.zeros_like(H)).reshape(B, K, -1)
    peak2, i2 = cand.max(dim=2)
    y2, x2 = (i2 // w).float(), (i2 % w).float()
    dist_px = torch.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) * stride
    dist_px = torch.where(peak2 > 0, dist_px, torch.full_like(dist_px, float("nan")))
    n_peaks = (lm & (H > 0.3 * peak1.view(B, K, 1, 1))).reshape(B, K, -1).sum(dim=2).float()
    P = flat / flat.sum(dim=2, keepdim=True).clamp_min(1e-12)
    entropy = -(P * torch.log(P.clamp_min(1e-12))).sum(dim=2) / math.log(h * w)
    near = (d * stride <= 12.0).reshape(B, K, -1)
    mass_near = (flat * near).sum(dim=2) / flat.sum(dim=2).clamp_min(1e-12)
    return torch.stack([peak1, peak2, peak2 / peak1.clamp_min(1e-6), dist_px, n_peaks, entropy, mass_near], dim=2).cpu().numpy()


def build_runner(shuffle, batch, device, store):
    from deeplabcut.pose_estimation_pytorch.apis.utils import get_pose_inference_runner
    folder = next(MODELS.glob(f"*shuffle{shuffle}"))
    snap = folder / "train" / "snapshot-120.pt"
    if not snap.exists():
        snap = next((folder / "train").glob("snapshot-best-120.pt"))
    runner = get_pose_inference_runner(model_config=folder / "train" / "pytorch_config.yaml", snapshot_path=snap,
                                       batch_size=batch, device=device,
                                       inference_cfg={"multithreading": {"enabled": False}})
    pred = runner.model.heads["bodypart"].predictor
    orig = pred.forward

    def forward(stride, outputs):
        with torch.no_grad():
            H = outputs["heatmap"]
            if pred.apply_sigmoid:
                H = torch.sigmoid(H)
            # this model's predictor does not apply a sigmoid (clip_scores instead): the raw map can be slightly
            # negative or above 1, so negative values are set to 0 before the mass and entropy are computed
            store.append(heatmap_stats(H.float().clamp(min=0), float(stride)))
        return orig(stride, outputs)

    pred.forward = forward
    return runner, snap


def frames(video_path, limit):
    from deeplabcut.pose_estimation_pytorch.apis.videos import VideoIterator
    it = VideoIterator(str(video_path))
    n = it.get_n_frames()
    for i, f in enumerate(it):
        if limit and i >= limit:
            break
        yield f
    print("frames in video:", n)


def run(video, shuffle, limit, batch, device):
    unit = unit_of(video)
    video_path = HERE / unit / f"{unit}.mp4"
    store = []
    runner, snap = build_runner(shuffle, batch, device, store)
    print("model", snap)
    t0 = time.time()
    preds = []
    done = 0
    for p in runner.inference(images=frames(video_path, limit)):
        preds.append(np.asarray(p["bodyparts"])[0, :, :3])  # (K, 3): x, y, conf of the single individual
        done += 1
        if done % 5000 == 0:
            print(f"{done} frames  {(time.time() - t0) / 60:.1f} min", flush=True)
    poses = np.stack(preds)  # (N, K, 3)
    S = np.concatenate(store)[: len(poses)]  # (N, K, len(STATS))
    assert len(S) == len(poses), (len(S), len(poses))
    cols = {"frame": np.arange(len(poses))}
    for k, bp in enumerate(BPS):
        cols[f"{bp}_x"], cols[f"{bp}_y"], cols[f"{bp}_conf"] = poses[:, k, 0], poses[:, k, 1], poses[:, k, 2]
        for j, s in enumerate(STATS):
            cols[f"{bp}_{s}"] = S[:, k, j]
    D = pd.DataFrame(cols)
    tag = f"video{video:02d}_shuffle{shuffle}" + (f"_first{limit}" if limit else "")
    D.to_csv(OUT / f"{tag}.csv", index=False, float_format="%.4f")
    print(f"written {OUT / f'{tag}.csv'}  {len(D)} frames  {(time.time() - t0) / 60:.1f} min")
    return D


def check_against_saved(D, h5):
    """The regressed coordinates must match the whole-video prediction file of the same model."""
    P = pd.read_hdf(h5)
    P.columns = [f"{b}_{c}" for _, b, c in P.columns]
    P = P.iloc[: len(D)]
    dxy = max(np.abs(P[f"{bp}_{c}"].values - D[f"{bp}_{c}"].values).max() for bp in BPS for c in ("x", "y"))
    dcf = max(np.abs(P[f"{bp}_likelihood"].values - D[f"{bp}_conf"].values).max() for bp in BPS)
    print(f"check against {h5.name}: max |dx,dy| = {dxy:.3f} px, max |dconf| = {dcf:.4f}")
    return dxy, dcf


def frame_summaries(D):
    """Frame-level scores from the per-point numbers (higher = more suspicious)."""
    pup, eye = BPS[:4], BPS[4:]
    S = pd.DataFrame({"frame": D.frame})
    for name, bps in (("pupil", pup), ("all", BPS)):
        S[f"hm_ratio_max_{name}"] = D[[f"{b}_ratio" for b in bps]].max(axis=1)
        S[f"hm_ratio_mean_{name}"] = D[[f"{b}_ratio" for b in bps]].mean(axis=1)
        S[f"hm_entropy_mean_{name}"] = D[[f"{b}_entropy" for b in bps]].mean(axis=1)
        S[f"hm_entropy_max_{name}"] = D[[f"{b}_entropy" for b in bps]].max(axis=1)
        S[f"hm_mass_min_{name}"] = -D[[f"{b}_mass_near" for b in bps]].min(axis=1)
        S[f"hm_npeaks_sum_{name}"] = D[[f"{b}_n_peaks" for b in bps]].sum(axis=1)
        S[f"hm_peak1_min_{name}"] = -D[[f"{b}_peak1" for b in bps]].min(axis=1)
        S[f"hm_ratio_dist_{name}"] = (D[[f"{b}_ratio" for b in bps]].values
                                     * np.nan_to_num(D[[f"{b}_dist_px" for b in bps]].values, nan=0)).max(axis=1)
    return S


def evaluate(video, D):
    from unreliable_frames_sweep import auc, roc
    T = pd.read_csv(HERE / "results" / "unreliable_frames_sweep" / "signals_all_frames.csv")
    T = T[T.video == video].copy()
    S = frame_summaries(D)
    E = T.merge(S, on="frame", how="inner")
    assert len(E) == len(T), (len(E), len(T))
    ref = {"fit_pupil": 1, "fitting": 1, "jump": 1, "min_conf": -1}
    rows = []
    hm_cols = [c for c in S.columns if c.startswith("hm_")]
    for col, sg in list(ref.items()) + [(c, 1) for c in hm_cols]:
        s = sg * E[col]
        r = {"signal": col}
        for w in ("wrong20", "wrong50"):
            r[f"auc_{w}"] = auc(s[E[w]], s[~E[w]])
            R = roc(s, E[w])
            for f in (5, 10, 20):
                ok = R[R.flagged <= f / 100 + 1e-9]
                r[f"recall_{w}_flag{f}"] = ok.recall.max() if len(ok) else 0.0
        rows.append(r)
    A = pd.DataFrame(rows)
    # rank average of fit_pupil with the best heatmap score (by AUC wrong20)
    best = A[A.signal.str.startswith("hm_")].sort_values("auc_wrong20", ascending=False).signal.iloc[0]
    comb = (E.fit_pupil.rank(pct=True) + E[best].rank(pct=True)) / 2
    r = {"signal": f"rank_avg(fit_pupil,{best})"}
    for w in ("wrong20", "wrong50"):
        r[f"auc_{w}"] = auc(comb[E[w]], comb[~E[w]])
        R = roc(comb, E[w])
        for f in (5, 10, 20):
            ok = R[R.flagged <= f / 100 + 1e-9]
            r[f"recall_{w}_flag{f}"] = ok.recall.max() if len(ok) else 0.0
    A = pd.concat([A, pd.DataFrame([r])], ignore_index=True)
    A.to_csv(OUT / f"pilot_video{video:02d}.csv", index=False, float_format="%.3f")
    E.to_csv(OUT / f"pilot_video{video:02d}_frames.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(f"video {video}: {len(E)} truth frames, wrong20 {int(E.wrong20.sum())}, wrong50 {int(E.wrong50.sum())}")
    print(A.round(3).to_string(index=False))
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=int, required=True)
    ap.add_argument("--shuffle", type=int)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--eval", action="store_true", help="only evaluate an existing full-video csv")
    a = ap.parse_args()
    shuffle, step, h5 = plateau_shuffle(a.video)
    if a.shuffle:
        shuffle = a.shuffle
    print(f"video {a.video}  unit {unit_of(a.video)}  plateau step {step}  shuffle {shuffle}")
    if a.eval:
        D = pd.read_csv(OUT / f"video{a.video:02d}_shuffle{shuffle}.csv")
    else:
        D = run(a.video, shuffle, a.limit, a.batch, a.device)
    check_against_saved(D, h5)
    if not a.limit:
        evaluate(a.video, D)


if __name__ == "__main__":
    main()
