#!/usr/bin/env python3
"""Error modes of one video from the model's own output: cluster every frame, show each cluster, pick frames to label.

    python error_modes.py --video 8

Idea (user, 2026-10-26): instead of a fixed quota of frames per key point, let the data say how many ways the model
goes wrong. Every frame of the whole video is described only by what the model produced on it (no label is used):
  geometry   the 8 predicted points in the eye frame (origin = midpoint of the two corners, x axis along the corners,
             unit = eye width W), pupil width / height / aspect / axes offset, eye opening, all divided by W
  confidence the 8 confidences
  heatmap    per point: normalised entropy, second-peak ratio, number of peaks, mass near the peak
             (results/heatmap_signals/videoNN_shuffleSSSS.csv, the plateau model re-run on the whole video)
  fitting    per point: one-step SARIMAX deviation / W (results/unreliable_frames_sweep/cache/videoNN_fwd301.npy)
Features are standardised and reduced by PCA (95 % of the variance); k-means with k chosen by the silhouette on a
20 000-frame subsample (k = 4 … 14). Per cluster: size, mean suspicion (rank average of fit_pupil and the mean pupil
heatmap entropy), the frames of the 70 truth frames that fall in it and how many of them are wrong (> 20 px, > 50 px,
area error > 20 %) — the truth frames are used only to JUDGE the clusters, never to build them or to pick frames —
and the four features whose cluster mean deviates most from the video mean. One contact sheet per cluster (20 frames:
10 nearest to the centre, 10 random; red = model; green ring = human label where the frame is a truth frame).
Frames to label (label-free, only from the training pool = the video minus its validation and test windows): quota per
cluster proportional to size × suspicion, at least 1, at most 4, 20 in all; inside a cluster the quota frames are the
medoids of a k-means with k = quota (spread over the cluster), not the most extreme frames.

Outputs in results/error_modes/videoNN/: clusters_summary.csv, cluster_features.csv, selection.csv, frames_clusters.csv
(all frames, local only), sheets cluster_KK.jpg (local only). Nothing else is modified."""
import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pupil_trace as pt  # noqa: E402
from scale_step import HERE  # noqa: E402

RES = HERE / "results"
PL = pd.read_csv(RES / "labels_to_plateau.csv")
BPS = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right", "eyelid_top", "eyelid_bottom", "eye_nasal_corner", "eye_temporal_corner"]
P4 = BPS[:4]
HM = ["entropy", "ratio", "n_peaks", "mass_near"]
SEED = 20261026
TILE_W, TILE_H, COLS, ROWS = 300, 225, 5, 4
RED, GREEN = (255, 60, 60), (60, 255, 60)
try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    FONT_B = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
except OSError:
    FONT = FONT_B = ImageFont.load_default()


def load_inputs(video):
    rr = PL[PL.video == video].iloc[0]
    step = max(1, int(rr.labels_at_plateau) // 20)
    h5 = sorted(glob.glob(str(HERE / rr.unit / "training-data" / f"predictions_step{step:02d}" / "*snapshot*120*.h5")))[-1]
    d = pt.load(h5)
    hm = sorted(glob.glob(str(RES / "heatmap_signals" / f"video{video:02d}_shuffle*.csv")))
    if not hm:
        raise SystemExit(f"no heatmap file for video {video}: run heatmap_signals.py first")
    H = pd.read_csv(hm[-1]).set_index("frame").sort_index()
    fwd = np.load(RES / "unreliable_frames_sweep" / "cache" / f"video{video:02d}_fwd301.npy")
    assert len(d) == len(H) == len(fwd), (len(d), len(H), len(fwd))
    truth = pd.read_csv(RES / "error_detector" / "scores_lovo.csv")
    truth = truth[truth.plateau & (truth.video == video)].drop_duplicates("frame").set_index("frame")
    truth["area_err_pct"] = (np.pi / 4 * np.abs(truth.pred_pupil_right_x - truth.pred_pupil_left_x) * np.abs(truth.pred_pupil_bottom_y - truth.pred_pupil_top_y)
                             - np.pi / 4 * np.abs(truth.human_pupil_right_x - truth.human_pupil_left_x) * np.abs(truth.human_pupil_bottom_y - truth.human_pupil_top_y)).abs() \
        / (np.pi / 4 * np.abs(truth.human_pupil_right_x - truth.human_pupil_left_x) * np.abs(truth.human_pupil_bottom_y - truth.human_pupil_top_y)) * 100
    return rr, d, H, fwd, truth


def features(d, H, fwd):
    xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
    N, T = xy["eye_nasal_corner"], xy["eye_temporal_corner"]
    C = (N + T) / 2
    u = T - N
    W = np.linalg.norm(u, axis=1)
    Wm = np.nanmedian(W)
    u = u / W[:, None]
    v = np.c_[-u[:, 1], u[:, 0]]
    F = {}
    for b in BPS:
        r = xy[b] - C
        F[f"geo_{b}_along"], F[f"geo_{b}_across"] = (r * u).sum(1) / Wm, (r * v).sum(1) / Wm
    F["geo_pupil_w"] = np.linalg.norm(xy["pupil_right"] - xy["pupil_left"], axis=1) / Wm
    F["geo_pupil_h"] = np.linalg.norm(xy["pupil_bottom"] - xy["pupil_top"], axis=1) / Wm
    F["geo_pupil_aspect"] = F["geo_pupil_h"] / np.maximum(F["geo_pupil_w"], 1e-3)
    F["geo_axes_off"] = np.linalg.norm((xy["pupil_right"] + xy["pupil_left"]) / 2 - (xy["pupil_bottom"] + xy["pupil_top"]) / 2, axis=1) / Wm
    F["geo_opening"] = ((xy["eyelid_bottom"] - xy["eyelid_top"]) * v).sum(1) / Wm
    F["geo_eye_width"] = W / Wm
    for b in BPS:
        F[f"conf_{b}"] = d[b]["likelihood"].to_numpy(float)
        for s in HM:
            x = H[f"{b}_{s}"].to_numpy(float)
            F[f"hm_{b}_{s}"] = np.log1p(x) if s == "n_peaks" else x
    for k, b in enumerate(BPS):
        F[f"fit_{b}"] = np.log1p(fwd[:, k] / Wm * 100)
    X = pd.DataFrame(F)
    X = X.fillna(X.median())
    lo, hi = X.quantile(0.005), X.quantile(0.995)
    return X.clip(lo, hi, axis=1), Wm, C, W


def suspicion(X, fwd):
    fit_pupil = np.nanmean(fwd[:, :4], axis=1)
    ent = X[[f"hm_{b}_entropy" for b in P4]].mean(1).to_numpy()
    r = lambda a: pd.Series(a).rank(pct=True).to_numpy()
    return (r(fit_pupil) + r(ent)) / 2, fit_pupil, ent


def cluster(X):
    Z = StandardScaler().fit_transform(X)
    P = PCA(n_components=0.95, random_state=SEED).fit_transform(Z)
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(P), min(20000, len(P)), replace=False)
    best = None
    rows = []
    for k in range(4, 15):
        km = KMeans(k, n_init=5, random_state=SEED).fit(P[sub])
        s = silhouette_score(P[sub], km.labels_, sample_size=10000, random_state=SEED)
        rows.append({"k": k, "silhouette": round(float(s), 4)})
        if best is None or s > best[0]:
            best = (s, k)
    k = best[1]
    km = KMeans(k, n_init=10, random_state=SEED).fit(P)
    dist = np.linalg.norm(P - km.cluster_centers_[km.labels_], axis=1)
    return km.labels_, dist, pd.DataFrame(rows), Z, P


def read_frame(cap, f):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
    ok, img = cap.read()
    assert ok, f
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def tile(cap, f, d, C, W, truth, label, cap_text):
    img = Image.fromarray(read_frame(cap, f))
    cx, cy = C[f]
    w = max(1.7 * W[f], 240)
    h = w * TILE_H / TILE_W
    box = (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))
    crop = img.crop(box).resize((TILE_W, TILE_H))
    sc = TILE_W / (box[2] - box[0])
    dr = ImageDraw.Draw(crop)
    for b in BPS:
        px, py = (d[b]["x"].iloc[f] - box[0]) * sc, (d[b]["y"].iloc[f] - box[1]) * sc
        if np.isfinite(px) and np.isfinite(py):
            dr.ellipse([px - 4, py - 4, px + 4, py + 4], fill=RED)
        if f in truth.index:
            hx, hy = (truth.loc[f, f"human_{b}_x"] - box[0]) * sc, (truth.loc[f, f"human_{b}_y"] - box[1]) * sc
            if np.isfinite(hx) and np.isfinite(hy):
                dr.ellipse([hx - 5, hy - 5, hx + 5, hy + 5], outline=GREEN, width=3)
    dr.rectangle([0, 0, TILE_W, 18], fill=(0, 0, 0))
    dr.text((4, 1), cap_text, fill=(255, 255, 255), font=FONT)
    return crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=int, required=True)
    ap.add_argument("--n-select", type=int, default=20)
    ap.add_argument("--fps", type=float, default=60.0)
    a = ap.parse_args()
    out = RES / "error_modes" / f"video{a.video:02d}"
    out.mkdir(parents=True, exist_ok=True)
    rr, d, H, fwd, truth = load_inputs(a.video)
    X, Wm, C, W = features(d, H, fwd)
    susp, fit_pupil, ent = suspicion(X, fwd)
    labels, dist, ks, Z, Pc = cluster(X)
    npc = Pc.shape[1]
    ks.to_csv(out / "silhouette_by_k.csv", index=False)
    k = labels.max() + 1
    print(f"video {a.video}: {len(X)} frames, {X.shape[1]} features, {npc} PCs, k = {k}")
    print(ks.to_string(index=False))

    frames = pd.DataFrame({"frame": np.arange(len(X)), "cluster": labels, "dist_to_centre": dist, "suspicion": susp,
                           "fit_pupil_px": fit_pupil, "pupil_entropy_mean": ent})
    frames["truth"] = frames.frame.isin(truth.index)
    frames["minute"] = (frames.frame / 3600).round(2)
    frames["frame_rmse_px"] = frames.frame.map(truth.frame_rmse_px)
    frames["area_err_pct"] = frames.frame.map(truth.area_err_pct)


    Zd = pd.DataFrame(Z, columns=X.columns)
    rows, frows = [], []
    for c in range(k):
        m = labels == c
        t = frames[m & frames.truth]
        zc = Zd[m].mean()
        top = zc.abs().sort_values(ascending=False).index[:4]
        rows.append({"cluster": c, "frames": int(m.sum()), "share_pct": round(100 * m.mean(), 1), "mean_suspicion": round(float(susp[m].mean()), 3),
                     "fit_pupil_median_px": round(float(np.nanmedian(fit_pupil[m])), 1), "pupil_entropy_median": round(float(np.nanmedian(ent[m])), 3),
                     "truth_frames": len(t), "truth_gt20px": int((t.frame_rmse_px > 20).sum()), "truth_gt50px": int((t.frame_rmse_px > 50).sum()),
                     "truth_area_gt20pct": int((t.area_err_pct > 20).sum()), "truth_rmse_median_px": round(float(t.frame_rmse_px.median()), 1) if len(t) else np.nan,
                     "top_features": "; ".join(f"{f} {zc[f]:+.1f}" for f in top)})
        for f in X.columns:
            frows.append({"cluster": c, "feature": f, "mean_z": round(float(zc[f]), 3), "mean_raw": round(float(X[f][m].mean()), 4)})
    S = pd.DataFrame(rows)
    S.to_csv(out / "clusters_summary.csv", index=False)
    pd.DataFrame(frows).to_csv(out / "cluster_features.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.drop(columns="top_features").to_string(index=False))
    for r in S.itertuples():
        print(f"cluster {r.cluster}: {r.top_features}")

    # label-free selection: quota per cluster proportional to size x suspicion (1..4); inside a cluster the quota
    # frames are the medoids of a k-means with k = quota on the same PCA features (spread over the cluster), not the
    # most extreme frames (those are the sporadic big errors, which a label does not teach the model much about)
    # only frames of the training pool may be labeled (select_frames.py: the last 10 % of the video is the test window,
    # the 10 % before it the validation window, 2-s guards): pool_hi = n - 2 * seg - 2 * guard
    n = len(X)
    seg = int(round(max(0.10 * n / a.fps, 60.0) * a.fps))
    guard = int(round(2.0 * a.fps))
    pool_hi = n - 2 * seg - 2 * guard
    frames["in_pool"] = frames.frame < pool_hi
    print(f"pool = frames < {pool_hi} ({pool_hi / a.fps / 60:.1f} min); cluster sizes inside the pool:",
          frames[frames.in_pool].cluster.value_counts().sort_index().to_dict())
    wgt = S.frames * S.mean_suspicion
    quota = np.maximum(1, np.round(wgt / wgt.sum() * a.n_select)).astype(int)
    quota = np.minimum(quota, 4)
    while quota.sum() > a.n_select:
        quota[quota.argmax()] -= 1
    while quota.sum() < a.n_select:
        quota[(wgt / quota).argmax()] += 1
    sel = []
    for c in range(k):
        idx = np.flatnonzero((labels == c) & frames.in_pool.to_numpy())
        q = int(quota[c])
        if q == 1 or len(idx) <= q:
            picks = [int(frames.iloc[idx].sort_values("dist_to_centre").frame.iloc[0])]
        else:
            sub = KMeans(q, n_init=5, random_state=SEED).fit(Pc[idx])
            picks = []
            for j in range(q):
                mem = idx[sub.labels_ == j]
                dj = np.linalg.norm(Pc[mem] - sub.cluster_centers_[j], axis=1)
                picks.append(int(mem[dj.argmin()]))
        sel += [{"frame": f, "cluster": c, "reason": "medoid of sub-cluster"} for f in picks]
    SEL = pd.DataFrame(sel).merge(frames[["frame", "suspicion", "fit_pupil_px", "pupil_entropy_mean", "in_pool"]], on="frame").sort_values(["cluster", "frame"])
    assert SEL.in_pool.all() and len(SEL) == a.n_select
    SEL["minute"] = (SEL.frame / 3600).round(1)
    SEL.round(3).to_csv(out / "selection.csv", index=False)
    frames.round(4).to_csv(out / "frames_clusters.csv", index=False)
    print(SEL.to_string(index=False))

    # contact sheets
    cap = cv2.VideoCapture(str(HERE / rr.unit / f"{rr.unit}.mp4"))
    # check that mp4 frame f equals the extracted PNG of a truth frame
    f0 = int(truth.index[0])
    png = glob.glob(str(HERE / rr.unit / "training-data" / "labels" / truth.loc[f0, "set"] / f"img{f0:06d}.png"))[0]
    diff = np.abs(read_frame(cap, f0).astype(float) - np.array(Image.open(png).convert("RGB")).astype(float)).mean()
    print(f"mp4 vs png check on frame {f0}: mean abs diff {diff:.2f}")
    assert diff < 6, "mp4 seek does not reproduce the extracted frame"
    rng = np.random.default_rng(SEED)
    for c in range(k):
        g = frames[frames.cluster == c]
        near = g.sort_values("dist_to_centre").frame.iloc[:COLS * ROWS // 2].tolist()
        rest = g[~g.frame.isin(near)].frame.to_numpy()
        rand = rng.choice(rest, min(COLS * ROWS - len(near), len(rest)), replace=False).tolist() if len(rest) else []
        sheet = Image.new("RGB", (COLS * TILE_W, ROWS * TILE_H + 40), (30, 30, 30))
        dr = ImageDraw.Draw(sheet)
        r = S.iloc[c]
        dr.text((8, 8), f"video {a.video}  cluster {c}: {r.frames} frames ({r.share_pct}%), suspicion {r.mean_suspicion}, truth {r.truth_frames} frames / {r.truth_gt20px} > 20 px.  "
                        f"row 1-2 nearest to centre, row 3-4 random.  red = model, green ring = human label", fill=(255, 255, 255), font=FONT_B)
        for j, f in enumerate(near + rand):
            txt = f"f{f}  fit {fit_pupil[f]:.0f}px  ent {ent[f]:.2f}" + (f"  RMSE {truth.loc[f, 'frame_rmse_px']:.0f}" if f in truth.index else "")
            sheet.paste(tile(cap, f, d, C, W, truth, labels, txt), ((j % COLS) * TILE_W, 40 + (j // COLS) * TILE_H))
        sheet.save(out / f"cluster_{c:02d}.jpg", quality=88)
    print("sheets written to", out)


if __name__ == "__main__":
    main()
