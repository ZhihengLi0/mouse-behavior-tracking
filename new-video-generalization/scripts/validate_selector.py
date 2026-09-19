#!/usr/bin/env python3
"""Label-free validation of the batch-1 selector on a video whose eye state
is known (old video: production-model predictions, 98% of points confident).

Question: do the 20 k-means picks cover the real range of eye states better
than (a) 20 frames evenly spaced in time and (b) 20 random frames?

Eye state per frame (from predictions): pupil area, eye opening, pupil
center x, pupil center y. Coverage metrics over the training pool:
  decile coverage   fraction of the 10 quantile bins of each variable that
                    contain at least one pick (averaged over 4 variables)
  joint coverage    fraction of occupied cells hit in a 5x5 quantile grid of
                    (pupil area x eye opening)
  extremes          picks inside the lowest / highest 5% of eye opening and
                    of pupil area (the rare states)
The selector code is the same recipe as select_scale_frames.py batch 1.
"""
import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
CLIP = ROOT / "active-learning/training-data/face_first4min.mp4"
H5 = sorted(glob.glob(str(ROOT / "active-learning/training-data/face_first4minDLC*shuffle60*.h5")))[-1]
SEED, STRIDE, N = 42, 5, 20

pr = pd.read_hdf(H5)
pr.columns = pr.columns.droplevel(0)
while pr.columns.nlevels > 2:
    pr.columns = pr.columns.droplevel(0)
area = (np.pi / 4 * np.abs(pr["pupil_right"]["x"] - pr["pupil_left"]["x"])
        * np.abs(pr["pupil_bottom"]["y"] - pr["pupil_top"]["y"])).to_numpy()
opening = np.abs(pr["eyelid_bottom"]["y"] - pr["eyelid_top"]["y"]).to_numpy()
cx = ((pr["pupil_left"]["x"] + pr["pupil_right"]["x"]) / 2).to_numpy()
cy = ((pr["pupil_top"]["y"] + pr["pupil_bottom"]["y"]) / 2).to_numpy()
STATE = {"pupil_area": area, "eye_opening": opening, "center_x": cx, "center_y": cy}

cap = cv2.VideoCapture(str(CLIP))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pool_hi = int(0.8 * n_frames)
cand = np.arange(0, pool_hi, STRIDE)

feats, ids = [], []
for i in range(pool_hi):
    if not cap.grab():
        break
    if i % STRIDE:
        continue
    ok, im = cap.retrieve()
    if not ok:
        continue
    v = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (32, 24)).ravel().astype(np.float32)
    feats.append((v - v.mean()) / (v.std() + 1e-6))
    ids.append(i)
cap.release()
feats, ids = np.asarray(feats), np.asarray(ids)

km = KMeans(n_clusters=N, n_init=10, max_iter=300, random_state=SEED).fit(feats)
kmeans_picks = []
for c in range(N):
    m = np.where(km.labels_ == c)[0]
    kmeans_picks.append(int(ids[m[np.argmin(np.linalg.norm(feats[m] - km.cluster_centers_[c], axis=1))]]))
uniform_picks = np.linspace(0, pool_hi - 1, N).astype(int)

pool = np.arange(pool_hi)
edges = {k: np.quantile(v[pool], np.linspace(0, 1, 11)) for k, v in STATE.items()}
ea = np.quantile(area[pool], np.linspace(0, 1, 6))
eo = np.quantile(opening[pool], np.linspace(0, 1, 6))
cell = lambda f: (np.clip(np.searchsorted(ea, area[f], "right") - 1, 0, 4),
                  np.clip(np.searchsorted(eo, opening[f], "right") - 1, 0, 4))
occupied = {cell(f) for f in pool[::3]}
lo_o, hi_o = np.quantile(opening[pool], [0.05, 0.95])
lo_a, hi_a = np.quantile(area[pool], [0.05, 0.95])


def score(picks):
    picks = np.asarray(picks)
    dec = np.mean([len(set(np.clip(np.searchsorted(edges[k], v[picks], "right") - 1, 0, 9))) / 10
                   for k, v in STATE.items()])
    joint = len({cell(f) for f in picks} & occupied) / len(occupied)
    ext = dict(open_low=int((opening[picks] <= lo_o).sum()), open_high=int((opening[picks] >= hi_o).sum()),
               area_low=int((area[picks] <= lo_a).sum()), area_high=int((area[picks] >= hi_a).sum()))
    return dec, joint, ext


rng = np.random.RandomState(0)
rand = [score(rng.choice(pool, N, replace=False)) for _ in range(500)]
rd, rj = np.array([r[0] for r in rand]), np.array([r[1] for r in rand])
r_ext = {k: np.mean([r[2][k] for r in rand]) for k in rand[0][2]}
r_allfour = np.mean([all(v > 0 for v in r[2].values()) for r in rand])

print(f"old video pool: {pool_hi} frames; {N} picks per method\n")
print(f"{'method':14s} {'decile cov':>11s} {'joint 5x5':>10s}   extremes (open_low/open_high/area_low/area_high)")
for name, p in [("k-means", kmeans_picks), ("uniform-time", uniform_picks)]:
    d, j, e = score(p)
    print(f"{name:14s} {d:11.2f} {j:10.2f}   {e['open_low']}/{e['open_high']}/{e['area_low']}/{e['area_high']}"
          f"   all four extremes hit: {all(v > 0 for v in e.values())}")
print(f"{'random x500':14s} {rd.mean():8.2f}±{rd.std():.2f} {rj.mean():7.2f}±{rj.std():.2f}   "
      f"{r_ext['open_low']:.1f}/{r_ext['open_high']:.1f}/{r_ext['area_low']:.1f}/{r_ext['area_high']:.1f}"
      f"   all four extremes hit in {r_allfour:.0%} of draws")
kd, kj, _ = score(kmeans_picks)
print(f"\nk-means percentile among random draws: decile {np.mean(rd <= kd):.0%}, joint {np.mean(rj <= kj):.0%}")
