#!/usr/bin/env python3
"""Extract the frozen test/val sets and one scale batch for the new video.

    python select_scale_frames.py --stage test|val|batch --batch-no N

Implements FROZEN_PARAMETERS.md: test = 100 even frames over the final 2
minutes; val = 20 even frames over minute 15-16; batch r = k-means (k=20r,
seed 42) on the 0-15 min pool, collision-safe +20 new frames. Each stage
writes PNGs + machinelabels (production_v1 prelabels) into
training-data/labels/<stage>/ ready for napari.
"""
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

UNIT = Path(__file__).resolve().parents[1]
TD = UNIT / "training-data"
VIDEO = TD / "20251031_Pluto_spont_1.mp4"
H5 = sorted(glob.glob(str(TD / "*shuffle60*.h5")))[-1]
FPS = 60
SEED = 42

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["test", "val", "batch"])
ap.add_argument("--batch-no", type=int, default=1)
a = ap.parse_args()

cap = cv2.VideoCapture(str(VIDEO))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if a.stage == "test":
    lo, hi = n_frames - 120 * FPS, n_frames - 1
    picks = np.linspace(lo, hi, 100).astype(int)
    name = "test100"
elif a.stage == "val":
    lo, hi = n_frames - 240 * FPS, n_frames - 120 * FPS
    picks = np.linspace(lo, hi - 1, 20).astype(int)
    name = "val20"
else:
    pool_hi = n_frames - 240 * FPS
    stride = 5
    idxs = np.arange(0, pool_hi, stride)
    # already-labeled frames from earlier batches
    done = set()
    for d in sorted((TD / "labels").glob("batch*")):
        done |= {int(p.stem.replace("img", "")) for p in d.glob("img*.png")}
    feats = []
    step = max(len(idxs) // 6000, 1)          # cap ~6000 samples for kmeans
    sub = idxs[::step]
    for i in sub:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, im = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        v = cv2.resize(g, (32, 24)).ravel().astype(np.float32)
        v = (v - v.mean()) / (v.std() + 1e-6)   # per-frame normalization:
        feats.append(v)                          # cluster by shape, not brightness
    feats = np.asarray(feats, dtype=np.float32)
    from scipy.cluster.vq import kmeans2
    np.random.seed(SEED)
    k = 20 * a.batch_no
    while True:
        cents, lab = kmeans2(feats, k, minit="++", seed=SEED)
        picks = []
        for c in range(k):
            members = np.where(lab == c)[0]
            if not len(members):
                continue
            j = members[np.argmin(np.linalg.norm(feats[members] - cents[c], axis=1))]
            f = int(sub[j])
            if f not in done and f not in picks:
                picks.append(f)
        new = [f for f in picks if f not in done][:  20 * a.batch_no]
        new = new[-20:] if a.batch_no > 1 else new[:20]
        if len(new) >= 20 or k > 20 * a.batch_no + 60:
            picks = sorted(new[:20])
            break
        k += 10
    name = f"batch{a.batch_no:02d}"

out = TD / "labels" / name
out.mkdir(parents=True, exist_ok=True)

pred = pd.read_hdf(H5)
scorer = pred.columns.get_level_values(0)[0]
rows, imgs = [], []
for f in picks:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
    ok, im = cap.read()
    if not ok:
        continue
    img = f"img{int(f):06d}.png"
    cv2.imwrite(str(out / img), im)
    rows.append(pred.iloc[int(f)])
    imgs.append(img)
cap.release()

ml = pd.DataFrame(rows)
ml.index = pd.MultiIndex.from_tuples([("labeled-data", name, i) for i in imgs])
ml.columns = pred.columns
ml.to_hdf(out / "machinelabels.h5", key="df_with_missing", mode="w")
ml.to_csv(out / "machinelabels.csv")
print(f"{name}: {len(imgs)} frames -> {out}")
