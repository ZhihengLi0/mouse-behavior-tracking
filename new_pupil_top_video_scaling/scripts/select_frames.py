#!/usr/bin/env python3
"""Extract the frozen test/val sets and one scale batch for a video - NEW PUPIL STANDARD era (2026-09-20).

Copy of old_pupil_top/new-video-generalization/scripts/select_scale_frames.py with the SAME split rule and the
SAME selectors (k-means on DoG fingerprints for batch 1, jump + k-means afterwards). Differences:
  * one folder per video: new_pupil_top/<unit>/training-data/labels/ and new_pupil_top/<unit>/results/ (--unit);
  * --n-test / --n-val set the set sizes (advisor plan 2026-09-20: 50 test, 20 validation);
  * no machine pre-labels are written here: frame selection depends on no model at all. Pre-labels in the
    new standard (pupil = ellipse endpoints) are produced separately by make_prelabels.py.
Original docstring follows.


    python select_scale_frames.py --stage test|val
    python select_scale_frames.py --stage batch --batch-no 1
    python select_scale_frames.py --stage batch --batch-no N --pred-h5 <h5 of latest model>

Implements FROZEN_PARAMETERS.md exactly (rewritten 2026-09-18 after audit):

  split      test = final 10% of the video (>= 60 s), val = the 10% before it
             (>= 60 s), pool = everything earlier; a GUARD of 2 s is removed
             on both sides of val so no pool/val/test frames are near-duplicates.
             fps and length are read from the video, never hardcoded.
  batch 1    k-means (sklearn, k=20, n_init=10, random_state=42) on 32x24
             difference-of-Gaussians fingerprints (sigma 1 - sigma 6 on a
             128x96 downscale, then per-frame normalized) of every 5th pool
             frame; one medoid per cluster.
  batch >=2  error-guided: `jump` detector on the latest model's predictions
             (any keypoint moving more than EPS_FRAC * median eye width
             between consecutive frames), then k-means (k=20) among the
             flagged pool frames; falls back to the largest-jump frames if
             too few are flagged.
  all picks  at least MIN_GAP_S apart from each other and from every frame
             already labeled in earlier batches.
  idempotent refuses to overwrite an existing set unless --force (which wipes it).
"""
import argparse
import glob
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

UNIT = Path(__file__).resolve().parents[1]
TD = UNIT / "training-data"
SEED = 42
STRIDE = 5
GUARD_S = 2.0
MIN_GAP_S = 1.0
EPS_FRAC = 0.03          # jump threshold as a fraction of median eye width
FLOOR_S = 60.0

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["test", "val", "val_extra", "batch"])
ap.add_argument("--batch-no", type=int, default=1)
ap.add_argument("--video", default=None)
ap.add_argument("--pred-h5", default=None, help="latest model's predictions (batch >= 2)")
ap.add_argument("--force", action="store_true")
ap.add_argument("--unit", required=True, help="video folder inside new_pupil_top/, e.g. first5minvedio")
ap.add_argument("--n-test", type=int, default=50)
ap.add_argument("--n-val", type=int, default=20)
a = ap.parse_args()

if not a.video:
    sys.exit("pass --video")
VIDEO = Path(a.video)
LABELS = UNIT / a.unit / "training-data" / "labels"
cap = cv2.VideoCapture(str(VIDEO))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = float(cap.get(cv2.CAP_PROP_FPS)) or 60.0
dur = n_frames / fps

seg = max(0.10 * dur, FLOOR_S)
guard = int(round(GUARD_S * fps))
test_lo = n_frames - int(round(seg * fps))
val_hi = test_lo - guard                      # exclusive
val_lo = val_hi - int(round(seg * fps))
pool_hi = val_lo - guard                      # exclusive
if pool_hi < int(60 * fps):
    sys.exit("video too short for the 80/10/10 rule with 60 s floors")

name = {"test": f"test{a.n_test}", "val": f"val{a.n_val}", "val_extra": "val30_extra"}.get(a.stage, f"batch{a.batch_no:02d}")
out = LABELS / name
if out.exists() and any(out.glob("img*.png")):
    if not a.force:
        sys.exit(f"{out} already populated; refusing to overwrite (use --force to wipe and redo)")
    shutil.rmtree(out)
out.mkdir(parents=True, exist_ok=True)


def labeled_elsewhere():
    done = set()
    for d in LABELS.glob("batch*"):
        if d != out:
            done |= {int(p.stem.replace("img", "")) for p in d.glob("img*.png")}
    return done


def spaced(cands, taken, n):
    """Greedy: keep candidates (in given order) at least MIN_GAP_S from all taken."""
    gap = int(round(MIN_GAP_S * fps))
    chosen = []
    ref = sorted(taken)
    for f in cands:
        if all(abs(f - g) >= gap for g in ref + chosen):
            chosen.append(int(f))
        if len(chosen) == n:
            break
    return chosen


def fingerprints(frame_ids):
    """Sequential decode of the requested frames -> normalized 768-d vectors."""
    want = np.zeros(n_frames, bool)
    want[np.asarray(frame_ids)] = True
    last = int(np.max(frame_ids))
    feats, kept = [], []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(last + 1):
        if not cap.grab():
            break
        if not want[i]:
            continue
        ok, im = cap.retrieve()
        if not ok:
            continue
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sm = cv2.resize(gray, (128, 96), interpolation=cv2.INTER_AREA)
        sm = cv2.GaussianBlur(sm, (0, 0), 1.0) - cv2.GaussianBlur(sm, (0, 0), 6.0)   # edges/shape, not slow lighting
        v = cv2.resize(sm, (32, 24), interpolation=cv2.INTER_AREA).ravel()
        feats.append((v - v.mean()) / (v.std() + 1e-6))
        kept.append(i)
    return np.asarray(feats, np.float32), np.asarray(kept)


def medoids(feats, ids, k):
    km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=SEED).fit(feats)
    order = np.argsort(-np.bincount(km.labels_, minlength=k))      # big clusters first
    picks = []
    for c in order:
        m = np.where(km.labels_ == c)[0]
        picks.append(int(ids[m[np.argmin(np.linalg.norm(feats[m] - km.cluster_centers_[c], axis=1))]]))
    return picks, km


state = None
if a.stage == "test":
    picks = np.linspace(test_lo, n_frames - 1, a.n_test).astype(int).tolist()
elif a.stage == "val":
    picks = np.linspace(val_lo, val_hi - 1, a.n_val).astype(int).tolist()
elif a.stage == "val_extra":
    # 2026-09-19: validation set enlarged 20 -> 50 so the best-mAP snapshot rule is less of a
    # lottery (20 frames = 160 points; a 0.25 mAP gap picked an undertrained epoch at step 2).
    # 30 more evenly spaced frames in the SAME segment, at least 1 s from every original one.
    orig = np.linspace(val_lo, val_hi - 1, 20).astype(int)
    grid = np.linspace(val_lo, val_hi - 1, 31)
    picks = []
    for f in ((grid[:-1] + grid[1:]) / 2).astype(int):
        while np.min(np.abs(orig - f)) < int(round(MIN_GAP_S * fps)):
            f += int(round(1.5 * MIN_GAP_S * fps))
        picks.append(int(f))
    assert len(set(picks)) == 30 and max(picks) < val_hi and not set(picks) & set(orig.tolist())
else:
    done = labeled_elsewhere()
    if a.batch_no == 1:
        cand = np.arange(0, pool_hi, STRIDE)
        mode = "kmeans-bootstrap"
    else:
        if not a.pred_h5:
            sys.exit("batch >= 2 is error-guided: pass --pred-h5 (latest model's predictions)")
        pr = pd.read_hdf(a.pred_h5)
        pr.columns = pr.columns.droplevel(0)
        while pr.columns.nlevels > 2:
            pr.columns = pr.columns.droplevel(0)
        bps = pr.columns.get_level_values(0).unique()
        width = np.hypot(pr["eye_temporal_corner"]["x"] - pr["eye_nasal_corner"]["x"],
                         pr["eye_temporal_corner"]["y"] - pr["eye_nasal_corner"]["y"])
        eps = EPS_FRAC * float(np.nanmedian(width))
        jump = np.zeros(len(pr))
        for bp in bps:
            d = np.hypot(np.diff(pr[bp]["x"], prepend=np.nan), np.diff(pr[bp]["y"], prepend=np.nan))
            jump = np.fmax(jump, np.nan_to_num(d))
        flagged = np.where(jump[:pool_hi] > eps)[0]
        print(f"jump: eps={eps:.1f}px ({EPS_FRAC} x eye width), {len(flagged)} pool frames flagged")
        if len(flagged) > 12000:
            flagged = flagged[:: int(np.ceil(len(flagged) / 12000))]
        cand = flagged if len(flagged) >= 40 else np.argsort(-jump[:pool_hi])[:2000]
        mode = "jump-guided"
    feats, ids = fingerprints(cand)
    ranked, km = medoids(feats, ids, 20)
    picks = spaced(ranked, done, 20)
    k = 20
    while len(picks) < 20 and k < 80:          # collisions/gaps removed some: widen and top up
        k += 10
        more, _ = medoids(feats, ids, k)
        picks += spaced([f for f in more if f not in picks], done | set(picks), 20 - len(picks))
    if len(picks) < 20:
        sys.exit(f"only {len(picks)} valid frames found - refusing a short batch")
    picks = sorted(picks[:20])
    state = dict(feats=feats, labels=km.labels_, frame_idx=ids, centroids=km.cluster_centers_,
                 picks=np.asarray(picks), k=20, mode=mode)

imgs = []
for f in picks:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
    ok, im = cap.read()
    if not ok:
        sys.exit(f"cannot read frame {f}")
    img = f"img{int(f):06d}.png"
    cv2.imwrite(str(out / img), im)
    imgs.append(img)
cap.release()
if state:
    np.savez_compressed(out / "kmeans_state.npz", **state)
print(f"{name}: {len(imgs)} frames | video {dur:.1f}s @ {fps:.0f}fps | pool [0,{pool_hi}) "
      f"val [{val_lo},{val_hi}) test [{test_lo},{n_frames}) | guard {guard} frames")
