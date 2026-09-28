#!/usr/bin/env python3
"""Fewer-labels study (user request 2026-09-27): do 5 or 10 labels of a new video already do what 20 do?
Picks subsets of the video's batch01 (the 20 frames already labeled), so no new labeling is needed.

    python make_label_subsets.py --unit 5_20251029_Pluto_spont_1 --sizes 5 10

Model-free, like batch01 itself: batch01's 20 k-means clusters (kmeans_state.npz) are merged into k groups by k-means
on their centroids, weighted by cluster size; each group contributes one of the 20 labeled frames.
  subset a = the labeled frame closest to its group centre
  subset b = the second closest (the closest again when a group holds one labeled frame)
Writes <unit>/training-data/labels/subsets/sub<k><a|b>.txt (one PNG name per line). Labels are not touched."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--unit", required=True)
ap.add_argument("--sizes", type=int, nargs="+", default=[5, 10])
a = ap.parse_args()
lab = HERE / a.unit / "training-data" / "labels"
z = np.load(lab / "batch01" / "kmeans_state.npz")
feats, fidx, picks, cent = z["feats"], z["frame_idx"], z["picks"], z["centroids"]
names = sorted(p.name for p in (lab / "batch01").glob("img*.png"))
assert sorted(f"img{p:06d}.png" for p in picks) == names, "batch01 PNGs do not match its k-means picks"
row = {f: i for i, f in enumerate(fidx)}
pf = feats[[row[p] for p in picks]]                                   # fingerprints of the 20 labeled frames
w = np.bincount(z["labels"], minlength=len(cent)).astype(float)
out = lab / "subsets"
out.mkdir(exist_ok=True)
for k in a.sizes:
    km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(cent, sample_weight=w)
    grp = km.predict(pf)
    sub = {"a": [], "b": []}
    for g in range(k):
        members = np.where(grp == g)[0]
        if len(members) == 0:                                          # no labeled frame in this group: nearest overall
            members = np.arange(len(picks))
        d = np.linalg.norm(pf[members] - km.cluster_centers_[g], axis=1)
        order = members[np.argsort(d)]
        sub["a"].append(order[0])
        sub["b"].append(order[1] if len(order) > 1 else order[0])
    for s, ids in sub.items():
        ids = sorted(set(ids))
        # a group can reuse a frame another group already took; top up from the unused frames nearest any centre
        if len(ids) < k:
            rest = [i for i in np.argsort(np.min(np.linalg.norm(pf[:, None] - km.cluster_centers_[None], axis=2), axis=1)) if i not in ids]
            ids = sorted(ids + rest[: k - len(ids)])
        (out / f"sub{k:02d}{s}.txt").write_text("".join(f"img{picks[i]:06d}.png\n" for i in ids))
        print(f"sub{k:02d}{s}: {len(ids)} frames, t = {', '.join(f'{picks[i] / 60:.0f}' for i in ids)} s")
