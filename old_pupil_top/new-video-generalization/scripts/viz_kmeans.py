#!/usr/bin/env python3
"""Visualize the batch-1 k-means selection (advisor request, 2026-09-18):
step 3 - what do the clusters look like, are they distinct?
step 4 - are the 20 representatives actually diverse?

    python viz_kmeans.py [--batch-no 1]

Reads training-data/labels/batchNN/kmeans_state.npz written by
select_scale_frames.py and the selected PNGs; writes two figures to results/.
"""
import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

UNIT = Path(__file__).resolve().parents[1]
FPS = 60.0

ap = argparse.ArgumentParser()
ap.add_argument("--batch-no", type=int, default=1)
a = ap.parse_args()
name = f"batch{a.batch_no:02d}"
d = UNIT / "training-data" / "labels" / name
st = np.load(d / "kmeans_state.npz")
feats, labels, fidx, cents, picks = (st["feats"], st["labels"], st["frame_idx"],
                                     st["centroids"], st["picks"])
k = int(st["k"])
t = fidx / FPS
pick_pos = np.array([int(np.where(fidx == p)[0][0]) for p in picks])
pick_cluster = labels[pick_pos]

# ---- cluster clarity: silhouette (sklearn if present) + PCA projection ----
try:
    from sklearn.metrics import silhouette_score
    sil = float(silhouette_score(feats, labels, sample_size=min(3000, len(feats)), random_state=0))
except Exception:
    sil = float("nan")
mu = feats.mean(0)
U, S, Vt = np.linalg.svd(feats - mu, full_matrices=False)
proj = (feats - mu) @ Vt[:2].T
expl = (S[:2] ** 2 / (S ** 2).sum()).sum()
sizes = np.bincount(labels, minlength=k)

fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1.3, 1], hspace=0.3, wspace=0.2)
cmap = plt.get_cmap("tab20", k)

ax = fig.add_subplot(gs[0, 0])
ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap=cmap, s=4, alpha=0.45, vmin=0, vmax=k - 1)
ax.scatter(proj[pick_pos, 0], proj[pick_pos, 1], marker="*", s=260, c="black", zorder=5)
for pp, cl in zip(pick_pos, pick_cluster):
    ax.annotate(str(cl), (proj[pp, 0], proj[pp, 1]), fontsize=8, color="white",
                ha="center", va="center", zorder=6)
ax.set_title(f"Step 3: {len(feats)} sampled frames in appearance space (PCA of 768-d edge fingerprints, "
             f"{expl:.0%} variance shown)\n{k} k-means clusters by color; black stars = the {len(picks)} "
             f"picked representatives (cluster id inside). silhouette = {sil:.2f}", fontsize=10)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

ax = fig.add_subplot(gs[0, 1])
order = np.argsort(-sizes)
ax.barh(range(k), sizes[order], color=[cmap(i) for i in order])
ax.set_yticks(range(k)); ax.set_yticklabels([str(i) for i in order], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("frames in cluster (of sampled)"); ax.set_title("Cluster sizes (id)", fontsize=10)
for y, i in enumerate(order):
    if i in set(pick_cluster):
        ax.text(sizes[i], y, "  picked", va="center", fontsize=7)

ax = fig.add_subplot(gs[1, :])
ax.scatter(t, labels, c=labels, cmap=cmap, s=3, alpha=0.6, vmin=0, vmax=k - 1)
ax.scatter(picks / FPS, pick_cluster, marker="*", s=220, c="black", zorder=5)
ax.set_xlabel("time in video (s)"); ax.set_ylabel("cluster id")
ax.set_title("Clusters over time: recurring eye states appear as horizontal bands; "
             "black stars = when each picked frame occurs", fontsize=10)
ax.set_yticks(range(k))
fig.suptitle(f"{name}: k-means diversity selection on the Pluto training pool (first 80%)", fontsize=13)
fig.savefig(UNIT / "results" / f"01_{name}_kmeans_clusters.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---- step 4: montage of the 20 picks ---------------------------------------
pngs = sorted(d.glob("img*.png"))
n = len(pngs)
ncol = 5
nrow = int(np.ceil(n / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.3 * nrow))
for ax, p in zip(np.ravel(axs), pngs):
    f = int(p.stem.replace("img", ""))
    im = cv2.imread(str(p))
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    cl = int(labels[np.where(fidx == f)[0][0]])
    ax.set_title(f"t={f / FPS:.1f}s  cluster {cl} ({sizes[cl]} frames)", fontsize=9)
    ax.axis("off")
for ax in np.ravel(axs)[n:]:
    ax.axis("off")
fig.suptitle(f"Step 4: the {n} picked representatives - one per cluster, sorted by time", fontsize=13)
fig.tight_layout()
fig.savefig(UNIT / "results" / f"02_{name}_selected_frames.png", dpi=120)
print(f"silhouette {sil:.3f}; cluster sizes min/median/max {sizes.min()}/{int(np.median(sizes))}/{sizes.max()}; "
      f"picks span {picks.min() / FPS:.0f}-{picks.max() / FPS:.0f}s, min gap {np.diff(np.sort(picks)).min() / FPS:.1f}s")
