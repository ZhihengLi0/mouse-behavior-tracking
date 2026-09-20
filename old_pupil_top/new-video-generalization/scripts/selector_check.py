#!/usr/bin/env python3
"""Label-free robustness check of the batch-1 selector across videos.

    python selector_check.py <video> [<video> ...]

Runs the exact batch-1 recipe of select_scale_frames.py (80/10/10 split with
2-s guards, every 5th pool frame, 32x24 normalized grayscale fingerprints,
scikit-learn k-means k=20 seed 42, medoid per cluster, >=1 s spacing) and
scores it against criteria declared BEFORE looking at the results:

  C1 runs and yields exactly 20 picks inside the pool
  C2 temporal coverage: picks fall in >= 7 of the pool's 10 time deciles
  C3 no near-duplicates: minimum gap between picks >= 1 s
  C4 clusters are eye STATES, not time chunks: median cluster spans >= 25%
     of the pool duration (5th-95th percentile of its members' times)
  C5 not illumination-driven: cluster labels explain < 50% of the variance
     of BACKGROUND brightness (top quarter of the frame = fur, no eye). The
     first version used whole-frame brightness, which cannot separate lighting
     from eye state (a dilated pupil darkens the frame); replaced 2026-09-19
     after an illumination-blind feature gave the same whole-frame eta^2.
  C6 picks are more diverse than chance: mean pairwise fingerprint distance
     of the 20 picks >= 75th percentile of 300 random 20-frame draws

Writes nothing under training-data/labels; figures go to
results/selector_check/.
"""
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

UNIT = Path(__file__).resolve().parents[1]
OUT = UNIT / "results" / "selector_check"
OUT.mkdir(parents=True, exist_ok=True)
SEED, STRIDE, N, GUARD_S, MIN_GAP_S, FLOOR_S = 42, 5, 20, 2.0, 1.0, 60.0
import os
FEAT = os.environ.get("FEAT", "raw")


def check(video):
    video = Path(video)
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 60.0
    w, h = int(cap.get(3)), int(cap.get(4))
    seg = max(0.10 * n / fps, FLOOR_S)
    guard = int(round(GUARD_S * fps))
    test_lo = n - int(round(seg * fps))
    val_hi = test_lo - guard
    val_lo = val_hi - int(round(seg * fps))
    pool_hi = val_lo - guard

    feats, ids, bright = [], [], []
    for i in range(pool_hi):
        if not cap.grab():
            break
        if i % STRIDE:
            continue
        ok, im = cap.retrieve()
        if not ok:
            continue
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bright.append(gray[: gray.shape[0] // 4].mean())   # fur-only strip: illumination proxy, no eye
        if FEAT == "dog":      # difference of Gaussians: keeps edges/shape, drops slow illumination
            small = cv2.resize(gray, (128, 96), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), 1.0) - cv2.GaussianBlur(small, (0, 0), 6.0)
            g = cv2.resize(small, (32, 24), interpolation=cv2.INTER_AREA).ravel()
        else:
            g = cv2.resize(gray, (32, 24)).ravel()
        feats.append((g - g.mean()) / (g.std() + 1e-6))
        ids.append(i)
    feats, ids, bright = np.asarray(feats), np.asarray(ids), np.asarray(bright)

    km = KMeans(n_clusters=N, n_init=10, max_iter=300, random_state=SEED).fit(feats)
    lab = km.labels_
    order = np.argsort(-np.bincount(lab, minlength=N))
    gap = int(round(MIN_GAP_S * fps))
    picks, pos = [], []
    for c in order:
        m = np.where(lab == c)[0]
        ranked = m[np.argsort(np.linalg.norm(feats[m] - km.cluster_centers_[c], axis=1))]
        for j in ranked:                         # nearest member that respects spacing
            if all(abs(ids[j] - p) >= gap for p in picks):
                picks.append(int(ids[j])); pos.append(int(j)); break
    picks, pos = np.asarray(picks), np.asarray(pos)

    t = ids / fps
    deciles = len(set(np.minimum((picks / pool_hi * 10).astype(int), 9)))
    min_gap = float(np.diff(np.sort(picks)).min() / fps)
    spans = []
    for c in range(N):
        tc = t[lab == c]
        spans.append((np.percentile(tc, 95) - np.percentile(tc, 5)) / (pool_hi / fps))
    span_med = float(np.median(spans))
    grand = bright.mean()
    eta2 = float(sum((lab == c).sum() * (bright[lab == c].mean() - grand) ** 2 for c in range(N))
                 / ((bright - grand) ** 2).sum())

    def mpd(idx):
        f = feats[idx]
        d = np.linalg.norm(f[:, None] - f[None], axis=2)
        return d[np.triu_indices(len(idx), 1)].mean()
    rng = np.random.RandomState(0)
    rand = np.array([mpd(rng.choice(len(feats), N, replace=False)) for _ in range(300)])
    div_pct = float((rand <= mpd(pos)).mean())
    sil = float(silhouette_score(feats, lab, sample_size=min(3000, len(feats)), random_state=0))

    crit = {
        "C1 20 picks in pool": len(picks) == N and picks.max() < pool_hi,
        "C2 time deciles >= 7": deciles >= 7,
        "C3 min gap >= 1 s": min_gap >= 1.0,
        "C4 clusters span >= 25% of pool": span_med >= 0.25,
        "C5 brightness eta2 < 0.5": eta2 < 0.5,
        "C6 diversity >= 75th pct of random": div_pct >= 0.75,
    }

    # figure: timeline of clusters + montage of the picks
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(5, 5, height_ratios=[1.3, 1, 1, 1, 1], hspace=0.28, wspace=0.05)
    ax = fig.add_subplot(gs[0, :])
    ax.scatter(t, lab, c=lab, cmap=plt.get_cmap("tab20", N), s=3, alpha=0.6, vmin=0, vmax=N - 1)
    ax.scatter(picks / fps, lab[pos], marker="*", s=200, c="black", zorder=5)
    ax.set_xlabel("time (s)"); ax.set_ylabel("cluster id"); ax.set_yticks(range(N))
    ok_all = all(crit.values())
    ax.set_title(f"{video.name}  {n / fps / 60:.1f} min @ {fps:.0f} fps, {w}x{h} | pool 0-{pool_hi / fps:.0f}s | "
                 f"silhouette {sil:.2f} | deciles hit {deciles}/10 | min gap {min_gap:.1f}s | "
                 f"cluster span {span_med:.0%} | brightness eta2 {eta2:.2f} | diversity pct {div_pct:.0%} | "
                 f"{'ALL CRITERIA PASS' if ok_all else 'CRITERIA FAILED'}", fontsize=10)
    for k, f in enumerate(np.sort(picks)):
        axi = fig.add_subplot(gs[1 + k // 5, k % 5])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        okr, im = cap.read()
        if okr:
            axi.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        axi.set_title(f"t={f / fps:.1f}s", fontsize=9); axi.axis("off")
    cap.release()
    fig.savefig(OUT / f"{video.stem}_selector_check_{FEAT}.png", dpi=95, bbox_inches="tight")
    plt.close(fig)

    print(f"\n=== [{FEAT}] {video.name}: {n / fps / 60:.1f} min, {fps:.0f} fps, {w}x{h}, "
          f"pool [0,{pool_hi}) val [{val_lo},{val_hi}) test [{test_lo},{n})")
    print(f"    silhouette {sil:.3f} | deciles {deciles}/10 | min gap {min_gap:.1f}s | median cluster span "
          f"{span_med:.0%} | brightness eta2 {eta2:.2f} | diversity percentile {div_pct:.0%}")
    for k, v in crit.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    return ok_all


results = {v: check(v) for v in sys.argv[1:]}
print("\nSUMMARY:", {Path(k).name: ("PASS" if v else "FAIL") for k, v in results.items()})
