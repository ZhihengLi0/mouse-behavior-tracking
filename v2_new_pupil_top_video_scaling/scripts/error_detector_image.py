#!/usr/bin/env python3
"""D2: the image error detector (stage 2 of P1, docs/next_phase_plan_2026-10-24.md). Needs the outputs of
error_detector.py (results/error_detector/features_pairs.csv). Read-only elsewhere; writes to results/error_detector/.

    python error_detector_image.py                 # crops, 5 video-grouped folds, D2 with and without synthetic errors, tables, figure
    python error_detector_image.py --epochs 12 --size 192 128

Input per pair (one model's prediction on one labeled frame it never trained on): the eye crop of the frame
(centre = midpoint of the two PREDICTED corners, ±0.72 W × ±0.5 W, W = the video's median corner distance; same box
as the figures of step 10) resized to 192 × 128, as three channels: the grey image; the four predicted pupil points
drawn as Gaussian blobs plus the two pupil axes; the four predicted eye points drawn as blobs. Nothing from the labels
enters the input. Targets: for each of the 8 keypoints "its error > 20 px" (masked where the labeler left the point
empty) and "frame RMSE > 20 px"; 9 sigmoid outputs of an ImageNet-pretrained ResNet-18, fine-tuned end to end.
Frame score = the frame output (d2_frame) or the largest keypoint output (d2_maxpoint).

Synthetic errors (variant d2_*_syn): during training, with probability 0.35 a pair's predicted points are displaced
before drawing (whole pupil shifted 15–120 px, or one pupil point 20–100 px, or the pupil scaled × 0.4–1.8 about its
centre, or one eyelid point 20–100 px); the targets are recomputed from the displaced points against the human labels.
Real errors from the pairs stay in; the synthetic ones only add positives.

Folds: 5 groups of videos (videos assigned to groups so that each holds a similar number of > 20 px eval frames);
the eval frames of a group (plateau model, test50 + val20) are scored by the network trained on the other groups'
pairs only (strict: nothing of a held-out video, not even its batch frames, is seen). D1 (gradient boosting of
error_detector.py, all features, all pairs) is re-run on the same folds so the two are comparable; fit_pupil is the
reference. Evaluation identical to error_detector.py.
Outputs: crops cache (local), scores_image.csv, auc_table_image.csv, operating_points_image.csv, detector_image.png."""
import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_step import BPS, HERE  # noqa: E402
from error_detector import OUT, CACHE, auc, evaluate, fit_predict, roc  # noqa: E402

PUPIL, EYE = BPS[:4], BPS[4:]
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MARGIN = 0.08          # extra crop margin cached for the training-time jitter


def frame_png(unit, set_, frame):
    return HERE / unit / "training-data" / "labels" / set_ / f"img{frame:06d}.png"


def crop_box(row):
    nx, ny = row["pred_eye_nasal_corner_x"], row["pred_eye_nasal_corner_y"]
    tx, ty = row["pred_eye_temporal_corner_x"], row["pred_eye_temporal_corner_y"]
    W = row["eye_width_px"]
    cx, cy = (nx + tx) / 2, (ny + ty) / 2
    return cx, cy, 0.72 * W * (1 + MARGIN), 0.5 * W * (1 + MARGIN)


def build_crops(D, size):
    """Grey crops (with margin) for every pair, resized to size*(1+MARGIN); cached as one npz keyed by pair order."""
    w, h = size
    ww, hh = int(round(w * (1 + MARGIN))), int(round(h * (1 + MARGIN)))
    p = CACHE / f"crops_{ww}x{hh}.npz"
    if p.exists():
        z = np.load(p)
        if len(z["crops"]) == len(D) and (z["key"] == D.pair_key.to_numpy()).all():
            return z["crops"], z["boxes"]
    crops, boxes = np.zeros((len(D), hh, ww), np.uint8), np.zeros((len(D), 4), np.float32)
    cache_img, t0 = {}, time.time()
    for n, (i, row) in enumerate(D.iterrows()):
        f = frame_png(row.unit, row["set"], int(row.frame))
        if f not in cache_img:
            cache_img.clear()
            cache_img[f] = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        im = cache_img[f]
        cx, cy, hw, hhp = crop_box(row)
        x0, x1, y0, y1 = int(round(cx - hw)), int(round(cx + hw)), int(round(cy - hhp)), int(round(cy + hhp))
        pad = np.zeros((y1 - y0, x1 - x0), np.uint8)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(im.shape[1], x1), min(im.shape[0], y1)
        pad[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = im[sy0:sy1, sx0:sx1]
        crops[n] = cv2.resize(pad, (ww, hh), interpolation=cv2.INTER_AREA)
        boxes[n] = (x0, y0, (x1 - x0) / ww, (y1 - y0) / hh)   # origin and px-per-pixel scale of the cached crop
        if n % 1000 == 0:
            print(f"  crop {n}/{len(D)} ({time.time() - t0:.0f} s)", flush=True)
    np.savez_compressed(p, crops=crops, boxes=boxes, key=D.pair_key.to_numpy().astype(str))
    return crops, boxes


def render(points, ox, oy, sx, sy, w, h, sigma=2.5):
    """Two channels (pupil blobs + axes, eye blobs) at crop resolution from 8 points in frame px."""
    ch = np.zeros((2, h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for k, (px, py) in enumerate(points):
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        u, v = (px - ox) / sx, (py - oy) / sy
        c = 0 if k < 4 else 1
        ch[c] = np.maximum(ch[c], np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * sigma ** 2)))
    img = np.zeros((h, w), np.float32)
    for a, b in ((0, 1), (2, 3)):
        (ax_, ay), (bx, by) = points[a], points[b]
        if np.all(np.isfinite([ax_, ay, bx, by])):
            cv2.line(img, (int(round((ax_ - ox) / sx)), int(round((ay - oy) / sy))), (int(round((bx - ox) / sx)), int(round((by - oy) / sy))), 0.6, 1)
    ch[0] = np.maximum(ch[0], img)
    return ch


class Pairs(torch.utils.data.Dataset):
    def __init__(self, D, crops, boxes, size, train, synthetic, seed=0):
        self.D, self.crops, self.boxes, self.size, self.train, self.syn = D.reset_index(drop=True), crops, boxes, size, train, synthetic
        self.pred = self.D[[f"pred_{b}_{c}" for b in BPS for c in "xy"]].to_numpy(np.float32).reshape(-1, 8, 2)
        self.hum = self.D[[f"human_{b}_{c}" for b in BPS for c in "xy"]].to_numpy(np.float32).reshape(-1, 8, 2)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.D)

    def targets(self, pred, hum):
        err = np.hypot(*(pred - hum).T)                     # (8,)
        ok = np.isfinite(err)
        y = np.zeros(9, np.float32); m = np.zeros(9, np.float32)
        y[:8][ok] = err[ok] > 20; m[:8][ok] = 1
        if ok.any():
            y[8] = np.sqrt(np.mean(err[ok] ** 2)) > 20; m[8] = 1
        return y, m

    def perturb(self, pred):
        p = pred.copy()
        mode = self.rng.integers(4)
        ang = self.rng.uniform(0, 2 * np.pi); d = lambda lo, hi: self.rng.uniform(lo, hi) * np.array([np.cos(ang), np.sin(ang)], np.float32)
        if mode == 0:
            p[:4] += d(15, 120)
        elif mode == 1:
            p[self.rng.integers(4)] += d(20, 100)
        elif mode == 2:
            c = p[:4].mean(0); p[:4] = c + (p[:4] - c) * self.rng.uniform(0.4, 1.8)
        else:
            p[4 + self.rng.integers(4)] += d(20, 100)
        return p

    def __getitem__(self, i):
        w, h = self.size
        crop, (x0, y0, sx, sy) = self.crops[i], self.boxes[i]
        H, Wc = crop.shape
        if self.train:
            ox, oy = self.rng.integers(0, Wc - w + 1), self.rng.integers(0, H - h + 1)
        else:
            ox, oy = (Wc - w) // 2, (H - h) // 2
        img = crop[oy:oy + h, ox:ox + w].astype(np.float32) / 255
        if self.train:
            img = np.clip(img * self.rng.uniform(0.75, 1.3) + self.rng.uniform(-0.1, 0.1), 0, 1)
        pred = self.pred[i]
        if self.train and self.syn and self.rng.random() < 0.35:
            pred = self.perturb(pred)
        y, m = self.targets(pred, self.hum[i])
        pts = render(pred, x0 + ox * sx, y0 + oy * sy, sx, sy, w, h)
        x = np.concatenate([img[None], pts], 0)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)


def make_model():
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 9)
    return m.to(DEV)


def train_fold(Dtr, Dte, crops, boxes, idx_tr, idx_te, size, epochs, synthetic, seed=0):
    torch.manual_seed(seed)
    ds_tr = Pairs(Dtr, crops[idx_tr], boxes[idx_tr], size, True, synthetic, seed)
    ds_te = Pairs(Dte, crops[idx_te], boxes[idx_te], size, False, False)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=48, shuffle=True, num_workers=0, drop_last=True)
    dl_te = torch.utils.data.DataLoader(ds_te, batch_size=96, shuffle=False, num_workers=0)
    m = make_model()
    pos = np.clip((Dtr[[f"err_{b}" for b in BPS]] > 20).mean().to_numpy(), 0.01, 1)
    pw = torch.tensor(np.r_[np.minimum(8, (1 - pos) / pos), min(8, (1 - Dtr.wrong20.mean()) / Dtr.wrong20.mean())], dtype=torch.float32, device=DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=epochs * len(dl_tr), pct_start=0.15)
    crit = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
    for ep in range(epochs):
        m.train(); tot, n, t0 = 0.0, 0, time.time()
        for x, y, mk in dl_tr:
            x, y, mk = x.to(DEV), y.to(DEV), mk.to(DEV)
            loss = (crit(m(x), y) * mk).sum() / mk.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss) * len(x); n += len(x)
        print(f"    epoch {ep + 1}/{epochs} loss {tot / n:.3f} ({time.time() - t0:.0f} s)", flush=True)
    m.eval(); out = []
    with torch.no_grad():
        for x, _, _ in dl_te:
            out.append(torch.sigmoid(m(x.to(DEV))).cpu().numpy())
    return np.concatenate(out)


def video_groups(E, k=5):
    pos = E.groupby("video").wrong20.sum().sort_values(ascending=False)
    groups, load = {}, np.zeros(k)
    for v, n in pos.items():
        g = int(np.argmin(load)); groups[v] = g; load[g] += n + 0.5
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--size", type=int, nargs=2, default=(192, 128))
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    size = tuple(a.size)
    D = pd.read_csv(OUT / "features_pairs.csv")
    D["pair_key"] = D.video.astype(str) + "_" + D.model_step.astype(str) + "_" + D["set"] + "_" + D.frame.astype(str)
    print(f"{len(D)} pairs, {D.wrong20.sum()} wrong20; device {DEV}", flush=True)
    crops, boxes = build_crops(D, size)
    E = D[D["eval"]].copy()
    groups = video_groups(E, a.folds)
    D["group"] = D.video.map(groups)
    print("video groups:", {g: sorted(v for v, gg in groups.items() if gg == g) for g in range(a.folds)}, flush=True)
    feat_cols = [c for c in D.columns if c not in pd.read_csv(OUT / "truth_pairs.csv", nrows=1).columns and c not in ("pair_key", "group")]
    for c in ("d2_frame", "d2_maxpoint", "d2_frame_syn", "d2_maxpoint_syn", "d1_pairs_all_5fold"):
        E[c] = np.nan
    for g in range(a.folds):
        tr, te = (D.group != g).to_numpy(), (D["eval"] & (D.group == g)).to_numpy()
        print(f"fold {g}: {tr.sum()} training pairs, {te.sum()} eval frames", flush=True)
        E.loc[te[D["eval"].to_numpy()], "d1_pairs_all_5fold"] = fit_predict(D.loc[tr, feat_cols], D.loc[tr, "wrong20"], D.loc[te, feat_cols])
        for syn, suf in ((False, ""), (True, "_syn")):
            print(f"  D2{suf or ' plain'}", flush=True)
            P = train_fold(D[tr], D[te], crops, boxes, np.where(tr)[0], np.where(te)[0], size, a.epochs, syn)
            E.loc[te[D["eval"].to_numpy()], f"d2_frame{suf}"] = P[:, 8]
            E.loc[te[D["eval"].to_numpy()], f"d2_maxpoint{suf}"] = P[:, :8].max(1)
            for c in (f"d2_frame{suf}", f"d2_maxpoint{suf}"):
                s = E[c].dropna(); y = E.loc[s.index, "wrong20"]
                print(f"    {c} so far: AUC20 {auc(s[y], s[~y]):.3f}", flush=True)
    E["fit_pupil_ref"] = E["fit_pupil"]
    names = ["d2_frame", "d2_maxpoint", "d2_frame_syn", "d2_maxpoint_syn", "d1_pairs_all_5fold", "fit_pupil_ref"]
    E.to_csv(OUT / "scores_image.csv", index=False)
    A, O, R = evaluate(E.reset_index(drop=True), names)
    A.to_csv(OUT / "auc_table_image.csv", index=False); O.to_csv(OUT / "operating_points_image.csv", index=False)
    print(A.to_string(index=False)); print(O[O.truth == "wrong20"].to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4))
    for j, w in enumerate(("wrong20", "wrong50")):
        for i, n in enumerate(names):
            Rr = roc(E[n], E[w])
            ax[j].plot(Rr.flagged * 100, Rr.recall * 100, lw=1.6, label=f"{n} (AUC {float(A.loc[A.signal == n, 'AUC_' + w].iloc[0]):.3f})")
        for x in (5, 10):
            ax[j].axvline(x, color="0.6", ls=":", lw=1)
        ax[j].set_xlim(0, 60); ax[j].set_ylim(0, 101); ax[j].set_xlabel("frames flagged (%)"); ax[j].set_ylabel("recall (%)")
        ax[j].set_title(f"{'> 20 px' if w == 'wrong20' else '> 50 px'} ({int(E[w].sum())} of {len(E)} frames)"); ax[j].legend(fontsize=8, loc="lower right"); ax[j].grid(alpha=0.3)
    fig.suptitle(f"Image detector D2 vs tabular D1 vs fit_pupil, {a.folds} video-grouped folds, {len(E)} step-8 frames", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "detector_image.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
