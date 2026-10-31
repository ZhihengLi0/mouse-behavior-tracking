#!/usr/bin/env python3
"""How the repair works, on real frames: the heatmap of the thrown point with its two peaks.

    python heatmap_example_figure.py

For a few repaired frames that the check by eye judged right (results/flag_and_repair/spotcheck_review/), the plateau
model of the video is run again on that one frame and the full heatmap of the thrown point is kept (the saved
predictions keep only the regressed coordinate; heatmap_signals.py kept only a few numbers per map). Each row: the
frame with the model's 8 points before the repair (red), the heatmap of the thrown point with peak 1 (red, the model's
answer) and peak 2 (blue, the repair's answer), and the frame after the repair. The regressed coordinates of the rerun
are checked against the saved predictions. Output: results/flag_and_repair/spotcheck_review/heatmap_examples.png.
Nothing existing is modified."""
import re
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heatmap_signals import MODELS  # noqa: E402
from second_peak_repair import BPS, RES, Geometry, load, peak2_pixels, pooled_bands  # noqa: E402

D = RES / "flag_and_repair" / "spotcheck_review"
EXAMPLES = [("video10_f028582", "pupil_left"), ("video12_f019587", "pupil_right"), ("video01", None)]  # video01: the right repair with the largest deviation


def pick():
    I = pd.read_csv(D / "items.csv").merge(pd.read_csv(D / "spotcheck_review_verdicts.csv"), on="item")
    R = I[(I.status == "repaired") & (I.verdict == "yes")]
    rows = []
    for item, b in EXAMPLES:
        if b is None:
            r = R[R.item.str.startswith(item)].sort_values("fit_rel", ascending=False).iloc[0]
            b = r.points.split("+")[0]
        else:
            r = R[R.item == item].iloc[0]
        rows.append((int(r.video), int(r.frame), b, r.points.split("+")))
    return rows


def heatmaps(video, frames):
    rr, d, Pk, H, truth, fwd, sig = load(video)
    h5 = sorted((RES.parent / rr.unit / "training-data").glob("predictions_step*/*snapshot*120*.h5"))
    shuffle = int(re.search(r"shuffle(\d+)_", str(Pk.attrs.get("src", "")) or "").group(1)) if False else None
    pk = sorted((RES / "heatmap_signals").glob(f"video{video:02d}_shuffle*_peaks.csv"))[-1]
    shuffle = int(re.search(r"shuffle(\d+)", pk.name).group(1))
    from deeplabcut.pose_estimation_pytorch.apis.utils import get_pose_inference_runner
    folder = next(MODELS.glob(f"*shuffle{shuffle}"))
    snap = folder / "train" / "snapshot-120.pt"
    if not snap.exists():
        snap = next((folder / "train").glob("snapshot-best-120.pt"))
    runner = get_pose_inference_runner(model_config=folder / "train" / "pytorch_config.yaml", snapshot_path=snap, batch_size=1, device="cpu",
                                       inference_cfg={"multithreading": {"enabled": False}})
    pred = runner.model.heads["bodypart"].predictor
    orig, store = pred.forward, []

    def forward(stride, outputs):
        with torch.no_grad():
            Hm = outputs["heatmap"]
            if pred.apply_sigmoid:
                Hm = torch.sigmoid(Hm)
            store.append((Hm.float().clamp(min=0)[0].cpu().numpy(), float(stride)))
        return orig(stride, outputs)

    pred.forward = forward
    cap = cv2.VideoCapture(str(RES.parent / rr.unit / f"{rr.unit}.mp4"))
    imgs = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        assert ok
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cap.release()
    out = runner.inference(images=iter(imgs))
    xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
    p2, off = peak2_pixels(d, Pk)
    for f, o in zip(frames, out):
        got = np.asarray(o["bodyparts"])[0, :, :2]
        saved = np.array([xy[b][f] for b in BPS])
        assert np.abs(got - saved).max() < 1.5, (video, f, np.abs(got - saved).max())
    B, scale = pooled_bands(RES / "second_peak_repair")
    G = Geometry(xy, scale)
    Wd = np.linalg.norm(xy["eye_temporal_corner"] - xy["eye_nasal_corner"], axis=1)
    return imgs, store, xy, p2, off, G, Wd


def main():
    rows = pick()
    fig, axes = plt.subplots(len(rows), 4, figsize=(17, 3.7 * len(rows)), constrained_layout=True)
    heads = ["before: red = the model's 8 points", "heatmap of the thrown point, whole frame, 8 px per cell\nred ring = peak 1 (reported), blue ring = peak 2, dashed = the view",
             "the same heatmap in the view", "after: blue = the point moved to peak 2"]
    for i, (v, f, b, pts) in enumerate(rows):
        imgs, store, xy, p2, off, G, Wd = heatmaps(v, [f])
        img, (Hm, s) = imgs[0], store[0]
        k = BPS.index(b)
        hm = Hm[k]
        hmask = np.ma.masked_where(hm < 0.06 * hm.max(), hm)
        cx, cy = G.C[f]
        w = max(1.7 * Wd[f], 240)
        h = w * 0.75
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        px, py = xy[b][f]
        nx, ny = p2[b][f]
        hext = (off[b][0] - s / 2, hm.shape[1] * s + off[b][0] - s / 2, hm.shape[0] * s + off[b][1] - s / 2, off[b][1] - s / 2)
        v1 = hm.max()
        v2 = hm[int(round((ny - off[b][1]) / s)), int(round((nx - off[b][0]) / s))]
        outside = not (x0 <= px <= x1 and y0 <= py <= y1)
        for c, ax in enumerate(axes[i]):
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(heads[c], fontsize=11)
        ax = axes[i][0]
        ax.imshow(img)
        for bb in BPS:
            ax.plot(*xy[bb][f], "o", mfc="none", mec="#ff3c3c", mew=2, ms=10)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_xlabel(f"video {v}, frame {f}: {b.replace('_', ' ')} thrown" + (" out of this view" if outside else "") + f"\npeak 1 scores {v1:.2f}, peak 2 scores {v2:.2f}", fontsize=10, loc="left")
        ax = axes[i][1]
        ax.imshow(img)
        ax.imshow(hmask, extent=hext, cmap="autumn", alpha=0.95, vmin=0, vmax=hm.max(), interpolation="nearest")
        ax.plot(px, py, "o", mfc="none", mec="#ff3c3c", mew=2.5, ms=16)
        ax.plot(nx, ny, "o", mfc="none", mec="#5078ff", mew=2.5, ms=16)
        ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec="white", lw=1.2, ls="--"))
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax = axes[i][2]
        ax.imshow(img)
        ax.imshow(hmask, extent=hext, cmap="autumn", alpha=0.95, vmin=0, vmax=hm.max(), interpolation="nearest")
        ax.plot(nx, ny, "o", mfc="none", mec="#5078ff", mew=2.5, ms=16)
        if not outside:
            ax.plot(px, py, "o", mfc="none", mec="#ff3c3c", mew=2.5, ms=16)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax = axes[i][3]
        ax.imshow(img)
        for bb in BPS:
            rep = bb in pts
            ax.plot(*(p2[bb][f] if rep else xy[bb][f]), "o", mfc="none", mec="#5078ff" if rep else "#ff3c3c", mew=2, ms=10)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
    fig.savefig(D / "heatmap_examples.png", dpi=150)
    print("figure", D / "heatmap_examples.png")


if __name__ == "__main__":
    main()
