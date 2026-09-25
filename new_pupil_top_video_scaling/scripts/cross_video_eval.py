#!/usr/bin/env python3
"""Every model ever trained in this unit, scored on every video's frozen test set (user request 2026-09-23).

    python cross_video_eval.py            # score all (model, test video) pairs not yet in the matrix, then plot
    python cross_video_eval.py --plot     # only redraw from results/cross_video_matrix.csv

Models are the final snapshots (epoch 120) of every completed step, in the order they were trained; the training
set of each model = all labels of the earlier videos + the labels of its own video up to that step. The same
metric code as scale_step.py (median over the 50 test frames of the per-frame RMSE over the labeled keypoints).
Outputs: results/cross_video_matrix.csv, results/cross_video_curves.png (error of every test set as the model
history advances), results/cross_video_matrix.png (heatmap)."""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_step import BPS, CONFIG, EPOCHS, HERE, PROJECT, flat  # noqa: E402

# (unit, mouse, date, [(step, shuffle, labels of this video)])   - chronological order of training
# "auto:NN" = discover the finished steps of that video from its shuffles NN1..NN9 (step = last digit)
VIDEOS = [
    ("0_first5minvedio", "mouse A", "5-min recording", [(1, 111, 20), (2, 112, 40), (3, 113, 60), (4, 114, 80), (5, 115, 100)]),
    ("1_20251031_Pluto_spont_1", "mouse B (Pluto)", "2025-10-31", [(1, 211, 20), (2, 212, 40), (3, 213, 60), (4, 214, 80), (5, 225, 100), (6, 216, 120), (7, 217, 140)]),
    ("2_20251031_pluton2", "mouse B (Pluto)", "2025-10-31", [(1, 321, 20), (2, 322, 40), (3, 323, 60), (4, 324, 80)]),
    ("3_20251031_pluto3", "mouse B (Pluto)", "2025-10-31", [(1, 411, 20), (2, 422, 40), (3, 423, 60)]),
    ("4_20251030_Pluto_spont_1", "mouse B (Pluto)", "2025-10-30", "auto:51"),
    ("5_20251029_Pluto_spont_1", "mouse B (Pluto)", "2025-10-29", "auto:61"),
    ("6_20251028_Pluto_spont_1", "mouse B (Pluto)", "2025-10-28", "auto:71"),
    ("7_20251027_Pluto_spont1", "mouse B (Pluto)", "2025-10-27", "auto:81"),
]
# labels of a video that are carried into later videos (the "prior" used when the next video started)
CARRIED = {"0_first5minvedio": 100, "1_20251031_Pluto_spont_1": 100, "2_20251031_pluton2": 60, "3_20251031_pluto3": 60}
# videos the labeler judged hard to read by eye (pupil boundary barely visible); marked in the figures
POOR_QUALITY = {"4_20251030_Pluto_spont_1"}
PALETTE = ["#2F6B9A", "#D1495B", "#2A9D8F", "#E08E45", "#7B4EA3", "#8C6D31", "#444444"]


def steps_of(spec):
    if not isinstance(spec, str):
        return spec
    base = int(spec.split(":")[1])
    out = []
    for k in range(1, 10):
        try:
            tsi_of(base * 10 + k)
        except (StopIteration, ValueError, IndexError, FileNotFoundError):
            continue
        out.append((k, base * 10 + k, 20 * k))
    return out


def test_units():
    """videos whose frozen test set exists (a new video joins as soon as its test50 has been labeled and frozen)"""
    return [v[0] for v in VIDEOS if (HERE / v[0] / "training-data" / "labels" / "test_frozen" / "test50_labels.h5").exists()]


def label_of(u):
    i = int(u.split("_")[0])
    lab = f"video {i} (5 min, mouse A)" if i == 0 else f"video {i} (Pluto, {next(v[2] for v in VIDEOS if v[0] == u)})"
    return lab + (" [POOR QUALITY: pupil hard to see]" if u in POOR_QUALITY else "")


R = HERE / "results"
OUT = R / "cross_video_matrix.csv"


def models():
    rows, idx, carried = [], 0, 0
    for u, mouse, date, spec in VIDEOS:
        steps = steps_of(spec)
        for step, shuffle, n in steps:
            rows.append(dict(model_idx=idx, model_unit=u, mouse=mouse, date=date, step=step, shuffle=shuffle,
                             labels_this_video=n, total_labels=carried + n))
            idx += 1
        carried += CARRIED.get(u, steps[-1][2] if steps else 0)
    return pd.DataFrame(rows)


def tsi_of(shuffle):
    d = next((PROJECT / "dlc-models-pytorch" / "iteration-0").glob(f"*shuffle{shuffle}"))
    pct = int(d.name.split("trainset")[1].split("shuffle")[0]) / 100
    fr = [round(float(x), 2) for x in yaml.safe_load(CONFIG.read_text())["TrainingFraction"]]
    snaps = sorted(p.name for p in (d / "train").glob("snapshot-*.pt"))
    final = next(k for k, s in enumerate(snaps) if s in (f"snapshot-{EPOCHS:03d}.pt", f"snapshot-best-{EPOCHS:03d}.pt"))
    return fr.index(round(pct, 2)), final


def score(shuffle, tsi, final_idx, test_unit):
    import deeplabcut
    test = HERE / test_unit / "training-data" / "labels" / "test50"
    gt = flat(pd.read_hdf(HERE / test_unit / "training-data" / "labels" / "test_frozen" / "test50_labels.h5"))
    out = HERE / test_unit / "training-data" / "eval" / "cross" / f"shuffle{shuffle}"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("image_predictions_*"):
        old.unlink()
    deeplabcut.analyze_images(str(CONFIG), [str(test)], frame_type=".png", destfolder=str(out), shuffle=shuffle,
                              trainingsetindex=tsi, save_as_csv=False, plotting=False, pcutoff=0.0, device="cpu", snapshot_index=final_idx)
    pr = flat(pd.read_hdf(sorted(out.glob("image_predictions_*.h5"))[-1])).loc[gt.index]
    e = pd.DataFrame({b: np.hypot(pr[b]["x"] - gt[b]["x"], pr[b]["y"] - gt[b]["y"]) for b in BPS})
    lk = np.stack([pr[b]["likelihood"].to_numpy(float) for b in BPS], axis=1)
    frame_rmse = np.sqrt((e ** 2).mean(axis=1))
    res = {"median_frame_rmse_px": round(float(frame_rmse.median()), 2), "p90_frame_rmse_px": round(float(frame_rmse.quantile(0.9)), 2),
           "frac_frames_rmse_gt_50px": round(float((frame_rmse > 50).mean()), 3),
           "frac_points_conf_ge_0.6": round(float((lk[e.notna().to_numpy()] >= 0.6).mean()), 3)}
    res.update({f"median_{b}_px": round(float(e[b].median()), 2) for b in BPS})
    return res


def plot(m):
    units = test_units()
    colors = {v[0]: PALETTE[i % len(PALETTE)] for i, v in enumerate(VIDEOS)}
    short = {v[0]: label_of(v[0]) for v in VIDEOS}
    mods = models()
    m = m[m.shuffle.isin(mods.shuffle)]
    fig, ax = plt.subplots(figsize=(19, 8), constrained_layout=True)
    for tu in units:
        s = m[m.test_unit == tu].sort_values("model_idx")
        ax.plot(s.model_idx, s.median_frame_rmse_px, "o-", color=colors[tu], lw=2, ms=6, label=f"test set of {short[tu]}")
        own = mods[mods.model_unit == tu].model_idx.min()
        if not np.isfinite(own):
            continue
        ax.axvline(own - 0.5, color=colors[tu], ls=":", lw=1)
        ax.annotate("own labels start", (own - 0.5, ax.get_ylim()[1] if False else 1.2), color=colors[tu], fontsize=8, rotation=90, va="bottom", ha="right")
    ax.set_yscale("log"); ax.set_ylabel("median frame RMSE on that video's 50 frozen test frames (px, log scale)")
    ax.set_xticks(mods.model_idx)
    ax.set_xticklabels([f"{r.labels_this_video}\n({r.total_labels})" for r in mods.itertuples()], fontsize=8)
    ax.set_xlabel("model, in training order: labels of its own video (total labels in its training set)")
    # video spans below the axis
    for u, mouse, date, _ in VIDEOS:
        g = mods[mods.model_unit == u]
        if g.empty:
            continue
        x0, x1 = g.model_idx.min() - 0.5, g.model_idx.max() + 0.5
        ax.axvspan(x0, x1, color=colors[u], alpha=0.06)
        head = f"video {u.split('_')[0]}" + (" (POOR QUALITY)" if u in POOR_QUALITY else "")
        ax.text((x0 + x1) / 2, 1.02, f"{head}\n{date}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color=colors[u])
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Every model of the sequence, scored on every video's frozen test set (final snapshot, epoch 120)\n"
                 "left of a dotted line = that video not yet in the training set (its 0-label regime); labels are 20 per step", fontsize=11, pad=34)
    fig.savefig(R / "cross_video_curves.png", dpi=130)
    # heatmap
    piv = m.pivot(index="test_unit", columns="model_idx", values="median_frame_rmse_px").reindex(units)
    fig, ax = plt.subplots(figsize=(19, 4.2), constrained_layout=True)
    im = ax.imshow(np.log10(piv.to_numpy()), aspect="auto", cmap="viridis_r")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}" if v < 100 else f"{v:.0f}", ha="center", va="center", fontsize=7.5, color="w" if v > 6 else "k")
    ax.set_yticks(range(len(units))); ax.set_yticklabels([short[u] for u in units], fontsize=9)
    ax.set_xticks(mods.model_idx); ax.set_xticklabels([f"{r.model_unit.split('_')[0]}:{r.labels_this_video}" for r in mods.itertuples()], fontsize=8, rotation=90)
    ax.set_xlabel("model (video index : labels of that video)"); ax.set_title("median frame RMSE (px) of each model on each test set", fontsize=11)
    fig.colorbar(im, ax=ax, label="log10 px", fraction=0.02)
    fig.savefig(R / "cross_video_matrix.png", dpi=130)
    print("saved", R / "cross_video_curves.png", R / "cross_video_matrix.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--plot", action="store_true"); a = ap.parse_args()
    mods = models()
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["shuffle", "test_unit"])
    if not a.plot:
        rows = []
        for r in mods.itertuples():
            tsi, final = tsi_of(r.shuffle)
            for tu in test_units():
                if len(done) and ((done.shuffle == r.shuffle) & (done.test_unit == tu)).any():
                    continue
                res = score(r.shuffle, tsi, final, tu)
                row = {**r._asdict(), "test_unit": tu, **res}; row.pop("Index", None)
                done = pd.concat([done, pd.DataFrame([row])], ignore_index=True); done.to_csv(OUT, index=False)
                print(f"[{r.model_idx:2d}] {r.model_unit} step {r.step} (shuffle {r.shuffle}) -> {tu}: {res['median_frame_rmse_px']} px", flush=True)
    plot(pd.read_csv(OUT))
