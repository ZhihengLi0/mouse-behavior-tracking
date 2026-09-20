#!/usr/bin/env python3
"""One scale step in the NEW pupil standard: train from scratch on N labels of a video, score on its frozen test set.

    python scale_step.py --unit first5minvedio --step 1 --shuffle 101              # train (+ evaluate if test labels exist)
    python scale_step.py --unit first5minvedio --step 1 --shuffle 101 --eval-only  # score an already trained step
    python scale_step.py --unit <video 2> --step 1 --shuffle 201 --prior first5minvedio:5

  train      = batches 01..step of --unit (20 labels each), plus ALL training batches of every --prior unit
               (name:last_batch) - this is how earlier videos carry over to a new one.
  validation = val20 of --unit (DeepLabCut's internal "test" split; monitors training only).
  test       = test50 of --unit, scored once per snapshot rule, never used for any decision.
Human labels are used exactly as saved (no correction, no snapping).

Recipe: ResNet-50, batch 2, from scratch, seed 42, snapshot every 10 epochs - as in the earlier units - but 120 epochs
with LR milestones [96, 114] (user decision 2026-09-20: with only 20-100 training images an epoch is 10-50 iterations,
and the 100-epoch run of step 1 was still improving on validation). Headline = FINAL snapshot (epoch 120),
robustness = best validation mAP among epochs >= 96. One metric: median over test frames of the per-frame RMSE over the labeled keypoints.
Own DeepLabCut project (dlc_projects/EyePupilEllipse-Zhiheng-2026-09-20) so old-standard labels never mix in.
"""
import argparse
import glob
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
OLD_CONFIG = ROOT / "dlc_projects/EyePupilBlink-Zhiheng-2026-08-17/config.yaml"
PROJECT = ROOT / "dlc_projects/EyePupilEllipse-Zhiheng-2026-09-20"
CONFIG = PROJECT / "config.yaml"
TASK, DATE, SCORER = "EyePupilEllipse", "Sep20", "Zhiheng"
EPOCHS, BATCH, SAVE_EVERY = 120, 2, 10
MILESTONES = [96, 114]            # the earlier [80, 95] of 100 epochs, scaled to 120 (same share of training at each learning rate)
BPS = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right", "eyelid_top", "eyelid_bottom", "eye_nasal_corner", "eye_temporal_corner"]


def log(msg):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def ensure_project(units):
    PROJECT.mkdir(parents=True, exist_ok=True)
    for d in ("labeled-data", "training-datasets", "dlc-models-pytorch", "videos"):
        (PROJECT / d).mkdir(exist_ok=True)
    cfg = yaml.safe_load(CONFIG.read_text()) if CONFIG.exists() else yaml.safe_load(OLD_CONFIG.read_text())
    cfg.update(Task=TASK, date=DATE, scorer=SCORER, project_path=str(PROJECT))
    if not CONFIG.exists():
        cfg["video_sets"], cfg["TrainingFraction"] = {}, [0.5]
    for u in units:                                     # labeled-data/<unit>/ is found through this key
        cfg["video_sets"].setdefault(str(PROJECT / "videos" / f"{u}.mp4"), {"crop": "0, 928, 0, 736"})
    CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


def ensure_fraction(f):
    cfg = yaml.safe_load(CONFIG.read_text())
    fr = [round(float(x), 2) for x in cfg["TrainingFraction"]]
    if round(f, 2) not in fr:
        cfg["TrainingFraction"].append(round(f, 2))
        CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        fr.append(round(f, 2))
    return fr.index(round(f, 2))


def human_table(folder):
    f = sorted(glob.glob(str(folder / "CollectedData_*.h5")))
    if not f:
        raise SystemExit(f"no saved labels in {folder}")
    return pd.read_hdf(f[0])


def stage(unit, last_batch, with_val):
    """Copy PNGs + the human labels of one unit into the DLC project, exactly as saved. Returns (train names, val names)."""
    lab = HERE / unit / "training-data" / "labels"
    dest = PROJECT / "labeled-data" / unit
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    parts, train, val = [], [], []
    sets = [f"batch{b:02d}" for b in range(1, last_batch + 1)] + (["val20"] if with_val else [])
    for s in sets:
        t = human_table(lab / s)
        names = [i[-1] if isinstance(i, tuple) else Path(str(i)).name for i in t.index]
        t.index = pd.MultiIndex.from_tuples([("labeled-data", unit, n) for n in names])
        for n in names:
            shutil.copy2(lab / s / n, dest / n)
        (val if s == "val20" else train).extend(names)
        parts.append(t)
    df = pd.concat(parts)
    assert not df.index.duplicated().any(), "a frame appears in two sets"
    df.to_hdf(dest / f"CollectedData_{SCORER}.h5", key="df_with_missing", mode="w")
    df.to_csv(dest / f"CollectedData_{SCORER}.csv")
    return train, val


def flat(df):
    df = df.copy()
    df.columns = df.columns.droplevel(0)
    while df.columns.nlevels > 2:
        df.columns = df.columns.droplevel(0)
    df.index = [i[-1] if isinstance(i, tuple) else Path(str(i)).name for i in df.index]
    return df


def evaluate(unit, shuffle, tsi, label, n_new, snapshot_index, rule, seed):
    """Same metric code as old_pupil_top/new-video-generalization/scripts/scale_step.py::evaluate."""
    import deeplabcut
    test = HERE / unit / "training-data" / "labels" / "test50"
    gt = flat(human_table(test))
    out = HERE / unit / "training-data" / "eval" / label
    out.mkdir(parents=True, exist_ok=True)
    deeplabcut.analyze_images(str(CONFIG), [str(test)], frame_type=".png", destfolder=str(out), shuffle=shuffle,
                              trainingsetindex=tsi, save_as_csv=True, plotting=False, pcutoff=0.0, device="cpu",
                              snapshot_index=snapshot_index)
    h5 = sorted(out.glob("image_predictions_*.h5"))[-1]
    pr = flat(pd.read_hdf(h5)).loc[gt.index]
    per = pd.DataFrame(index=gt.index)
    for b in BPS:
        per[f"{b}_error_px"] = np.hypot(pr[b]["x"] - gt[b]["x"], pr[b]["y"] - gt[b]["y"])
        per[f"{b}_likelihood"] = pr[b]["likelihood"]
    per.to_csv(out / "per_frame_errors.csv")
    e = per[[f"{b}_error_px" for b in BPS]]
    lk = per[[f"{b}_likelihood" for b in BPS]].to_numpy()
    labeled = e.notna().to_numpy()
    frame_rmse = np.sqrt((e ** 2).mean(axis=1))
    row = {"label": label, "unit": unit, "seed": seed, "training_frames_this_video": n_new,
           "median_frame_rmse_px": round(float(frame_rmse.median()), 2),
           "median_frame_mean_abs_px": round(float(e.mean(axis=1).median()), 2),
           "overall_rmse_px": round(float(np.sqrt(np.nanmean(e.to_numpy() ** 2))), 2),
           "p90_frame_rmse_px": round(float(frame_rmse.quantile(0.9)), 2),
           "frac_frames_rmse_gt_50px": round(float((frame_rmse > 50).mean()), 3),
           "frac_points_conf_ge_0.6": round(float((lk[labeled] >= 0.6).mean()), 3),
           "n_test_frames": len(e), "n_test_points": int(labeled.sum()),
           "snapshot_rule": f"{rule} ({h5.stem.split('snapshot_')[-1]})"}
    for b in BPS:
        row[f"median_{b}_px"] = round(float(e[f"{b}_error_px"].median()), 2)
    curve = HERE / unit / "results" / "scale_curve.csv"
    cur = pd.read_csv(curve) if curve.exists() else pd.DataFrame()
    cur = cur[cur["label"] != label] if len(cur) else cur
    pd.concat([cur, pd.DataFrame([row])], ignore_index=True).to_csv(curve, index=False)
    log(f"EVAL {label}: median frame RMSE {row['median_frame_rmse_px']} px | p90 {row['p90_frame_rmse_px']} px | "
        f">50px {row['frac_frames_rmse_gt_50px']:.1%} | conf>=0.6 {row['frac_points_conf_ge_0.6']:.0%}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--shuffle", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--prior", nargs="*", default=[], help="earlier videos as name:last_batch")
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    import deeplabcut

    prior = [(p.split(":")[0], int(p.split(":")[1])) for p in a.prior]
    ensure_project([u for u, _ in prior] + [a.unit])
    train_names = {}
    for u, last in prior:
        train_names[u], _ = stage(u, last, with_val=False)
    train_names[a.unit], val_names = stage(a.unit, a.step, with_val=True)
    n_new = len(train_names[a.unit])

    # the order of the merged table is whatever DeepLabCut produces: read it, then address rows by name
    from deeplabcut.generate_training_dataset.trainingsetmanipulation import merge_annotateddatasets
    cfg = deeplabcut.auxiliaryfunctions.read_config(str(CONFIG))
    tsf = Path(deeplabcut.auxiliaryfunctions.get_training_set_folder(cfg))
    (PROJECT / tsf).mkdir(parents=True, exist_ok=True)
    merged = merge_annotateddatasets(cfg, PROJECT / tsf)
    keys = [(i[-2], i[-1]) if isinstance(i, tuple) else tuple(Path(str(i)).parts[-2:]) for i in merged.index]
    want_train = {(u, n) for u, ns in train_names.items() for n in ns}
    want_val = {(a.unit, n) for n in val_names}
    train_idx = [k for k, key in enumerate(keys) if key in want_train]
    val_idx = [k for k, key in enumerate(keys) if key in want_val]
    assert len(train_idx) == len(want_train) and len(val_idx) == len(want_val) and len(keys) == len(train_idx) + len(val_idx), \
        (len(keys), len(train_idx), len(val_idx))
    fraction = round(len(train_idx) / len(keys), 2)
    tsi = ensure_fraction(fraction)
    pct = int(round(fraction * 100))
    base = f"x{n_new:03d}_step{a.step:02d}"
    tag = f"_{a.tag}" if a.tag else ""
    mdir = PROJECT / "dlc-models-pytorch" / "iteration-0" / f"{TASK}{DATE}-trainset{pct}shuffle{a.shuffle}" / "train"

    if not a.eval_only:
        log(f"{a.unit} step {a.step}: {len(train_idx)} train ({n_new} from this video, prior {dict(prior)}) / {len(val_idx)} val, "
            f"fraction {fraction}, tsi {tsi}, shuffle {a.shuffle}, seed {a.seed}")
        deeplabcut.create_training_dataset(str(CONFIG), Shuffles=[a.shuffle], trainIndices=[train_idx], testIndices=[val_idx],
                                           net_type="resnet_50", userfeedback=False, engine=deeplabcut.Engine.PYTORCH)
        pc = mdir / "pytorch_config.yaml"
        c = yaml.safe_load(pc.read_text())
        c["train_settings"].update(batch_size=BATCH, epochs=EPOCHS, seed=a.seed)
        c["runner"]["scheduler"]["params"]["milestones"] = MILESTONES
        c["runner"]["snapshots"].update(save_epochs=SAVE_EVERY, max_snapshots=14)
        pc.write_text(yaml.safe_dump(c, sort_keys=False))
        assert yaml.safe_load(pc.read_text())["runner"]["scheduler"]["params"]["milestones"] == MILESTONES
        log(f"recipe patched: batch {BATCH}, {EPOCHS} epochs, milestones {MILESTONES}, snapshot every {SAVE_EVERY}; training from scratch on mps")
        deeplabcut.train_network(str(CONFIG), shuffle=a.shuffle, trainingsetindex=tsi, epochs=EPOCHS, save_epochs=SAVE_EVERY,
                                 max_snapshots_to_keep=14, batch_size=BATCH, device="mps", snapshot_path=None)
        log("TRAINING DONE")

    snaps = sorted(p.name for p in mdir.glob("snapshot-*.pt"))
    find = lambda ep: next(k for k, n in enumerate(snaps) if n in (f"snapshot-{ep:03d}.pt", f"snapshot-best-{ep:03d}.pt"))
    print(f"TSI={tsi}\nFINAL_INDEX={find(EPOCHS)}", flush=True)
    test = HERE / a.unit / "training-data" / "labels" / "test50"
    if not glob.glob(str(test / "CollectedData_*.h5")):
        log("test50 is not labeled yet: skipping evaluation (rerun with --eval-only once it is)")
        return
    evaluate(a.unit, a.shuffle, tsi, f"{base}_final{tag}", n_new, find(EPOCHS), "final snapshot", a.seed)
    st = pd.read_csv(mdir / "learning_stats.csv")
    have = [e_ for e_ in (100, 110, 120) if any(n in (f"snapshot-{e_:03d}.pt", f"snapshot-best-{e_:03d}.pt") for n in snaps)]
    st = st[st["step"].isin(have) & st["metrics/test.mAP"].notna()]
    ep = int(st.loc[st["metrics/test.mAP"].idxmax(), "step"])
    evaluate(a.unit, a.shuffle, tsi, f"{base}_mAPlate{tag}", n_new, find(ep), f"best validation mAP among epochs >= {MILESTONES[0]}", a.seed)


if __name__ == "__main__":
    main()
