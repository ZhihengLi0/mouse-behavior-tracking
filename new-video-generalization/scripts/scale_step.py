#!/usr/bin/env python3
"""One step of the new-video scale experiment: build the dataset, train, evaluate.

    python scale_step.py --step 1 --shuffle 61
    python scale_step.py --baseline            # score production_v1 as x = 0

Dataset for step N (FROZEN_PARAMETERS.md):
  train = the 676 old-video reviewed frames (80 seed-train + 596 merged)
          + Pluto batch01..batchN (20 each)
  val   = the 20 Pluto validation frames (snapshot selection only)
  the 20 old-video validation frames are left unused.
Every step trains FROM SCRATCH (frozen recipe, 100 epochs from ImageNet
weights) - advisor decision 2026-09-19: with only 20 new frames, continuing
from an existing model risks overfitting to them.
Evaluation: the frozen 59-frame Pluto test set, one definition of the metric
(median over frames of the RMSE over the labeled keypoints of that frame).
"""
import argparse
import glob
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

UNIT = Path(__file__).resolve().parents[1]
ROOT = UNIT.parent
sys.path.insert(0, str(ROOT / "active-learning" / "scripts"))
import round_train_pipeline as rtp  # noqa: E402

TD = UNIT / "training-data"
LABELS = TD / "labels"
PLUTO = "pluto_spont_1"
PLUTO_DIR = rtp.PROJECT / "labeled-data" / PLUTO
TEST_GT = LABELS / "test_frozen" / "test59_labels.h5"
TEST_FRAMES = LABELS / "test100"
CURVE = UNIT / "results" / "scale_curve.csv"
PROD_SHUFFLE, PROD_TSI = 60, 11
BPS = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right",
       "eyelid_top", "eyelid_bottom", "eye_nasal_corner", "eye_temporal_corner"]


def flat(df):
    df = df.copy()
    df.columns = df.columns.droplevel(0)
    while df.columns.nlevels > 2:
        df.columns = df.columns.droplevel(0)
    df.index = [i[-1] if isinstance(i, tuple) else Path(i).name for i in df.index]
    return df


def evaluate(shuffle, tsi, label, arm, n_new, snapshot_index=-1, rule="best validation mAP"):
    import deeplabcut
    out = TD / "eval" / label
    out.mkdir(parents=True, exist_ok=True)
    deeplabcut.analyze_images(str(rtp.CONFIG), [str(TEST_FRAMES)], frame_type=".png",
                              destfolder=str(out), shuffle=shuffle, trainingsetindex=tsi,
                              save_as_csv=True, plotting=False, pcutoff=0.0, device="cpu",
                              snapshot_index=snapshot_index)
    gt = flat(pd.read_hdf(TEST_GT))
    pr = flat(pd.read_hdf(sorted(out.glob("image_predictions_*.h5"))[-1])).loc[gt.index]
    per = pd.DataFrame(index=gt.index)
    for b in BPS:
        per[f"{b}_error_px"] = np.hypot(pr[b]["x"] - gt[b]["x"], pr[b]["y"] - gt[b]["y"])
        per[f"{b}_likelihood"] = pr[b]["likelihood"]
    per.to_csv(out / "per_frame_errors.csv")
    e = per[[f"{b}_error_px" for b in BPS]]
    lk = per[[f"{b}_likelihood" for b in BPS]].to_numpy()
    labeled = e.notna().to_numpy()
    row = {
        "label": label, "arm": arm, "pluto_training_frames": n_new,
        "median_frame_rmse_px": round(float(np.sqrt((e ** 2).mean(axis=1)).median()), 2),
        "median_frame_mean_abs_px": round(float(e.mean(axis=1).median()), 2),
        "overall_rmse_px": round(float(np.sqrt(np.nanmean(e.to_numpy() ** 2))), 2),
        # the project's goal is the error TAIL (find and fix failing frames); the median saturates early
        "p90_frame_rmse_px": round(float(np.sqrt((e ** 2).mean(axis=1)).quantile(0.9)), 2),
        "frac_frames_rmse_gt_50px": round(float((np.sqrt((e ** 2).mean(axis=1)) > 50).mean()), 3),
        "frac_points_conf_ge_0.6": round(float((lk[labeled] >= 0.6).mean()), 3),
        "n_test_frames": len(e), "n_test_points": int(labeled.sum()),
        "snapshot_rule": f"{rule} ({sorted(out.glob('image_predictions_*.h5'))[-1].stem.split('snapshot_')[-1]})",
    }
    cur = pd.read_csv(CURVE) if CURVE.exists() else pd.DataFrame()
    cur = cur[cur["label"] != label] if len(cur) else cur
    pd.concat([cur, pd.DataFrame([row])], ignore_index=True).to_csv(CURVE, index=False)
    rtp.log(f"EVAL {label}: median frame RMSE {row['median_frame_rmse_px']} px | mean-abs median "
            f"{row['median_frame_mean_abs_px']} px | overall RMSE {row['overall_rmse_px']} px | "
            f"conf>=0.6 {row['frac_points_conf_ge_0.6']:.0%}")
    return row


def stage_old():
    parts = []
    for branch in rtp.BRANCHES:
        parts += [rtp.cumulative_table(branch, 11), rtp.normalized_branch_table(branch, 12)]
    merged = pd.concat(parts)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    if rtp.STAGING.exists():
        shutil.rmtree(rtp.STAGING)
    rtp.STAGING.mkdir(parents=True)
    merged.to_hdf(rtp.STAGING / "CollectedData_Zhiheng.h5", key="df_with_missing", mode="w")
    for _, _, img in merged.index:
        for branch in rtp.BRANCHES:
            cands = [rtp.UNIT / "frames" / branch / img, rtp.UNIT / "frames" / "round12" / branch / img]
            cands += [rtp.branch_store(branch) / f"round{r}_frames" / img for r in range(1, 12)]
            hit = next((c for c in cands if c.exists()), None)
            if hit:
                shutil.copy(hit, rtp.STAGING / img)
                break
        else:
            raise RuntimeError(f"png not found for {img}")
    return merged


def stage_pluto(step):
    if PLUTO_DIR.exists():
        shutil.rmtree(PLUTO_DIR)
    PLUTO_DIR.mkdir(parents=True)
    tabs, n_train = [], 0
    for name in [f"batch{k:02d}" for k in range(1, step + 1)] + ["val20"]:
        d = LABELS / name
        t = pd.read_hdf(d / "CollectedData_Zhiheng.h5")
        imgs = [i[-1] if isinstance(i, tuple) else Path(i).name for i in t.index]
        t.index = pd.MultiIndex.from_tuples([("labeled-data", PLUTO, i) for i in imgs])
        for i in imgs:
            shutil.copy(d / i, PLUTO_DIR / i)
        tabs.append(t)
        if name != "val20":
            n_train += len(t)
    table = pd.concat(tabs)               # batches first, val last - order is relied upon below
    table.to_hdf(PLUTO_DIR / "CollectedData_Zhiheng.h5", key="df_with_missing", mode="w")
    table.to_csv(PLUTO_DIR / "CollectedData_Zhiheng.csv")
    cfg = yaml.safe_load(rtp.CONFIG.read_text())
    key = str(rtp.PROJECT / "videos" / f"{PLUTO}.mp4")
    if key not in cfg["video_sets"]:
        cfg["video_sets"][key] = {"crop": "0, 928, 0, 736"}
        rtp.CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return table, n_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--shuffle", type=int)
    ap.add_argument("--baseline", action="store_true")
    a = ap.parse_args()

    if a.baseline:
        evaluate(PROD_SHUFFLE, PROD_TSI, "x000_production_v1", "unadapted", 0)
        return

    import deeplabcut
    old = stage_old()
    pl, n_new = stage_pluto(a.step)
    seed = pd.read_hdf(rtp.SEED_LABELS)
    seed_names = [str(v[-1] if isinstance(v, tuple) else v) for v in seed.index]
    old_val = set(rtp.SPLIT["validation_filenames"])
    n_seed, n_old = len(seed), len(old)
    train_idx = [i for i, n in enumerate(seed_names) if n not in old_val]
    train_idx += list(range(n_seed, n_seed + n_old + n_new))
    val_idx = list(range(n_seed + n_old + n_new, n_seed + n_old + len(pl)))
    fraction = round(len(train_idx) / (len(train_idx) + len(val_idx)), 2)
    tsi = rtp.ensure_fraction(fraction)
    pct = int(round(fraction * 100))
    rtp.log(f"step {a.step}: {len(train_idx)} train ({n_new} Pluto) / {len(val_idx)} Pluto val, "
            f"fraction {fraction}, tsi {tsi}, shuffle {a.shuffle}")

    deeplabcut.create_training_dataset(str(rtp.CONFIG), Shuffles=[a.shuffle], trainIndices=[train_idx],
                                       testIndices=[val_idx], net_type="resnet_50", userfeedback=False,
                                       engine=deeplabcut.Engine.PYTORCH)
    # verify DLC's combined table really has the order the indices assume
    comb = pd.read_hdf(glob.glob(str(rtp.PROJECT / "training-datasets" / "iteration-0" / "*" /
                                     "CollectedData_Zhiheng.h5"))[0])
    names = [(i[-2], i[-1]) if isinstance(i, tuple) else tuple(Path(i).parts[-2:]) for i in comb.index]
    assert len(comb) == n_seed + n_old + len(pl), (len(comb), n_seed, n_old, len(pl))
    assert all(names[i][0] == PLUTO for i in val_idx), "validation indices do not point at Pluto rows"
    val_imgs = {i[-1] for i in pl.index[n_new:]}
    assert {names[i][1] for i in val_idx} == val_imgs, "validation rows are not the val20 frames"
    assert sum(names[i][0] == PLUTO for i in train_idx) == n_new, "Pluto training rows miscounted"
    assert not any(names[i][1] in old_val and names[i][0] == "face" for i in train_idx)
    rtp.log("index check passed: val = Pluto val20, train includes exactly the Pluto batch rows")
    rtp.patch_model_config(pct, a.shuffle, rtp.BATCH)

    label = f"x{n_new:03d}_step{a.step:02d}_scratch"
    rtp.run([rtp.PYTHON, rtp.ROOT / "scripts/train_eye_model.py", "--shuffle", a.shuffle,
             "--batch-size", rtp.BATCH, "--epochs", rtp.EPOCHS, "--device", "mps",
             "--trainset-fraction", pct, "--trainingsetindex", tsi, "--save-epochs", 10,
             "--max-snapshots", 12, "--no-resume"])
    # Score under BOTH snapshot rules until the rule is decided (2026-09-19: best-mAP on 20
    # validation frames picked an undertrained epoch-20 snapshot at step 2).
    evaluate(a.shuffle, tsi, label, "scratch", n_new)
    tdir = glob.glob(str(rtp.PROJECT / "dlc-models-pytorch" / "iteration-0" / f"*shuffle{a.shuffle}" / "train"))[0]
    snaps = sorted(Path(x).name for x in glob.glob(tdir + "/snapshot-*.pt"))
    fin = snaps.index(f"snapshot-{int(rtp.EPOCHS):03d}.pt")
    evaluate(a.shuffle, tsi, label.replace("_scratch", "_final"), "scratch", n_new,
             snapshot_index=fin, rule="final snapshot")
    print(f"FINAL_INDEX={fin}")
    print(f"TSI={tsi}")


if __name__ == "__main__":
    main()
