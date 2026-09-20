#!/usr/bin/env python3
"""Re-score the completed active-learning experiment under the FINAL-snapshot
rule (epoch 100), for consistency with the new-video experiment.

The original points used DeepLabCut's default (best validation mAP over all
snapshots), which chose a pre-LR-decay snapshot (epoch < 80) in 20 of 34
models. Same frozen 100-frame test set, same metric definitions; CPU only.
Writes active-learning/results/convergence_final_snapshot.csv.
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_eye_test_set as ev  # noqa: E402  (paths of the frozen test set)
import deeplabcut  # noqa: E402

P = ROOT / "dlc_projects/EyePupilBlink-Zhiheng-2026-08-17"
B = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right",
     "eyelid_top", "eyelid_bottom", "eye_nasal_corner", "eye_temporal_corner"]
BASE = {"uncertain": 31, "jump": 41, "fitting": 51}
OUT = ROOT / "active-learning/results/convergence_final_snapshot.csv"
conv = pd.read_csv(ROOT / "active-learning/results/convergence.csv")
fractions = [round(float(x), 2) for x in yaml.safe_load(open(P / "config.yaml"))["TrainingFraction"]]


def flat(p):
    d = pd.read_hdf(p)
    d.index = [ev.frame_name(i) for i in d.index]
    return ev.drop_scorer_level(d).sort_index()


gt = flat(ev.TEST_LABELS)
done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
cache = {}
for _, r in conv.iterrows():
    lab = r["evaluated_label"]
    if len(done) and ((done["branch"] == r["branch"]) & (done["round"] == r["round"])).any():
        continue
    if lab not in cache:
        sh = 30 if r["round"] == 0 else BASE[r["branch"]] + int(r["round"]) - 1
        frac = round(r["cumulative_training_frames"] / (r["cumulative_training_frames"] + 20), 2)
        tdir = glob.glob(str(P / "dlc-models-pytorch/iteration-0" / f"*trainset{int(round(frac * 100))}shuffle{sh}" / "train"))[0]
        snaps = sorted(os.path.basename(x) for x in glob.glob(tdir + "/snapshot-*.pt"))
        final = "snapshot-100.pt" if "snapshot-100.pt" in snaps else "snapshot-best-100.pt"
        best = [s for s in snaps if "best" in s][0]
        dest = ev.TEST_ROOT / f"predictions_final_{lab}"
        dest.mkdir(exist_ok=True)
        deeplabcut.analyze_images(str(ev.MAIN_CONFIG), [str(ev.TEST_LABEL_DIR)], frame_type=".png", destfolder=str(dest),
                                  shuffle=sh, trainingsetindex=fractions.index(frac), save_as_csv=False, plotting=False,
                                  pcutoff=0.0, device="cpu", snapshot_index=snaps.index(final))
        pr = flat(sorted(dest.glob("image_predictions_*.h5"))[-1]).loc[gt.index]
        e = pd.DataFrame({b: np.hypot(pr[b]["x"] - gt[b]["x"], pr[b]["y"] - gt[b]["y"]) for b in B})
        fr = np.sqrt((e ** 2).mean(axis=1))
        lk = np.array([pr[b]["likelihood"] for b in B]).T
        cache[lab] = dict(final_snapshot=final, mAP_best_snapshot=best,
                          external_overall_rmse_px=round(float(np.sqrt(np.nanmean(e.to_numpy() ** 2))), 2),
                          median_frame_rmse_px=round(float(fr.median()), 2),
                          median_frame_mean_abs_px=round(float(e.mean(axis=1).median()), 2),
                          p90_frame_rmse_px=round(float(fr.quantile(0.9)), 2),
                          frac_frames_rmse_gt_50px=round(float((fr > 50).mean()), 3),
                          points_likelihood_ge_0_6=int((lk >= 0.6).sum()))
    row = dict(branch=r["branch"], round=int(r["round"]), cumulative_training_frames=int(r["cumulative_training_frames"]),
               evaluated_label=lab, **cache[lab])
    done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
    done.to_csv(OUT, index=False)
    print(f"{lab}: final {row['median_frame_rmse_px']} px (mAP-best was {r['median_frame_rmse_px']} px, chose {row['mAP_best_snapshot']})", flush=True)
print("RESCORE DONE")
