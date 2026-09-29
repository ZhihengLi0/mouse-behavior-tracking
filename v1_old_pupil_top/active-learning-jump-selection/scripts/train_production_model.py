#!/usr/bin/env python3
"""Consolidated production model (post-experiment, 2026-09-14).

The experiment kept branches isolated by design. With the arc closed, train
ONE model on the union of every human-reviewed label we own:

    seed 100 (80 train / frozen 20 val)
  + uncertain rounds 1-11  + jump rounds 1-11  + fitting rounds 1-11
  + the labeled-but-untrained round-12 sets (all three branches)

deduplicated by frame (keep last). Recipe identical to the frozen protocol
(ResNet-50, batch 2, 100 epochs from scratch, best snapshot by the frozen
20-frame validation). The report-only test set is scored ONCE, labeled
al_production_v1, and the number is recorded outside convergence.csv.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round_train_pipeline as rtp

SHUFFLE = 60
LABEL = "al_production_v1"

merged_parts = []
for branch in rtp.BRANCHES:
    merged_parts.append(rtp.cumulative_table(branch, 11))
    merged_parts.append(rtp.normalized_branch_table(branch, 12))
merged = pd.concat(merged_parts)
merged = merged[~merged.index.duplicated(keep="last")].sort_index()
rtp.log(f"production: merged pool = {len(merged)} unique reviewed frames (+100 seed)")

# stage the merged pool
if rtp.STAGING.exists():
    shutil.rmtree(rtp.STAGING)
rtp.STAGING.mkdir(parents=True)
merged.to_hdf(rtp.STAGING / "CollectedData_Zhiheng.h5", key="df_with_missing", mode="w")
for _, _, img in merged.index:
    found = False
    for branch in rtp.BRANCHES:
        cands = [rtp.UNIT / "frames" / branch / img,
                 rtp.UNIT / "frames" / "round12" / branch / img]
        cands += [rtp.branch_store(branch) / f"round{r}_frames" / img for r in range(1, 12)]
        for cand in cands:
            if cand.exists():
                shutil.copy(cand, rtp.STAGING / img)
                found = True
                break
        if found:
            break
    if not found:
        raise RuntimeError(f"png not found for {img}")
rtp.log(f"production: staged {len(merged)} frames")

train_idx, val_idx, total = rtp.combined_indices(len(merged))
fraction2 = round(len(train_idx) / total, 2)
tsi = rtp.ensure_fraction(fraction2)
fraction_pct = int(round(fraction2 * 100))
rtp.log(f"production: {len(train_idx)} train / {len(val_idx)} val "
        f"(fraction {fraction2}, trainingsetindex {tsi}, shuffle {SHUFFLE})")

import deeplabcut

deeplabcut.create_training_dataset(
    str(rtp.CONFIG), Shuffles=[SHUFFLE],
    trainIndices=[train_idx], testIndices=[val_idx],
    net_type="resnet_50", userfeedback=False,
    engine=deeplabcut.Engine.PYTORCH)
rtp.patch_model_config(fraction_pct, SHUFFLE, rtp.BATCH)

rtp.run([rtp.PYTHON, rtp.ROOT / "scripts/train_eye_model.py",
         "--shuffle", SHUFFLE, "--batch-size", rtp.BATCH, "--epochs", rtp.EPOCHS,
         "--device", "mps", "--trainset-fraction", fraction_pct,
         "--trainingsetindex", tsi, "--save-epochs", 10,
         "--max-snapshots", 12, "--no-resume"])

rtp.run([rtp.PYTHON, rtp.ROOT / "scripts/evaluate_eye_test_set.py",
         "--train-frames", 100, "--shuffle", SHUFFLE,
         "--trainingsetindex", tsi, "--snapshot-index", -1,
         "--model-label", LABEL])

rtp.unstage()
rtp.log("PRODUCTION MODEL COMPLETE")
