#!/usr/bin/env python3
"""The x = 0 point of a video's scale curve: the model of the EARLIER video(s) applied unchanged to this video's
frozen test set (no label of this video in training).

    python score_prior_model.py --unit 20251031_Pluto_spont_1 --prior-unit first5minvedio --shuffle 115 --tsi 4

Writes the rows x000_step00_final / x000_step00_mAPlate into <unit>/results/scale_curve.csv with the same metric
code (scale_step.evaluate) and the same two snapshot rules as every other step: final snapshot (epoch 120) and
the snapshot the prior video's own curve selected by validation mAP (read from its scale_curve.csv)."""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scale_step as ss  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--unit", required=True)
ap.add_argument("--prior-unit", required=True, help="the earlier video whose last model is applied")
ap.add_argument("--shuffle", type=int, required=True, help="DLC shuffle of that model")
ap.add_argument("--tsi", type=int, required=True, help="training-set index of that model")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

mdirs = list((ss.PROJECT / "dlc-models-pytorch" / "iteration-0").glob(f"*shuffle{a.shuffle}"))
assert len(mdirs) == 1, mdirs
snaps = sorted(p.name for p in (mdirs[0] / "train").glob("snapshot-*.pt"))
find = lambda ep: next(k for k, n in enumerate(snaps) if n in (f"snapshot-{ep:03d}.pt", f"snapshot-best-{ep:03d}.pt"))

prior = pd.read_csv(ss.HERE / a.prior_unit / "results" / "scale_curve.csv")
last = prior[prior["label"].str.endswith("_mAPlate")].sort_values("training_frames_this_video").iloc[-1]
ep_map = int(re.search(r"\((?:best-)?(\d+)\)", last["snapshot_rule"]).group(1))
print(f"prior model: {mdirs[0].name}; final snapshot = epoch {ss.EPOCHS}; its validation-mAP-selected snapshot = epoch {ep_map}")

ss.evaluate(a.unit, a.shuffle, a.tsi, "x000_step00_final", 0, find(ss.EPOCHS),
            f"final snapshot; {a.prior_unit} model applied unchanged", a.seed)
ss.evaluate(a.unit, a.shuffle, a.tsi, "x000_step00_mAPlate", 0, find(ep_map),
            f"best validation mAP among epochs >= {ss.MILESTONES[0]} on {a.prior_unit}; model applied unchanged", a.seed)
