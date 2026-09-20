#!/usr/bin/env python3
"""Score the snapshot-rule comparison points that scale_step.py did not produce at the time
(steps 1-3 predate the ">= 80" rule; step 4 never scored DLC's default pick).

Report-only: these rows are secondary marks in 03_scale_curve.png and feed no decision.
The epoch for each rule is read from learning_stats.csv (validation mAP), never from the test set.

    python backfill_snapshot_rules.py
"""
import glob
from pathlib import Path

import pandas as pd

import scale_step as ss

rtp = ss.rtp
STEPS = [(1, 61, 97, 20), (2, 72, 97, 40), (3, 73, 97, 60), (4, 84, 94, 80)]   # step, shuffle, trainset %, Pluto frames
done = set(pd.read_csv(ss.CURVE)["label"])

for step, shuffle, pct, n_new in STEPS:
    tdir = glob.glob(str(rtp.PROJECT / "dlc-models-pytorch" / "iteration-0" / f"*trainset{pct}shuffle{shuffle}" / "train"))[0]
    snaps = sorted(Path(x).name for x in glob.glob(tdir + "/snapshot-*.pt"))
    find = lambda ep: next(k for k, n in enumerate(snaps) if n in (f"snapshot-{ep:03d}.pt", f"snapshot-best-{ep:03d}.pt"))
    tsi = rtp.ensure_fraction(pct / 100)
    st = pd.read_csv(tdir + "/learning_stats.csv")
    st = st[st["metrics/test.mAP"].notna()]
    ep_all = int(st.loc[st["metrics/test.mAP"].idxmax(), "step"])
    late = st[st["step"] >= 80]
    ep_late = int(late.loc[late["metrics/test.mAP"].idxmax(), "step"])
    base = f"x{n_new:03d}_step{step:02d}"
    for label, ep, rule in [(f"{base}_mAP80plus", ep_late, "best validation mAP among epochs >= 80"),
                            (f"{base}_mAPall", ep_all, f"best validation mAP -> epoch {ep_all}")]:
        legacy = f"{base}_scratch"            # steps 1-3 already hold the all-epochs row under this label
        if label in done or (label.endswith("_mAPall") and legacy in done):
            continue
        ss.evaluate(shuffle, tsi, label, "seed42", n_new, snapshot_index=find(ep), rule=rule)
