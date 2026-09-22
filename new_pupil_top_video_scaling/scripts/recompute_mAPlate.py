#!/usr/bin/env python3
"""Recompute the secondary "_mAPlate" rows of a video's scale curve after its val20 labels were corrected.

    python recompute_mAPlate.py --unit 20251031_Pluto_spont_1 --prior first5minvedio:5 --steps 1:211 2:212 3:213 4:214 5:225

Nothing is retrained. For every step the DeepLabCut project is staged exactly as scale_step.py staged it for that
step (batches 01..N of this video + val20 + the earlier videos), so the merged table has the same row order as
the shuffle's Documentation pickle; the val20 rows now carry the corrected labels. Then DeepLabCut's own
evaluate_network scores the snapshots after the first LR drop (epochs >= MILESTONES[0]) on the validation split,
the snapshot with the highest validation mAP is chosen and scored on the frozen test set with scale_step.evaluate
(same metric code, label x0N0_stepNN_mAPlate). The headline "_final" rows are not touched.
Writes <unit>/results/val_mAP_by_snapshot.csv with every validation mAP that was computed."""
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scale_step as ss  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--unit", required=True)
ap.add_argument("--prior", nargs="*", default=[], help="earlier videos as name:last_batch (same as scale_step.py)")
ap.add_argument("--steps", nargs="+", required=True, help="step:shuffle pairs")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
import deeplabcut  # noqa: E402
from deeplabcut.generate_training_dataset.trainingsetmanipulation import merge_annotateddatasets  # noqa: E402

prior = [(p.split(":")[0], int(p.split(":")[1])) for p in a.prior]
cfg = deeplabcut.auxiliaryfunctions.read_config(str(ss.CONFIG))
tsf = ss.PROJECT / Path(deeplabcut.auxiliaryfunctions.get_training_set_folder(cfg))
rows = []
for spec in a.steps:
    step, shuffle = (int(v) for v in spec.split(":"))
    # 1) stage the project as it was for this step, with the CURRENT (corrected) label files
    train_names = {}
    for u, last in prior:
        train_names[u], _ = ss.stage(u, last, with_val=False)
    train_names[a.unit], val_names = ss.stage(a.unit, step, with_val=True)
    merged = merge_annotateddatasets(cfg, tsf)
    keys = [(i[-2], i[-1]) if isinstance(i, tuple) else tuple(Path(str(i)).parts[-2:]) for i in merged.index]
    n_train = sum(len(v) for v in train_names.values())
    fraction = round(n_train / len(keys), 2)
    tsi = ss.ensure_fraction(fraction)
    pct = int(round(fraction * 100))
    # 2) the shuffle's recorded split must point at exactly the val20 frames -> row order unchanged
    doc = tsf / f"Documentation_data-{cfg['Task']}_{pct}shuffle{shuffle}.pickle"
    with open(doc, "rb") as f:
        meta = pickle.load(f)
    val_idx = [int(i) for i in meta[2]]
    got = sorted(keys[i][1] for i in val_idx)
    assert got == sorted(val_names), f"step {step}: the recorded validation split does not match val20 ({got[:3]}...)"
    assert len(meta[1]) == n_train, (len(meta[1]), n_train)
    # 3) validation mAP of the late snapshots with the corrected labels
    mdir = ss.PROJECT / "dlc-models-pytorch" / "iteration-0" / f"{ss.TASK}{ss.DATE}-trainset{pct}shuffle{shuffle}" / "train"
    snaps = sorted(p.name for p in mdir.glob("snapshot-*.pt"))
    find = lambda ep: next(k for k, n in enumerate(snaps) if n in (f"snapshot-{ep:03d}.pt", f"snapshot-best-{ep:03d}.pt"))
    late = [e for e in range(10, ss.EPOCHS + 1, 10) if e >= ss.MILESTONES[0]
            and any(n in (f"snapshot-{e:03d}.pt", f"snapshot-best-{e:03d}.pt") for n in snaps)]
    names = [snaps[find(e)][:-3] for e in late]
    deeplabcut.evaluate_network(str(ss.CONFIG), Shuffles=[shuffle], trainingsetindex=tsi, snapshots_to_evaluate=names,
                                device="cpu", plotting=False)
    ev = ss.PROJECT / "evaluation-results-pytorch" / "iteration-0" / f"{ss.TASK}{ss.DATE}-trainset{pct}shuffle{shuffle}"
    # one "<scorer>_<snapshot>-results.csv" per snapshot; columns "Training epochs", "test mAP" (test = the validation split)
    res = pd.concat([pd.read_csv(p) for p in ev.rglob("*-results.csv")], ignore_index=True)
    res.columns = [c.strip() for c in res.columns]
    maps = {}
    for e, nm in zip(late, names):
        sel = res[(res["Training epochs"] == e) & (res["Shuffle number"] == shuffle)]
        maps[e] = float(sel["test mAP"].iloc[-1])
        rows.append({"step": step, "shuffle": shuffle, "epoch": e, "snapshot": nm, "val_mAP_corrected_val20": maps[e],
                     "val_rmse_px": float(sel["test rmse"].iloc[-1])})
    best_ep = max(maps, key=maps.get)
    ss.log(f"step {step} (shuffle {shuffle}): validation mAP with corrected val20 {maps} -> epoch {best_ep}")
    # 4) score the selected snapshot on the frozen test set (same metric code as every other row)
    n_new = len(train_names[a.unit])
    ss.evaluate(a.unit, shuffle, tsi, f"x{n_new:03d}_step{step:02d}_mAPlate", n_new, find(best_ep),
                f"best validation mAP among epochs >= {ss.MILESTONES[0]}", a.seed)
out = ss.HERE / a.unit / "results" / "val_mAP_by_snapshot.csv"
pd.DataFrame(rows).to_csv(out, index=False)
ss.log(f"wrote {out}")
