#!/usr/bin/env python3
"""Audit 2026-11-08: the minimum distance of the heatmap second peak from the first (heatmap_signals.MIN_CELLS, 3 cells
= 24 px at stride 8, set by hand) tested at 2, 3 and 4 cells on the 1330 truth frames.

    python second_peak_distance_test.py [--cells 2 3 4]

For every video the plateau model's heatmap peaks are recomputed on the truth frames only (heatmap_signals.peaks_for_frames,
one file per setting: results/flag_and_repair/second_peak_distance/videoNN_cells{c}.csv), then flag_and_repair.py is run
with --peaks-csv on that file (output in a scratch folder, whole-video counts meaningless because only the truth frames
have peaks) and the truth-frame summary is collected: second_peak_distance.csv (per video and setting) and
second_peak_distance_totals.csv. The 3-cell row must reproduce the committed run (15 / 4 / 1). Nothing existing is modified."""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heatmap_signals as HS  # noqa: E402
from second_peak_repair import PL, RES, load  # noqa: E402

OUT = RES / "flag_and_repair" / "second_peak_distance"
PIPE = ["--finder", "either", "--accept", "labelregion", "--points", "all", "--replace", "smallest"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, nargs="+", default=[2, 3, 4])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in sorted(PL[PL.status == "final"].video.astype(int)):
        shuffle, step, h5 = HS.plateau_shuffle(v)
        folder = next(HS.MODELS.glob(f"*shuffle{shuffle}"))
        snap = folder / "train" / "snapshot-120.pt"
        if not snap.exists():
            snap = next((folder / "train").glob("snapshot-best-120.pt"))
        truth = load(v)[4]
        want = truth.index.to_numpy()
        for c in a.cells:
            peaks = OUT / f"video{v:02d}_cells{c}.csv"
            if not peaks.exists():
                HS.MIN_CELLS = c
                HS.peaks_for_frames(v, folder, snap, h5, want, peaks)
            out = OUT / "runs" / f"video{v:02d}_cells{c}"
            if not (out / "summary.csv").exists():
                with open(OUT / f"video{v:02d}_cells{c}.log", "w") as lf:
                    subprocess.run([sys.executable, "flag_and_repair.py", "--video", str(v), *PIPE, "--peaks-csv", str(peaks), "--out", str(out)],
                                   check=True, cwd=str(Path(__file__).resolve().parent), stdout=lf, stderr=subprocess.STDOUT)
            S = pd.read_csv(out / "summary.csv").iloc[0]
            rows.append({"video": v, "cells": c, "px": c * 8, "wrong_before": int(S.wrong_before), "wrong_repaired_to_good": int(S.wrong_repaired_to_good),
                         "wrong_still_flagged": int(S.wrong_flagged_not_repaired), "closed": int(S.closed), "good_made_worse": int(S.good_made_worse),
                         "good_repaired": int(S.good_repaired), "median_after_pct": S.median_error_after_pct})
            print(rows[-1])
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "second_peak_distance.csv", index=False)
    T = D.groupby("cells")[["wrong_before", "wrong_repaired_to_good", "wrong_still_flagged", "closed", "good_made_worse", "good_repaired"]].sum().reset_index()
    T["px_at_stride_8"] = T.cells * 8
    T.to_csv(OUT / "second_peak_distance_totals.csv", index=False)
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
