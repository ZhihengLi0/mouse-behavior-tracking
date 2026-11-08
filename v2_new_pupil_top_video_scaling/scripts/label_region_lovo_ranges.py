#!/usr/bin/env python3
"""Leave-one-video-out acceptance ranges, one set per held-out video, for the audit of the repair (2026-11-07).

    python label_region_lovo_ranges.py

The acceptance ranges in use (label_region_margin.csv, label_region_outline.csv) are built from the labels of all 19
videos, so when the repair is judged on the truth frames of video v the range already contains v's own labels. This
script writes, for every video v, the same ranges built from the other 18 videos only, with the same margins
(0.42 pupil, 0.23 outline), so that `flag_and_repair.py --lovo-range` can repair video v as if it were a new video.
Outputs results/acceptance_thresholds/label_region_lovo_ranges.csv and label_region_outline_lovo_ranges.csv
(held_out_video, quantity, min, max, margin). Nothing existing is modified."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acceptance_thresholds import QUANTS, label_quantities, region  # noqa: E402
from outline_region import OQUANTS, region as region_out  # noqa: E402
from second_peak_repair import BPS, PL, RES, Geometry, load, pooled_bands  # noqa: E402

OUT = RES / "acceptance_thresholds"


def main():
    B, scale = pooled_bands(RES / "second_peak_repair")
    L = []
    for v in sorted(PL[PL.status == "final"].video.astype(int)):
        rr, d, P, H, truth, fwd, sig = load(v)
        xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
        Q = label_quantities(truth, Geometry(xy, scale))
        Q.insert(0, "video", v)
        L.append(Q)
        print(f"video {v}: {len(Q)} labeled frames")
    L = pd.concat(L, ignore_index=True)
    L.round(4).to_csv(OUT / "label_quantities.csv", index=False)
    m = float(pd.read_csv(OUT / "label_region_margin.csv").margin.iloc[0])
    rows = []
    for v in sorted(L.video.unique()):
        R = region(L[L.video != v], m)
        R.insert(0, "held_out_video", v)
        R["margin"] = m
        rows.append(R)
    pd.concat(rows, ignore_index=True).round(4).to_csv(OUT / "label_region_lovo_ranges.csv", index=False)
    Lo = pd.read_csv(OUT / "label_outline_quantities.csv")
    mo = float(pd.read_csv(OUT / "label_region_outline.csv").margin.iloc[0])
    rows = []
    for v in sorted(Lo.video.unique()):
        R = region_out(Lo[Lo.video != v], mo).reset_index()
        R.insert(0, "held_out_video", v)
        R["margin"] = mo
        rows.append(R)
    pd.concat(rows, ignore_index=True).round(4).to_csv(OUT / "label_region_outline_lovo_ranges.csv", index=False)
    print(f"pupil margin {m}, outline margin {mo}: ranges for {L.video.nunique()} held-out videos written")


if __name__ == "__main__":
    main()
