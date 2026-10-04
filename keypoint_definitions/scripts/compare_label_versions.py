#!/usr/bin/env python3
"""How far each keypoint moved at every change of the labelling definition, measured on frames labelled under both versions.

    python compare_label_versions.py

Writes ../results/label_changes.csv (one row per change and keypoint): number of frames with the point in both versions,
how many moved by more than 2 px, the median displacement, and the mean shift in x and y (image coordinates: +x = right,
+y = down)."""
import os

import numpy as np
import pandas as pd

from kp_render import LID, PUPIL, ROOT, V2, load

OLD = os.path.join(ROOT, "v1_old_pupil_top/new-video-generalization/training-data/labels")
BK = os.path.join(ROOT, "local_data/backups/newstd_labels_cornerV1_20260920_1818/first5minvedio/training-data/labels")
L = "training-data/labels"
CHANGES = [  # (change, old label sets, new label sets)
    ("A: visible-edge standard -> ellipse standard + outer corners (Pluto 2025-10-31, same frames)",
     [f"{OLD}/batch01", f"{OLD}/val20"], [f"{V2}/1_20251031_Pluto_spont_1/{L}/batch01", f"{V2}/1_20251031_Pluto_spont_1/{L}/val20"]),
    ("B: first corner definition -> outer corners (5-min video, same frames, both under the ellipse standard)",
     [f"{BK}/{s}" for s in ("test50", "val20", "batch01")], [f"{V2}/0_first5minvedio/{L}/{s}" for s in ("test50", "val20", "batch01")]),
    ("C: convention v1 -> v2, double contour = left one (video 6 test set)",
     [f"{V2}/6_20251028_Pluto_spont_1/training-data/_archive_2026-09-26_convention1/labels/test50"], [f"{V2}/6_20251028_Pluto_spont_1/{L}/test50"]),
    ("C: convention v1 -> v2 (video 7 test set)",
     [f"{V2}/7_20251027_Pluto_spont1/{L}/_archive_2026-09-29_convention1/test50"], [f"{V2}/7_20251027_Pluto_spont1/{L}/test50"]),
    ("C: convention v1 -> v2 (video 8 test set)",
     [f"{V2}/8_20251024_Pluto_spont1/{L}/_archive_2026-09-29_convention1/test50"], [f"{V2}/8_20251024_Pluto_spont1/{L}/test50"]),
]
rows = []
for name, olds, news in CHANGES:
    d = pd.concat([load(n).sub(load(o)).dropna(how="all") for o, n in zip(olds, news)])
    for bp in PUPIL + LID:
        dx, dy = d[(bp, "x")].dropna(), d[(bp, "y")].dropna()
        r = np.hypot(dx, dy)
        rows.append(dict(change=name, keypoint=bp, frames=len(r), moved_gt_2px=int((r > 2).sum()), median_shift_px=round(r.median(), 1),
                         mean_dx_px=round(dx.mean(), 1), mean_dy_px=round(dy.mean(), 1)))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "label_changes.csv"), index=False)
print(out.to_string(index=False))
