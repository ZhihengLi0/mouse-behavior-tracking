#!/usr/bin/env python3
"""Where each keypoint sits in every human label set of the 5-minute video (first mouse), before and after 2026-09-20.

    python definition_check.py

The camera and the eye position are the same in all sets, so the median image coordinates of a point can be compared
between label sets even when the frames differ. Writes ../results/definition_by_label_set.csv: one row per label set with
the median x / y of every keypoint and a few derived distances. Used to check (1) that all label sets made before
2026-09-20 place the points in the same way and (2) how the placement differs from the present definitions."""
import glob
import os

import numpy as np
import pandas as pd

from kp_render import LID, PUPIL, ROOT, V2

OLDP = os.path.join(ROOT, "dlc_projects/EyePupilBlink-Zhiheng-2026-08-17/labeled-data")
AL = os.path.join(ROOT, "v1_old_pupil_top/active-learning-jump-selection/frames")
SETS = [  # (period, label set, files)
    ("before 2026-09-20", "training pool, 100 frames (batch-size and model selection)", [f"{OLDP}/face/CollectedData_Zhiheng.h5"]),
    ("before 2026-09-20", "active learning, branch jump, rounds 1-11", sorted(glob.glob(f"{AL}/round*/jump/CollectedData_*.h5"))),
    ("before 2026-09-20", "active learning, branch uncertain, rounds 1-11", sorted(glob.glob(f"{AL}/round*/uncertain/CollectedData_*.h5"))),
    ("before 2026-09-20", "active learning, branch fitting, rounds 1-11", sorted(glob.glob(f"{AL}/round*/fitting/CollectedData_*.h5"))),
    ("before 2026-09-20", "test set, 100 frames of the final minute", [f"{ROOT}/local_data/test_sets/eye_last_minute_100/dlc_label_project/labeled-data/eye_last_minute_100/CollectedData_Zhiheng.h5"]),
    ("2026-09-20 18:18", "video 0 test set, 50 frames of the final minute (pupil as ellipse, corners not yet moved)",
     [f"{ROOT}/local_data/backups/newstd_labels_cornerV1_20260920_1818/first5minvedio/training-data/labels/test50/CollectedData_Zhiheng.h5"]),
    ("present", "video 0 test set, 50 frames of the final minute", [f"{V2}/0_first5minvedio/training-data/labels/test50/CollectedData_Zhiheng.h5"]),
]


def load(f):
    d = pd.read_hdf(f)
    d.columns = pd.MultiIndex.from_tuples([(c[-2], c[-1]) for c in d.columns])
    return d.astype(float)


rows = []
for period, name, files in SETS:
    d = pd.concat([load(f) for f in files])
    g = lambda b, c: d[(b, c)]
    r = {"period": period, "label_set": name, "frames": len(d)}
    for b in PUPIL + LID:
        r[f"{b}_x"], r[f"{b}_y"] = round(g(b, "x").median()), round(g(b, "y").median())
    r["corner_to_corner_px"] = round(np.hypot(g("eye_temporal_corner", "x") - g("eye_nasal_corner", "x"), g("eye_temporal_corner", "y") - g("eye_nasal_corner", "y")).median())
    r["pupil_top_above_eyelid_top_px"] = round((g("eyelid_top", "y") - g("pupil_top", "y")).median())      # negative = pupil top BELOW the eyelid point
    r["pupil_left_higher_than_right_px"] = round((g("pupil_right", "y") - g("pupil_left", "y")).median())
    rows.append(r)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "definition_by_label_set.csv"), index=False)
pd.set_option("display.width", 250)
print(out[["period", "label_set", "frames", "eye_nasal_corner_x", "eye_temporal_corner_x", "corner_to_corner_px", "eyelid_top_y", "eyelid_bottom_y",
           "pupil_top_y", "pupil_top_above_eyelid_top_px", "pupil_left_higher_than_right_px"]].to_string(index=False))
