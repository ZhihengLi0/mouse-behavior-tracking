#!/usr/bin/env python3
"""Pre-labels in the NEW pupil standard for one extracted frame set (labeling aid only).

    python make_prelabels.py <folder with img*.png> [--shuffle 60 --tsi 11]

1. The old-standard production model predicts the eight keypoints on the PNGs (CPU, a few minutes).
2. The four pupil points are replaced by the endpoints of the axis-aligned ellipse through the predicted
   left / right / bottom points - the three pupil points shown to be true pupil edges (2026-09-20):
       centre = (mean x of L,R ; mean y of L,R)   half-width a = |R.x - L.x| / 2   half-height b = B.y - centre.y
       left = (cx - a, cy)  right = (cx + a, cy)  top = (cx, cy - b)  bottom = (cx, cy + b)
   The old pupil_top prediction (visible upper edge) is discarded. Eyelids and eye corners are kept as predicted.
3. Written as machinelabels.h5/.csv so napari-deeplabcut shows them as editable points.

Frame SELECTION never uses a model; these points are only a starting position that the human corrects.
"""
import argparse
import glob
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "dlc_projects/EyePupilBlink-Zhiheng-2026-08-17/config.yaml"


def to_ellipse_standard(df):
    """df: DLC table (scorer, bodyparts, coords). Returns a copy with the four pupil points on the ellipse."""
    out = df.copy()
    col = {(c[-2], c[-1]): c for c in df.columns}                 # works with or without an "individuals" level
    g = lambda b, c: df[col[(b, c)]].to_numpy(float)
    cx, cy = (g("pupil_left", "x") + g("pupil_right", "x")) / 2, (g("pupil_left", "y") + g("pupil_right", "y")) / 2
    a, b = np.abs(g("pupil_right", "x") - g("pupil_left", "x")) / 2, g("pupil_bottom", "y") - cy
    sign = np.sign(g("pupil_right", "x") - g("pupil_left", "x"))
    for bp, x, y in (("pupil_left", cx - sign * a, cy), ("pupil_right", cx + sign * a, cy), ("pupil_top", cx, cy - b), ("pupil_bottom", cx, cy + b)):
        out[col[(bp, "x")]], out[col[(bp, "y")]] = x, y
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--shuffle", type=int, default=60)
    ap.add_argument("--tsi", type=int, default=11)
    ap.add_argument("--snapshot-index", type=int, default=-1)
    ap.add_argument("--new-standard-model", action="store_true",
                    help="predict with a model of the NEW-standard project (EyePupilEllipse); its points are used as they are")
    a = ap.parse_args()
    folder = Path(a.folder).resolve()
    import deeplabcut
    config = ROOT / "dlc_projects/EyePupilEllipse-Zhiheng-2026-09-20/config.yaml" if a.new_standard_model else CONFIG

    with tempfile.TemporaryDirectory() as tmp:
        deeplabcut.analyze_images(str(config), [str(folder)], frame_type=".png", destfolder=tmp, shuffle=a.shuffle,
                                  trainingsetindex=a.tsi, save_as_csv=False, plotting=False, pcutoff=0.0, device="cpu",
                                  snapshot_index=a.snapshot_index)
        pred = pd.read_hdf(sorted(glob.glob(tmp + "/image_predictions_*.h5"))[-1])
    if pred.columns.nlevels == 4:                                  # single-animal project: drop the "individuals" level
        pred.columns = pred.columns.droplevel(1)
    names = [i[-1] if isinstance(i, tuple) else Path(str(i)).name for i in pred.index]
    pred.index = pd.MultiIndex.from_tuples([("labeled-data", folder.name, n) for n in names])
    pred = pred.sort_index()
    ml = pred if a.new_standard_model else to_ellipse_standard(pred)
    ml.to_hdf(folder / "machinelabels.h5", key="df_with_missing", mode="w")
    ml.to_csv(folder / "machinelabels.csv")
    print(f"{folder.name}: {len(ml)} frames pre-labeled ({'new-standard model, points as predicted' if a.new_standard_model else 'old model converted to ellipse endpoints'}; shuffle {a.shuffle}, tsi {a.tsi})")
