#!/usr/bin/env python3
"""Make the four human pupil points exact ellipse endpoints (new pupil standard, 2026-09-20).

    python snap_to_ellipse.py <label folder>

The standard says pupil_left/right/top/bottom ARE the endpoints of one axis-aligned ellipse:
    left = (cx - a, cy)   right = (cx + a, cy)   top = (cx, cy - b)   bottom = (cx, cy + b)
Four hand-placed points never satisfy that exactly, so they are replaced by the least-squares closest
endpoint set, which has a closed form:
    cx = mean of the four x      a = (R.x - L.x) / 2
    cy = mean of the four y      b = (B.y - T.y) / 2
Frames with a missing pupil point are left untouched (and listed). Eyelids and eye corners are never changed.
The untouched human file is kept once as CollectedData_<scorer>_raw_human.h5/.csv.
"""
import glob
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

folder = Path(sys.argv[1]).resolve()
h5 = Path(sorted(glob.glob(str(folder / "CollectedData_*.h5")))[0])
h5 = next(p for p in map(Path, sorted(glob.glob(str(folder / "CollectedData_*.h5")))) if "raw_human" not in p.name)
raw = h5.with_name(h5.stem + "_raw_human.h5")
if not raw.exists():
    shutil.copy2(h5, raw); shutil.copy2(h5.with_suffix(".csv"), raw.with_suffix(".csv"))
df = pd.read_hdf(raw)                                   # always snap from the untouched human labels
sc = df.columns.get_level_values(0)[0]
P = ["pupil_left", "pupil_right", "pupil_top", "pupil_bottom"]
X = np.stack([df[(sc, p, "x")].to_numpy(float) for p in P]); Y = np.stack([df[(sc, p, "y")].to_numpy(float) for p in P])
ok = np.isfinite(X).all(0) & np.isfinite(Y).all(0)
cx, cy = X.mean(0), Y.mean(0)
a, b = (X[1] - X[0]) / 2, (Y[3] - Y[2]) / 2
new = {"pupil_left": (cx - a, cy), "pupil_right": (cx + a, cy), "pupil_top": (cx, cy - b), "pupil_bottom": (cx, cy + b)}
out = df.copy(); moved = {}
for k, p in enumerate(P):
    nx, ny = np.where(ok, new[p][0], X[k]), np.where(ok, new[p][1], Y[k])
    moved[p] = np.hypot(nx - X[k], ny - Y[k])[ok]
    out[(sc, p, "x")], out[(sc, p, "y")] = nx, ny
out.to_hdf(h5, key="df_with_missing", mode="w"); out.to_csv(h5.with_suffix(".csv"))
names = [i[-1] if isinstance(i, tuple) else str(i) for i in df.index]
print(f"{folder.name}: snapped {int(ok.sum())}/{len(df)} frames; untouched (missing pupil point): {[n for n, o in zip(names, ok) if not o]}")
print("  distance each human point moved (px), median / max: " + ", ".join(f"{p.split('_')[1]} {np.median(v):.1f}/{v.max():.1f}" for p, v in moved.items()))
print(f"  height/width median {np.median((b / a)[ok]):.2f}; raw human file kept as {raw.name}")
