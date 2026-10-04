"""Shared helpers for the keypoint-definition figures: load a label set, draw one frame with its eight keypoints."""
import glob
import os

import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
V2 = os.path.join(ROOT, "v2_new_pupil_top_video_scaling")
PUPIL = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right"]
LID = ["eyelid_top", "eyelid_bottom", "eye_nasal_corner", "eye_temporal_corner"]
SHORT = {"pupil_top": "PT", "pupil_bottom": "PB", "pupil_left": "PL", "pupil_right": "PR", "eyelid_top": "ET",
         "eyelid_bottom": "EB", "eye_nasal_corner": "NC", "eye_temporal_corner": "TC"}
RED, CYAN, YELLOW = "#ff4d4d", "#35c9f0", "#ffd84d"


def load(label_dir):
    """Label set -> DataFrame indexed by image file name, columns (bodypart, coord)."""
    df = pd.read_hdf(glob.glob(os.path.join(label_dir, "CollectedData_*.h5"))[0])
    df.index = [i[-1] if isinstance(i, tuple) else os.path.basename(i) for i in df.index]
    df.columns = pd.MultiIndex.from_tuples([(c[-2], c[-1]) for c in df.columns])
    return df


def image(label_dir, name, alt_dirs=()):
    for d in (label_dir, *alt_dirs):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return np.asarray(Image.open(p).convert("L"))
    raise FileNotFoundError(name)


def draw(ax, img, row, ellipse=True, names=True, color_pupil=RED, color_lid=CYAN, only=None, ms=5, alpha=1.0, gain=1.0):
    """Draw the frame and the labelled points of one row (a Series indexed by (bodypart, coord))."""
    if img is not None:
        ax.imshow(np.clip(img.astype(float) * gain, 0, 255), cmap="gray", vmin=0, vmax=255)
    pts = {b: (row[(b, "x")], row[(b, "y")]) for b in PUPIL + LID if (b, "x") in row.index and np.isfinite(row[(b, "x")])}
    if ellipse and all(b in pts for b in PUPIL):
        t, b, l, r = (np.array(pts[k]) for k in PUPIL)
        c = (l + r) / 2
        ax.add_patch(Ellipse(c, np.linalg.norm(r - l), np.linalg.norm(b - t), angle=np.degrees(np.arctan2(*(r - l)[::-1])),
                             fill=False, ec=color_pupil, lw=0.9, alpha=0.8 * alpha))
    for b, (x, y) in pts.items():
        if only and b not in only:
            continue
        col = color_pupil if b in PUPIL else color_lid
        ax.plot(x, y, "o", ms=ms, mfc=col, mec="white", mew=0.6, alpha=alpha)
        if names:
            ax.annotate(SHORT[b], (x, y), xytext=(4, -5), textcoords="offset points", color=col, fontsize=7, weight="bold")
    ax.set_xticks([]); ax.set_yticks([])


def crop(ax, row, pad=70, img_shape=(736, 928), box=None):
    """Zoom to the eye: bounding box of the labelled points plus a margin (or an explicit box x0, x1, y0, y1)."""
    if box is None:
        xs = np.array([row[(b, "x")] for b in PUPIL + LID if (b, "x") in row.index]); ys = np.array([row[(b, "y")] for b in PUPIL + LID if (b, "y") in row.index])
        box = (np.nanmin(xs) - pad, np.nanmax(xs) + pad, np.nanmin(ys) - pad, np.nanmax(ys) + pad)
    ax.set_xlim(max(0, box[0]), min(img_shape[1], box[1])); ax.set_ylim(min(img_shape[0], box[3]), max(0, box[2]))
