"""How large is 10 px? One test frame of video 16 (human labels, frozen test set), full frame + 3x zoom on the eye, with a
10 px scale bar and a 10 px circle around every labeled keypoint (a prediction inside the circle is within 10 px).
Eye width = corner-to-corner distance of this frame's labels; pupil size = left-right and top-bottom distances.
Output: results/scale_10px/scale_10px_video16.png. Asked by Kaiwen on 2026-10-09 ("10 个 pixel 大概在这个视频上距离会有多远")."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
HERE = Path(__file__).resolve().parent.parent
U = "16_20251010_Terra_spont_1"; L = HERE / U / "training-data" / "labels"
d = pd.read_hdf(L / "test_frozen" / "test50_labels.h5"); sc = d.columns.get_level_values(0)[0]
xy = lambda r, b: (float(d.loc[r, (sc, b, "x")]), float(d.loc[r, (sc, b, "y")]))
P = ["pupil_top", "pupil_bottom", "pupil_left", "pupil_right", "eyelid_top", "eyelid_bottom", "eye_nasal_corner", "eye_temporal_corner"]
r = d.index[len(d) // 2]
name = r[-1] if isinstance(r, tuple) else Path(str(r)).name
img = plt.imread(L / "test50" / name)
pts = {b: xy(r, b) for b in P}
W = np.hypot(*np.subtract(pts["eye_temporal_corner"], pts["eye_nasal_corner"]))
pw = np.hypot(*np.subtract(pts["pupil_right"], pts["pupil_left"])); ph = np.hypot(*np.subtract(pts["pupil_bottom"], pts["pupil_top"]))
col = {"pupil": "#2a9df4", "eyelid": "#f4a62a", "eye": "#3cb44b"}
fig, ax = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
for a, zoom in zip(ax, (False, True)):
    a.imshow(img, cmap="gray"); a.set_axis_off()
    for b, (x, y) in pts.items():
        c = col[b.split("_")[0]]
        a.add_patch(Circle((x, y), 10, fill=False, ec=c, lw=1.6 if zoom else 1.0))
        a.plot(x, y, "+", color=c, ms=8 if zoom else 5, mew=1.5)
    cx = np.mean([p[0] for p in pts.values()]); cy = np.mean([p[1] for p in pts.values()])
    if zoom:
        qx = (pts["pupil_left"][0] + pts["pupil_right"][0]) / 2; qy = (pts["pupil_top"][1] + pts["pupil_bottom"][1]) / 2
        h = 0.36 * W; a.set_xlim(qx - h, qx + h); a.set_ylim(qy + 0.8 * h, qy - 0.8 * h); zf = img.shape[1] / (2 * h)
        bx, by = qx - 0.9 * h, qy + 0.68 * h
    else:
        bx, by = 40, img.shape[0] - 40
        qx = (pts["pupil_left"][0] + pts["pupil_right"][0]) / 2; qy = (pts["pupil_top"][1] + pts["pupil_bottom"][1]) / 2
        a.add_patch(Rectangle((qx - 0.36 * W, qy - 0.288 * W), 0.72 * W, 0.576 * W, fill=False, ec="w", lw=1, ls="--"))
    a.plot([bx, bx + 10], [by, by], color="#e6194b", lw=4 if zoom else 3, solid_capstyle="butt")
    a.text(bx + 14, by, "10 px", color="#e6194b", va="center", fontsize=12 if zoom else 10, fontweight="bold")
ax[0].set_title(f"Video 16 (Terra, 2025-10-10), test frame {name}, human labels\nfull frame {img.shape[1]} x {img.shape[0]} px; dashed box = zoom", fontsize=10)
ax[1].set_title(f"Zoom on the pupil ({zf:.1f}x): red bar = 10 px;\ncircle of radius 10 px around each labeled point", fontsize=10)
fig.text(0.5, 0.015, f"In this frame: eye width (corner to corner) {W:.0f} px, so 10 px = {1000 / W:.1f}% of the eye width; pupil {pw:.0f} px wide and {ph:.0f} px tall, "
         f"so 10 px = {1000 / pw:.1f}% of its width.\nBlue = pupil points, orange = eyelid points, green = eye corners.", ha="center", fontsize=10)
fig.tight_layout(rect=(0, 0.08, 1, 0.97))
out = HERE / "results" / "scale_10px" / "scale_10px_video16.png"; fig.savefig(out, dpi=150); print("saved", out, f"| W {W:.0f} px, pupil {pw:.0f} x {ph:.0f} px")
