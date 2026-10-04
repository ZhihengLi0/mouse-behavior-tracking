#!/usr/bin/env python3
"""Example figures for the keypoint definitions (../figures/*.jpg), drawn from the human label sets on disk.

    python make_figures.py

All frames are brightened for display (gamma 0.45) so that the dark eye region can be read; the labels were placed on
the same frames in napari. Red = pupil points and the ellipse through them, cyan = eyelid and corner points,
yellow = the earlier position of a point in the figures that show a change."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kp_render import CYAN, LID, PUPIL, RED, ROOT, V2, YELLOW, crop, draw, image, load

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
L = "training-data/labels"
OLD = os.path.join(ROOT, "v1_old_pupil_top/new-video-generalization/training-data/labels")
BK = os.path.join(ROOT, "local_data/backups/newstd_labels_cornerV1_20260920_1818/first5minvedio/training-data/labels")
MICE = [("mouse A, 5-min video (unit 0)", f"{V2}/0_first5minvedio/{L}/test50", "img015001.png"),
        ("Pluto, 2025-10-31 (unit 1)", f"{V2}/1_20251031_Pluto_spont_1/{L}/test50", "img068086.png"),
        ("Terra, 2025-10-17 (unit 11)", f"{V2}/11_20251017_Terra_spont_1/{L}/test50", "img068086.png")]


def bright(img):
    return 255 * (img / 255.0) ** 0.45


def xy(row, bp):
    return np.array([row[(bp, "x")], row[(bp, "y")]])


def margins(ax, row):
    """Dashed guide: a parabola through nasal corner - lid midpoint - temporal corner for each eyelid margin."""
    n, t = xy(row, "eye_nasal_corner"), xy(row, "eye_temporal_corner")
    for lid in ("eyelid_top", "eyelid_bottom"):
        m = xy(row, lid)
        if not np.isfinite([*n, *t, *m]).all():
            continue
        s = np.linspace(0, 1, 50)[:, None]
        c = 2 * m - (n + t) / 2  # control point of the quadratic Bezier that passes through m at s = 0.5
        ax.plot(*(((1 - s) ** 2) * n + 2 * s * (1 - s) * c + s ** 2 * t).T, "--", color=CYAN, lw=0.8, alpha=0.7)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=110, pil_kwargs={"quality": 88}); plt.close(fig); print("saved", name)


def change_panel(ax, old_dir, new_dir, name, points, pad=45):
    """One frame: earlier positions (yellow) -> current positions (red / cyan), arrows for the points that moved."""
    A, B = load(old_dir).loc[name], load(new_dir).loc[name]
    draw(ax, bright(image(new_dir, name, (old_dir,))), B, names=True)
    for bp in points:
        a, b = xy(A, bp), xy(B, bp)
        if np.isfinite([*a, *b]).all() and np.hypot(*(b - a)) > 2:
            ax.plot(*a, "o", ms=6, mfc="none", mec=YELLOW, mew=1.5)
            ax.annotate("", b, a, arrowprops=dict(arrowstyle="->", color=YELLOW, lw=1.2))
    crop(ax, B, pad=pad)


# 1. all eight points on one frame of each mouse
fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=True)
for ax, (title, d, name) in zip(axes, MICE):
    row = load(d).loc[name]
    draw(ax, bright(image(d, name)), row, ms=6); margins(ax, row); crop(ax, row, pad=60); ax.set_title(title, fontsize=11)
fig.suptitle("The eight keypoints. PT/PB/PL/PR = pupil top/bottom/left/right (ends of the two axes of the pupil ellipse); ET/EB = midpoints of the upper/lower "
             "eyelid margin; NC/TC = nasal/temporal corner. Dashed = eyelid margins (guide only)", fontsize=10)
save(fig, "fig1_all_keypoints.jpg")

# 2. the two corners, zoomed
fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), constrained_layout=True)
for j, (title, d, name) in enumerate(MICE):
    row = load(d).loc[name]
    for i, (bp, lab) in enumerate((("eye_nasal_corner", "nasal corner (left in the image)"), ("eye_temporal_corner", "temporal corner (right in the image)"))):
        ax = axes[i, j]; draw(ax, bright(image(d, name)), row, ms=7); margins(ax, row)
        x, y = xy(row, bp); crop(ax, row, box=(x - 130, x + 130, y - 95, y + 95)); ax.set_title(f"{lab} - {title}", fontsize=9.5)
fig.suptitle("Corners = where the upper and the lower eyelid margin meet. Nasal: on the outer (left) border of the shadow. "
             "Temporal: outer apex of the dark area to the right of the bright reflection bands, not on a reflection", fontsize=10)
save(fig, "fig2_corners_zoom.jpg")

# 3. eyelid midpoints, open eye and nearly closed eye
BL = [(f"{V2}/2_20251031_Pluto_spont_3/{L}/batch02", "img020908.png"), (f"{V2}/9_20251023_Pluto1/{L}/batch02", "img028167.png"),
      (f"{V2}/7_20251027_Pluto_spont1/{L}/batch02", "img021876.png")]
fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
for ax, (title, d, name) in zip(axes[0], MICE):
    row = load(d).loc[name]; draw(ax, bright(image(d, name)), row, only=LID, ellipse=False, ms=7); margins(ax, row); crop(ax, row, pad=60); ax.set_title("open eye - " + title, fontsize=9.5)
for ax, (d, name) in zip(axes[1], BL):
    row = load(d).loc[name]; draw(ax, bright(image(d, name)), row, ellipse=False, ms=7); margins(ax, row); crop(ax, row, pad=70)
    ax.set_title(f"eye nearly closed - video {d.split('/')[-4].split('_')[0]}, {name[:-4]}", fontsize=9.5)
fig.suptitle("Eyelid points = midpoint of the upper and of the lower eyelid margin (edge between lid and eye). Their distance is the eye opening. In blinks the lids and corners are still labelled; the pupil is left empty when it cannot be seen", fontsize=10)
save(fig, "fig3_eyelids_open_and_blink.jpg")

# 4. change A: visible-edge standard -> ellipse standard (Pluto, same frames)
new1 = f"{V2}/1_20251031_Pluto_spont_1/{L}/batch01"
fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
for ax, name in zip(axes, ("img005160.png", "img020855.png", "img004835.png")):
    change_panel(ax, f"{OLD}/batch01", new1, name, PUPIL + LID); ax.set_title(f"Pluto 2025-10-31 {name[:-4]}", fontsize=10)
fig.suptitle("Change of 2026-09-20 (yellow = position before, arrow -> current). pupil_top moved from the visible edge under the lid to the top of the full ellipse; "
             "left/right moved to the mid-height of the ellipse; both corners moved outwards", fontsize=10)
save(fig, "fig4_change_ellipse_standard.jpg")

# 5. change B: first corner definition -> outer corners (5-min video)
new0 = f"{V2}/0_first5minvedio/{L}/test50"
fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
for ax, name in zip(axes, ("img015001.png", "img016176.png", "img017351.png")):
    change_panel(ax, f"{BK}/test50", new0, name, LID, pad=55); ax.set_title(f"5-min video {name[:-4]}", fontsize=10)
fig.suptitle("Corner change of 2026-09-20 evening (yellow = first definition). Nasal corner: from inside the shadow to its outer (left) border. Temporal corner: from the inner "
             "line next to the reflection to the outer line. eyelid_top moved up to the edge of the lid", fontsize=10)
save(fig, "fig5_change_outer_corners.jpg")

# 6. change C: convention v2 (double contour -> the left one), video 6 test set
new6 = f"{V2}/6_20251028_Pluto_spont_1/{L}/test50"
old6 = f"{V2}/6_20251028_Pluto_spont_1/training-data/_archive_2026-09-26_convention1/labels/test50"
d = (load(new6) - load(old6))
r = sum(np.hypot(d[(b, "x")], d[(b, "y")]).fillna(0) for b in PUPIL).sort_values(ascending=False)
fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
for ax, name in zip(axes, r.index[:3]):
    change_panel(ax, old6, new6, name, PUPIL, pad=30); ax.set_title(f"video 6 (2025-10-28) {name[:-4]}", fontsize=10)
fig.suptitle("Convention v2 of 2026-09-26 (yellow = before). Where the pupil edge looks doubled the left contour is used; the shifts are a few pixels (video 6: pupil_top re-placed in 14 of 50 frames, pupil_bottom in 6)", fontsize=10)
save(fig, "fig6_change_convention_v2.jpg")


# 7-8. construction lines: how each point is derived (cyan / red) against the placement that is easy to get wrong (yellow)
def bezier(n, m, t, s0=0.0, s1=1.0):
    s = np.linspace(s0, s1, 60)[:, None]
    return (((1 - s) ** 2) * n + 2 * s * (1 - s) * (2 * m - (n + t) / 2) + s ** 2 * t).T


def lid_lines(ax, row, color, solid=(0.22, 0.78), lw=1.3):
    """Each eyelid margin: solid where it is seen (middle part), dashed where it is extended to the corner."""
    n, t = xy(row, "eye_nasal_corner"), xy(row, "eye_temporal_corner")
    for lid in ("eyelid_top", "eyelid_bottom"):
        m = xy(row, lid)
        ax.plot(*bezier(n, m, t, *solid), "-", color=color, lw=lw)
        ax.plot(*bezier(n, m, t, 0, solid[0]), "--", color=color, lw=lw); ax.plot(*bezier(n, m, t, solid[1], 1), "--", color=color, lw=lw)


def note(ax, text, at, to, color):
    ax.annotate(text, at, to, color=color, fontsize=9, weight="bold", ha="center", va="center", annotation_clip=False,
                bbox=dict(fc="black", ec="none", alpha=0.65, pad=3), arrowprops=dict(arrowstyle="-", color=color, lw=0.8))


name = "img015001.png"
A, B = load(f"{BK}/test50").loc[name], load(new0).loc[name]
img = bright(image(new0, name))
fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), constrained_layout=True)
for ax in axes:
    ax.imshow(img, cmap="gray", vmin=0, vmax=255); ax.set_xticks([]); ax.set_yticks([])
    lid_lines(ax, A, YELLOW, lw=1.1); lid_lines(ax, B, CYAN)
    for bp in ("eye_nasal_corner", "eye_temporal_corner"):
        ax.plot(*xy(A, bp), "x", color=YELLOW, ms=9, mew=2); ax.plot(*xy(B, bp), "o", mfc=CYAN, mec="white", ms=8)
    for bp in ("eyelid_top", "eyelid_bottom"):
        ax.plot(*xy(B, bp), "o", mfc=CYAN, mec="white", ms=6)
crop(axes[0], B, pad=55); axes[0].set_title("whole eye: solid = eyelid margin where it is seen, dashed = its extension to the corner", fontsize=10)
n, t = xy(B, "eye_nasal_corner"), xy(B, "eye_temporal_corner")
crop(axes[1], B, box=(n[0] - 70, n[0] + 190, n[1] - 90, n[1] + 110)); axes[1].set_title("nasal corner (left)", fontsize=10)
note(axes[1], "correct: crossing of the two\nextended margins, on the LEFT\nborder of the shadow", n, n + (95, -62), CYAN)
note(axes[1], "wrong: inside the shadow\n(margins cut short)", xy(A, "eye_nasal_corner"), xy(A, "eye_nasal_corner") + (95, 45), YELLOW)
crop(axes[2], B, box=(t[0] - 190, t[0] + 70, t[1] - 100, t[1] + 100)); axes[2].set_title("temporal corner (right)", fontsize=10)
note(axes[2], "correct: lower margin extended\nupwards (outer line), apex of the dark\narea right of the reflections", t, t + (-75, -78), CYAN)
note(axes[2], "wrong: inner line,\nnext to the reflection", xy(A, "eye_temporal_corner"), xy(A, "eye_temporal_corner") + (-95, 70), YELLOW)
fig.suptitle("How the corners are constructed (5-min video, img015001). Cyan = current definition, yellow = the placement that is easy to get wrong (first definition of 2026-09-20). "
             "Lines are guides drawn through the labelled points", fontsize=10)
save(fig, "fig7_construction_corners.jpg")

fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), constrained_layout=True)
for ax, name in zip(axes, ("img020855.png", "img005160.png", "img004835.png")):
    A, B = load(f"{OLD}/batch01").loc[name], load(new1).loc[name]
    ax.imshow(bright(image(new1, name)), cmap="gray", vmin=0, vmax=255); ax.set_xticks([]); ax.set_yticks([])
    draw(ax, None, B, names=True, only=PUPIL, ms=7)
    ax.plot(*np.c_[xy(B, "pupil_left"), xy(B, "pupil_right")], "--", color=RED, lw=1.2); ax.plot(*np.c_[xy(B, "pupil_top"), xy(B, "pupil_bottom")], "--", color=RED, lw=1.2)
    lid_lines(ax, B, CYAN, lw=1.0)
    ok = [b for b in PUPIL if np.isfinite(xy(A, b)).all()]
    order = [b for b in ("pupil_left", "pupil_top", "pupil_right", "pupil_bottom", "pupil_left") if b in ok]
    ax.plot(*np.array([xy(A, b) for b in order]).T, "--", color=YELLOW, lw=1.1)
    for b in ok:
        ax.plot(*xy(A, b), "x", color=YELLOW, ms=9, mew=2)
    crop(ax, B, pad=45); ax.set_title(f"Pluto 2025-10-31 {name[:-4]}", fontsize=10)
B = load(new1).loc["img020855.png"]; A = load(f"{OLD}/batch01").loc["img020855.png"]
note(axes[0], "correct: top of the full ellipse,\nabove the upper eyelid margin", xy(B, "pupil_top"), xy(B, "pupil_top") + (150, -22), RED)
note(axes[0], "wrong: highest VISIBLE\npupil edge at the lid", xy(A, "pupil_top"), xy(A, "pupil_top") + (165, 40), YELLOW)
fig.suptitle("How the pupil points are constructed. Red = ellipse through the whole pupil, dashed red = its two axes, the four points are the ends of the axes. "
             "Yellow = visible-edge placement used before 2026-09-20. Cyan = eyelid margins", fontsize=10)
save(fig, "fig8_construction_pupil.jpg")
