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
MICE = [("first mouse, 5-min video (video 0)", f"{V2}/0_first5minvedio/{L}/test50", "img015001.png"),
        ("Pluto, 2025-10-31 (video 1)", f"{V2}/1_20251031_Pluto_spont_1/{L}/test50", "img068086.png"),
        ("Terra, 2025-10-17 (video 11)", f"{V2}/11_20251017_Terra_spont_1/{L}/test50", "img068086.png")]


def bright(img):
    """Display version of a frame: percentile stretch, gamma 0.6 and local contrast (CLAHE), so that edges stay visible."""
    import cv2
    lo, hi = np.percentile(img, (1, 99.5))
    g = (np.clip((img.astype(float) - lo) / (hi - lo), 0, 1) ** 0.6 * 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(g).astype(float)


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


def smooth(points, n=200):
    """Smooth curve through points picked by eye on the image (chord-length cubic spline)."""
    from scipy.interpolate import CubicSpline
    pts = np.array(points, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return CubicSpline(s, pts)(np.linspace(0, s[-1], n)).T


# Fig 7 follows the user's sketch sent to the advisor on 2026-09-20 (Slack): the eyelid margins are traced where they are
# sharp (solid), continued to where they cross (dashed), and the crossing is circled. Traced by eye on this frame.
name = "img015001.png"
A, B = load(f"{BK}/test50").loc[name], load(new0).loc[name]
img = bright(image(new0, name))
n, t = xy(B, "eye_nasal_corner"), xy(B, "eye_temporal_corner")
et, eb = xy(B, "eyelid_top"), xy(B, "eyelid_bottom")
UPPER = [(420, 350), (480, 339), (550, 332), tuple(et), (700, 336), (765, 358), (812, 398)]      # upper margin, sharp part
UPPER_L = [tuple(n + (-14, 9)), tuple(n), (385, 359), (420, 350)]                               # continued to the left
UPPER_R = [(812, 398), (840, 430), tuple(t), tuple(t + (5, 16))]                                # continued to the right
LOWER = [(410, 482), (440, 540), (480, 581), (540, 611), tuple(eb), (700, 620), (775, 603)]      # lower margin, sharp part
LOWER_L = [tuple(n + (-7, -22)), tuple(n), (383, 428), (410, 482)]                              # continued up to the crossing
LOWER_R = [(775, 603), (838, 572), (874, 524), (870, 484), tuple(t)]                             # outer line, right of the reflections
an, at = xy(A, "eye_nasal_corner"), xy(A, "eye_temporal_corner")      # the alternative placement that was discussed and not adopted
# yellow: the INNER border of the shadow / of the reflections, followed down to near the lower-eyelid point
ALT_N = [(440, 351), (418, 372), tuple(an), (418, 452), (441, 500), (468, 542), (504, 577), (550, 601), (600, 619)]
ALT_T = [(765, 358), (794, 400), tuple(at), (801, 492), (786, 530), (768, 560), (742, 582), (700, 599), (650, 613)]
SHADOW = np.vstack([smooth(UPPER_L[1:]).T, smooth(ALT_N).T, smooth([(600, 621), tuple(eb)]).T, smooth(LOWER[:5]).T[::-1], smooth(LOWER_L[1:]).T[::-1]])


def corner_lines(ax):
    ax.add_patch(plt.Polygon(SHADOW, closed=True, fc=YELLOW, ec="none", alpha=0.16, hatch="////"))
    for solid in (UPPER, LOWER):
        ax.plot(*smooth(solid), "-", color=RED, lw=2.4)
    for dashed in (UPPER_L, UPPER_R, LOWER_L, LOWER_R):
        ax.plot(*smooth(dashed), "--", color=RED, lw=2.2)
    for alt in (ALT_N, ALT_T):
        ax.plot(*smooth(alt), ":", color=YELLOW, lw=2.6)
    for c in (n, t):
        ax.add_patch(plt.Circle(c, 13, fill=False, ec=RED, lw=2.4))
    ax.plot(*an, "x", color=YELLOW, ms=13, mew=3); ax.plot(*at, "x", color=YELLOW, ms=13, mew=3)
    for bp in ("eye_nasal_corner", "eye_temporal_corner"):
        ax.plot(*xy(B, bp), "o", mfc=CYAN, mec="white", ms=9, mew=1.2)
    for bp in ("eyelid_top", "eyelid_bottom"):
        ax.plot(*xy(B, bp), "o", mfc=CYAN, mec="white", ms=7)


BOXES = [("whole eye", (n[0] - 70, t[0] + 70, et[1] - 70, eb[1] + 60)),
         ("nasal corner (left)", (275, 635, 300, 660)),
         ("temporal corner (right)", (610, 928, 318, 660))]
fig, axes = plt.subplots(3, 2, figsize=(15, 19.5), constrained_layout=True)
for (title, box), (a0, a1) in zip(BOXES, axes):
    for ax in (a0, a1):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(box[0], box[1]); ax.set_ylim(box[3], box[2])
    corner_lines(a1)
    a0.set_title(f"{title}: the frame without any line", fontsize=12); a1.set_title(f"{title}: the same frame with the construction", fontsize=12)
note(axes[1, 1], "CORRECT (red circle, cyan dot): the lower-left margin (outer border of the\nshadow) and the upper margin are continued until they cross", n, (455, 322), CYAN)
note(axes[1, 1], "NOT USED (yellow cross): crossing on the\nINNER border of the shadow (dotted yellow)", an, (535, 455), YELLOW)
note(axes[1, 1], "shadow (hatched), narrower\nbut still present further down", (392, 440), (345, 590), YELLOW)
note(axes[2, 1], "CORRECT (red circle, cyan dot): the lower margin continued upwards\nalong the OUTER line, right of the reflections, meets the upper margin", t, (765, 338), CYAN)
note(axes[2, 1], "NOT USED (yellow cross): crossing on the INNER line,\nalong the inner edge of the reflection and of the shadow below it", at, (725, 640), YELLOW)
fig.suptitle("How the two corners are constructed (5-min video, img015001), after the sketch of 2026-09-20. Left column: the frame as it is. Right column: the same frame with the lines.\n"
             "Red solid = eyelid margin traced by eye, red dashed = its continuation, red circle = crossing = corner, cyan = labelled point.\n"
             "Yellow dotted + cross = the construction that was discussed and is NOT used (inner border of the shadow / of the reflections)", fontsize=11)
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
