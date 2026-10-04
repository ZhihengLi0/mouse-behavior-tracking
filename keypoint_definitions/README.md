# Keypoint definitions (mouse eye, 8 points)

Where each of the eight keypoints is placed, how the placement changed over the project, and which placements are easy
to get wrong. This is the reference for every label set under `v2_new_pupil_top_video_scaling/` (current state:
ellipse standard + outer corners of 2026-09-20, convention v2 of 2026-09-26, pupil-band decision of 2026-09-29).
Chinese version: [README_zh.md](README_zh.md).

All example frames are brightened for display (gamma 0.45); the eye region is too dark to read otherwise. In these
videos the nasal side is on the left of the image and the temporal side on the right.

![all keypoints](figures/fig1_all_keypoints.jpg)

## Terms

| term used here | anatomical name | meaning |
|---|---|---|
| eyelid margin | upper / lower eyelid margin (palpebral margin) | the edge where the eyelid skin ends and the eye surface begins |
| eye opening | palpebral fissure | the opening between the two eyelid margins |
| nasal corner | medial canthus | where the two eyelid margins meet on the nose side (left in the image) |
| temporal corner | lateral canthus | where the two eyelid margins meet on the ear side (right in the image) |
| reflection / glint | corneal reflection of the infrared lamps | bright spots and bands; they are not anatomy |

## Definitions

### Pupil: `pupil_top`, `pupil_bottom`, `pupil_left`, `pupil_right`

Treat the pupil as an ellipse. The four points are the ends of its two axes: left and right are the ends of the
horizontal axis, top and bottom are the ends of the vertical axis.

1. The ellipse covers the whole pupil, including the part hidden under the upper eyelid or under fur. `pupil_top` is
   therefore often above `eyelid_top` (fig. 1, mouse A).
2. The ellipse may be tilted. The points only have to sit at the ends of the axes; left and right need not be at the
   same height, but they are at the mid-height of the ellipse, not at the widest visible part.
3. The pupil edge is the outer edge of the darkened region. A lighter band around the black core (fur in front of it,
   or the infrared lamps lightening it) is still pupil.
4. Where the edge looks doubled (two contours), use the left contour; `pupil_top` and `pupil_bottom` move left with it.
5. If the pupil looks irregular (rhombic, dented on one side), still approximate it with an ellipse. A side that cannot
   be seen is placed from the ellipse through the other three points.
6. A reflection that sits on the pupil edge is ignored: the edge continues under it (`pupil_right` in fig. 1 is placed where the pupil edge
   passes the bright spot).
7. If the pupil cannot be seen at all (eye closed), the four pupil points are left empty.

Construction of the pupil points: the ellipse covers the whole pupil, the dashed red lines are its two axes, the points are their ends. Yellow = the visible-edge placement.

![construction pupil](figures/fig8_construction_pupil.jpg)

### Eyelids: `eyelid_top`, `eyelid_bottom`

`eyelid_top` is the midpoint of the upper eyelid margin and `eyelid_bottom` the midpoint of the lower eyelid margin
(midpoint of the arc between the two corners). Lashes and fur are ignored: the margin is the edge of the dark eye
region. The distance between the two points is the eye opening, so they are not required to be vertically aligned.
As for every point: labelled when it can be seen clearly, left empty when its position cannot be judged accurately.

![eyelids](figures/fig3_eyelids_open_and_blink.jpg)

### Corners: `eye_nasal_corner`, `eye_temporal_corner`

Each corner is the point where the upper and the lower eyelid margin meet; where the meeting point itself is hidden,
extend the two margins until they cross.

- **Nasal corner (left).** There is usually a shadow at this corner. Follow the left (outer) border of the shadow
  and take its crossing with the upper eyelid margin. Do not label inside the shadow or on its right border.
- **Temporal corner (right).** There are bright reflection bands next to this corner. The corner is the right apex of
  the dark area to the right of the reflections, reached by extending the lower eyelid margin upwards (the outer line,
  which stays visible in every frame). Do not label on a reflection or on the inner line next to it.
  Why the outer side: in the discussion of 2026-09-20 it was concluded that fur does not reflect, so the bright
  reflections are still inside the eye and the corner must lie outside them.

Both corners and both eyelid points only serve blink detection, so a robust position that can be found in every frame
is preferred over an exact one.

![corners](figures/fig2_corners_zoom.jpg)

Construction of the corners (redrawn after the hand sketch used in the discussion of 2026-09-20): the three rows are the whole eye, the nasal corner and the temporal corner; in each row the left image is the frame without any line and the right image the same frame with the construction. Solid red = eyelid margin traced on the image, dashed red = its continuation, red circle = the crossing, which is the corner; cyan = the labelled point. Nasal corner: the shadow at the corner narrows downwards and continues to near the lower eyelid; its outer border is taken as the margin and continued until it crosses the continued upper margin. Temporal corner: the lower margin is continued upwards along the outer line, around the right of the reflections, until it meets the upper margin. Dotted yellow lines and yellow crosses = the construction that was compared in the discussion and not adopted: along the inner border of the shadow (nasal) and along the inner edge of the reflection and of the shadow below it (temporal), both drawn down to near the lower-eyelid point; hatched = the shadow at the nasal corner.

![construction corners](figures/fig7_construction_corners.jpg)

## What changed, measured on frames labelled under both versions

Numbers from `scripts/compare_label_versions.py` (`results/label_changes.csv`); +x = right, +y = down.

**A. 2026-09-20, ellipse standard and outer corners** (Pluto 2025-10-31, 40 frames labelled before and after):

| keypoint | frames | median shift | mean shift (x, y) | what changed |
|---|---|---|---|---|
| pupil_top | 32 | 52.4 px | −8.9, −45.9 | from the highest visible pupil edge to the top of the full ellipse |
| pupil_right | 36 | 14.9 px | −0.6, −15.2 | to the end of the horizontal axis (mid-height of the ellipse) |
| pupil_left | 33 | 14.2 px | −2.0, −9.7 | same |
| pupil_bottom | 38 | 13.5 px | −4.2, −4.3 | end of the vertical axis |
| eyelid_top | 40 | 16.5 px | −2.1, −10.9 | up to the eyelid margin |
| eyelid_bottom | 40 | 20.1 px | −17.6, −1.5 | to the midpoint of the lower margin between the new corners |
| eye_nasal_corner | 37 | 38.5 px | −37.3, −11.2 | outwards, to the left border of the shadow |
| eye_temporal_corner | 37 | 67.9 px | +61.6, +6.0 | outwards, to the outer line |

**B. The corner part alone** (5-min video, 90 frames, both versions under the ellipse standard): nasal corner 39.8 px
(left and up) and temporal corner 56.3 px (right and down) in all 90 frames; `eyelid_top` 16.9 px (up) in 62 of 90
frames; the pupil points and `eyelid_bottom` did not move. So change A is two changes made on the same day: the pupil
points changed with the ellipse standard, the corners and `eyelid_top` with the corner definition.

**C. 2026-09-26, convention v2 (double contour = left one):** only pupil points moved, by a few pixels. Video 6 test
set: `pupil_top` re-placed in 14 of 50 frames, `pupil_bottom` in 6, `pupil_right` in 4. Video 7: at most 4 of 50 per
point. Video 8, a video where the pupil edge is hard to see: 20-31 of 50 frames per pupil point.

**D. 2026-09-29, pupil band:** no label changed. The decision confirmed the existing placement (rule 3 above).

![change A](figures/fig4_change_ellipse_standard.jpg)
![change B](figures/fig5_change_outer_corners.jpg)
![change C](figures/fig6_change_convention_v2.jpg)

## Placements that are easy to get wrong

| mistake | correct placement | example |
|---|---|---|
| `pupil_top` on the visible pupil edge at the eyelid | top of the full ellipse, often above the upper-eyelid point (a position covered by the lid) | fig. 8, yellow crosses |
| `pupil_left` / `pupil_right` at different heights on the visible outline | ends of the horizontal axis of the ellipse | fig. 4 |
| pupil point on the border of a reflection | on the pupil edge, which continues under the reflection | fig. 1, `PR` |
| pupil edge taken at the black core, leaving out the lighter band | outer edge of the darkened region | rule 3 |
| right contour of a doubled pupil edge | left contour | fig. 6 |
| nasal corner inside the shadow | left border of the shadow, at its crossing with the upper margin | fig. 7, middle panel |
| temporal corner on the inner line or on a reflection band | outer apex of the dark area to the right of the reflections | fig. 7, right panel |
| a point dragged far away, or the two corners swapped | caught by the geometry check that runs on every saved batch | – |

## Files

- `scripts/kp_render.py` – loads a label set, draws a frame with its keypoints.
- `scripts/make_figures.py` – writes `figures/fig1`-`fig8`.
- `scripts/compare_label_versions.py` – writes `results/label_changes.csv`.

The label sets and frames read by these scripts are local only (not in git).
