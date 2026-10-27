# Second-peak repair of sporadic big errors (P2 step 3, pilot), 2026-10-26

Script: `scripts/second_peak_repair.py --video N [--k-mad 5] [--bands pooled|own] [--sheet]`. Inputs: the plateau model's
whole-video predictions, the heatmap peak coordinates written by `scripts/heatmap_signals.py --peaks` (new option, one
inference pass of ≈ 90 min per video; `results/heatmap_signals/videoNN_shuffleSSSS_peaks.csv`, local only) and the
human labels of the video's 70 truth frames (last 4 minutes) only to judge. Nothing temporal: every frame is judged
alone, so a repair has no timing effect and changes only the frames that fail the check.

Rule per frame, label-free. Eye frame = corners define the axis, unit = eye width. Each pupil point gets a scale-free
position relative to the other three (pupil_left: nasal offset from the top/bottom midpoint divided by the pupil
height; pupil_top: upward offset from the left/right midpoint divided by the pupil width; and so on), compared with the
video's own median; the scale is the median over the 19 videos of each video's MAD (`geometry_bands.csv`; `--bands own`
uses the video's own MAD). A point fails at |z| > 5, outside the corners or the eyelid band, or when left/right or
top/bottom are swapped. A failing frame is tried with the failing point(s) replaced by their own second heatmap peak
(cell × stride + the per-point median offset of the regressed point from the first-peak cell, 3–5 px); the smallest
replacement that makes the frame pass is kept ("repaired"); otherwise the frame is "drop". Eyes less than half open
("closed") are left alone.

## Video 8 (`video08/`): not the right case, and the method does not repair it

Whole video (71 748 frames, pooled bands, k = 5): 58 595 ok, 213 closed, 3 244 repaired (4.5 %, pupil_left in 2 564),
9 696 drop (13.5 %). Of the 9 069 frames flagged by the time-series rule (fit_o200_max > 44 px, 12.6 %), 1 647 are
repaired and 4 418 dropped by the geometry rule, 2 794 pass it.

On the 70 truth frames (9 above 50 px, 38 above 20 px): 8 repaired, 19 dropped, 43 untouched.

| | frames | outcome |
|---|---|---|
| > 50 px | 9 | 3 repaired (2 end below 50 px: 86 → 50, 50 → 32; one gets worse 54 → 61), 6 dropped, none below 20 px |
| 20–50 px | 29 | 3 repaired, all to below 20 px (38 → 19, 33 → 15, 31 → 13); 10 dropped |
| ≤ 20 px | 32 | 2 repaired and made worse (15 → 25, 11 → 24) = false repairs; 3 dropped = false drops |

Median frame RMSE of the kept frames 21.5 → 16.7 px. With the video's own bands (`--bands own`) the rule is quieter
(2.1 % repaired, 8.4 % dropped; 0 false repairs, 2 false drops) and repairs 1 of the 9 (86 → 50).

Why it cannot work here (the contact sheet `video08/repaired_sheet.jpg`, local): in all 9 frames the model's four pupil
points form a self-consistent small ellipse on the dark core, while the label is the wider ring (pupil_left ≈ 120 px
further nasal, pupil_bottom ≈ 60 px lower, pupil_top ≈ 40 px higher; see the labeling decision of 2026-09-29, ring =
pupil). The frame's geometry is normal to any check, so the geometry rule either passes it or blames the wrong point
(the > 50 px frames are "repaired"/dropped through pupil_top, whose ratio shrinks when the width is the core's). The
second peak of pupil_left does sit near the ring edge in 6 of 9 frames (28–61 px from the label) and elsewhere
(> 300 px) in 3; the other three points have no useful second peak. Moving one point to the ring while the other three
stay on the core is not a repair, which is what the numbers show (none of the 9 reaches 20 px). Video 8's 9 frames are
the only > 50 px truth frames of this kind: a large-magnitude member of family B (systematic core-vs-ring mode, 12–16 %
of the video's frames; `results/error_modes/`), to be addressed by labels of that mode, not per frame.

The other 23 of the 32 > 50 px truth frames (videos 1, 2, 4, 10, 11, 14) are family A: one or two points thrown
200–600 px with confidence ≈ 0 and a fitting deviation of 120–580 px. Video 1 (8 frames, 6 of them pupil_left alone
230–420 px off, the model's other points within 10–40 px) is the proper test of this step; its peaks pass is running
(started 2026-10-26 22:00, ≈ 90 min) and its result will be added below.

Note for reruns of `scripts/error_modes.py`: it globs `video08_shuffle*.csv` and takes the last match; the folder now
also holds `video08_shuffle911_run1_2026-10-25.csv` (byte-identical to the regenerated file) and the `_peaks.csv`.
