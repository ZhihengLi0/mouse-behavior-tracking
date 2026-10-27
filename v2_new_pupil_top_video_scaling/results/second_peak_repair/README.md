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

## Videos 1, 2, 11, 4 (overnight 2026-10-26/27, same script, same defaults; `video01/`, `video02/`, `video11/`, `video04/`)

Chosen with the user: two clean videos with sporadic throws (2: truth median 6.9 px, 4 frames > 50 px; 11: 8.9 px,
4 frames), the family-A video 1 (12.7 px, 8 frames) and the second blurry video 4 (14.7 px, 2 frames; 57 % of its
truth frames have an area error > 20 %, like video 8). Pooled bands, k = 5, all frames of each video.

| video | truth > 50 px | repaired to < 50 px (to < 20 px) | dropped | left alone (eye closed) | > 50 px left among kept | frames ≤ 20 px made worse / dropped | whole video repaired / dropped | flagged by fit_o200_max (44 px) |
|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 6 (5) | 2 | 0 | 0 | 0 / 1 | 5.4 % / 3.9 % | 16.5 % |
| 2 | 4 | 0 (0) | 4 | 0 | 0 | 0 / 0 | 0.5 % / 1.1 % | 6.1 % |
| 11 | 4 | 2 (2) | 0 | 2 | 2 (the closed ones) | 0 / 0 | 0.7 % / 0.3 % | 4.3 % |
| 4 | 2 | 2 (0) | 0 | 0 | 0 | 0 / 0 | 14.2 % / 3.1 % | 5.0 % |
| 8 | 9 | 2 (0), one made worse | 6 | 0 | 1 | 2 / 3 | 4.5 % / 13.5 % | 12.6 % |
| all | 27 | 12 (7) | 12 | 2 | 3 | 2 / 4 | | |

- **Video 1 (family A proper).** The 6 frames with pupil_left thrown 230–420 px into the fur or outside the eye are
  all repaired by their second peak, which sits on the pupil edge: 148 → 13, 139 → 17, 117 → 16, 114 → 10, 111 → 15,
  85 → 31 px (`video01/repaired_sheet.jpg`). The 2 frames with several points thrown (354 px with four points; 83 px
  with pupil_left and pupil_bottom) are dropped. Of the 56 frames ≤ 20 px none is changed and one is dropped (17.7 px,
  its pupil_top 31 px off). After the rule no truth frame of the video is above 50 px. Whole video: 3 858 frames
  repaired (3 554 through pupil_left), 2 822 dropped; 87 % of them are also flagged by fit_o200_max, which flags 16.5 %
  of this video.
- **Video 2.** All 4 frames > 50 px are dropped, none repaired: pupil_top thrown 196 / 210 / 59 px (its second peak is
  not at a valid place) and one frame whose error is the temporal corner (410 px; the pupil is within 21 px). No frame
  ≤ 20 px is changed or dropped; one 20–50 px frame is made worse (23 → 40 px, pupil_left moved from 19 to 94 px off).
  Whole video 0.5 % repaired, 1.1 % dropped (fit_o200_max flags 6.1 %).
- **Video 11.** The 2 frames with pupil_left thrown 483 / 494 px are repaired to 11 / 12 px. The other 2 frames > 50 px
  are "closed" (eye opening below half the video's median: one with pupil_bottom 495 px off, one with the nasal corner
  455 px off) and are left alone by design; the area series masks such frames anyway. Nothing else changes (0.7 %
  repaired, 0.3 % dropped on the whole video).
- **Video 4 (blurry, but not the video-8 mode).** Both frames > 50 px are repaired (118 → 33 px: pupil_bottom 320 → 20;
  117 → 38 px: pupil_top 331 → 64, but the pair replacement also moved pupil_left from 1 to 85 px off), and 8 more truth
  frames are changed, every one for the better (42 → 16, 41 → 22, 35 → 25, 27 → 24, 27 → 20, 19 → 9, 18 → 16, 16 → 6 px):
  the model's pupil_right sits on the corneal reflection next to the small blurry pupil and its second peak is the
  pupil edge (`video04/repaired_sheet.jpg`). This is a third kind of error — a wrong bright target, sustained, not
  jumpy — which the second peak does fix and the time series does not see: 14.2 % of the whole video is repaired
  (pupil_right in 6 052 frames) while fit_o200_max flags 5.0 %, and only 1 572 of the 10 153 repaired frames are
  flagged. Whole-video effect unverified beyond the 70 truth frames (median 14.7 → 13.7 px).

Reading across the five videos: the rule removes or repairs 24 of the 27 > 50 px frames (the remaining 3 are two
closed-eye frames and one video-8 frame); on the 238 frames ≤ 20 px of videos 1, 2, 11, 4 it changes nothing for the
worse and drops one. Its failures are video 8's core-vs-ring mode (geometry is normal, see above) and multi-point
throws (dropped, not repaired). The share of a video it touches runs from 1 % (clean videos) to 18 % (video 8) and
is not predicted by the time-series flag share (video 4: 17 % touched, 5 % flagged), so the two finders are
complementary: the time series sees jumps, the geometry sees sustained wrong targets. Under the constraint "repair
only flagged frames" the video-4 repairs would mostly not be applied.

Script note: `scripts/error_modes.py` (2026-10-27) now skips the `_peaks.csv` files when it looks for a video's heatmap
statistics (it took the last file in sorted order, which after a `--peaks` run was the peaks file; videos 1 and 2 were
re-clustered after the fix, video 8 was unaffected).
