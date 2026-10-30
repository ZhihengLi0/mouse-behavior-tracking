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

## Two systems, the same experiments on every video (2026-10-27 night)

User, 10-27: the geometry check above is itself a "find the suspicious frame" rule, and the finder and the fixer should
be one system. `results/geometry_detector/README.md` ranks the geometry check as a detector on the 1330 truth frames:
AUC 0.86 against 0.99 for fit_o200_max, 27 against 31 of the 32 > 50 px frames at the same flagged share, and the OR of
the two catches nothing more. So two complete systems are compared, both zero-human, per frame, without smoothing:

- **System A (geometry)** — find = geometry check; fix = peak 2 if the frame then passes the check, else drop.
  (`scripts/second_peak_repair.py`, folders `videoNN/`; parameter variants `videoNN_k3/`, `videoNN_k7/`, `videoNN_own/`
  via the `--tag` option added 10-27, default output unchanged.)
- **System B (time series triggered)** — find = fit_o200_max (largest-of-8 fitting deviation, SARIMAX (2,0,0), pooled
  threshold 44 px = the detector chosen at the 50-px tier); fix = on flagged open frames only, replace the geometry-
  violating pupil points or the point with the largest deviation by peak 2 and accept the smallest replacement after
  which the geometry check passes, else drop; a flagged frame that passes the check untouched stays as it is
  ("flagged_ok"); unflagged frames are never touched even if they violate the geometry ("unflagged_violation").
  (`scripts/triggered_repair.py`, folders `videoNN_trig/`, same summary columns.)

Truth frames (70 per video), the > 50 px frames and the frames ≤ 20 px:

| video | > 50 px | A: repaired < 50 (< 20) / dropped / left | B: repaired < 50 (< 20) / dropped / left | false repairs / false drops A → B | whole video repaired + dropped A → B | B: unflagged geometry violations |
|---|---|---|---|---|---|---|
| 1 | 8 | 6 (5) / 2 / 0 | 6 (5) / 2 / 0 | 0 / 1 → 0 / 1 | 9.3 % → 8.1 % | 1.3 % |
| 2 | 4 | 0 / 4 / 0 | 1 (1) / 3 / 0 | 0 / 0 → 0 / 0 | 1.5 % → 1.5 % | 0.1 % |
| 11 | 4 | 2 (2) / 0 / 2 closed | 2 (2) / 0 / 2 closed | 0 / 0 → 0 / 0 | 1.0 % → 1.0 % | 0.0 % |
| 4 | 2 | 2 (0) / 0 / 0 | 2 (0) / 0 / 0 | 0 / 0 → 0 / 0 | 17.3 % → 2.9 % | 14.4 % |
| 8 | 9 | 2 (0) / 6 / 1 | 2 (0) / 3 / 4 | 2 / 3 → 0 / 0 | 18.0 % → 8.5 % | 9.6 % |

- **B is the coherent system and the safer one**: 0 false repairs and 0 false drops on all five videos (A: 2 and 3, all
  in video 8), the same > 50 px frames repaired in videos 1, 11, 4 and one more in video 2 (frame 68672, pupil_top 74 → 7 px:
  the largest-deviation point was added to the candidates), and half the drops on the whole videos (video 8 13.5 → 5.4 %,
  video 1 3.9 → 2.7 %).
- **What B gives up**: (1) the sustained glint mode of video 4 — 14 % of its frames violate the geometry but are never
  flagged by the time series, so B leaves them (A repaired 10 of the 70 truth frames there, median error 14.7 → 13.7 px);
  (2) four video-8 frames at 50–55 px whose fitting deviation is 36–40 px, under the pooled 44 px threshold (the
  leave-one-video-out threshold of `tier50_reeval` is lower for video 8 and catches them). Both are the two finders'
  known difference (jumps vs sustained wrong targets) and belong in the 10-31 discussion, not in the pipeline.
- **Band width** (`videoNN_k3/`, `_k7/`, `_own/`; k = 5 with the pooled scale is the default): k = 3 produces false
  drops on every video (video 1: 4 of 61 frames ≤ 20 px; video 8: 9, and 24 % of the whole video dropped) for one
  extra repair on video 2; k = 7 and the video's own MAD drop less (video 8: 6–8 % instead of 13.5 %) but leave more
  video-8 frames unrepaired (1 of 9 instead of 2) and k = 7 keeps one > 50 px frame. The default is kept.

Every video that gets the peaks pass gets the same five folders (`videoNN`, `_k3`, `_k7`, `_own`, `_trig`); the
whole-video shares of A and B per video are in each folder's `whole_video.csv`.

## The chosen pipeline on all nineteen videos (2026-10-27 to 10-29, `videoNN_trig/` folders, `scripts/triggered_repair.py`)

User, 10-27 night: keep one pipeline, the one whose finder is the detector chosen at the 50-px tier, and run it the same
way on every video — no clustering, no band-width variants, no geometry-first version from here on. The pipeline:
**find** = largest-of-8 fitting deviation (SARIMAX (2,0,0), pooled threshold 44 px); **fix** = on a flagged open frame,
replace the geometry-violating pupil points or the point with the largest deviation by the heatmap's second peak and
accept the smallest replacement after which the geometry check passes; **else drop**; a flagged frame that already
passes the check and every unflagged frame stay as they are. Videos 10, 14, 17, 18, 15 got the peaks pass tonight
(88 min each) and the repair (seconds); the five earlier videos are re-listed with the same pipeline. Videos 5, 3, 9 and 13 followed on the night of
10-28/29; none of them has a truth frame above 50 px, so on them the run measures only what the pipeline does where
nothing is wrong. Videos 12, 0, 16, 6 and 7 followed on 10-29, so every video now has the pipeline.

| video | labels | truth > 50 px | repaired to < 50 (< 20) | dropped | left: closed / unflagged / passes the check | frames ≤ 20 px made worse / dropped | median error before → after (kept) | flagged | whole video repaired / dropped |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 12.7 px | 8 | 6 (5) | 2 | 0 / 0 / 0 | 0 / 1 | 12.7 → 12.4 | 16.5 % | 5.4 % / 2.7 % |
| 2 | 6.9 | 4 | 1 (1) | 3 | 0 / 0 / 0 | 0 / 0 | 6.9 → 6.5 | 6.1 % | 0.5 % / 0.9 % |
| 11 | 8.9 | 4 | 2 (2) | 0 | 2 / 0 / 0 | 0 / 0 | 8.9 → 8.9 | 4.3 % | 0.7 % / 0.3 % |
| 4 | 14.7 | 2 | 2 (0) | 0 | 0 / 0 / 0 | 0 / 0 | 14.7 → 14.7 | 5.0 % | 2.2 % / 0.7 % |
| 8 | 21.5 | 9 | 2 (0) | 3 | 0 / 4 / 0 | 0 / 0 | 21.5 → 19.2 | 12.6 % | 3.1 % / 5.4 % |
| 10 | 10.3 | 3 | 0 | 1 | 2 / 0 / 0 | 0 / 0 | 10.3 → 10.3 | 2.9 % | 0.6 % / 0.7 % |
| 14 | 10.8 | 2 | 0 | 1 | 0 / 1 / 0 | 0 / 0 | 10.8 → 10.7 | 4.3 % | 1.1 % / 0.8 % |
| 17 | 4.7 | 0 | – | – | – | 0 / 0 | 4.7 → 4.7 | 1.5 % | 0.2 % / 0.4 % |
| 18 | 12.4 | 0 | – | – | – | 0 / 0 | 11.2 → 11.2 | 9.7 % | 3.8 % / 1.4 % |
| 15 | 9.7 | 0 | – | – | – | 0 / 0 | 10.4 → 10.4 | 9.4 % | 2.2 % / 1.1 % |
| 5 | 13.3 | 0 | – | – | – | 0 / 0 | 13.3 → 13.3 | 1.1 % | 0.03 % / 0.1 % |
| 3 | 3.3 | 0 | – | – | – | 0 / 0 | 3.3 → 3.3 | 3.2 % | 0.2 % / 0.3 % |
| 9 | 9.0 | 0 | – | – | – | 0 / 0 | 9.0 → 9.0 | 2.2 % | 0.4 % / 0.6 % |
| 13 | 9.4 | 0 | – | – | – | 0 / 0 | 9.4 → 9.4 | 1.7 % | 0.4 % / 0.4 % |
| 12 | 6.0 | 0 | – | – | – | 0 / 0 | 6.0 → 6.0 | 4.3 % | 0.7 % / 0.7 % |
| 0 | 11.1 | 0 | – | – | – | 0 / 0 | 11.1 → 11.1 | 1.1 % | 0.1 % / 0.4 % |
| 16 | 7.0 | 0 | – | – | – | 0 / 0 | 7.0 → 7.0 | 0.2 % | 0.08 % / 0.01 % |
| 6 | 6.8 | 0 | – | – | – | 0 / 0 | 6.8 → 6.8 | 6.2 % | 1.9 % / 1.0 % |
| 7 | 5.7 | 0 | – | – | – | 0 / 0 | 5.7 → 5.7 | 0.8 % | 0.2 % / 0.2 % |

- **32 > 50 px frames in the nineteen videos (all of them in the first ten): 13 repaired (8 to below 20 px), 10 dropped, 9 left.** Of the 9 left: 4 are
  marginal video-8 frames (50–55 px, deviation 36–40 px, under the pooled 44 px threshold), 2 + 2 are frames the eyelid
  rule calls closed (video 11, and video 10 frames 61053 / 70136 where the model's eyelid points are themselves wrong, so
  "closed" is the model's error, not the eye's), 1 is video 14 frame 61431 (51 px, deviation 11 px: no jump, no geometry
  violation — the frame missed by every detector). Among the kept frames 9 remain above 50 px; none of the kept frames
  that were below 50 px went above it.
- **False changes: 0 repairs made worse, 1 drop** (video 1, a 16 px frame) on the 1068 frames ≤ 20 px of the nineteen videos.
  On the clean videos 17, 5, 3, 9, 13, 12, 0, 16, 6 and 7 no truth frame is made worse (the one change, video 13 frame
  67940, takes a 27 px frame to 10 px); on the whole videos 0.03–1.9 % repaired and 0.01–1.0 % dropped is the
  pipeline's cost where nothing is wrong. Video 6 is the expensive clean video (6.2 % flagged, 2.9 % touched), video 12
  the next (4.3 % flagged); on both the truth frames show no error, so the touched frames there are either errors the
  70 truth frames do not sample or false alarms of the time-series rule — to be checked by eye before the 10-31 meeting.
- **Whole-video cost**: 0.1–9.3 % of frames touched (repaired + dropped) per video; the share follows the flag share
  (video 1 8 %, video 8 8.5 %, videos 18 / 15 5 % / 3 %, the clean videos ≈ 1 %). Drops are 0.01–5.4 % per video; nothing
  is interpolated or smoothed, so the only timing effect is the dropped frames.
- Video 10 frame 70282 and video 14 frame 64865 are multi-point throws (3–4 pupil points at once): dropped, as in
  video 2. Video 14 frame 59165 (27 px, pupil_right replaced, 27 → 41 px) is the one repair of a 20–50 px frame that got
  worse — the only such case in nineteen videos.

## Across the nineteen videos (`all_videos/`, `scripts/triggered_repair_all_videos.py`, 2026-10-29)

One script reads the nineteen `videoNN_trig/` folders and writes the tables side by side (`whole_video_all.csv`,
`truth_summary_all.csv`, `totals.csv`), the time structure of the touched frames (`touched_structure.csv`), the figure
`touched_share_per_video.png` and a spot-check sample drawn by the same rule on every video (below).

- **Truth frames, all videos together**: 1330 frames, 32 above 50 px. The pipeline repairs 13 of them to below 50 px
  (8 to below 20 px), drops 10 and leaves 9; among the kept frames 9 remain above 50 px, all of them frames it left
  alone. Of the 1175 frames at or below 20 px, none is made worse by a repair and one is dropped; five frames between
  20 and 50 px are dropped.
- **Whole videos**: 1.31 million frames, 5.1 % flagged by the time-series rule, 1.3 % repaired, 1.0 % dropped. The
  touched share per video runs from 0.09 % (video 16) to 8.5 % (video 8), median 1.2 %; the five videos above 2 % are
  1, 8, 18, 15 and 6, the first four being the videos with the most > 50 px truth frames or the worst image quality.
- **How the touched frames sit in time**: mostly in short episodes (median length one to three frames; a third to a
  half of the episodes are single frames), but in videos 1, 6, 7, 8 and 12 a quarter to a half of the touched frames
  lie in episodes of one second or longer, the longest 150–220 frames: there the model stays wrong for a while, and a
  drop removes a whole stretch rather than a frame.
- **Which point, how far**: the replaced point is pupil_left on most Pluto videos (video 1: 3538 of 3886 repairs,
  video 6: 1369 of 1382, video 8: 1997 of 2394), pupil_right on videos 4 and 18, pupil_top on videos 13 and 15. On
  videos 0, 1, 2, 3, 6, 10 and 11 the accepted second peak lies 260–530 px from the model's point: the model had put the
  point at the image border (in video 6 pupil_left sits at the top edge of the image, the second peak on the pupil);
  on the other videos the repairs move a point 70–130 px.
- **Spot check by eye, same rule on every video** (`spotcheck_sample.csv`, sheets `spotcheck/videoNN.jpg` local only):
  per video 10 repaired and 10 dropped frames, one frame per episode, episodes drawn at random (seed 0); 374 frames in
  all (video 16 has only four dropped episodes). The `verdict` column (real error / false alarm / cannot tell) is empty
  and is to be filled in by hand; until then the touched share of a video is not its error rate. A first look at
  videos 6 and 12: the repaired frames of video 6 are the pupil_left-at-the-border case throughout, a real error mode
  that this video's 70 truth frames do not contain, and its dropped frames are frames where the four pupil points
  collapse onto the glint or the lid; video 12 is mixed, with some repairs moving pupil_top a short way along the
  upper lid and some dropped frames showing a narrow, dark eye with plausible-looking points.
