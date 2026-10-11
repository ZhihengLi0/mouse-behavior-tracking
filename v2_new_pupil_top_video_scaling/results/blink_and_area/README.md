# Blink and pupil area, all finished videos (2026-10-04; rerun 2026-10-06 with videos 12 and 13, 2026-10-07 with video 14, 2026-10-08 with video 15, 2026-10-10 with video 16, 2026-10-11 with video 17)

Question: which rule gives the most accurate pupil area, and which rule best detects a closed eye, checked on every
finished video of the v2 line with one script and one set of formulas.

**Basic information.** Videos 0-17 (1 of mouse A, 10 of Pluto, 7 of Terra). Labels: keypoint definitions made explicit
on 2026-09-20 (labeling convention v2). Model of a video = the model at its plateau point (`../labels_to_plateau.csv`;
video 7, plateau at 0 labels, uses its 20-label model); ResNet-50, batch 2, 120 epochs, final snapshot, prediction of
every frame of the whole video, no confidence cut-off. Script: `../../scripts/blink_and_area.py` (formulas in its
header); it reads the human labels and the whole-video predictions and changes nothing else.

| file | made by | content |
|---|---|---|
| `blink_area_area_methods_all_videos.csv` / `.png` | `scripts/blink_and_area.py` part A | area error of 9 rules, per video and pooled |
| `blink_area_closure_frames_all_videos.csv` | part B | one row per human-labeled frame used (closed / open) with its signals |
| `blink_area_closure_signals_all_videos.csv` / `.png` | part B | AUC of each signal, per video and pooled; figure |
| `blink_area_closure_rules_all_videos.csv` | part B | threshold rules: detected closed frames and flagged open frames |
| `blink_area_closed_frames_all_videos.jpg` | part B | all 39 frames used as "closed" truth |
| `blink_area_timeseries_videoNN.png` | part C | one figure per video (NN = video index): area, eye opening, confidence, pupil centre |
| `blink_area_timeseries_all_videos.csv` / `.png` | part C | summary table; area trace of every video in one figure |

## A. Area

Frames: the test frames (50 per video, final 10% of the video) with all four pupil points labeled, 893 in total.
Truth `A = pi/4 * |xR - xL| * |yB - yT|` from the human points; error of a frame `= 100 * |A_hat - A| / A`.

| rule | median error | 90th percentile | median signed error |
|---|---|---|---|
| four points | 5.18% | 32.9% | -1.0% |
| no top (rule of `pupil_trace.py`) | 8.28% | 32.6% | 1.9% |
| no bottom / no left / no right | 10.17 / 10.82 / 9.17% | 38.4 / 35.3 / 37.3% | |
| drop the lowest-confidence point | 8.83% | 32.9% | 1.1% |
| four, fallback to drop-lowest when a confidence < 0.6 | 5.54% | 31.9% | -1.1% |
| four + 5-frame median / + 31-frame (0.5 s) median | 4.97 / 5.10% | 33.0 / 32.5% | |

1. The four-point rule has the lowest pooled median error (5.2%; three-point rule 8.3%).
2. Per video it is lower than the three-point rule on 14 of 18 videos; the three-point rule is lower on videos 1, 2, 5, 7.
3. A median filter over time does not help: the error is not frame-to-frame jitter.
4. Videos 4 and 8 have 43.8% and 24.0% (four-point); these are the videos with the largest keypoint error
   (15.01 and 25.34 px). The pooled 90th percentile is 33%.
5. Why the four-point rule is better: both rules use the same width; the three-point rule has no top point and assumes
   that the left and right points lie exactly at mid-height of the ellipse (height = twice the distance from the bottom
   point to the line through them). A labeler cannot place them exactly at mid-height; the four-point rule uses only
   their x coordinates and is not affected, the three-point rule doubles the offset into the height. The three-point
   formula applied to the human points themselves (rows `no_top_on_human_labels` of the CSV, no model involved)
   differs from the human four-point area by a median of 5.1% (90th percentile 12.1%; 13.0% on video 3, 9.0% on video 6).

## B. Eye closure

Truth: closed = the labeler left all four pupil points empty (39 frames, videos 1-9, 14, 15 and 17); open = all four labeled (2209
frames). Only frames the predicting model was not trained on (test50, val20: plateau model; batch N >= 2: model of
step N-1). AUC pooled: mean pupil confidence 0.996, lowest pupil confidence 0.992, pupil-axes midpoint offset 0.957,
eye opening / video median 0.925, eye opening / 5-s median 0.904, pupil height / width 0.391.

| rule | closed frames detected | open frames flagged |
|---|---|---|
| mean pupil confidence < t (t leave-one-video-out; 0.407 on all frames) | 38 of 39 | 64 of 2209 (2.9%) |
| lowest pupil confidence < t (0.295) | 36 of 39 | 116 (5.3%) |
| eye opening / 5-s median < t (0.767) | 34 of 39 | 54 (2.4%) |
| eye opening / video median < t (0.695) | 33 of 39 | 85 (3.8%) |
| rule of `pupil_trace.py` (left/right/bottom confidence < 0.6 or opening < 95% of its 5-s median, +-3 frames) | 39 of 39 | 1121 (50.7%) |

Limits: 39 closed frames only, none in videos 0, 10-13 and 16 (videos 14, 15 and 17 are the only Terra videos with closed frames); they come from batches picked by the jump
rule, not from a random sample; "closed" = all four pupil points left empty by the labeler.

The threshold of the mean pupil confidence chosen on all frames was 0.527 up to the rerun with video 16 and is 0.407 in
the rerun with video 17 (5 more closed frames). `blink_final_rule.py`, `blink_notopen_rule.py` and
`blink_per_mouse_threshold.py` have 0.527 as a fixed constant and were not changed.

## C. Time series

Rule `mean pupil confidence < 0.407` (the threshold of part B in this rerun; 0.527 in the reruns up to video 16),
flagged frames closer than 0.25 s merged into events (`blink_area_timeseries_all_videos.csv`): 0.0-1.4% of the frames
flagged on the 18 videos (0-5.0 events per minute; the most on video 8 with 1.4% and video 10 with 1.3%; with the
earlier threshold 0.527 video 8 had 12.9% and video 15 3.7%). The rule of `pupil_trace.py` flags 4-84% of the same videos.

Not solved: the area traces of videos 1, 2, 4, 6 and 8 contain stretches where neighbouring frames jump back and
forth with the eye open (one pupil point alternating between two positions); the closure rule does not flag them.
The area correlates with the eye opening (r = 0.65-0.94 on 17 videos, 0.36 on video 1); this number alone does not
separate a real joint change from the upper eyelid covering the pupil. Next: a rule for the jumps checked against
human-labeled frames, and an event-by-event check against the original video.

The earlier analyses (`../pupil_area_variants.*`, `../blink_area_consistency*`, `<video>/results/blink_area_analysis/`,
2026-09-23/25, videos 0-7) are kept as they are; this folder repeats them on all finished videos with the current labels.
