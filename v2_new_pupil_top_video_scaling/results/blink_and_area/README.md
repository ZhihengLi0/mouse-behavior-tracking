# Blink and pupil area, all finished videos (2026-10-04; rerun 2026-10-06 with videos 12 and 13, 2026-10-07 with video 14)

Question: which rule gives the most accurate pupil area, and which rule best detects a closed eye, checked on every
finished video of the v2 line with one script and one set of formulas.

**Basic information.** Videos 0-14 (1 of mouse A, 10 of Pluto, 4 of Terra). Labels: keypoint definitions made explicit
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
| `blink_area_closed_frames_all_videos.jpg` | part B | all 31 frames used as "closed" truth |
| `blink_area_timeseries_videoNN.png` | part C | one figure per video (NN = video index): area, eye opening, confidence, pupil centre |
| `blink_area_timeseries_all_videos.csv` / `.png` | part C | summary table; area trace of every video in one figure |

## A. Area

Frames: the test frames (50 per video, final 10% of the video) with all four pupil points labeled, 743 in total.
Truth `A = pi/4 * |xR - xL| * |yB - yT|` from the human points; error of a frame `= 100 * |A_hat - A| / A`.

| rule | median error | 90th percentile | median signed error |
|---|---|---|---|
| four points | 5.68% | 33.8% | -1.7% |
| no top (rule of `pupil_trace.py`) | 7.59% | 36.9% | 0.4% |
| no bottom / no left / no right | 10.38 / 11.13 / 9.63% | 42.4 / 36.6 / 39.4% | |
| drop the lowest-confidence point | 8.42% | 34.1% | -0.9% |
| four, fallback to drop-lowest when a confidence < 0.6 | 5.84% | 33.8% | -1.9% |
| four + 5-frame median / + 31-frame (0.5 s) median | 5.50 / 5.74% | 34.1 / 34.0% | |

1. The four-point rule has the lowest pooled median error (5.7%; three-point rule 7.6%).
2. Per video it is lower than the three-point rule on 11 of 15 videos; the three-point rule is lower on videos 1, 2, 5, 7.
3. A median filter over time does not help: the error is not frame-to-frame jitter.
4. Videos 4 and 8 have 43.8% and 24.0% (four-point); these are the videos with the largest keypoint error
   (15.01 and 25.34 px). The pooled 90th percentile is 34%.
5. Why the four-point rule is better: both rules use the same width; the three-point rule has no top point and assumes
   that the left and right points lie exactly at mid-height of the ellipse (height = twice the distance from the bottom
   point to the line through them). A labeler cannot place them exactly at mid-height; the four-point rule uses only
   their x coordinates and is not affected, the three-point rule doubles the offset into the height. The three-point
   formula applied to the human points themselves (rows `no_top_on_human_labels` of the CSV, no model involved)
   differs from the human four-point area by a median of 4.8% (90th percentile 12.2%; 13.0% on video 3, 9.0% on video 6).

## B. Eye closure

Truth: closed = the labeler left all four pupil points empty (31 frames, videos 1-9 and 14); open = all four labeled (1828
frames). Only frames the predicting model was not trained on (test50, val20: plateau model; batch N >= 2: model of
step N-1). AUC pooled: mean pupil confidence 0.995, lowest pupil confidence 0.991, pupil-axes midpoint offset 0.948,
eye opening / video median 0.938, eye opening / 5-s median 0.911, pupil height / width 0.390.

| rule | closed frames detected | open frames flagged |
|---|---|---|
| mean pupil confidence < t (t leave-one-video-out; 0.527 on all frames) | 30 of 31 | 66 of 1828 (3.6%) |
| lowest pupil confidence < t (0.295) | 28 of 31 | 108 (5.9%) |
| eye opening / 5-s median < t (0.768) | 27 of 31 | 45 (2.5%) |
| eye opening / video median < t (0.695) | 26 of 31 | 59 (3.2%) |
| rule of `pupil_trace.py` (left/right/bottom confidence < 0.6 or opening < 95% of its 5-s median, +-3 frames) | 31 of 31 | 977 (53.4%) |

Limits: 31 closed frames only, none in videos 0, 10-13 (video 14 is the first Terra video with closed frames); they come from batches picked by the jump
rule, not from a random sample; "closed" = all four pupil points left empty by the labeler.

## C. Time series

Rule `mean pupil confidence < 0.527`, flagged frames closer than 0.25 s merged into events
(`blink_area_timeseries_all_videos.csv`): 0.1-2.6% of the frames flagged on 14 videos (0.6-8.5 events per minute),
12.9% on video 8 (second half, low confidence overall). The rule of `pupil_trace.py` flags 10-84% of the same videos.

Not solved: the area traces of videos 1, 2, 4, 6 and 8 contain stretches where neighbouring frames jump back and
forth with the eye open (one pupil point alternating between two positions); the closure rule does not flag them.
The area correlates with the eye opening (r = 0.72-0.92 on 14 videos, 0.36 on video 1); this number alone does not
separate a real joint change from the upper eyelid covering the pupil. Next: a rule for the jumps checked against
human-labeled frames, and an event-by-event check against the original video.

The earlier analyses (`../pupil_area_variants.*`, `../blink_area_consistency*`, `<video>/results/blink_area_analysis/`,
2026-09-23/25, videos 0-7) are kept as they are; this folder repeats them on all finished videos with the current labels.
