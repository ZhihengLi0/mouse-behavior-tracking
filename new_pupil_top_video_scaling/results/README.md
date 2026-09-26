# Across videos

The final deliverable of this unit: labels needed to reach the plateau (y) against the number of videos
already in the training set (x). One point per video, taken from `<video>/results/`. Videos are indexed from 0, so the folder index = the number of earlier videos in the training set (= x).

Plateau rule (same for every video): running best of the final-snapshot series (median frame RMSE on that
video's 50 frozen test frames); the plateau is reached when two consecutive 20-label steps each improve the
running best by <= 3%. The plateau point is the last step that still improved it.

Snapshot rules (fixed, same for every video): headline = final snapshot (epoch 120); secondary column = the
snapshot with the best validation mAP after the first learning-rate drop (epochs >= 96 in the 120-epoch recipe).

| video | mouse / length | labels of earlier videos in the training set | final-snapshot series (px, 0/20/40/60/80/100 labels) | plateau | error at plateau |
|---|---|---|---|---|---|
| 0 `0_first5minvedio` | mouse A, 5.0 min | 0 | – / 12.93 / 13.00 / 12.04 / 13.07 / 12.29 | **60 labels** | 12.04 px |
| 1 `1_20251031_Pluto_spont_1` | mouse B, 19.9 min | 100 (all of video 0) | 208.2 / 10.37 / 12.17 / 11.35 / 11.59 / 10.94 | **20 labels** | 10.37 px |
| 2 `2_20251031_pluton2` | mouse B, 19.9 min (2nd recording) | 200 (videos 0 + 1) | 9.99 / 7.73 / 7.57 / 7.83 / 7.56 / – | **20 labels** (rule fired at 60; user stopped selecting at 60 labels on 2026-09-23; batch04 was already labeled and was trained + scored, no further selection) | 7.73 px |
| 3 `3_20251031_pluto3` | mouse B, 19.9 min (3rd recording) | 260 (videos 0 + 1 + 60 of video 2) | 4.43 / 3.40 / 3.87 / 3.63 / – / – | **20 labels** (rule fired at 60) | 3.40 px |
| 4 `4_20251030_Pluto_spont_1` | mouse B, 19.9 min, 2025-10-30 (poor image quality) | 320 (videos 0-3 as for video 3) | 22.80 / 21.75 / 18.19 / 17.16 / 16.40 / 15.38 / 17.10 (120) / 14.74 (140) / 14.89 (160) | not reached (first non-improving step at 160) | – |

Notes
- With 50 test frames the median moves by about 1 px between neighbouring steps from sampling alone, so the
  3% rule is at the noise floor; differences of 1 px between steps of the same video are not meaningful.
- Video 2, label-free signals kept improving after the median flattened: pool frames flagged by the jump rule
  21% -> 11% -> 6% -> 6% (steps 1-4), test keypoints with confidence >= 0.6: 59% -> 75% -> 74% -> 82% -> 89%.
  The user therefore chose to keep selecting correction batches for video 2 beyond the plateau (batch 06).
- The 0-label point of a video = the earlier videos' model applied unchanged (`scripts/score_prior_model.py`);
  video 1 has no earlier model, so no 0 point.

## Cross-video back-test (2026-09-23)

Every model of the sequence (19 final snapshots: video 0 steps 1-5, video 1 steps 1-7, video 2 steps 1-4,
video 3 steps 1-3) scored on every video's 50 frozen test frames with `scripts/cross_video_eval.py`
(`cross_video_matrix.csv`, `cross_video_curves.png`, `cross_video_matrix.png`). x = models in training order;
left of a video's dotted line the video is not yet in the training set, so those points are its 0-label regime.

| test set | mouse-A models only (video 0, 20-100 labels) | first model with own labels | later models (own labels + later videos) |
|---|---|---|---|
| video 0 (mouse A) | 12.0-13.1 px | – | 11.9-12.9 px; never degrades while 220 mouse-B labels are added |
| video 1 (Pluto 1) | 187-302 px (different mouse: useless) | 10.4 px (20 labels) | 11.3-12.2 px, flat |
| video 2 (Pluto 2) | 172-216 px | 20.3 -> 9.1 px with video-1 labels alone (0-label regime) | 7.0-7.8 px once its own labels enter |
| video 3 (Pluto 3) | 178-291 px | 8.7 px with 20 video-1 labels; 107 px outlier at 40 video-1 labels; 4.4-5.4 px in the 0-label regime with video-2 labels | 3.4-3.9 px |

Reading: (1) a mouse-A-only model does not transfer to mouse B at all; 20 labels of mouse B bring every
mouse-B video to about 10 px. (2) Old videos do not get worse as later videos are added (video 0 and video 1
rows are flat within the 1 px sampling noise). (3) Same-day videos transfer strongly: video 3 already sits at
4-5 px before any of its own labels, driven by the video-1/video-2 labels. (4) Single models can be off by a
lot in the 0-label regime (video 3 under the 40-label video-1 model: 107 px), so a 0-shot number from one
model is not reliable on its own.

## Pupil-area rules: which pupil points to use (2026-09-23, kaiwen's direction 3)

`scripts/pupil_area_variants.py` -> `pupil_area_variants.csv/.png`, `pupil_area_labeler_prior.csv`. Frozen test frames
of all four videos, last trained model of each video, truth = 4-point ellipse area from the human labels (all four
pupil points are ellipse endpoints under the new standard). Median |area error| in %:

| rule | video 0 (mouse A) | video 1 | video 2 | video 3 | pooled (196 frames) |
|---|---|---|---|---|---|
| current 3-point (pupil_top unused, `pupil_trace.py`) | 6.5 | 6.7 | 3.1 | 12.8 | 6.9 |
| 4-point | 2.2 | 6.2 | 4.4 | 3.2 | **3.7** |
| 3 of 4, drop bottom / left / right | 5.9 / 3.1 / 4.4 | 12.4 / 8.5 / 15.8 | 7.7 / 11.4 / 5.5 | 19.3 / 27.4 / 22.8 | 9.1 / 10.4 / 10.8 |
| 3 of 4 chosen by the labeler prior | 6.5 | 6.7 | 3.1 | 12.8 | 6.9 |
| 3 of 4, lowest model confidence dropped per frame | 5.4 | 7.1 | 4.5 | 25.5 | 7.9 |
| oracle (best 3 of 4 per frame, lower bound) | 1.4 | 2.8 | 1.3 | 11.6 | 2.5 |

- Labeler prior = the pupil point the labeler corrected most often in the pre-labeled training batches (moved > 2 px or
  emptied). It is `pupil_top` in every video (20-40% of pre-labels corrected, vs 3-23% for the other points), so the
  prior picks exactly the current rule. Dropping the hardest point does not help: using pupil_top still halves the
  pooled error (6.9% -> 3.7%), and in video 3 the 3-point rule is biased by -10.5%.
- Per-frame point selection has headroom (oracle 2.5%), but model confidence does not find it (7.9%, worse than
  4-point) - consistent with kaiwen's distrust of confidence. Any learned weighting would have to beat 3.7%.
- Recommendation: switch the production area to the 4-point rule; `pupil_trace.py` is unchanged until that is decided.

## Video quality flags

- Video 4 (`4_20251030_Pluto_spont_1`, 2025-10-30): **poor quality** - the pupil boundary is hard to see even by
  eye (labeler, 2026-09-24). Its errors are not directly comparable with the clear 2025-10-31 videos; marked in the
  cross-video figures.

## Per-video eye time series (2026-09-25, `scripts/eye_timeseries.py`)

Each video now has `results/blink_area_analysis/timeseries.png` + `timeseries_summary.csv`: pupil centre, pupil area
(4-point rule, with the current 3-point rule in grey), eye opening, lowest pupil confidence, and rasters of the
production blink rule (`pupil_trace.py`, unchanged) and of the earlier study's residual rule (> 5 MADs from a 1-s
rolling median), on the newest whole-video prediction of each video (all 11 videos as of 2026-09-26).

| video | model | median 4-pt area (px²) | 3-pt vs 4-pt | corr 3-pt/4-pt | untrusted (production rule) | pupil conf < 0.6 | any residual flag |
|---|---|---|---|---|---|---|---|
| 0 `0_first5minvedio` | 115 | 63,105 | +6.2% | 0.945 | 10.9% (92 runs) | 4.8% | 6.0% |
| 1 `1_20251031_Pluto_spont_1` | 217 | 27,028 | -3.3% | 0.987 | 26.1% (683 runs) | 17.4% | 10.2% |
| 2 `2_20251031_pluton2` | 323 | 31,996 | +0.8% | 0.991 | 30.8% (944 runs) | 12.1% | 12.5% |
| 3 `3_20251031_pluto3` | 423 | 26,022 | -2.9% | 0.993 | 37.8% (909 runs) | 17.0% | 13.8% |
| 4 `4_20251030_Pluto_spont_1` | 515 | 28,591 | -1.1% | 0.988 | 50.5% (725 runs) | 46.9% | 14.6% |
| 5 `5_20251029_Pluto_spont_1` | 515 | 40,141 | +1.7% | 0.986 | 11.7% (545 runs) | 92.8% | 5.3% |
| 6 `6_20251028_Pluto_spont_1` | 515 | 37,751 | -3.2% | 0.925 | 44.8% (872 runs) | 99.2% | 19.7% |
| 7 `7_20251027_Pluto_spont1` | 515 | 28,486 | +2.6% | 0.854 | 39.1% (910 runs) | 95.5% | 28.6% |
| 8 `8_20251024_Pluto_spont1` | 518 | 38,447 | +3.4% | 0.961 | 56.7% (484 runs) | 90.1% | 14.2% |
| 9 `9_20251023_Pluto1` | 518 | 25,851 | +3.2% | 0.969 | 52.7% (774 runs) | 90.2% | 12.6% |
| 10 `10_20251022_Pluto1` | 518 | 14,704 | -12.6% | 0.990 | 70.4% (1115 runs) | 95.8% | 11.7% |

First observations (no conclusions drawn yet; the blink / area algorithm comparison is the next step):
- The production blink rule marks 26-50% of every mouse-B video as untrusted, far more than real blinks: it was tuned
  on video 0 (11%) and on mouse B it is driven by pupil confidence < 0.6 and by normal fluctuations of the eye opening.
- The 3- and 4-point areas move together (corr 0.95-0.99); the 3-point rule is biased by -3% to +6% depending on the video.
- Video 4: pupil centre x oscillates between two positions for long stretches (e.g. 14.6-15.6 min), the signature of
  one pupil point (pupil_left) jumping between the pupil edge and another attractor.
- On mouse B the pupil area rises and falls together with the eye opening; whether this is real (arousal) or the lid
  cutting the tracked ellipse needs to be separated before the area is trusted.

## Blink and pupil-area rules across all videos (2026-09-25, `scripts/blink_area_consistency.py`)

`blink_area_consistency.png` + `_area.csv`, `_blink.csv`, `_blink_frames.csv`, `_coupling.csv`.

**Area (every frozen test set; truth = human 4-point ellipse area).** The 4-point rule is better on 7 of 8 videos
(exception: video 2, 3.1% vs 4.4%). Its bias stays within -3% to +4% on every video, while the 3-point rule's bias
swings from -13% to +6% depending on the video (on the human labels alone the 3-point area differs from the
4-point area by -13% ... +3%): 3-point areas are not comparable across videos, 4-point areas are.
(Videos 5-7: test pre-labels came from the scoring model's line, so their absolute errors are optimistic.)

| video | 0 | 1 | 2 | 3 | 4 (poor) | 5* | 6* | 7* |
|---|---|---|---|---|---|---|---|---|
| 3-point, median abs error | 6.5% | 6.7% | 3.1% | 12.8% | 16.9% | 2.5% | 9.9% | 3.3% |
| 4-point, median abs error | 2.2% | 6.2% | 4.4% | 3.2% | 13.5% | 1.9% | 1.6% | 0.2% |

**Blink (human judgement = labeled frames whose four pupil points were left empty; only frames the predicting model
never trained on; videos 0-4).** Only 16 closed-eye frames exist (2-6 per mouse-B video, none in video 0), so these
numbers are indicative. The lowest pupil confidence separates every closed frame from the open ones in every video
(AUC 1.00; closed frames all < 0.20). Eye opening relative to the video median (AUC 0.79-0.99) and the pupil-axis
centre offset (0.75-1.00) are good but video-dependent; eye opening relative to its 5-s median (the production
trigger) is inconsistent (0.52-1.00); pupil height/width does not work (AUC < 0.5).
The production rule catches all 16 closed frames but also flags 26-75% of open labeled frames; its confidence
threshold 0.6 alone flags 48% of open frames. A threshold of 0.2 would catch all 16 closed frames and flag 1.5% of
open frames - chosen on these same frames, so it must be validated on new closed-eye labels before adoption.

**Area vs eye opening.** On trusted frames the 4-point area correlates with the eye opening at r = 0.69-0.94 in every
video: the lid coupling is a consistent property of the measured area and must be separated (real pupil change vs
the lid cutting the ellipse) before areas are compared across states.

Consistent conclusions: (1) use the 4-point area; (2) the current blink rule over-flags on mouse B, mainly through the
0.6 confidence cut; (3) the lowest pupil confidence with a much lower cut (about 0.2) is the most consistent
closed-eye signal found so far; (4) the area moves with the eye opening in every video. `pupil_trace.py` is unchanged.
