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
| 2 `2_20251031_Pluto_spont_3` | mouse B, 19.9 min (10-31 session spont_3) | 200 (videos 0 + 1) | 9.99 / 7.73 / 7.57 / 7.83 / 7.56 / – | **20 labels** (rule fired at 60; user stopped selecting at 60 labels on 2026-09-23; batch04 was already labeled and was trained + scored, no further selection) | 7.73 px |
| 3 `3_20251031_Pluto_spont_2` | mouse B, 19.9 min (10-31 session spont_2) | 260 (videos 0 + 1 + 60 of video 2) | 4.43 / 3.40 / 3.87 / 3.63 / – / – | **20 labels** (rule fired at 60) | 3.40 px |
| 4 `4_20251030_Pluto_spont_1` | mouse B, 19.9 min, 2025-10-30 (poor image quality) | 320 (videos 0-3) | run 2 (convention v2): 32.11 / 24.64 / 15.01 / 16.82 / 17.54 | **40 labels** (run 2; run 1 under v1: 140 labels) | 15.01 px |
| 5 `5_20251029_Pluto_spont_1` | mouse B, 19.9 min, 2025-10-29 | 360 (videos 0-3 + 40 of video 4 run 2) | convention v2: 11.90 / 15.85 / 16.58 / – / – / – | **20 labels** (user decision: the median rule fires at 0, but the 0-label model is > 50 px off on 12% of test frames; 20 labels remove all of them) | 15.85 px |
| 6 `6_20251028_Pluto_spont_1` | mouse B, 19.9 min, 2025-10-28 | 380 (videos 0-3 + 40 of video 4 + 20 of video 5) | convention v2: 9.36 / 6.75 / 8.46 / 10.68 | **20 labels** (rule fired at 60; user decision 2026-09-29) | 6.75 px |
| 7 `7_20251027_Pluto_spont1` | mouse B, 19.9 min, 2025-10-27 | 400 (videos 0-3 + 40 of video 4 + 20 of video 5 + 20 of video 6) | convention v2: 5.36 / 5.86 / 5.84 | **0 labels** (rule fired at 40; user decision 2026-09-29) | 5.36 px |
| 8 `8_20251024_Pluto_spont1` | mouse B, 19.9 min, 2025-10-24 | 400 (videos 0-3 + 40 of video 4 + 20 of video 5 + 20 of video 6) | convention v2: 27.46 / 25.34 / 30.03 / 33.77 | **20 labels** (rule fired at 60; user decision 2026-09-30) | 25.34 px |
| 9 `9_20251023_Pluto1` | mouse B, 19.9 min, 2025-10-23 | 420 (videos 0-3 + 40 of video 4 + 20 each of videos 5, 6, 8) | convention v2: 23.41 / 10.39 / 8.55 / 8.67 / 8.20 / 9.39 / 8.63 (120 labels) | **80 labels** (rule fired at 120; user decision 2026-10-01) | 8.20 px |
| 10 `10_20251022_Pluto1` | mouse B, 19.9 min, 2025-10-22 | 500 (videos 0-3 + 40 of video 4 + 20 each of videos 5, 6, 8 + 80 of video 9) | convention v2: 18.13 / 11.83 / 10.32 / 10.49 / 10.76 | **40 labels** (rule fired at 80; user decision 2026-10-03) | 10.32 px |
| 11 `11_20251017_Terra_spont_1` | mouse C (Terra), 19.9 min, 2025-10-17 | 540 (videos 0-3 + 40 of video 4 + 20 each of videos 5, 6, 8 + 80 of video 9 + 40 of video 10) | convention v2: 25.96 / 9.24 / 8.86 / 8.77 / 8.65 | **40 labels** (rule fired at 80; user decision 2026-10-04) | 8.86 px |
| 12 `12_20251016_Terra_spont_2` | mouse C (Terra), 19.9 min, 2025-10-16 | 580 (as for video 11 + 40 of video 11) | convention v2: 5.13 (0 labels; biased low by the pre-labels; see the video's README) / 5.77 / 5.84 | **0 labels** (rule fired at 40; the video-11 model already fits) | 5.13 px |
| 13 `13_20251015_Terra_spont_1` | mouse C (Terra), 19.9 min, 2025-10-15 | 580 (video 12 carries none) | convention v2: 69.13 / 9.43 / 9.44 | in progress (one flat step at 40) | – |

`labels_to_plateau.png` / `labels_to_plateau.csv` (2026-09-30): the plateau point of every video in the sequence as one figure - labels needed (left) and the error at the plateau vs the 0-label error (right); `scripts/plot_labels_to_plateau.py`. The csv is the table above in machine-readable form and is updated when a video finishes.

Notes
- With 50 test frames the median moves by about 1 px between neighbouring steps from sampling alone, so the
  3% rule is at the noise floor; differences of 1 px between steps of the same video are not meaningful.
- Video 2, label-free signals kept improving after the median flattened: pool frames flagged by the jump rule
  21% -> 11% -> 6% -> 6% (steps 1-4), test keypoints with confidence >= 0.6: 59% -> 75% -> 74% -> 82% -> 89%.
  The user therefore chose to keep selecting correction batches for video 2 beyond the plateau (batch 06).
- The 0-label point of a video = the earlier videos' model applied unchanged (`scripts/score_prior_model.py`);
  video 1 has no earlier model, so no 0 point.

## Labels carried into later videos (rule fixed 2026-10-01)

Fixed rule for every video from video 4 on and for all future videos: a video is finished when it reaches its
plateau, the next video starts from there, and only the labels up to the plateau point go into the training sets of
later videos. This mirrors real use, where labeling stops once the plateau is reached. Labels made beyond the plateau
point stay on disk and are not used by later videos. Every model is trained from scratch on its whole training set.

Videos 0-3 carry the labels that existed when the next video started, which is more than their plateau point
(320 labels instead of 120). These 320 labels are the same in the training set of every later model, so the
comparison between later videos is not affected.

| video | labeled in total | plateau point | carried into later videos |
|---|---|---|---|
| 0 | 100 | 60 | 100 |
| 1 | 140 | 20 | 100 |
| 2 | 80 | 20 | 60 |
| 3 | 60 | 20 | 60 |
| 4 (run 2) | 80 | 40 | 40 |
| 5 | 40 | 20 | 20 |
| 6 | 60 | 20 | 20 |
| 7 | 40 | 0 | 0 |
| 8 | 60 | 20 | 20 |
| 9 | 120 | 80 | 80 |
| 10 | 80 | 40 | 40 |
| 11 | 80 | 40 | 40 |

Sum carried into video 9: 420 labels; into video 10: 500 labels; into video 11: 540 labels; into video 12: 580 labels (the "labels of earlier videos" column of the table at the top).

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
rolling median), on the newest whole-video prediction of each video. Videos 4-10 were drawn with the discarded convention-v1 branch models (515/518) and were removed on 2026-09-27 (kept at tag `v0.8.0-video4-convention1`); they are redrawn with convention-v2 models as the videos are processed.

| video | model | median 4-pt area (px²) | 3-pt vs 4-pt | corr 3-pt/4-pt | untrusted (production rule) | pupil conf < 0.6 | any residual flag |
|---|---|---|---|---|---|---|---|
| 0 `0_first5minvedio` | 115 | 63,105 | +6.2% | 0.945 | 10.9% (92 runs) | 4.8% | 6.0% |
| 1 `1_20251031_Pluto_spont_1` | 217 | 27,028 | -3.3% | 0.987 | 26.1% (683 runs) | 17.4% | 10.2% |
| 2 `2_20251031_Pluto_spont_3` | 323 | 31,996 | +0.8% | 0.991 | 30.8% (944 runs) | 12.1% | 12.5% |
| 3 `3_20251031_Pluto_spont_2` | 423 | 26,022 | -2.9% | 0.993 | 37.8% (909 runs) | 17.0% | 13.8% |

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

## Labeling convention v2 (2026-09-26)

On videos where the pupil edge appears doubled (a second contour next to the real edge), points are now placed on the
LEFT contour and pupil_top / pupil_bottom are shifted left accordingly. Video 4 restarts from 50/20/20 under v2 (run 1
archived, tag `v0.8.0-video4-convention1`); video 5 restarts after it. Rows of the run-1 models (511-519, 611) and of
the video 4/5 test sets were removed from `cross_video_matrix.csv` (copy of the old matrix in the tag). The test sets
of videos 5-10 were labeled under v1 and are to be relabeled under v2 where the double contour occurs.

### Test-set relabeling under v2 (2026-09-26)

Test sets of later videos are relabeled under v2 with the user's own v1 labels as pre-labels (so only double-contour
points move and no model's predictions anchor the labels), then re-frozen and re-scored by every model.

- Video 5 (2025-10-29): 12 of 400 points changed (9 pupil_top, 3 pupil_bottom; median move 8-16 px); pupil width
  median unchanged (241 px). v1 and v2 agree on this video.
- ~~The first run-2 model of video 4 (shuffle 531) scores 91.6 px on video 5 ... carries that definition over.~~ **Retracted 2026-09-27**: the next run-2 model (532, 40 v2 labels) scores 11.9 px on video 5 and 3.9-12.1 px on
  videos 0-3, like every other model; 531's 91.6 px was a single-model failure of the kind seen before (e.g. 107 and
  178 px), not evidence that the video-4 v2 labels bias other videos. Open question unchanged: the v2 labels of video 4
  mark a much smaller pupil (width 92 vs 215 px) - which one is right is discussed with Kaiwen Sheng.

### v1 branch removed (2026-09-27)

At the user's request the convention-v1 branch after video 3 was deleted locally to free space (about 12 GB):
video-4 run-1 labels/predictions/models (shuffles 511-519), the first video-5 attempt (611), and whole-video
predictions / time series of videos 4-10 made with those models. Everything that was published remains at tag
`v0.8.0-video4-convention1`. Kept: the v1 test labels of videos 6-10 (pre-labels for their v2 relabel) and the
video-4 zero-shot prediction of the pre-fork model 423.

## Fewer-labels study (2026-09-27 / 28): are 5 or 10 labels enough?

Step 1 of every video retrained with 5 or 10 of its 20 batch01 labels (two model-free subsets each,
`scripts/make_label_subsets.py`; `scale_step.py --subset`). Everything else is exactly the step-1 training set of that
time (earlier videos only, never later ones), same recipe, same frozen test set and scoring. Per-video figures:
`<video>/results/fewer_labels.png`. Median frame RMSE (share of test frames > 50 px):

| video | 0 labels | 5 labels (a / b) | 10 labels (a / b) | 20 labels |
|---|---|---|---|---|
| 0 (mouse A, no earlier video) | – | 78.5 / 81.5 (70% / 72%) | 20.5 / 20.8 (14% / 8%) | 12.93 (4%) |
| 1 (first video of mouse B) | 208.2 (98%) | 96.0 / 41.4 (56% / 48%) | 20.2 / 16.1 (42% / 34%) | 10.37 (4%) |
| 2 | 9.99 (10%) | 8.2 / 11.5 (10% / 18%) | 10.5 / 9.9 (22% / 12%) | 7.73 (8%) |
| 3 | 4.43 (0%) | 3.8 / 3.9 (0% / 0%) | 3.8 / 3.5 (0% / 0%) | 3.40 (0%) |
| 4 (poor quality, v2 labels) | 32.11 (22%) | 30.7 / 32.2 (10% / 14%) | 45.3 / 44.8 (42% / 32%) | 24.64 (6%) |
| 5 (v2 labels) | 11.90 (12%) | 13.0 / 9.4 (0% / 4%) | 12.6 / 12.7 (0% / 4%) | 15.85 (0%) |
| 6 (v2 labels; added 2026-10-03) | 9.36 (0%) | 6.7 / 12.8 (0% / 2%) | 7.6 / 10.1 (0% / 2%) | 6.75 (0%) |
| 7 (v2 labels; added 2026-10-03) | 5.36 (0%) | 6.4 / 6.1 (0% / 0%) | 5.8 / 5.8 (0% / 0%) | 5.86 (0%) |
| 8 (v2 labels; added 2026-10-02) | 27.46 (14%) | 31.1 / 31.5 (30% / 28%) | 23.5 / 22.4 (12% / 6%) | 25.34 (14%) |
| 9 (v2 labels; added 2026-10-02) | 23.41 (8%) | 11.7 / 12.6 (6% / 8%) | 11.6 / 11.1 (0% / 0%) | 10.39 (8%) |
| 10 (v2 labels; added 2026-10-03) | 18.13 (8%) | 15.0 / 12.5 (4% / 2%) | 11.7 / 13.1 (8% / 2%) | 11.83 (2%) |
| 11 (Terra, first video of mouse C; added 2026-10-04) | 25.96 (16%) | 9.7 / 10.0 (2% / 2%) | 9.5 / 8.9 (0% / 2%) | 9.24 (20%) |

- What this study compares: the size of the FIRST batch. The 5 and 10 labels are taken out of the 20 batch01 labels that were
  already made (appearance-based, model-free, like batch01 itself). It is not a procedure of 5 labels per step (label 5,
  select with the jump rule, label 5 more); that procedure was not run and these numbers must not be read as its result. The
  regular procedure stays unchanged: 20 labels in the first batch and 20 per step (user decision 2026-10-03).
- 5 labels were enough on 2 of 6 videos (3 and 5), both of which the earlier videos already covered well.
- On the other 4, 5 and 10 labels fall short of 20: the first video of a mouse (0, 1) improves steadily with labels,
  video 2 does not improve on 0 labels until 20, and on video 4 10 labels are worse than none (both subsets agree;
  not explained yet).
- 2026-10-02: the study is being extended to videos 6-10 and to every later video (subset models use shuffles 2000 + 10 * video + 1..4, trained while the main line is idle). Video 8, a hard day: 5 labels are worse than none, 10 labels are at least as good as 20. Video 9: 5 labels already halve the error (23.4 -> about 12 px); 10 and 20 labels add about 1 px each.
- Two subsets of the same size can differ a lot (video 1: 96 vs 41 px at 5 labels), so a single small batch is
  unreliable. 20 labels was the smallest size that worked on every video.

## Analyses across all finished videos (2026-10-04)

| folder | script (in `../scripts/`) | content |
|---|---|---|
| `blink_and_area/` | `blink_and_area.py` | pupil-area rules, eye-closure signals and rules, time series; one figure per video (`blink_area_timeseries_videoNN.png`) and all videos together. Four-point area: median error 6.6% on 594 test frames (three-point rule 8.3%); rule "mean pupil confidence < 0.527": 27 of 28 human closed-eye frames detected, 4.1% of open frames flagged |
| `keypoint_confidence_error/` | `keypoint_confidence_error.py` | error and confidence of each of the 8 keypoints; one figure per video (`keypoint_confidence_error_videoNN.png`) and all videos together. Pupil top is the least accurate point (median 10.9 px; the others 3.9-5.8 px) |

| `unreliable_frames/` | `unreliable_frames.py` | check of the jump rule and of the confidence on 910 evenly spaced human-labeled frames (test50 + val20 of videos 0-12): the jump rule flags 12.6% of the frames and finds 20 of the 30 frames above 50 px; lowest confidence < 0.6 finds all 30 but flags 52.9% |
| `blink_eyelid_distance/` | `blink_by_eyelid_distance.py`, `blink_final_rule.py`, `blink_spotcheck_round2.py` | first-round closure rule from the eyelid distance (r < 0.70 of the video median) with the handling of failed eyelid points; 137-item human spot check (2026-10-05) used to choose the final rule D (precision 96.9%, recall 86.2% among the sampled items); 512 closure events on videos 0-12 (`blink_final_*`); a second, blind 100-item spot check drawn afterwards (`blink_r2_*`, verdicts pending) |

Each folder has its own README with the basic information, the file-to-script table and the results.

## All videos in one figure, three views (2026-10-04)

`scripts/plot_all_videos_views.py` draws the same figure (blocks = videos in training order, x inside a block = labels
of that video, y = median frame RMSE of each test set, from `cross_video_matrix.csv` and `labels_to_plateau.csv`) three
times with a different emphasis:

| file | emphasis |
|---|---|
| `all_videos_view_main.png` | main line: each video's own test set while its own labels are added |
| `all_videos_view_fewer.png` | first batch of 5 / 10 labels: the band spans the two subsets (a, b) of each size. These models were only trained and scored afterwards; no jump selection and no further labeling followed them |
| `all_videos_view_backtest.png` | back-test: each test set after its own block, one colour per video |

## Videos 4, 5 and 8: larger errors (note of 2026-10-04)

Plateau errors 15.01, 15.85 and 25.34 px. The labeler's judgement: these videos are blurry and the pupil edge and
point positions are hard to identify even by eye, so the human labels themselves are less certain. Not confirmed by a
number: whole-image sharpness (variance of the Laplacian) and brightness of the 50 test frames do not differ between
these three and the other Pluto videos (sharpness 738-833 for videos 4, 5, 8; 774-854 for videos 1-3, 6, 7, 9).
