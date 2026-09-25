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
| 4 `4_20251030_Pluto_spont_1` | mouse B, 19.9 min, 2025-10-30 (poor image quality) | 320 (videos 0-3 as for video 3) | 22.80 / 21.75 / 18.19 / 17.16 / 16.40 / – | not reached (in progress) | – |

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
