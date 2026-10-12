# Video 18: 2025-10-08 Terra spont 1 (mouse C, 19.9 min; index 18) - labeling convention v2

The last video of the planned set. Label scaling under convention v2, continuing the v2 line after video 17 (plateau at
40 labels; its batches 1-2 are carried); the eighth video of mouse C (Terra) and the earliest Terra day. This video had no
earlier human labels: test50, val20 and batch01 were extracted on 2026-10-11 with the same model-free rule as every
video, pre-labeled with the video-17 plateau model (1812, 40 labels) and corrected by the labeler on 2026-10-11 (test50:
260 of 400 points moved, 5 left empty; val20: 111 of 160 moved, 1 empty; batch01: 84 of 160 moved, 6 empty: one frame
without pupil points and one frame without the left and right pupil points), then the test set was frozen. Training set of
every step = the carried labels of the earlier videos (740 as in video 17 + video 17 batches 1-2: 40; 780 in total) + this
video's labels; shuffles 1911, 1912, ...

Label QC notes: in test50, 14 points with pre-label confidence below 0.6 were left at the pre-label; three test frames
(eye nearly closed) keep only some of the pupil points (1, 3 and 3 of 4), as labeled; in batch01 one frame
(`img014960.png`) has a pupil width of 325 px, above the QC limit of 320 px (kept as labeled). Batch02 (saved 2026-10-11 22:49: 96 points moved, 7 emptied, 57 left at the pre-label) passed the geometry QC; 17
of its unchanged points have a pre-label confidence below 0.6 (12 eye-corner points, 1 eyelid, 4 pupil points).

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-17 40-label model 1812, applied unchanged) | 38.24 px | 57.5 | 22% | 59% | – |
| 20 (shuffle 1911) | 13.13 px | 37.3 | 0% | 72% | 17664 of 57158 (31%) |
| 40 (shuffle 1912) | 12.38 px | 33.9 | 0% | 88% | pending |

The 0-label point is the worst of the Terra videos since video 13 (69 px): the eyelid points are fine (median 0-6 px) but
every pupil point is 38-54 px off, and the model is confident on only 59% of the points. No existing model does well on
this day: the best is the video-17 80-label model 1814 at 24.3 px, then the video-12 40-label model 1312 at 25.8 px; the
other Terra models give 34-40 px and above (`../results/cross_video_matrix.csv`). Step 1 (20 labels, 2026-10-11 16:18): 13.13 px, 66% better than the 0-label 38.24 px; p90 from 57.5 to 37.3 px, no test
frame above 50 px, 72% of keypoints confident (running best 13.13 px; the next step improves if it is below 12.74 px).
Batch 2 (20 frames, `selection_sheets/selection_040_frames.png`, window opened 2026-10-11 17:40) was selected from the 20-label model's whole-video prediction: the jump rule (eps 13.3 px = 0.03 x the median eye width of this video) flags 17664 of the 57158 pool frames (31%), far more than on video 17 (8.6% at the same step), consistent with the 20-label model still being 13 px off on this day (`jump_flagged.csv`).

Step 2 (40 labels, 2026-10-12 01:20): 12.38 px, 5.7% better than the running best of 13.13 px (it had to be below 12.74 px
to count), so the running best moves to 12.38 px; the next step improves if it is below 12.01 px. p90 from 37.3 to 33.9 px,
88% of keypoints confident (72% at 20 labels). What is left is the pupil (per-point medians: pupil top 12.9, bottom 11.0,
right 8.9, left 5.6 px) and the nasal corner (14.6 px); the eyelid points and the temporal corner are at 3.6-6.9 px.
Batch 3 is being selected from the 40-label model's whole-video prediction.
