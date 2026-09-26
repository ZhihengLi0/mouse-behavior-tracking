# Video 4: 2025-10-30 Pluto spont 1 (mouse B, 19.9 min; index 4 = four earlier videos in the training set)

First recording of a different day (the earlier mouse-B videos 1-3 are all from 2025-10-31). Started 2026-09-24.
Protocol as for every video: 50 frozen test frames (final 10%), 20 validation frames, batches of 20; batch01 by
k-means, later batches by jump rule + k-means. The video copy is `4_20251030_Pluto_spont_1.mp4` (local only).

> **Poor-quality video (labeler's judgement, 2026-09-24).** In this recording the pupil boundary is hard to see even
> by eye (dark, low-contrast eye; the pupil/iris edge is often not visible), so the human labels themselves are less
> certain than in videos 1-3. Keep this in mind when comparing its errors with the other videos.

## Result so far (median frame RMSE over the 50 frozen test frames; final snapshot = epoch 120)

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-3 step-3 model, 320 labels, applied unchanged) | 22.80 px | 44.5 | 4% | 90% | – |
| 20 | 21.75 px | 44.0 | 6% | 87% | 8,548 (15.0%) |
| 40 | 18.19 px | 35.3 | 4% | 80% | 10,536 (18.4%) |
| 60 | 17.16 px | 32.1 | 2% | 77% | 7,537 (13.2%) |
| 80 | 16.40 px | 34.1 | 0% | 83% | 7,725 (13.5%) |
| 100 | 15.38 px | 32.5 | 0% | 80% | 7,810 (13.7%) |
| 120 | 17.10 px | 33.4 | 2% | 84% | 6,984 (12.2%) |
| 140 | 14.74 px | 34.6 | 0% | 89% | 6,208 (10.9%) |
| 160 | 14.89 px | 35.6 | 0% | 88% | 5,799 (10.1%) |

Training set of step 1 (shuffle 511): 100 labels of video 0 + 100 of video 1 + 60 of video 2 + 60 of video 3 +
20 of this video (340 frames); validation = the 20 val frames of this video.

- The 0-label error is about five times that of the same-day videos (video 3 started at 4.43 px), and all 19
  earlier models score 22-353 px on this test set (`../../results/cross_video_matrix.csv`): a new day is a new domain.
- The first 20 labels fixed pupil_right (22.6 -> 6.5 px) but not pupil_left / pupil_bottom (about 30 px each);
  the model under-sizes the pupil on this day (pre-label width median 157 px vs 215 px in the human test labels).
- Anchoring: the test50 pre-labels came from the same model as the 0-label point (shuffle 423) and the labeler moved
  146 of 400 points, so the 0-label score is optimistic wherever the labeler agreed with that model.

Plateau rule after step 6 (2026-09-25): the 120-label model (17.10 px) did not improve the running best of 15.38 px
(100 labels), so this is the first step with <= 3% improvement; the rule fires only if step 7 (140 labels) also fails
to improve it by more than 3%. With 50 test frames on a video whose pupil edge is hard to see, a 1-2 px swing between
steps is within noise.

After step 7 (2026-09-25): 140 labels = 14.74 px, 4.2% better than the previous best (15.38 px at 100 labels), so the
plateau rule restarts: the running best is now 14.74 px and the rule fires only after two further steps with <= 3% gain.

After step 8: 160 labels = 14.89 px, no improvement over the running best (14.74 px at 140) - first step toward the plateau rule; step 9 (180 labels) decides.
