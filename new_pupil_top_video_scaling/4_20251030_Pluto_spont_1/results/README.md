# Video 4: 2025-10-30 Pluto spont 1 (mouse B, 19.9 min; index 4 = four earlier videos in the training set)

First recording of a different day (the earlier mouse-B videos 1-3 are all from 2025-10-31). Started 2026-09-24.
Protocol as for every video: 50 frozen test frames (final 10%), 20 validation frames, batches of 20; batch01 by
k-means, later batches by jump rule + k-means. The video copy is `4_20251030_Pluto_spont_1.mp4` (local only).

## Result so far (median frame RMSE over the 50 frozen test frames; final snapshot = epoch 120)

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-3 step-3 model, 320 labels, applied unchanged) | 22.80 px | 44.5 | 4% | 90% | – |
| 20 | 21.75 px | 44.0 | 6% | 87% | 8,548 (15.0%) |

Training set of step 1 (shuffle 511): 100 labels of video 0 + 100 of video 1 + 60 of video 2 + 60 of video 3 +
20 of this video (340 frames); validation = the 20 val frames of this video.

- The 0-label error is about five times that of the same-day videos (video 3 started at 4.43 px), and all 19
  earlier models score 22-353 px on this test set (`../../results/cross_video_matrix.csv`): a new day is a new domain.
- The first 20 labels fixed pupil_right (22.6 -> 6.5 px) but not pupil_left / pupil_bottom (about 30 px each);
  the model under-sizes the pupil on this day (pre-label width median 157 px vs 215 px in the human test labels).
- Anchoring: the test50 pre-labels came from the same model as the 0-label point (shuffle 423) and the labeler moved
  146 of 400 points, so the 0-label score is optimistic wherever the labeler agreed with that model.
