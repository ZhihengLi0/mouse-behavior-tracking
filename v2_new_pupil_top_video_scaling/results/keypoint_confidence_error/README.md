# Error and confidence of each keypoint, all finished videos (2026-10-04; rerun 2026-10-06 with videos 12 and 13)

Question: which of the 8 keypoints are the least accurate, which get the lowest model confidence, and does a low
confidence go with a large error.

**Basic information.** Videos 0-13. Frames: the 50 frozen test frames of each video (final 10% of the video, never
used for training). Labels: keypoint definitions made explicit on 2026-09-20 (labeling convention v2). Models: every
step of every video (ResNet-50, batch 2, 120 epochs, final snapshot); the all-videos figure uses the model at each
video's plateau point (video 7: its 20-label model). Error of a point = Euclidean distance (px) between prediction and
human label, no confidence cut-off, points without a human label excluded; confidence = the model's likelihood of the
point. Data = the `per_frame_errors.csv` files written when each step was scored.

| file | made by | content |
|---|---|---|
| `keypoint_confidence_error_videoNN.png` | `scripts/keypoint_confidence_error.py` | one figure per video (NN = video index): median error and mean confidence of each keypoint at each step; confidence against error for every test point of the plateau-point model |
| `keypoint_confidence_error_all_videos.png` | same script | per keypoint: bar = all videos pooled, dots = the 14 videos, for the median error and for the mean confidence; median error by confidence bin |
| `keypoint_confidence_error_all_videos.csv` | same script | per video, step and keypoint: n, median and p90 error, mean confidence, share with confidence >= 0.6, Spearman correlation of confidence and error |
| `keypoint_confidence_error_bins_all_videos.csv` | same script | pooled: points, median error and share of errors > 20 px in each confidence bin, per keypoint |

Pooled over the 14 videos (plateau-point models, about 700 points per keypoint):

| keypoint | median error | p90 error | mean confidence | confidence >= 0.6 | Spearman (confidence, error) |
|---|---|---|---|---|---|
| pupil top | 10.17 px | 33.3 | 0.75 | 76% | -0.39 |
| pupil bottom | 4.21 px | 22.5 | 0.85 | 91% | -0.42 |
| pupil left | 4.88 px | 37.4 | 0.76 | 78% | -0.57 |
| pupil right | 4.83 px | 15.9 | 0.83 | 94% | -0.22 |
| eyelid top | 5.81 px | 14.9 | 0.81 | 90% | -0.40 |
| eyelid bottom | 3.98 px | 13.0 | 0.84 | 95% | -0.34 |
| nasal corner | 4.36 px | 15.3 | 0.81 | 94% | -0.35 |
| temporal corner | 4.92 px | 14.7 | 0.76 | 77% | -0.51 |

1. Pupil top is the least accurate point (median 10.2 px, about twice every other point) and has the lowest mean
   confidence (0.75) together with pupil left (0.76). Pupil left has the largest 90th percentile (37 px).
2. For every keypoint a lower confidence goes with a larger error (Spearman -0.22 to -0.57). Points with confidence
   0.8-1 have a median error of 2.6-4.5 px (pupil top 8.0 px); points with confidence 0.2-0.4 have 6.6-45.7 px.
3. Pupil top is the exception in size: even its high-confidence points (0.8-1) have a median error of 8.0 px, so its
   confidence does not identify the accurate predictions as well as for the other points.
4. Videos differ: on videos 4 and 8 all four pupil points are poor (9.7-29.8 px), on video 5 only pupil top (38.8 px).
