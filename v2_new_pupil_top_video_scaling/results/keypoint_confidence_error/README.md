# Error and confidence of each keypoint, all finished videos (2026-10-04)

Question: which of the 8 keypoints are the least accurate, which get the lowest model confidence, and does a low
confidence go with a large error.

**Basic information.** Videos 0-11. Frames: the 50 frozen test frames of each video (final 10% of the video, never
used for training). Labels: keypoint definitions made explicit on 2026-09-20 (labeling convention v2). Models: every
step of every video (ResNet-50, batch 2, 120 epochs, final snapshot); the all-videos figure uses the model at each
video's plateau point (video 7: its 20-label model). Error of a point = Euclidean distance (px) between prediction and
human label, no confidence cut-off, points without a human label excluded; confidence = the model's likelihood of the
point. Data = the `per_frame_errors.csv` files written when each step was scored.

| file | made by | content |
|---|---|---|
| `keypoint_confidence_error_videoNN.png` | `scripts/keypoint_confidence_error.py` | one figure per video (NN = video index): median error and mean confidence of each keypoint at each step; confidence against error for every test point of the plateau-point model |
| `keypoint_confidence_error_all_videos.png` | same script | the two tables with one column per video plus the pooled column; median error by confidence bin |
| `keypoint_confidence_error_all_videos.csv` | same script | per video, step and keypoint: n, median and p90 error, mean confidence, share with confidence >= 0.6, Spearman correlation of confidence and error |
| `keypoint_confidence_error_bins_all_videos.csv` | same script | pooled: points, median error and share of errors > 20 px in each confidence bin, per keypoint |

Pooled over the 12 videos (plateau-point models, about 600 points per keypoint):

| keypoint | median error | p90 error | mean confidence | confidence >= 0.6 | Spearman (confidence, error) |
|---|---|---|---|---|---|
| pupil top | 10.92 px | 35.2 | 0.72 | 72% | -0.36 |
| pupil bottom | 4.82 px | 24.2 | 0.84 | 90% | -0.46 |
| pupil left | 4.88 px | 39.2 | 0.74 | 75% | -0.60 |
| pupil right | 4.84 px | 17.9 | 0.83 | 94% | -0.26 |
| eyelid top | 5.82 px | 15.0 | 0.80 | 88% | -0.44 |
| eyelid bottom | 3.89 px | 13.3 | 0.85 | 95% | -0.33 |
| nasal corner | 4.45 px | 17.0 | 0.80 | 94% | -0.38 |
| temporal corner | 4.75 px | 13.9 | 0.77 | 81% | -0.51 |

1. Pupil top is the least accurate point (median 10.9 px, about twice every other point) and has the lowest mean
   confidence (0.72) together with pupil left (0.74). Pupil left has the largest 90th percentile (39 px).
2. For every keypoint a lower confidence goes with a larger error (Spearman -0.26 to -0.60). Points with confidence
   0.8-1 have a median error of 2.4-4.5 px (pupil top 8.3 px); points with confidence 0.2-0.4 have 6.6-45.7 px.
3. Pupil top is the exception in size: even its high-confidence points (0.8-1) have a median error of 8.3 px, so its
   confidence does not identify the accurate predictions as well as for the other points.
4. Videos differ: on videos 4 and 8 all four pupil points are poor (9.7-29.8 px), on video 5 only pupil top (38.8 px).
