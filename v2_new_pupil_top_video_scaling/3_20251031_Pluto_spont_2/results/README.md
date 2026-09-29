
## Result so far (median frame RMSE over the 50 frozen test frames; final snapshot = epoch 120)

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-2 step-3 model, 60 labels, applied unchanged) | 4.43 px | 23.2 | 0% | 89% | – |
| 20 | **3.40 px** | 21.3 | 0% | 96% | 3,679 (6.4%) |
| 40 | 3.87 px | 15.8 | 0% | 95% | 3,328 (5.8%) |
| 60 | 3.63 px | 10.2 | 0% | 96% | 3866 (6.8%) |

Training set of step 1: 100 labels of video 0 + 100 of video 1 + 60 of video 2 + 20 of this video (280 frames).
The test50 pre-labels came from the video-2 step-2 model (40 labels) and the labeler moved 50 of 398 points, so
all scores of this video are anchored to that model wherever the labeler agreed with it (see the note in the
video-1 README); the 0-label point uses a different (later) video-2 model, so it is not the pre-labeling model itself.

Plateau rule (running best of the final-snapshot series, two consecutive steps with <= 3% improvement): neither the
40- nor the 60-label step improved the running best of 3.40 px (20 labels), so the rule fired at 60 labels and the
plateau point is **20 labels** - the third video in a row with a 20-label plateau, now at a 3-4 px floor. The tail
kept improving (p90 21.3 -> 15.8 -> 10.2 px). With the 0-label point included in the series the conclusion is the
same (0 -> 20 labels improved by 23%).

**Fewer-labels study (2026-09-28)**: step 1 retrained with 5 or 10 of batch01's 20 labels (two subsets each, training
set otherwise as at the time: videos 0-2 only) -> `fewer_labels.png` / `fewer_labels.csv`.

| labels of this video | median | p90 | frames > 50 px |
|---|---|---|---|
| 0 | 4.43 px | 23.2 | 0% |
| 5 (subset a / b) | 3.82 / 3.91 px | 22.9 / 15.4 | 0% / 0% |
| 10 (subset a / b) | 3.81 / 3.48 px | 21.7 / 22.5 | 0% / 0% |
| 20 | 3.40 px | 21.3 | 0% |

On this video (10-31 session spont_2, already 4.4 px with 0 labels) 5 labels are as good as 20.
