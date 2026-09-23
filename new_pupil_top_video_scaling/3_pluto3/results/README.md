
## Result so far (median frame RMSE over the 50 frozen test frames; final snapshot = epoch 120)

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-2 step-3 model, 60 labels, applied unchanged) | 4.43 px | 23.2 | 0% | 89% | – |
| 20 | **3.40 px** | 21.3 | 0% | 96% | 3,679 (6.4%) |
| 40 | 3.87 px | 15.8 | 0% | 95% | 3,328 (5.8%) |

Training set of step 1: 100 labels of video 0 + 100 of video 1 + 60 of video 2 + 20 of this video (280 frames).
The test50 pre-labels came from the video-2 step-2 model (40 labels) and the labeler moved 50 of 398 points, so
all scores of this video are anchored to that model wherever the labeler agreed with it (see the note in the
video-1 README); the 0-label point uses a different (later) video-2 model, so it is not the pre-labeling model itself.
