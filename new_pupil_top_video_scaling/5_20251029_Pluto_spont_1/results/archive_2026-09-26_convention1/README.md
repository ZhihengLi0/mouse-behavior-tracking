# Video 5: 2025-10-29 Pluto spont 1 (mouse B, 19.9 min; index 5)

Started as a test-set-only video (2026-09-24: frozen test set scored by every model). Per-video label scaling started
2026-09-26 after video 4 reached its plateau (140 labels). Training set of every step = all labels of the earlier
videos up to their plateau / stop point (videos 0-3 as before, video 4 batches 1-7 = 140 labels) + this video's labels.
Test pre-labels came from model 511 (video 4 step 1), so the 0-label point below (model 517) is not the pre-labeling model.

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-4 plateau model 517, 460 labels, applied unchanged) | 7.16 px | 48.2 | 10% | 73% | – |
| 20 | 5.92 px | 11.1 | 0% | 96% | – |

`blink_area_analysis/timeseries.png`: eye time series with model 515.
