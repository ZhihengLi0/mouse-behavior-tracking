# Video 9: 2025-10-23 Pluto 1 (mouse B, 19.9 min; index 9) - labeling convention v2

Label scaling under convention v2, continuing the v2 line after video 8. This video had no earlier human labels: test50,
val20 and batch01 were pre-labeled with the video-6 20-label model (711) and corrected by the labeler on 2026-09-30
(test50: 177 of 400 points moved), then the test set was frozen. Training set of every step = videos 0-3 + video 4 run 2
batches 1-2 (40) + video 5 batch 1 (20) + video 6 batch 1 (20) + video 8 batch 1 (20) + this video's labels (video 7
contributed none); shuffles 1011, 1012, ...

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-6 20-label model 711, applied unchanged) | 23.41 px | 41.4 | 8% | 36% | – |
| 20 (shuffle 1011) | 10.39 px | 24.3 | 8% | 55% | 10359 of 57158 (18.1%) |
