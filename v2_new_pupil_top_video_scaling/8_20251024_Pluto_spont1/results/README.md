# Video 8: 2025-10-24 Pluto spont 1 (mouse B, 19.9 min; index 8) - labeling convention v2

Label scaling under convention v2, continuing the v2 line after video 7 (which needed no labels of its own). The test
set (50 frames, last 10% of the video) was relabeled under v2 on 2026-09-29 (pre-labels = the labeler's own v1
labels; 113 of 400 points moved, 3 emptied) and frozen. val20 and batch01 were pre-labeled with the video-6 20-label
model (711) and corrected. Training set of every step = videos 0-3 + video 4 run 2 batches 1-2 (40) + video 5 batch 1
(20) + video 6 batch 1 (20) + this video's labels (video 7 contributed none); shuffles 911, 912, ...

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-6 20-label model 711, applied unchanged) | 27.46 px | 51.3 | 14% | 59% | – |
| 20 (shuffle 911) | 25.34 px | 50.6 | 14% | 68% | 9741 of 57158 (17.0%) |
| 40 (shuffle 912) | 30.03 px | 50.4 | 12% | 89% | 8734 of 57158 (15.3%) |
