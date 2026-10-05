# Video 13: 2025-10-15 Terra spont 1 (mouse C, 19.9 min; index 13) - labeling convention v2

Label scaling under convention v2, continuing the v2 line after video 12 (finished with its plateau at 0 labels, so it
carries none); the third video of mouse C (Terra). This video had no earlier human labels: test50, val20 and batch01
were pre-labeled with the video-11 40-label model (1212) and corrected by the labeler on 2026-10-04 (test50: 287 of 400
points moved; val20: 102 of 160; batch01: 96 of 160), then the test set was frozen. Training set of every step = the
carried labels of the earlier videos (videos 0-3: 100 + 100 + 60 + 60; video 4 run 2 batches 1-2: 40; videos 5, 6, 8
batch 1: 20 each; video 9 batches 1-4: 80; videos 10 and 11 batches 1-2: 40 each; videos 7 and 12 contributed none; 580
in total) + this video's labels; shuffles 1411, 1412, ...

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-11 40-label model 1212, applied unchanged) | 69.13 px | 116.6 | 56% | 24% | – |
| 20 (shuffle 1411) | 9.43 px | 16.0 | 0% | 86% | pending |

The model that fitted video 12 (the day after, 5.13 px) does not fit this day: 69 px at 0 labels, more than half of
the test frames above 50 px. The other models of videos 11 and 12 give 31.5-79.4 px on this test set
(`../results/cross_video_matrix.csv`).
