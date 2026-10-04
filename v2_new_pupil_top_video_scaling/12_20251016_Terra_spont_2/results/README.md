# Video 12: 2025-10-16 Terra spont 2 (mouse C, 19.9 min; index 12) - labeling convention v2

Label scaling under convention v2, continuing the v2 line after video 11 (finished at 40 labels); the second video of
mouse C (Terra). This video had no earlier human labels: test50, val20 and batch01 were pre-labeled with the video-11
40-label model (1212) and corrected by the labeler on 2026-10-04 (test50: 110 of 400 points moved; val20: 72 of 160
moved, 1 emptied; batch01: 74 of 160 moved, 4 emptied), then the test set was frozen. Training set of every step = the
carried labels of the earlier videos (videos 0-3: 100 + 100 + 60 + 60; video 4 run 2 batches 1-2: 40; videos 5, 6, 8
batch 1: 20 each; video 9 batches 1-4: 80; videos 10 and 11 batches 1-2: 40 each; video 7 contributed none; 580 in
total) + this video's labels; shuffles 1311, 1312, ...

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-11 40-label model 1212, applied unchanged) | 5.13 px | 9.4 | 0% | 77% | – |

The 0-label number is biased low: model 1212 produced the pre-labels of this test set and 290 of the 400 test points
were left as pre-labeled, so its error on those points is 0. The other models of video 11 (20 / 60 / 80 labels), which
did not produce the pre-labels, give 10.66 / 8.29 / 8.99 px on this test set (`../results/cross_video_matrix.csv`).
