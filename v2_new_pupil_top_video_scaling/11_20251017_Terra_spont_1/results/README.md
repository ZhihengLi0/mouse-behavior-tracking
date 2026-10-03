# Video 11: 2025-10-17 Terra spont 1 (mouse C, 19.9 min; index 11) - labeling convention v2

Label scaling under convention v2, continuing the v2 line after video 10 (finished at 40 labels); the first video of a
new mouse (Terra). This video had no earlier human labels: test50, val20 and batch01 were pre-labeled with the video-10
40-label model (1112) and corrected by the labeler on 2026-10-03 (test50: 249 of 400 points moved; val20: 92 of 160;
batch01: 99 of 160), then the test set was frozen. Training set of every step = the carried labels of the earlier videos
(videos 0-3: 100 + 100 + 60 + 60; video 4 run 2 batches 1-2: 40; videos 5, 6, 8 batch 1: 20 each; video 9 batches 1-4:
80; video 10 batches 1-2: 40; video 7 contributed none; 540 in total) + this video's labels; shuffles 1211, 1212, ...

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-10 40-label model 1112, applied unchanged) | 25.96 px | 80.1 | 16% | 42% | – |
| 20 (shuffle 1211) | 9.24 px | 166.4 | 20% | 86% | pending |

Back-test of the earlier models on this test set (`../results/cross_video_matrix.csv`): the models of the sequence
score between 16.5 and 313 px; the other models of video 10 (20 / 60 / 80 labels) give 16.54 / 29.18 / 33.41 px, so
the 0-label number of a single model is not stable on this new mouse.
