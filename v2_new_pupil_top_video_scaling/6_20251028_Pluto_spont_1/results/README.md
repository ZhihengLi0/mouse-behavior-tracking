# Video 6: 2025-10-28 Pluto spont 1 (mouse B, 19.9 min; index 6) - labeling convention v2

Label scaling under convention v2, continuing the v2 line after video 5 (finished at 20 labels). The test set (50
frames, last 10% of the video) was relabeled under v2 on 2026-09-28 (pre-labels = the labeler's own v1 labels; 26 of
400 points moved) and frozen. val20 and batch01 were pre-labeled with the video-5 20-label model (631) and corrected.
Training set of every step = videos 0-3 (as before) + video 4 run 2 batches 1-2 (40) + video 5 batch 1 (20) + this
video's labels; shuffles 711, 712, ...

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 20 (shuffle 711) | 6.75 px | 11.2 | 0% | 83% | 6801 of 57158 (11.9%) |

Back-test of model 711 on the other frozen test sets: videos 1-5 = 12.01 / 6.78 / 3.21 / 14.80 / 13.65 px, video 7 = 5.36 px, video 8 = 20.76 px (`../results/cross_video_matrix.csv`).
