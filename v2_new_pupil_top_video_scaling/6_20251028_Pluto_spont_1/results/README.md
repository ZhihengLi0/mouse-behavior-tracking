# Video 6: 2025-10-28 Pluto spont 1 (mouse B, 19.9 min; index 6) - labeling convention v2

Label scaling under convention v2, continuing the v2 line after video 5 (finished at 20 labels). The test set (50
frames, last 10% of the video) was relabeled under v2 on 2026-09-28 (pre-labels = the labeler's own v1 labels; 26 of
400 points moved) and frozen. val20 and batch01 were pre-labeled with the video-5 20-label model (631) and corrected.
Training set of every step = videos 0-3 (as before) + video 4 run 2 batches 1-2 (40) + video 5 batch 1 (20) + this
video's labels; shuffles 711, 712, ...

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 20 (shuffle 711) | 6.75 px | 11.2 | 0% | 83% | 6801 of 57158 (11.9%) |
| 40 (shuffle 712) | 8.46 px | 11.8 | 0% | 90% | 4948 of 57158 (8.7%) |
| 60 (shuffle 713) | 10.68 px | 14.9 | 2% | 82% | pending |

**Plateau rule fired at step 3 (2026-09-29):** neither 40 labels (8.46 px) nor 60 labels (10.68 px) improved the running best of 6.75 px, so the plateau point is **20 labels, 6.75 px**; the batch04 popup is held pending the user's decision.

Back-test of model 711 on the other frozen test sets: videos 1-5 = 12.01 / 6.78 / 3.21 / 14.80 / 13.65 px, video 7 = 5.36 px, video 8 = 20.76 px (`../results/cross_video_matrix.csv`).

**Fewer-labels study (2026-10-03)**: step 1 retrained with 5 or 10 of batch01's 20 labels (two subsets each, training
set otherwise as at the time: videos 0-3 + 40 of video 4 + 20 of video 5; shuffles 2061-2064) ->
`fewer_labels.png` / `fewer_labels.csv`.

| labels of this video | median | p90 | frames > 50 px |
|---|---|---|---|
| 0 | 9.36 px | 22.3 | 0% |
| 5 (subset a / b) | 6.73 / 12.84 px | 12.9 / 27.1 | 0% / 2% |
| 10 (subset a / b) | 7.63 / 10.11 px | 10.7 / 14.5 | 0% / 2% |
| 20 | 6.75 px | 11.2 | 0% |

On this video the two subsets of each size disagree: subset a reaches the 20-label level with 5 labels (6.73 px),
subset b is worse than no labels at both sizes (12.84 and 10.11 px against 9.36 px).
