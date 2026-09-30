# Video 7: 2025-10-27 Pluto spont 1 (mouse B, 19.9 min; index 7) - labeling convention v2

Label scaling under convention v2, continuing the v2 line after video 6 (finished at 20 labels). The test set (50
frames, last 10% of the video) was relabeled under v2 on 2026-09-29 (pre-labels = the labeler's own v1 labels; 16 of
400 points moved, one mis-placed eyelid_top corrected on re-check) and frozen. val20 and batch01 were pre-labeled with
the video-6 20-label model (711) and corrected (batch01: one pupil_left corrected on re-check). Training set of every
step = videos 0-3 (as before) + video 4 run 2 batches 1-2 (40) + video 5 batch 1 (20) + video 6 batch 1 (20) + this
video's labels; shuffles 811, 812, ...

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-6 20-label model 711, applied unchanged) | 5.36 px | 8.9 | 0% | 81% | – |
| 20 (shuffle 811) | 5.86 px | 7.6 | 0% | 100% | pending |
