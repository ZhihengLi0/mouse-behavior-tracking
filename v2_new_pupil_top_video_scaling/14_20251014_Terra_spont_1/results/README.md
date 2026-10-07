# Video 14: 2025-10-14 Terra spont 1 (mouse C, 19.9 min; index 14) - labeling convention v2

Label scaling under convention v2, continuing the v2 line after video 13 (plateau at 20 labels; its batch01 is carried);
the fourth video of mouse C (Terra). This video had no earlier human labels: test50, val20 and batch01 were pre-labeled
with the video-13 20-label model (1411) and corrected by the labeler on 2026-10-06 (test50: 149 of 400 points moved;
val20: 78 of 160; batch01: 68 of 160, 4 points left empty), then the test set was frozen. Training set of every step =
the carried labels of the earlier videos (videos 0-3: 100 + 100 + 60 + 60; video 4 run 2 batches 1-2: 40; videos 5, 6, 8
batch 1: 20 each; video 9 batches 1-4: 80; videos 10 and 11 batches 1-2: 40 each; video 13 batch 1: 20; videos 7 and 12
contributed none; 600 in total) + this video's labels; shuffles 1511, 1512, ...

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-13 20-label model 1411, applied unchanged) | 13.33 px | 170.8 | 16% | 71% | – |
| 20 (shuffle 1511) | 10.83 px | 28.7 | 2% | 88% | 14576 of 57158 (25.5%) |
| 40 (shuffle 1512) | 12.41 px | 25.8 | 2% | 90% | pending |

The other Terra models give 11.3-35.4 px on this test set (video-13 40- and 60-label models 11.28 / 11.59 px, video-12
models 24.5-26.9 px, video-11 models 26.6-35.4 px; `../results/cross_video_matrix.csv`).

Step 1 (20 labels): 10.83 px, 19% better than the 0-label 13.33 px (2026-10-06 15:57); frames > 50 px from 16% to 2%.

Step 2 (40 labels): 12.41 px, no gain over the running best 10.83 px (step 1): one flat step by the rule (2026-10-06 19:31). The p90
improved from 28.7 to 25.8 px; the median did not.
