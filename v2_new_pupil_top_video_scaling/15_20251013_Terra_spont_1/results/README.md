# Video 15: 2025-10-13 Terra spont 1 (mouse C, 19.9 min; index 15) - labeling convention v2

Label scaling under convention v2, continuing the v2 line after video 14 (plateau at 60 labels; its batches 1-3 are
carried); the fifth video of mouse C (Terra). This video had no earlier human labels. Frames were extracted on 2026-10-06
with the same model-free rule as every video (test50 = 50 evenly spaced frames of the last 10%, val20 = 20 of the 10%
before it, batch01 = 20 k-means picks from the first 80%).

**Deviation from the standard procedure (recorded, not corrected):** test50 was opened at the labeler's request on
2026-10-06 23:05, before video 14's plateau was decided, with pre-labels from the video-14 20-label model (1511, video 14's
running best at that time); the labeler corrected it (126 of 400 points moved) and it was frozen at 23:41. Video 14 went on
to a plateau at 60 labels, so the standard procedure would have pre-labeled test50 with model 1513. The pre-labels are only
the starting point of the correction, every point is placed by the labeler under the keypoint definitions made explicit on
2026-09-20, and the test set stays frozen as labeled. val20 and batch01 were pre-labeled with the plateau model 1513 on
2026-10-07 (standard) and corrected by the labeler (val20: 63 of 160 points moved; batch01: 101 of 160, none left empty). Training set of every step = the carried labels of the earlier videos
(videos 0-3: 100 + 100 + 60 + 60; video 4 run 2 batches 1-2: 40; videos 5, 6, 8 batch 1: 20 each; video 9 batches 1-4: 80;
videos 10 and 11 batches 1-2: 40 each; video 13 batch 1: 20; video 14 batches 1-3: 60; videos 7 and 12 contributed none;
660 in total) + this video's labels; shuffles 1611, 1612, ...

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-14 60-label model 1513, applied unchanged) | 10.57 px | 28.8 | 0% | 64% | – |
| 20 (shuffle 1611) | 11.42 px | 26.2 | 0% | 86% | 15807 of 57158 (27.7%) |
| 40 (shuffle 1612) | 9.73 px | 19.1 | 0% | 94% | 13953 of 57158 (24.4%) |
| 60 (shuffle 1613) | 9.61 px | 26.7 | 2% | 91% | 11698 of 57158 (20.5%) |
| 80 (shuffle 1614) | 10.24 px | 25.4 | 0% | 85% | pending (batch 5 selected, not labeled) |

The other video-14 models give 10.4-10.8 px on this test set (40 labels 10.40, 80 labels 10.77), the video-12 models
11.4-16.4 px, the video-13 plateau model 26.5 px (`../results/cross_video_matrix.csv`).

Step 1 (20 labels): 11.42 px, no gain over the 0-label 10.57 px (the video-14 plateau model already fits this day): one flat
step by the rule (2026-10-07 21:26). The share of confident keypoints rose from 64% to 86% and the p90 from 28.8 to 26.2 px;
the median did not improve. Batch 2 is selected from the 20-label model's whole-video prediction and opened for labeling.

Step 2 (40 labels): 9.73 px, 7.9% better than the running best 10.57 px (0 labels), so an improving step by the rule and the
flat-step count restarts at zero (2026-10-08 01:14); p90 from 26.2 to 19.1 px, 94% of keypoints confident. Batch 3 is selected from the
40-label model's whole-video prediction and opened for labeling.

Step 3 (60 labels): 9.61 px, 1.2% better than the running best 9.73 px (step 2), below the 3% threshold: one flat step by the
rule (2026-10-08 12:10); p90 worse (19.1 to 26.7 px), one test frame above 50 px. Batch 4 is selected from the 60-label model and
opened for labeling; the rule fires if 80 labels also fail to improve by 3% on 9.73 px.

Step 4 (80 labels): 10.24 px, worse than the running best 9.73 px (step 2): second flat step in a row, so the plateau rule
fired at 80 labels (2026-10-08 16:28). Plateau point = the last improving step, 40 labels (9.73 px). The batch-5 window is held;
awaiting the user's decision.

**Closed 2026-10-08 (user decision: "规则触发就标记然后下一个视频"): plateau at 40 labels, 9.73 px** (tag `v0.9.17-video15-plateau`).
Batches 1-2 (40 labels) are carried into the later videos (700 labels go into video 16); batches 3-4 were labeled for the rule only
and are not carried; batch 5 was selected from the 80-label model but never labeled.
