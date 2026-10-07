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
2026-10-07 (standard) and corrected by the labeler. Training set of every step = the carried labels of the earlier videos
(videos 0-3: 100 + 100 + 60 + 60; video 4 run 2 batches 1-2: 40; videos 5, 6, 8 batch 1: 20 each; video 9 batches 1-4: 80;
videos 10 and 11 batches 1-2: 40 each; video 13 batch 1: 20; video 14 batches 1-3: 60; videos 7 and 12 contributed none;
660 in total) + this video's labels; shuffles 1611, 1612, ...

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-14 60-label model 1513, applied unchanged) | 10.57 px | 28.8 | 0% | 64% | – |

The other video-14 models give 10.4-10.8 px on this test set (40 labels 10.40, 80 labels 10.77), the video-12 models
11.4-16.4 px, the video-13 plateau model 26.5 px (`../results/cross_video_matrix.csv`).
