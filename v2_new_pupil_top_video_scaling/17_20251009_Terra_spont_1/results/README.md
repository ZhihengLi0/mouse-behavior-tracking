# Video 17: 2025-10-09 Terra spont 1 (mouse C, 19.9 min; index 17) - labeling convention v2

Label scaling under convention v2, continuing the v2 line after video 16 (plateau at 40 labels; its batches 1-2 are
carried); the seventh video of mouse C (Terra). This video had no earlier human labels: test50, val20 and batch01 were
extracted on 2026-10-10 with the same model-free rule as every video, pre-labeled with the video-16 plateau model (1712,
40 labels) and corrected by the labeler on 2026-10-10 (test50: 76 of 400 points moved; val20: 50 of 160; batch01: 54 of
160 moved and the 4 pupil points of one frame left empty), then the test set was frozen. Training set of every step = the
carried labels of the earlier videos (700 as in video 16 + video 16 batches 1-2: 40; 740 in total) + this video's labels;
shuffles 1811, 1812, ...

Label QC notes: in test50, 45 points with pre-label confidence below 0.6 were left at the pre-label (36 of them
`eyelid_top`, confidence 0.40-0.59); in batch01 one frame (`img000970.png`) has a pupil width of 330 px, above the QC
limit of 320 px (eye width 495 px in that frame; kept as labeled).

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-16 40-label model 1712, applied unchanged) | 6.38 px | 8.7 | 0% | 82% | – |
| 20 (shuffle 1811) | 4.75 px | 6.7 | 0% | 96% | 4916 of 57158 (8.6%) |
| 40 (shuffle 1812) | 4.47 px | 6.2 | 0% | 95% | 4165 of 57158 (7.3%) |
| 60 (shuffle 1813) | 4.89 px | 7.0 | 0% | 96% | pending |

The other Terra models give 6.9-91 px on this test set (the video-15 60-label model 1513 is the next best at 6.88 px,
the video-15 plateau model 1512 7.13, the video-16 60-label model 1713 7.28; `../results/cross_video_matrix.csv`).
Step 1 (20 labels, 2026-10-10 14:59): 4.75 px, 26% better than the 0-label 6.38 px; p90 from 8.7 to 6.7 px, 96% of
keypoints confident. Step 2 (40 labels, 2026-10-10 18:52): 4.47 px, 5.9% better than 20 labels, above the 3% bar, so it counts as an
improving step (running best 4.47 px; the next step improves if it is below 4.34 px). Batch 3 was selected from
the 40-label model's whole-video prediction.
Step 3 (60 labels, 2026-10-10 23:26): 4.89 px, 9% worse than the running best of 4.47 px: the first step without a gain
of more than 3%. The rule needs two such steps in a row, so one more step is run. In batch03 one frame
(`img003154.png`, eye closed) had its top, bottom and right pupil points removed by the labeler while the pre-label
`pupil_left` (confidence 0.20, about 400 px from the eye) was still there; with the labeler's confirmation that point
was removed before training, so the frame has no pupil points.
