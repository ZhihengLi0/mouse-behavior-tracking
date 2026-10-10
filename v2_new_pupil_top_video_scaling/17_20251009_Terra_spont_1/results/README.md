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

The other Terra models give 6.9-91 px on this test set (the video-15 60-label model 1513 is the next best at 6.88 px,
the video-15 plateau model 1512 7.13, the video-16 60-label model 1713 7.28; `../results/cross_video_matrix.csv`).
Step 1 (20 labels, shuffle 1811) started on 2026-10-10 12:40.
