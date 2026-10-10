# Video 16: 2025-10-10 Terra spont 1 (mouse C, 19.9 min; index 16) - labeling convention v2

Label scaling under convention v2, continuing the v2 line after video 15 (plateau at 40 labels; its batches 1-2 are
carried); the sixth video of mouse C (Terra). This video had no earlier human labels: test50, val20 and batch01 were
extracted on 2026-10-08 with the same model-free rule as every video, pre-labeled with the video-15 plateau model (1612,
40 labels) and corrected by the labeler on 2026-10-08 (test50: 136 of 400 points moved; val20: 55 of 160; batch01: 76 of
160; no point left empty), then the test set was frozen. Training set of every step = the carried labels of the earlier
videos (videos 0-3: 100 + 100 + 60 + 60; video 4 run 2 batches 1-2: 40; videos 5, 6, 8 batch 1: 20 each; video 9 batches
1-4: 80; videos 10 and 11 batches 1-2: 40 each; video 13 batch 1: 20; video 14 batches 1-3: 60; video 15 batches 1-2: 40;
videos 7 and 12 contributed none; 700 in total) + this video's labels; shuffles 1711, 1712, ...

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-15 40-label model 1612, applied unchanged) | 11.35 px | 23.1 | 0% | 76% | – |
| 20 (shuffle 1711) | 7.72 px | 11.5 | 2% | 96% | 464 of 57158 (0.8%) |
| 40 (shuffle 1712) | 7.15 px | 11.3 | 0% | 98% | 347 of 57158 (0.6%) |
| 60 (shuffle 1713) | 7.26 px | 10.1 | 0% | 98% | 290 of 57158 (0.5%) |
| 80 (shuffle 1714) | 7.78 px | 12.2 | 0% | 97% | 260 of 57158 (0.5%) |

Step 3 (60 labels, 2026-10-09 18:14): 7.26 px, 1.5% worse than the running best 7.15 px (40 labels): the first step without a gain above 3%. p90 improved from 11.3 to 10.1 px. By the rule one more non-improving step is needed, so batch 4 was selected and is being labeled; step 4 (80 labels) counts as improving below 6.94 px.

Step 4 (80 labels, 2026-10-10 02:23): 7.78 px, worse than the running best 7.15 px (step 2): second flat step in a row, so the plateau rule
fired at 80 labels. Plateau point = the last improving step, 40 labels (7.15 px). Batch 5 was selected automatically (260 pool frames flagged) but its window is held;
awaiting the user's decision.

The other Terra models give 9.7-14.1 px on this test set (the video-12 20-label model 1311 is the best at 9.70 px, the
video-15 60-label model 11.52, the video-14 plateau model 12.08; `../results/cross_video_matrix.csv`).

Step 1 (20 labels): 7.72 px, 32% better than the 0-label 11.35 px (2026-10-08 23:43); p90 from 23.1 to 11.5 px, 96% of
keypoints confident; one test frame above 50 px. Batch 2 is selected from the 20-label model's whole-video prediction.

**Fewer-labels study (2026-10-09)**: step 1 retrained with 5 or 10 of batch01's 20 labels (two subsets each, training set
otherwise as at the time: the 700 carried labels; shuffles 2161-2164). Size of the first batch only; no jump selection followed.
5 labels = 9.18 / 10.42 px (subsets a / b; p90 15.6 / 14.9), 10 labels = 9.35 / 9.18 px (p90 15.3 / 13.8), against 11.35 px at
0 labels and 7.72 px at 20: every subset improves on the 0-label model, none reaches the 20-label number (19-35% above it);
figure `fewer_labels.png`.

Step 2 (40 labels): 7.15 px, 7.4% better than the running best 7.72 px (step 1): an improving step (2026-10-09 13:25); no test
frame above 50 px, 98% of keypoints confident. Batch 3 is selected from the 40-label model.
