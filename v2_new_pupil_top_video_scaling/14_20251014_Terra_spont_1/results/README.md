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
| 40 (shuffle 1512) | 12.41 px | 25.8 | 2% | 90% | 10512 of 57158 (18.4%) |
| 60 (shuffle 1513) | 10.30 px | 16.1 | 2% | 94% | 10420 of 57158 (18.2%) |
| 80 (shuffle 1514) | 10.45 px | 18.1 | 2% | 92% | 10523 of 57158 (18.4%) |
| 100 (shuffle 1515) | 10.24 px | 26.5 | 2% | 96% | pending |

The other Terra models give 11.3-35.4 px on this test set (video-13 40- and 60-label models 11.28 / 11.59 px, video-12
models 24.5-26.9 px, video-11 models 26.6-35.4 px; `../results/cross_video_matrix.csv`).

Step 1 (20 labels): 10.83 px, 19% better than the 0-label 13.33 px (2026-10-06 15:57); frames > 50 px from 16% to 2%.

Step 2 (40 labels): 12.41 px, no gain over the running best 10.83 px (step 1): one flat step by the rule (2026-10-06 19:31). The p90
improved from 28.7 to 25.8 px; the median did not.

Step 3 (60 labels): 10.30 px, 4.9% better than the running best 10.83 px (step 1), so an improving step by the rule and the
flat-step count restarts at zero (2026-10-07 01:12); p90 from 25.8 to 16.1 px. Batch 4 is selected from the 60-label model's
whole-video prediction and opened for labeling.

**Fewer-labels study (2026-10-06 to 2026-10-07)**: step 1 retrained with 5 or 10 of batch01's 20 labels (two subsets each,
training set otherwise as at the time: the 600 carried labels; shuffles 2141-2144). Size of the first batch only; no
jump selection followed. 5 labels = 10.93 / 10.22 px (subsets a / b; p90 16.7 / 169.1; 4% / 14% of frames > 50 px) against
13.33 px at 0 labels and 10.83 px at 20; 10 labels = 9.82 / 10.06 px (subsets a / b; p90 22.9 / 25.9; 8% / 2% of frames > 50 px).
On this video the medians of 5 and 10 labels are within 10% of the 20-label 10.83 px (three of the four below it), but with
more gross errors (subset 5b: 14% of frames > 50 px against 2% at 20 labels); figure `fewer_labels.png`.

Step 4 (80 labels): 10.45 px, no gain over the running best 10.30 px (step 3): one flat step by the rule (2026-10-07 11:03). Batch 5
is selected from the 80-label model and opened; the rule fires only if 100 labels also fail to improve by 3%.

Step 5 (100 labels): 10.24 px, 0.6% better than the running best 10.30 px (step 3), below the 3% threshold: second flat step in a
row, so the plateau rule fired at 100 labels (2026-10-07 15:38). Plateau point = the last improving step, 60 labels (10.30 px).
The batch-6 window is held; awaiting the user's decision. (Batch 5 was saved with one leftover pre-label point, pupil_bottom of
img022177 at (930, 33) on a closed eye; it was emptied with the user's approval and step 5 restarted from scratch before any
result was recorded.)
