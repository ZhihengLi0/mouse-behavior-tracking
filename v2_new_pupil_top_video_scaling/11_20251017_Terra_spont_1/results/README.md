# Video 11: 2025-10-17 Terra spont 1 (mouse C, 19.9 min; index 11) - labeling convention v2

Label scaling under convention v2, continuing the v2 line after video 10 (finished at 40 labels); the first video of a
new mouse (Terra). This video had no earlier human labels: test50, val20 and batch01 were pre-labeled with the video-10
40-label model (1112) and corrected by the labeler on 2026-10-03 (test50: 249 of 400 points moved; val20: 92 of 160;
batch01: 99 of 160), then the test set was frozen. Training set of every step = the carried labels of the earlier videos
(videos 0-3: 100 + 100 + 60 + 60; video 4 run 2 batches 1-2: 40; videos 5, 6, 8 batch 1: 20 each; video 9 batches 1-4:
80; video 10 batches 1-2: 40; video 7 contributed none; 540 in total) + this video's labels; shuffles 1211, 1212, ...

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-10 40-label model 1112, applied unchanged) | 25.96 px | 80.1 | 16% | 42% | – |
| 20 (shuffle 1211) | 9.24 px | 166.4 | 20% | 86% | 9762 of 57158 (17.1%) |
| 40 (shuffle 1212) | 8.86 px | 17.5 | 4% | 94% | 4456 of 57158 (7.8%) |
| 60 (shuffle 1213) | 8.77 px | 12.9 | 2% | 97% | 3350 of 57158 (5.9%) |
| 80 (shuffle 1214) | 8.65 px | 13.1 | 2% | 98% | pending |

The plateau rule fired at 80 labels: steps 3 and 4 (8.77, 8.65 px) each improved the running best by less than 3% (1.0% and 1.4%). Plateau point: 40 labels, 8.86 px; awaiting user decision (2026-10-04).

Back-test of the earlier models on this test set (`../results/cross_video_matrix.csv`): the models of the sequence
score between 16.5 and 313 px; the other models of video 10 (20 / 60 / 80 labels) give 16.54 / 29.18 / 33.41 px, so
the 0-label number of a single model is not stable on this new mouse.

**Fewer-labels study (2026-10-04)**: step 1 retrained with 5 or 10 of batch01's 20 labels (two subsets each, training
set otherwise as at the time: the 540 carried labels of videos 0-10; shuffles 2111-2114) ->
`fewer_labels.png` / `fewer_labels.csv`. This compares the size of the first batch (the 5 / 10 labels are taken out of
the 20 already labeled, picked by image appearance without a model); it is not a procedure of 5 labels per step.

| labels of this video | median | p90 | frames > 50 px |
|---|---|---|---|
| 0 | 25.96 px | 80.1 | 16% |
| 5 (subset a / b) | 9.69 / 9.97 px | 22.9 / 24.0 | 2% / 2% |
| 10 (subset a / b) | 9.54 / 8.87 px | 17.1 / 14.8 | 0% / 2% |
| 20 | 9.24 px | 166.4 | 20% |

On this first video of a new mouse 5 labels already bring the median from 26 px to 9.7-10.0 px, within 1 px of the
20-label model (9.24 px); 10 labels give 8.9-9.5 px. The 20-label model has more test frames above 50 px (20%) than
the 5- and 10-label models (0-2%); with one run per point this difference is not explained.
