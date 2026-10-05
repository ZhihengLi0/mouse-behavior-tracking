# Video 12: 2025-10-16 Terra spont 2 (mouse C, 19.9 min; index 12) - labeling convention v2

Label scaling under convention v2, continuing the v2 line after video 11 (finished at 40 labels); the second video of
mouse C (Terra). This video had no earlier human labels: test50, val20 and batch01 were pre-labeled with the video-11
40-label model (1212) and corrected by the labeler on 2026-10-04 (test50: 110 of 400 points moved; val20: 72 of 160
moved, 1 emptied; batch01: 74 of 160 moved, 4 emptied), then the test set was frozen. Training set of every step = the
carried labels of the earlier videos (videos 0-3: 100 + 100 + 60 + 60; video 4 run 2 batches 1-2: 40; videos 5, 6, 8
batch 1: 20 each; video 9 batches 1-4: 80; videos 10 and 11 batches 1-2: 40 each; video 7 contributed none; 580 in
total) + this video's labels; shuffles 1311, 1312, ...

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-11 40-label model 1212, applied unchanged) | 5.13 px | 9.4 | 0% | 77% | – |
| 20 (shuffle 1311) | 5.77 px | 7.7 | 0% | 99% | 7700 of 57158 (13.5%) |
| 40 (shuffle 1312) | 5.84 px | 7.7 | 0% | 96% | not computed (video finished) |

The 0-label number is biased low: model 1212 produced the pre-labels of this test set and 290 of the 400 test points
were left as pre-labeled, so its error on those points is 0. The other models of video 11 (20 / 60 / 80 labels), which
did not produce the pre-labels, give 10.66 / 8.29 / 8.99 px on this test set (`../results/cross_video_matrix.csv`).

Step 1 (20 labels): 5.77 px. Against the 0-label number (5.13 px, biased low by the pre-labels) this is no gain, so by the
rule it counts as one flat step; against the other video-11 models on this test set (8.29-10.66 px) it is clearly better.

Step 2 (40 labels): 5.84 px. The plateau rule fired at 40 labels: steps 1 and 2 (5.77, 5.84 px) did not improve the
running best, the 0-label number 5.13 px.

**Finished with the plateau at 0 labels, 5.13 px (user decision 2026-10-04 21:29).** The model of video 11 already fits this
recording day (as with video 7 after video 6); no labels of this video are carried into later videos. Batches 01-02 stay
on disk. The three numbers 5.13 / 5.77 / 5.84 px differ by less than 1 px.

**Fewer-labels study (2026-10-05)**: step 1 retrained with 5 or 10 of batch01's 20 labels (two subsets each, training
set otherwise as at the time: the 580 carried labels of videos 0-11; shuffles 2121-2124) -> `fewer_labels.png` /
`fewer_labels.csv`. This compares the size of the first batch (the 5 / 10 labels are taken out of the 20 already labeled,
picked by image appearance without a model); it is not a procedure of 5 labels per step and no jump selection followed.

| labels of this video | median | p90 | frames > 50 px |
|---|---|---|---|
| 0 | 5.13 px | 9.4 | 0% |
| 5 (subset a / b) | 5.79 / 6.63 px | 8.0 / 8.7 | 0% / 0% |
| 10 (subset a / b) | 5.99 / 5.96 px | 7.7 / 8.1 | 0% / 0% |
| 20 | 5.77 px | 7.7 | 0% |

On this video 0, 5, 10 and 20 labels all give 5.1-6.6 px: the video-11 model already fits this day, so the size of the
first batch makes no difference here.
