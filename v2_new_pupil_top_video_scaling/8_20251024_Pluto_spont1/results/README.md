# Video 8: 2025-10-24 Pluto spont 1 (mouse B, 19.9 min; index 8) - labeling convention v2

Label scaling under convention v2, continuing the v2 line after video 7 (which needed no labels of its own). The test
set (50 frames, last 10% of the video) was relabeled under v2 on 2026-09-29 (pre-labels = the labeler's own v1
labels; 113 of 400 points moved, 3 emptied) and frozen. val20 and batch01 were pre-labeled with the video-6 20-label
model (711) and corrected. Training set of every step = videos 0-3 + video 4 run 2 batches 1-2 (40) + video 5 batch 1
(20) + video 6 batch 1 (20) + this video's labels (video 7 contributed none); shuffles 911, 912, ...

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-6 20-label model 711, applied unchanged) | 27.46 px | 51.3 | 14% | 59% | – |
| 20 (shuffle 911) | 25.34 px | 50.6 | 14% | 68% | 9741 of 57158 (17.0%) |
| 40 (shuffle 912) | 30.03 px | 50.4 | 12% | 89% | 8734 of 57158 (15.3%) |
| 60 (shuffle 913) | 33.77 px | 52.0 | 16% | 84% | pending |

**Plateau rule fired at step 3 (2026-09-30):** neither 40 labels (30.03 px) nor 60 labels (33.77 px) improved the running best of 25.34 px (20 labels); by the rule the plateau point is **20 labels, 25.34 px**. The batch04 popup is held pending the user's decision. This day stays hard: 12-16% of test frames are > 50 px off at every step.

**Fewer-labels study (2026-10-02)**: step 1 retrained with 5 or 10 of batch01's 20 labels (two subsets each, training
set otherwise as at the time: videos 0-3 + 40 of video 4 + 20 each of videos 5 and 6; shuffles 2081-2084) ->
`fewer_labels.png` / `fewer_labels.csv`.

| labels of this video | median | p90 | frames > 50 px |
|---|---|---|---|
| 0 | 27.46 px | 51.3 | 14% |
| 5 (subset a / b) | 31.05 / 31.48 px | 192.4 / 196.4 | 30% / 28% |
| 10 (subset a / b) | 23.48 / 22.42 px | 50.8 / 45.8 | 12% / 6% |
| 20 | 25.34 px | 50.6 | 14% |

On this hard day 5 labels are worse than none (twice as many frames > 50 px); 10 labels (22.4-23.5 px) are at least as
good as 20 (25.34 px), and the two subsets of each size agree with each other.
