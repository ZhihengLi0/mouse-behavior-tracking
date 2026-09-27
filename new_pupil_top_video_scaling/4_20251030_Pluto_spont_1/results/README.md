# Video 4: 2025-10-30 Pluto spont 1 (mouse B, 19.9 min) - run 2, labeling convention v2

> **Poor-quality video**: the pupil boundary is hard to see even by eye.

**Restarted 2026-09-26 under labeling convention v2.** Where the pupil edge appears doubled (two contours, a
"ghost" next to the real edge), the point is placed on the LEFT contour; pupil_top / pupil_bottom are placed further
left accordingly (agreed with Kaiwen Sheng: the right contour is most likely the iris edge / a reflection). Run 1
(convention v1: plateau 140 labels, 14.74 px) is kept only at tag `v0.8.0-video4-convention1` (local copies and models 511-519 deleted 2026-09-27).

Same frames as run 1 (test50 / val20 / batch01 are model-free), relabeled from scratch with pre-labels from the
video-3 step-3 model (423) exactly as in run 1. Steps use shuffles 531, 532, ...; training set = videos 0-3 (as in
run 1) + this video's labels.

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-3 step-3 model 423, applied unchanged) | 32.11 px | – | – | – | – |
| 20 | 24.64 px | 45.2 | 6% | 72% | 17672 |
| 40 | 15.01 px | 24.1 | 2% | 90% | 7134 |
| 60 | 16.82 px | 25.1 | 2% | 79% | 8050 |
| 80 | 17.54 px | 44.7 | 8% | 86% | – |

**Run 2 plateau (2026-09-27):** neither 60 labels (16.82 px) nor 80 labels (17.54 px) improved the running best of
15.01 px by more than 3%, so the plateau point of run 2 (convention v2) is **40 labels, 15.01 px**. Run 1 (v1) needed
140 labels for 14.74 px on a different (larger) pupil definition, so the two numbers are not directly comparable.
