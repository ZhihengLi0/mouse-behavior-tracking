# Video 4: 2025-10-30 Pluto spont 1 (mouse B, 19.9 min) - run 2, labeling convention v2

> **Poor-quality video**: the pupil boundary is hard to see even by eye.

**Restarted 2026-09-26 under labeling convention v2.** Where the pupil edge appears doubled (two contours, a
"ghost" next to the real edge), the point is placed on the LEFT contour; pupil_top / pupil_bottom are placed further
left accordingly (agreed with Kaiwen Sheng: the right contour is most likely the iris edge / a reflection). Run 1
(convention v1: plateau 140 labels, 14.74 px) is kept in `archive_2026-09-26_convention1/` and at tag
`v0.8.0-video4-convention1`; its models (shuffles 511-519) are archived outside the DLC project and no longer scored.

Same frames as run 1 (test50 / val20 / batch01 are model-free), relabeled from scratch with pre-labels from the
video-3 step-3 model (423) exactly as in run 1. Steps use shuffles 531, 532, ...; training set = videos 0-3 (as in
run 1) + this video's labels.

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-3 step-3 model 423, applied unchanged) | 32.11 px | – | – | – | – |
| 20 | 24.64 px | 45.2 | 6% | 72% | 17672 |
| 40 | 15.01 px | 24.1 | 2% | 90% | 7134 |
