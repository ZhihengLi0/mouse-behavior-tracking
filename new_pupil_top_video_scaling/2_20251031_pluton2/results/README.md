# Video 2: 2_pluton2 (19.9 min, second recording of the mouse of video 1)

Scale curve within this video under the new pupil standard. Training = ALL 100 labels of video 0
(`0_first5minvedio`) + ALL 100 labels of video 1 (`1_20251031_Pluto_spont_1`) + N labels of this video, from
scratch; validation = this video's val20; test = this video's 50 frozen test frames. The x = 0 point is the
video-1 step-5 model (shuffle 225) applied unchanged.

Video file: `2_pluton2/2_pluton2.mp4` (a copy of `pluton2.mp4`, local only).

Frame extraction (2026-09-22): test50 / val20 / batch01 by the same model-free protocol as videos 0 and 1
(k-means on appearance fingerprints, medoid per cluster). Pre-labels for the labeler come from the video-2
step-5 model; they only save dragging time and do not influence which frames were chosen.

## Result (median frame RMSE over the 50 frozen test frames; final snapshot = epoch 120)

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-1 step-5 model, 100 labels, applied unchanged) | 9.99 px | 46.3 | 10% | 75% | – |
| 20 | **7.73 px** | 26.4 | 8% | 85% | 5,292 (9.3%) |
| 40 | 7.57 px | 25.1 | 2% | 89% | 3,294 (5.8%) |
| 60 | 7.83 px | 22.8 | 8% | 90% | 3,252 (5.7%) |
| 80 | 7.56 px | 19.7 | 2% | 92% | not predicted (see below) |

Plateau rule (running best of the final-snapshot series, two consecutive steps with <= 3% improvement): the
40-label step improved the running best by 2% and the 60-label step not at all, so the rule fired at 60 labels and
the plateau point is **20 labels** - the same as video 1, now with the 100 labels of video 1 also in the training
set. The user stopped the jump selection after 60 labels (2026-09-23); batch04 had already been labeled, so step 4
was trained and scored (no whole-video prediction, no batch05). Test set frozen 2026-09-22 (sha256 `36bf7e79…`).

Per keypoint: `pupil_top` is the only point still improving (18.3 -> 14.0 -> 10.2 -> 14.0 px); every other point
is at 1-5 px from the first step on. The test50 pre-labels came from the video-1 step-5 model and the labeler
moved 126 of 397 points, so scores are anchored to that model where the labeler agreed with it.
