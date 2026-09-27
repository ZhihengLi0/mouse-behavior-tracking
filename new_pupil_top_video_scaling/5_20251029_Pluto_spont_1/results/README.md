# Video 5: 2025-10-29 Pluto spont 1 (mouse B, 19.9 min; index 5) - labeling convention v2

Label scaling under convention v2 (started 2026-09-27, after video 4 run 2 reached its plateau at 40 labels).
The test set was relabeled under v2 on 2026-09-26 (12 of 400 points changed; pupil width unchanged, 241 px) and
frozen. val20 / batch01 are the same frames as in the first attempt, pre-labeled with the user's own v1 labels.
Training set of every step = videos 0-3 (as before) + video 4 run 2 batches 1-2 (40 labels, its plateau) + this
video's labels; shuffles 631, 632, ... The first attempt (convention v1, 0 labels 7.16 px / 20 labels 5.92 px) is kept
in `archive_2026-09-26_convention1/`.

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-4 run-2 plateau model 532, applied unchanged) | 11.90 px | 128.1 | 12% | 65% | – |

`blink_area_analysis/timeseries.png`: eye time series (model 515, run-1 model).
