# Video 2: 20251031_Pluto_spont_1 (19.9 min, a different mouse)

Scale curve within this video under the new pupil standard. Training = ALL 100 labels of video 1
(`first5minvedio`, batches 01-05) + N labels of this video, from scratch; validation = this video's val20;
test = this video's 50 frozen test frames (`training-data/labels/test_frozen/test50_labels.h5`,
sha256 `5411047aae9258bd…`; frozen 2026-09-22 after a read-only check, never edited since).
Reference from the old label standard (not comparable in px): trained with 676 old-mouse labels, this video
reached its plateau at about 80 labels.

## Result (final snapshot = epoch 120, median frame RMSE over the 50 test frames)

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 20 | **10.37 px** | 21.6 | 4% | 59% | 12,060 (21%) |
| 40 | 12.17 px | 20.3 | 0% | 75% | 6,229 (11%) |
| 60 | 11.35 px | 19.7 | 2% | 74% | 3,564 (6%) |
| 80 | 11.59 px | 19.8 | 0% | 82% | 3,521 (6%) |

Plateau rule fired at 60 labels: the running best (10.37 px at 20 labels) was not improved by the 40- and
60-label steps -> **plateau = 20 labels** for this video when the 100 labels of video 1 are in the training set.
The 1-px differences between steps are within the sampling noise of a 50-frame median; the tail (p90, failed
frames, confidence) and the jump-flagged fraction kept improving up to 60-80 labels.

Per keypoint (median error, px): pupil_left stays the worst pupil point (13.0 -> 12.6), pupil_right the best
(4.2 -> 3.3); the two eye corners are 7-14 px, eyelid_top ~7.5, eyelid_bottom ~5.

Files: `scale_curve.csv/png`, `jump_flagged.csv`, `selection_040/060/080/100_frames.png` (how each batch was
chosen: jump rule on the previous model's whole-video prediction, then k-means among the flagged frames).
Batch 05 (for 100 labels) is selected but not trained.
