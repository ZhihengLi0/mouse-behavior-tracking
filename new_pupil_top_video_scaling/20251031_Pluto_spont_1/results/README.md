# Video 2: 20251031_Pluto_spont_1 (19.9 min, a different mouse)

Scale curve within this video under the new pupil standard. Training = ALL 100 labels of video 1
(`first5minvedio`, batches 01-05) + N labels of this video, from scratch; validation = this video's val20;
test = this video's 50 frozen test frames (`training-data/labels/test_frozen/test50_labels.h5`,
sha256 `5411047aae9258bd…`; frozen 2026-09-22 after a read-only check, never edited since).
Reference from the old label standard (not comparable in px): trained with 676 old-mouse labels, this video
reached its plateau at about 80 labels.

## Result (median frame RMSE over the 50 frozen test frames)

Headline = final snapshot (epoch 120). Secondary = the snapshot with the best validation mAP after the first
learning-rate drop (epochs >= 96, i.e. 100/110/120); same two rules as video 1.

| labels of this video | final snapshot | best-val-mAP snapshot | p90 (final) | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|---|
| 0 (video-1 model applied unchanged) | 208.2 px | 225.8 px (ep. 100) | 258 | 98% | 19% | – |
| 20 | **10.37 px** | 9.86 (ep. 110) | 21.6 | 4% | 59% | 12,060 (21%) |
| 40 | 12.17 px | 10.96 (ep. 110) | 20.3 | 0% | 75% | 6,229 (11%) |
| 60 | 11.35 px | 11.50 (ep. 100) | 19.7 | 2% | 74% | 3,564 (6%) |
| 80 | 11.59 px | 11.98 (ep. 100) | 19.8 | 0% | 82% | 3,521 (6%) |
| 100 | 10.94 px | 11.29 (ep. 110) | 19.3 | 0% | 89% | (predicting, batch 06 being selected) |

Plateau rule (running best of the final-snapshot series, two consecutive steps with <= 3% improvement) fired at
60 labels: the running best (10.37 px at 20 labels) was not improved by the 40- and 60-label steps ->
**plateau = 20 labels** for this video when the 100 labels of video 1 are in the training set. The 1-px
differences between 20 and 100 labels are within the sampling noise of a 50-frame median; the tail (p90,
failed frames, confidence) and the jump-flagged fraction kept improving up to 100 labels, which is why the
user chose to keep selecting correction batches beyond the plateau.

The x = 0 point: the video-1 model (100 labels, shuffle 115) applied to this mouse without any label of this
video is unusable (98% of the test frames fail). Caveat: the pre-labels of test50 came from that same model,
and where the labeler judged a pre-label correct it was left untouched, so for pupil_bottom / pupil_right /
eyelid_bottom the x = 0 error is exactly 0 on many frames. Pre-label-assisted labeling anchors the test labels
to the pre-labeling model wherever the labeler agreed with it; this holds for every step's score.

Per keypoint (median error, px, final snapshot): pupil_left stays the worst pupil point (13.0 -> 12.4),
pupil_right the best (4.2 -> 3.3); the two eye corners 7-14 px, eyelid_top ~7.5, eyelid_bottom ~5.

Validation-set note: during training of steps 1-5 the val20 file still contained one or two unmoved pre-label
points (fixed 2026-09-22, only labels of val20 changed). The validation split never enters training; it only
drives the secondary column, which was therefore recomputed with the corrected val20
(`scripts/recompute_mAPlate.py`, per-snapshot values in `val_mAP_by_snapshot.csv`). The headline rows are
unaffected. Step 5 was trained twice: the first run (shuffle 215) was stopped at epoch ~45 when one unmoved
pre-label point was found in batch05; the corrected batch05 was trained as shuffle 225.

Files: `scale_curve.csv/png`, `val_mAP_by_snapshot.csv`, `jump_flagged.csv`, `selection_040/060/080/100_frames.png`
(how each batch was chosen: jump rule on the previous model's whole-video prediction, then k-means among the
flagged frames).
