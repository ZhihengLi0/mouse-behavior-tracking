# Video 1: 1_20251031_Pluto_spont_1 (19.9 min, a different mouse)

Scale curve within this video under the new pupil standard. Training = ALL 100 labels of video 0
(`0_first5minvedio`, batches 01-05) + N labels of this video, from scratch; validation = this video's val20;
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

Files: `scale_curve.csv/png`, `val_mAP_by_snapshot.csv`, `jump_flagged.csv`, `selection_sheets/selection_*_frames.png`
(how each batch was chosen: jump rule on the previous model's whole-video prediction, then k-means among the
flagged frames).

## Pupil-area and blink machinery under the new standard (2026-09-22, `scripts/compare_pupil_methods.py`)

Question from the user: does the earlier machinery (3-point endpoint ellipse, blink handling of 2026-09-19) still
fit the new label standard, or is there a more accurate variant now that `pupil_top` is a real ellipse endpoint?
Model = step 5 (100 Pluto labels + 100 video-1 labels). Full table in `blink_area_analysis/pupil_methods_comparison.csv`, figure
`blink_area_analysis/pupil_methods_comparison.png`.

| quantity, model vs human on the 50 frozen test frames | old 3-point rule (top unused) | new 4-point rule (height = B.y - T.y) |
|---|---|---|
| pupil area, median / p90 abs. error | 6.9% / 29.5% | **6.0% / 21.1%** |
| pupil height | 6.2% / 20.4% | **3.3% / 7.7%** |
| pupil width (same in both) | 6.0% / 21.5% | |
| eye opening (lid-to-lid distance) | 2.1% / 4.9% | |

- On the same human labels the two rules differ by a median -1.2% (|diff| 3.2%, p90 7.6%): the labeled pupils
  are vertically symmetric about the left-right line (top-to-centre / centre-to-bottom = 1.025), so the old rule
  has no systematic bias under the new standard; it just ignores the information in `pupil_top` and doubles the
  `pupil_bottom` error. On the whole video (36,152 trusted frames) the two areas correlate at 0.992, the old rule
  reading a median 4.4% lower.
- Recommendation: switch the area to the 4-point rule (width = |R.x - L.x|, height = |B.y - T.y|, area = pi/4 w h).
  The remaining area error is dominated by the width, i.e. by `pupil_left` (median 12 px), the weakest keypoint.
- Blink handling: the trigger of 2026-09-19 (eye opening < 95% of its 5-s median, or left/right/bottom confidence
  < 0.6) marks 24,032 of 71,748 frames (33.5%, 899 runs, longest 6.9 s) as untrusted on this video with this
  model; 9,528 frames are flagged by confidence alone, 2,386 by eye opening alone, 1,818 by both (the rest is the
  3-frame widening). Adding a `pupil_top` confidence condition raises this to 49.6%. The confidence part of the
  rule is therefore the limiting factor on this mouse: the model's confidence is lower than on the 5-minute video
  (89% of test keypoints >= 0.6) although its keypoint error is not worse, so the 0.6 threshold discards many
  usable frames. Re-tuning that threshold (or relying on eye opening + a lower cutoff) is a decision for the user;
  nothing was changed in the pipeline.

## Do the predicted pupil points betray a closed eye? (2026-09-22, `scripts/blink_pupil_dispersion.py`)

User observation while labeling: on closed-eye frames the model's pupil points scatter or jump a lot, so the pupil
geometry itself might detect blinks. Measured only; the pipeline is unchanged.

Signals per frame from the predicted pupil points: `aspect` (height / width), `centre_off` (distance between the
centre of top-bottom and the centre of left-right, in pupil widths), `jump` (largest frame-to-frame move of a pupil
point, in pupil widths), `min_conf` (lowest confidence of the 4 pupil points), `area_rel` (4-pt area / 5-s median).

1. Against the eyelid trigger on the whole video (71,748 frames, step-5 model; reference = eye opening < 95% of
   its 5-s median, 4,285 frames): AUC 0.62-0.70 for the dispersion signals, 0.79 for `area_rel`; at a matched
   number of flagged frames only 17-32% of them coincide with the eyelid trigger. Weak agreement - but the eyelid
   trigger is itself a heuristic, not ground truth.
2. Against the labeler (530 labeled frames of all units with pre-labels; 20 frames where the labeler left >= 1
   pupil point empty = pupil not visible): `min_conf` AUC 0.84 (median 0.23 on closed vs 0.57 on open frames),
   `centre_off` AUC 0.76 (0.25 vs 0.05 pupil widths), `aspect` 0.34 (useless), and the model's own lid distance
   only 0.55. So on frames a human calls closed, the scatter of the pupil points and their low confidence are far
   better indicators than the predicted eyelid distance. Caveats: n = 20, and these frames were themselves picked
   by the jump rule, i.e. frames where the model was unstable.

Reading: the hypothesis holds for the frames a human calls closed; the two triggers (lid distance vs pupil
scatter/confidence) flag largely different frames, so combining them - or replacing the lid distance - is a
decision for the user after the video-level counts are compared (`blink_area_analysis/blink_pupil_dispersion.csv`,
`blink_area_analysis/blink_pupil_dispersion_labeled_frames.csv`, figure `blink_area_analysis/blink_pupil_dispersion.png`).
