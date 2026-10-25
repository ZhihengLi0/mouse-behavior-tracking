# Stage 2 of P1: an error detector fitted to our own truth, 2026-10-25

Scripts: `scripts/error_detector.py` (truth pairs, features, tabular detector D1), `scripts/error_detector_image.py`
(image detector D2). Read-only with respect to every earlier result.

## Basic information
- Question: given only what the model produced on a frame, can a classifier trained on frames with known error find
  the frames whose prediction is > 20 px wrong, better than the best hand-made signal of stage 1 (`fit_pupil`)?
- Frames scored (the test): the 1330 frames of step 8 — 50 test + 20 validation frames of each of the 19 videos,
  evenly spaced, never trained on — predicted by each video's plateau-point model. Truth = frame RMSE against the human
  labels, > 20 px (155 frames) and > 50 px (32 frames). Identical to step 8 and to the stage-1 sweep.
- Training examples (no new labels): every (whole-video prediction file, labeled frame the model never trained on)
  pair. Files = all `predictions_stepNN` of a video (its 20/40/60-label models, 1–7 per video, 65 in all); frames =
  the batches labeled after that step plus test50 + val20. Per pair: per-keypoint error, frame RMSE, wrong20 / wrong50,
  predicted and human coordinates (`truth_pairs.csv`).
- Caveat on the truth: test50 / val20 were pre-labeled by an earlier video's model and the batches by the previous
  step's model, so wherever the labeler accepted a pre-label unchanged the measured error is biased low.
- Protocols: `lovo` = the held-out video's plateau frames are scored by a detector trained on the other 18 videos'
  pairs only; `deploy` = the held-out video's own batch frames may be trained on as well (its test/val never).
  D2 uses 5 video-grouped folds (strict, like `lovo`), and D1 is re-run on the same folds for the comparison.
- Features (D1, label-free, `features_pairs.csv`): 8 confidences; the 8 points relative to the corner midpoint / eye
  width; per-point jump forward and backward; per-point distance to the centred 31-frame median (a feature only,
  nothing is smoothed); per-point SARIMAX(3,0,1) deviation (DLC's fitting), its mean / pupil mean / eye mean / max and
  the count outside the 99% interval; pupil geometry; the same summaries at t−1, t+1 and over t−5..t+5.
- D2 input: the eye crop around the predicted corners (±0.72 W × ±0.5 W, 192 × 128) as three channels: grey image,
  predicted pupil points + axes, predicted eye points. 9 outputs (each keypoint > 20 px, frame > 20 px),
  ImageNet-pretrained ResNet-18 fine-tuned; variant with synthetic displaced predictions as extra positives.
- Evaluation: AUC at 20 / 50 px (pooled, per video), recall / precision at 2 / 5 / 10 % flagged, flagged share for
  recall 50 / 70 / 90 %. Decision rule (plan): recall ≥ 50 % at ≤ 5 % flagged, precision ≥ 50 %, at 20 px.

## Result
Neither detector beats `fit_pupil`; combining them adds at most 0.02 AUC and does not move the operating points.
Every method lands at ≈ 35 % recall of the > 20 px frames at 5 % flagged and ≈ 50 % at 9–10 % flagged. The stage-1
decision rule (recall ≥ 50 % at ≤ 5 % flagged) is not met by a fitted detector either.

| detector (held-out videos only) | AUC > 20 px | AUC > 50 px | 5 % flagged: recall / precision | 10 % flagged | flagged for recall 50 % |
|---|---|---|---|---|---|
| fit_pupil (stage 1, reference) | 0.850 | 0.973 | 35 % / 82 % | 52 % / 61 % | 9.2 % |
| D1, 1330 plateau frames, 93 features, lovo | 0.827 | 0.965 | 34 % / 80 % | 48 % / 56 % | 10.8 % |
| D1, 7010 pairs, 93 features, lovo | 0.838 | 0.927 | 30 % / 70 % | 43 % / 50 % | 13.4 % |
| D1, 7010 pairs, deploy protocol | 0.847 | 0.917 | 32 % / 76 % | 44 % / 51 % | 13.8 % |
| D1, 7010 pairs, 5 video-grouped folds | 0.840 | 0.931 | 30 % / 70 % | 44 % / 51 % | 12.5 % |
| D2 image, frame output, 5 folds | 0.769 | 0.900 | 26 % / 61 % | 36 % / 42 % | 20.3 % |
| D2 image, max over keypoints | 0.773 | 0.903 | 23 % / 53 % | 32 % / 37 % | 17.8 % |
| D2 image + synthetic errors | 0.761 | 0.855 | 18 % / 42 % | 32 % / 37 % | 24.0 % |
| lovo logistic regression fit_pupil + D2 + min_conf | 0.853 | 0.979 | 35 % / 82 % | 54 % / 62 % | 8.4 % |
| rank average fit_pupil + D2 + D1 | 0.869 | 0.963 | 36 % / 83 % | 48 % / 56 % | 10.2 % |

(`auc_table.csv`, `operating_points.csv`, `auc_table_image.csv`, `operating_points_image.csv`, `combination_table.csv`;
per-frame scores in `scores_lovo.csv` and `scores_image.csv`.)

Observations
- Truth pairs: 7010 (1235 > 20 px, 416 > 50 px); the 1330 plateau pairs reproduce step 8 exactly. The D1 sanity variant
  (sweep-like scalars, plateau frames only) gives AUC 0.799, near the sweep's logistic regression (0.815).
- More training pairs from the earlier, weaker models do not help and lower the > 50 px AUC (0.965 → 0.927): the
  weaker models fail differently from the plateau model, so their errors teach the wrong pattern.
- The image detector, which sees the crop and the drawn prediction, is the weakest (0.77) and synthetic displaced
  predictions make it worse (0.76): the real misses are not "the ellipse sits off the pupil" cases. D2 is only weakly
  correlated with fit_pupil (Spearman 0.35), yet the combination gains nothing at the operating points.
- The 101 frames > 20 px that fit_pupil misses at 5 % flagged have median RMSE 26 px (75 % below 34 px), median
  lowest confidence 0.52, no jump, and a median D2 score of 0.12 (non-error frames 0.02): to every detector they
  look like ordinary frames. 92 % of the 155 error frames have their largest error on a pupil point (pupil_left 79,
  pupil_right 27, pupil_top 22, pupil_bottom 15); the signed offsets have no common direction, so it is not a
  labeling-convention bias.

What this means: every signal that can be computed from the prediction, its time course and the image has been tried
(stage 1 + this stage) and all of them saturate at the same place. The remaining question is whether those 101 frames
are model errors at all or disagreements of 20–35 px between an ambiguous label and an acceptable prediction; that
can only be answered by a human looking at them (next step, to decide with Kaiwen).
