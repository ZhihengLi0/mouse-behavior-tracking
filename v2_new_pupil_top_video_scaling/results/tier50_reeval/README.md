# Every signal and detector re-evaluated at the 50-px tier, 2026-10-26

Script: `scripts/tier50_reeval.py`. Inputs: the per-frame scores already computed by the sweep (`results/unreliable_frames_sweep/
signals_all_frames.csv`), the tabular detector D1 (`results/error_detector/scores_lovo.csv`, leave-one-video-out scores) and
the image detector D2 (`results/error_detector/scores_image.csv`). Nothing recomputed, nothing modified. Truth: frame RMSE
> 50 px, 32 of the 1330 frames (videos 1: 8, 8: 9, 2: 4, 11: 4, 10: 3, 4: 2, 14: 2). Thresholds leave-one-video-out
(quantile of the score on the other 18 videos). Files: `signals_50px.csv` (45 scores), `caught_per_video.csv`, `combos.csv`.

Why: the sweep and 8.2 ranked signals by the 20-px tier (155 frames), where fit_pupil (mean of the 4 pupil points, SARIMAX
(3,0,1)) was best. With the 50-px tier as the target (decision of 2026-10-26) the ranking changes.

| score (what it is) | AUC 50 px (95 % interval) | AUC 20 px | caught of 32 at 1 % / 2 % / 5 % flagged | precision at 5 % | flagged share for 90 % recall |
|---|---|---|---|---|---|
| fit_o200_max: SARIMAX order (2,0,0), largest deviation of the 8 points | 0.986 (0.972–0.996) | 0.810 | 16 / 23 / **31** | 39 % | 6.0 % |
| fit_o311_max: order (3,1,1), largest of 8 | 0.976 (0.935–0.998) | 0.821 | 15 / 23 / **31** | 45 % | 3.8 % |
| fit_o100_max: order (1,0,0), largest of 8 | 0.975 (0.937–0.997) | 0.807 | 15 / 24 / **31** | 41 % | 4.7 % |
| fit_o501_max: order (5,0,1), largest of 8 | 0.982 (0.967–0.994) | 0.840 | 16 / 20 / 27 | 38 % | 7.3 % |
| fit_max: order (3,0,1), largest of 8 | 0.976 (0.958–0.991) | 0.828 | 16 / 22 / 26 | 35 % | 9.8 % |
| fitting (8.1, mean of 8) | 0.974 (0.955–0.990) | 0.838 | 16 / 23 / 25 | 34 % | 13.2 % |
| fit_pupil (8.2, mean of 4 pupil points) | 0.973 (0.951–0.990) | 0.850 | 16 / 20 / 25 | 34 % | 9.7 % |
| d1_plateau_all (gradient boosting, 93 features) | 0.965 (0.943–0.983) | 0.827 | 10 / 17 / 24 | 31 % | 13.7 % |
| min_conf | 0.941 (0.914–0.965) | 0.781 | 8 / 13 / 17 | 23 % | 16.0 % |
| pca_video | 0.920 (0.862–0.972) | 0.781 | 15 / 23 / 23 | 32 % | 34 % |
| d2_frame (ResNet-18 on the crop) | 0.900 (0.838–0.952) | 0.769 | 10 / 15 / 22 | 32 % | 32 % |
| ens_sdw_max (ensemble disagreement) | 0.888 (0.819–0.945) | 0.747 | 9 / 11 / 22 | 31 % | 45 % |
| jump (8.1) | 0.849 (0.755–0.928) | 0.715 | 14 / 20 / 21 | 31 % | 46 % |

- **At the 50-px tier the best rule is "largest deviation of the 8 points" with a low-order model**, not the pupil mean:
  fit_o200_max / fit_o311_max / fit_o100_max catch 31 of the 32 frames at 5 % flagged (fit_pupil 25, fitting 25), and 90 %
  of them at 3.8–6 % flagged (fit_pupil 9.7 %). Per video (`caught_per_video.csv`) the gain is in video 8: 9 of 9 instead of
  5 of 9; videos 1, 2, 4, 10, 11 are complete under both. The one frame missed by all three is video 14 frame 61431
  (51 px, low confidence, no motion).
- Why these were set aside before: at the 20-px tier they are 0.02–0.04 AUC below fit_pupil (the sweep README: orders
  (1,0,0) and (2,0,0) "slightly worse"), and 8.2 optimised the 20-px tier. The 50-px frames are mostly one point thrown
  far (pupil_left in 21 of 32), which the maximum over points sees and the mean over four dilutes.
- Combining with fit_pupil (rank average, `combos.csv`) does not beat the single score (30 of 32 at 5 % for the best pair).
- **Caveats.** 32 positives: one frame is 3 % recall; the AUC intervals of the top seven scores overlap. The best of 45
  scores was picked on the same frames, which flatters it; the difference to fit_pupil at 5 % flagged (6 frames, 4 of them
  video 8) is the thing to confirm on new truth. The learned detectors (D1, D2) and the ensemble do not reach the fitting
  family at this tier either.
- **Whole videos** (same thresholds applied to the cached whole-video deviations): fit_o200_max flags 5.4 % of the
  1 309 478 frames in 22 673 runs; fit_o311_max 6.5 % in 14 034 runs (fewer single-frame runs: 4 295 vs 12 944), fit_max
  5.7 % in 20 922 runs. Per-video shares 0.2–21 %.

Recommendation for the 10-31 meeting: at the 50-px tier the working detector should be the largest-of-8 fitting deviation
with order (2,0,0) or (3,1,1) (threshold ≈ 44 / 73 px, or the equivalent share of the eye width), flagging ≈ 5 % and catching
≈ 97 % of the > 50 px frames on the present truth; fit_pupil stays the better choice only if the 20-px tier matters.
