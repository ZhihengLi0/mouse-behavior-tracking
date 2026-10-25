# Stage 1 of the next phase: label-free signals for unreliable frames (S1-1 … S1-8), 2026-10-25

Script: `scripts/unreliable_frames_sweep.py` (read-only; the step-8 results are not modified). Frames, models and
truth are those of step 8: the 50 test + 20 validation frames of every video (1330), the plateau-point model of each
video, truth = frame RMSE against the human labels > 20 px (155 frames) or > 50 px (32 frames). Signal definitions
are in the script header; `signals_all_frames.csv` has every signal at every evaluated frame.

## Result in one line
No new signal beats DLC's `fitting` family. Memoryless signals from the literature (pose-PCA reconstruction error,
ensemble disagreement, geometry) are clearly worse on our data, and combining signals does not help. The stage-1
target (recall ≥ 50% at > 20 px with ≤ 5% of frames flagged) is not reached by any signal; reaching it costs ≈ 9–11%
flagged frames.

## AUC (pooled over the 1330 frames; `auc_per_signal.csv` has every signal and the per-video median / minimum)

| signal | AUC > 20 px | AUC > 50 px | note |
|---|---|---|---|
| fit_pupil (SARIMAX deviation, mean over the 4 pupil points) | 0.850 | 0.973 | best at 20 px |
| fitting (step 8: mean over 8 points, order (3,0,1)) | 0.838 | 0.974 | |
| fit_o501_max (order (5,0,1), largest of 8) | 0.840 | 0.982 | best at 50 px |
| fit_max (largest of 8) | 0.828 | 0.976 | |
| fit_fb_max (forward + backward) | 0.825 | 0.977 | backward adds nothing |
| combo_rank / combo_lr (S1-5) | 0.842 / 0.815 | 0.965 / 0.941 | no gain over fitting |
| pca_video (S1-7) | 0.781 | 0.920 | 17–59 labeled poses per video |
| min_conf / mean_conf | 0.781 / 0.777 | 0.941 / 0.912 | |
| ens_sdw_max (S1-8, EKS weighting) | 0.747 | 0.888 | 1–7 models per video, see limits |
| geo_axes_off | 0.733 | 0.849 | |
| jump | 0.715 | 0.849 | |
| pca_mouse | 0.699 | 0.861 | |
| fit_ci (points outside the 99% interval) | 0.598 | 0.840 | |
| geo_outside | 0.524 | 0.511 | no information |

Eye-width normalisation (`*_rel`) changes the AUC by ≤ 0.005. Orders (1,0,0) and (2,0,0) are slightly worse than
(3,0,1); (5,0,1) is equal at 20 px and best at 50 px.

## Operating points (`operating_points.csv`)

Flagged share needed to reach a recall on the 155 frames > 20 px, and the precision there:

| signal | recall 50% | recall 70% | recall 90% |
|---|---|---|---|
| fitting | 10.7% flagged, precision 55% | 21.4%, 38% | 56.8%, 19% |
| fit_pupil | 9.2%, 64% | 21.7%, 38% | 53.8%, 20% |
| combo_lr | 9.1%, 65% | 28.6%, 29% | 61.8%, 17% |
| pca_video | 15.6%, 38% | 38.0%, 22% | 60.2%, 18% |
| ens_sdw_max | 16.2%, 36% | 37.0%, 22% | 74.8%, 14% |

On the 32 frames > 50 px, recall 90% costs: fit_o501_max 7.3% flagged (precision 30%), fit_pupil 9.7%, fitting 13.2%,
min_conf 16.0%, pca_video 34%, ens_sdw_max 45%. At a fixed 5% flagged share every signal gives recall ≈ 30% at 20 px
and 60–80% at 50 px (table in `sweep_overview.png`). Sanity check: `fitting > 20 px` reproduces step 8 (2.0% flagged,
recall 72% / 15%, precision 85%).

## Why the literature's memoryless signals did not help here
- Pose-PCA was fitted on 17–59 labeled poses per video (Lightning Pose's r = 0.91 used 631); our poses vary little
  within a video (head-fixed eye), so the subspace is thin and a wrong pupil point moves the reconstruction error less
  than the measurement noise of 2–5 px.
- The ensembles are the training steps of the same video (1–7 models; videos 5, 7 and 12 have a single model, so
  their ensemble signals are 0), trained on nested label sets with the same backbone and the same frames; they share
  failures more than independently seeded members would.
- Geometry: the pupil endpoints of a stuck frame are still a plausible ellipse most of the time (the whole pupil is
  shifted or scaled, not torn).

## What follows (plan, stage 2)
The stage-1 decision rule is not met. Next: an inference pass that exports heatmap peaks (multiple-peak entropy,
Anipose-style Viterbi disagreement), the MEYE segmentation cross-check, and — if the 10% flagged share is acceptable
for human review — `fit_pupil` at ≈ 8 px as the working detector. Not run: rolling-window SARIMAX fits (S1-3), the
per-run evaluation (S1-6, needs clip labels).

Files: `signals_all_frames.csv`, `auc_per_signal.csv`, `operating_points.csv`, `roc_curves.csv`, `sweep_overview.png`;
`cache/` (per-video SARIMAX deviations, local only) and `run*.log` are not committed.
