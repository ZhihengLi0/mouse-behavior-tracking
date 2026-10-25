# Next phase (from 2026-10-24): unreliable frames — find them better, repair them cheaply

Status of the project on 2026-10-24: all 19 videos are at their plateau point; the report (`report/index.html`) is
accepted as it stands. The remaining question of the eye line is step 12's first item: after the model has saturated,
some frames are still predicted wrongly, and a human has to find and fix them. This phase splits it into two
sub-problems that are not independent:

- **P1 — detect.** Without labels, flag the frames whose prediction is unreliable, more precisely than the rules of
  step 8 do.
- **P2 — repair.** Bring the flagged frames back to an acceptable error at the smallest human cost.

A better P1 (high precision, high recall) makes P2 cheap; a cheap P2 (automatic repair) lowers the recall P1 must reach.
Both are evaluated on the same frames and with the same truth as step 8.

Next meeting: 2026-10-31. Deliverables for it are marked ▶.

## 0. Baseline and evaluation (fixed, reused from step 8)

- Frames: the 50 test frames + 20 validation frames of every video, 19 × 70 = 1330, evenly spaced, never trained on.
  Model = the plateau-point model of the video. Signals per frame are in
  `v2_new_pupil_top_video_scaling/results/unreliable_frames/unreliable_frames_all_videos.csv`.
- Truth: frame RMSE against the human labels. "very unreliable" > 50 px (32 frames), "unreliable" > 20 px (155 frames).
  Report both; 20 px is the main target because 32 positives make the 50-px numbers move by 3% per frame.
- Per rule: flagged share, recall, precision; per signal: AUC. Thresholds chosen on data are always chosen
  leave-one-video-out. Per-video numbers next to pooled numbers (videos differ a lot).
- Baseline rules (step 8):

  | rule | flags | recall >50 px | precision >50 px | recall >20 px | precision >20 px |
  |---|---|---|---|---|---|
  | fitting > 20 px (DLC default) | 2% | 72% | 85% | 16% | 89% |
  | jump > 3% of eye width | 12% | 66% | 14% | 44% | 44% |
  | lowest confidence < 0.6 | 52% | 100% | 5% | 88% | 20% |

  What P1 must beat: at a flagged share of a few percent, recall >20 px well above 16%.

## P1 — detect unreliable frames

### Stage 1 ▶ zero-cost signals on existing predictions and labels (no new labels, no code of step 8 changed)

New script `unreliable_frames_sweep.py`, outputs in a new folder `results/unreliable_frames_sweep/`. Experiment ids
are the ones used in `literature_unreliable_frames.md`.

- **S1-1 threshold sweep.** fitting epsilon from 2 to 60 px (log steps); full ROC and precision–recall at both truth
  levels; the operating points that give recall > 20 px of 50% / 70% / 90% and their flagged share. Also per-video
  adaptive thresholds (percentile / IQR) for confidence and jump, as refineDLC does.
- **S1-2 how the 8 deviations are combined.** Now mean over keypoints (DLC's own definition). Try the maximum; pupil
  points only; eye points only; the count of keypoints outside the SARIMAX confidence interval (computed but unused
  in DLC's code).
- **S1-3 model order and direction.** Orders (1,0,0), (2,0,0), (5,0,1), (3,1,1); forward + backward one-step prediction,
  larger deviation (a run is caught at entry by the forward model and at exit by the backward one); rolling 2-min
  windows instead of the whole video.
- **S1-4 normalisation.** Deviation / eye width of the video, so one threshold serves all videos.
- **S1-5 combination.** Logistic regression on all signals, leave-one-video-out (the self-consistency recipe);
  rank average as the simple alternative. Report whether it beats the best single signal at the same flagged share.
- **S1-6 per-run evaluation.** Group consecutive flagged frames into runs; count a run as caught if any frame is
  flagged. The evenly spaced test frames cannot measure this; if per-run truth is needed, label 5 clips × 60
  consecutive frames (300 labels; decide with Kaiwen first).
- **S1-7 Pose-PCA reconstruction error** (Lightning Pose). PCA on the video's labeled 8-point poses (16-dim, components
  explaining 99% of variance), per-frame reconstruction error of the predicted pose; memoryless, so it should fire on
  every frame of a stuck run. **S1-7b geometric constraints**: pupil axes crossing at their midpoints (`axes_off`),
  pupil between the eyelids, eye width constant within the video.
- **S1-8 ensemble disagreement.** Per keypoint, per frame, the standard deviation across the video's existing models
  (the 20/40/60-label steps; optionally the 5/10-label subset models after one whole-video inference pass each),
  confidence-weighted as in EKS. Must be measured: if all models share the failure it is silent.

Decision rule after stage 1: if some signal or combination reaches recall > 20 px ≥ 50% at ≤ 5% flagged
(precision ≥ 50%), adopt it as the detector and go to P2. Otherwise stage 2.

### Stage 2 (only if stage 1 is not enough): signals that need an extra inference pass or a new tool

Cheapest first, all label-free:

- **S2-3 MEYE cross-check.** Run the pretrained MEYE U-Net on the 1330 crops; |segmentation area − ellipse area| as a
  signal; calibrate against our "bright band = pupil" rule first.
- **S2-4 multiple-peak entropy** of the DLC heatmaps (needs a heatmap export for the 1330 frames).
- **S2-5 Viterbi path vs raw prediction** (Anipose): needs the k best heatmap peaks per keypoint from an inference
  pass; disagreement between the path and the top-1 prediction as a flag, and the path itself as a repair.
- **Learned detector** on all features (gradient boosting) only if the logistic regression of S1-5 is clearly limited.
- Lightning Pose retraining (temporal loss, temporal context network) last: a separate project.

### Literature ▶ done: `literature_unreliable_frames.md` (cards per source, two synthesis tables, references).

## P2 — repair the flagged frames at minimum human cost

Metric: human cost = frames (or points) a human touches per 1000 frames; benefit = reduction of frames > 20 px /
> 50 px and of the area error, measured on the flagged test frames. Report "px of error removed per labelled frame".

Ordered by human cost, each evaluated on the frames the P1 detector flags:

1. **P2-1 zero human: EKS** (ensemble median + Kalman smoothing with the ensemble variance as observation noise, generic
   single-camera version) on the existing per-video models' CSVs; baseline = single-model median / ARIMA / spline
   smoothing (two sources report that single-model smoothing does not remove glitches without over-smoothing).
   Error of the repaired frames vs the raw prediction, by run length.
2. **P2-2 zero human: drop.** Exclude flagged frames from the area and blink analyses (steps 9, 10) and measure how much
   the per-video numbers move.
3. **P2-3 zero human: segmentation fallback.** MEYE area where the ellipse is rejected, only if S2-3 shows its area
   error below the 4-point ellipse's on the test frames.
4. **P2-4 two labels per run.** Correct the two endpoints of a flagged run, linear interpolation between them (CVAT
   practice). **P2-4b one label per run**: correct one frame, propagate ± 30 frames by optical flow on the eye crop,
   accept where propagations from both ends agree within 20 px (Charles et al. 2016).
5. **P2-5 pick among peaks.** Show the annotator the 2–3 heatmap peaks of a flagged keypoint and let them click the right
   one (Liu & Ferrari 2017); needs the heatmap export of S2-4.
6. **Retrain with the corrected frames** (the existing active-learning step). Compare, per labelled frame, with 4:
   retraining helps all videos of the mouse, direct repair helps only the run.

Decision rule: pick, for the flagged share the detector produces, the cheapest option whose remaining frames > 20 px
is below an agreed level (to agree with Kaiwen; proposal: ≤ 2% of frames per video).

## Order of work

1. ▶ Stage 1 of P1 (S1-1 … S1-8), one script, one results folder, one figure (ROC + operating-point table).
2. ▶ P2-1 and P2-2 on the frames the current fitting rule flags (baseline for P2); EKS on 2–4 existing models per video.
3. Meeting 2026-10-31: decide stage 2 vs P2 routes, and whether per-run truth (300 labels) is worth it.

Nothing in `results/` of earlier steps is modified; every new result goes to a new folder and the page gets a new
section only when a result is final.
