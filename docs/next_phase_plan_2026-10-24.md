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

### Stage 1 ▶ fitting on its default algorithm: thresholds and parameters (no new labels, no code of step 8 changed)

New script `unreliable_frames_fitting_sweep.py`, outputs in a new folder `results/unreliable_frames_sweep/`.

1. **Threshold sweep.** epsilon from 2 to 60 px (log steps). Full ROC and precision–recall curves at both truth levels;
   table of operating points, e.g. the epsilon that gives recall >20 px of 50% / 70% / 90% and what it costs in flagged
   share. Answers "is 20 px just the wrong number".
2. **How the 8 deviations are combined.** Now: mean over the 8 keypoints. Try max over keypoints; mean over the 4 pupil
   points only and the 4 eye points only; number of keypoints above a per-point threshold; the largest two. The wrong
   frames of step 8 are mostly pupil-point errors, so the eye points may be diluting the signal.
3. **Model order and direction.** Now: SARIMAX(3,0,1) on the whole video, one-step forward prediction, points with
   confidence < 0.01 set to missing. Try orders (1,0,0), (2,0,0), (5,0,1), (3,1,1); forward + backward prediction and
   take the larger deviation (a run of wrong frames is only caught at its first frame by a forward model; the backward
   model catches its last frame); a rolling window (e.g. 2 min) instead of the whole video for non-stationary videos.
4. **Normalisation.** Deviation in px vs deviation / eye width of the video (as jump does), so one threshold serves
   videos of different scale.
5. **Combination.** fitting + mean confidence + jump as one score: rank average, and a logistic regression with
   leave-one-video-out. Report whether the combination beats the best single signal at the same flagged share.
6. **Per-run evaluation.** Group consecutive flagged frames into runs; count a run as caught if any of its frames is
   flagged. The evenly spaced test frames cannot measure this; use the 327 blind-judged frames' neighbourhoods and
   the whole-video predictions to at least measure run lengths. If per-run truth is needed, label 5 clips × 60
   consecutive frames (300 labels; decide with Kaiwen first).

Decision rule after stage 1: if some setting reaches recall >20 px ≥ 50% at ≤ 5% flagged (precision ≥ 50%), adopt it as
the detector and go to P2. Otherwise stage 2.

### Stage 2 (only if stage 1 is not enough): new signals, then a learned detector

Cheapest first, all label-free:

- **Model disagreement.** Every video already has several trained models (the 20/40/60-label steps, and the 5/10-label
  subset models a/b). Distance between the plateau model's prediction and the previous step's (or the subsets') on
  the same frame = an ensemble-disagreement signal at zero training cost. Lightning Pose and EKS use exactly this.
- **Geometric consistency.** The pupil is an ellipse with two axes that must cross at their midpoints (`axes_off`,
  already an AUC-0.92 signal for eye closure); eye width is constant within a video; the pupil lies between the eyelids.
  Residuals of these constraints per frame.
- **Smoother residual.** Distance of the prediction from a Kalman / median-smoothed trajectory (Anipose-style), as an
  alternative to SARIMAX.
- **Learned detector.** A small classifier (logistic regression or gradient boosting) on all signals, trained
  leave-one-video-out on the 1330 frames; report AUC and the operating points as above. 1330 frames with 155
  positives is enough for a linear model, not for anything larger.

### Literature to read for P1 and P2 ▶ (short written survey, one paragraph per source, what is transferable)

- DeepLabCut: Mathis et al. 2018 (Nat Neurosci), Nath et al. 2019 (Nat Protoc) — the jump / uncertain / fitting
  outlier detectors and the refine-and-retrain loop we already use.
- LEAP (Pereira et al. 2019, Nat Methods) and SLEAP (Pereira et al. 2022, Nat Methods) — their active-learning
  "suggestions" (image-feature clustering, prediction score, velocity) and their correction GUI.
- Lightning Pose (Biderman et al. 2024, Nat Methods) — unsupervised temporal / PCA losses, ensemble Kalman smoothing
  (EKS), and using the loss values as a per-frame reliability signal.
- Anipose (Karashchuk et al. 2021, Cell Rep) — median / Viterbi filters and spatiotemporal regularisation that repair
  outliers without labels.
- DeepGraphPose (Wu et al. 2020, NeurIPS) — temporal and spatial constraints during inference.
- Pupil-specific tools (e.g. DeepLabCut-based pupillometry, mEye / pupil segmentation nets) — how they handle blinks and
  occlusion.
- General: ensemble / MC-dropout uncertainty for keypoints, label-efficient video annotation (propagation across frames).

## P2 — repair the flagged frames at minimum human cost

Metric: human cost = frames (or points) a human touches; benefit = reduction of frames > 20 px / > 50 px and of the
area error, measured on the flagged test frames. Report "px of error removed per labelled frame".

Ordered by human cost, each evaluated on the frames the P1 detector flags:

1. **Zero human: interpolate / smooth over flagged frames.** Linear interpolation between the nearest unflagged
   neighbours, and a Kalman / EKS-style smoother across the run. Error of the repaired frames vs the raw prediction.
   Works for short runs; fails for long ones — report by run length.
2. **Zero human: drop.** Exclude flagged frames from the area and blink analyses (step 9, 10) and measure how much the
   per-video numbers move. At a 2–5% flagged share this is probably the cheapest acceptable answer for the downstream
   analyses, even if the frames themselves stay wrong.
3. **One label per run.** A human corrects one frame per flagged run; the correction is propagated to the run by
   interpolation anchored on it. Human cost = number of runs, not frames.
4. **Retrain with the corrected frames** (the existing active-learning step). Compare, per labelled frame, with 3:
   retraining helps all videos of the mouse, direct repair helps only the run.

Decision rule: pick, for the flagged share the detector produces, the cheapest option whose remaining frames > 20 px
is below an agreed level (to agree with Kaiwen; proposal: ≤ 2% of frames per video).

## Order of work

1. ▶ Stage 1 of P1 (items 1–5), one script, one results folder, one figure (ROC + operating-point table).
2. ▶ Literature survey, written as `docs/literature_unreliable_frames.md`.
3. ▶ P2 option 1 and 2 on the frames the current fitting rule flags (baseline for P2).
4. Meeting 2026-10-31: decide stage 2 vs P2, and whether per-run truth (300 labels) is worth it.

Nothing in `results/` of earlier steps is modified; every new result goes to a new folder and the page gets a new
section only when a result is final.
