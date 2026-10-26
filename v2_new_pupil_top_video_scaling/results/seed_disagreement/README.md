# Seed disagreement for finding unreliable frames (pilot)

Follow-up of `results/error_detector/` (page section 8.2) and `results/heatmap_signals/`. Two models trained on the
identical labels, differing only in the random seed, disagree most on the frames the model has not really learned.
The distance between their predictions is scored per frame and compared with fit_pupil on the truth frames of one
video. The main-line coordinates stay those of the plateau model; the second model is a second opinion only.

## Basic information

| item | value |
|---|---|
| video | 8 (`8_20251024_Pluto_spont1`), the video with the most wrong frames (38 of 70 frames > 20 px, 9 > 50 px) |
| main model | the plateau-point model of video 8, shuffle 911, seed 42 (unchanged) |
| twin model | shuffle 3081, seed 7; training set identical to shuffle 911 (checked image by image): 420 frames = videos 0:100, 1:100, 2:60, 3:60, 4:40, 5:20, 6:20 carried in, 20 of video 8 (batch 1); validation = video 8 val20; same recipe (ResNet-50, batch 2, 120 epochs, milestones 96/114, snapshot 120) via `scripts/scale_step.py --seed 7 --tag seed7 --no-eval` |
| truth frames | the 70 step-8 frames of video 8 (test50 + val20 PNGs); frame RMSE vs human labels > 20 px / > 50 px from `results/unreliable_frames_sweep/signals_all_frames.csv` |
| disagreement score | both models run on the 70 PNGs (the main model's PNG predictions equal its whole-video file to 0.00 px); per point the distance between the two predictions; per frame `dis_max_all`, `dis_mean_all`, `dis_max_pupil`, `dis_mean_pupil` |
| comparison | AUC and recall at 10 / 20 / 30 % flagged (5 % is 3 frames here, not informative) against fit_pupil, fitting, jump, min_conf, heatmap entropy; rank averages with fit_pupil; bootstrap interval of the AUC gain over the 70 frames |
| not done | one video, one extra seed; the twin is not run on the whole video; no label, model, earlier result or script changed |

Script: `scripts/seed_disagreement.py --video 8 --twin 3081`. Files: `pilot_video08.csv` (table), `pilot_video08_frames.csv`
(70 frames with both models' coordinates and every score), `train_video08_seed7.log` (local only).

## Result (video 8, 70 frames: 38 > 20 px, 9 > 50 px)

Full table: `pilot_video08.csv`; per frame: `pilot_video08_frames.csv`.

| score | AUC > 20 px | AUC > 50 px |
|---|---|---|
| fit_pupil (reference) | 0.894 | 0.729 |
| min_conf (lowest confidence) | 0.830 | 0.741 |
| heatmap entropy, 4 pupil points (`results/heatmap_signals`) | 0.868 | 0.811 |
| seed disagreement, mean over 8 points (best of 4 variants) | 0.762 | 0.536 |
| seed disagreement, other variants (max, pupil only) | 0.69–0.71 | 0.46–0.50 |
| rank average fit_pupil + disagreement | 0.888 | 0.649 |
| rank average fit_pupil + disagreement + heatmap entropy | 0.911 | 0.719 |

- The disagreement between the two seeds is a weak signal here: alone it is below the confidence, and added to
  fit_pupil it changes nothing (AUC gain −0.005, bootstrap interval −0.07 to +0.07; at > 50 px it hurts).
- The reason is that the two models make the same errors. Their per-frame RMSEs rank almost identically (Spearman
  0.89); all 38 frames the main model gets wrong by > 20 px the twin also gets wrong (plus 4 more); 6 of the 9
  > 50 px frames are shared. The two predictions differ by only 4.9 px (median) on the wrong frames against 3.4 px on
  the right ones, and their pupil_left offsets from the human label are correlated (0.64).
- The twin is slightly worse than the plateau model (median RMSE 25.3 vs 21.5 px over the 70 frames), the ordinary
  seed-to-seed spread.
- Reading: on this video the wrong frames are not places where training is unstable; any model trained on these
  labels lands at the same positions. That points to the labels / the definition of the pupil edge (or to a lack of
  training examples of these situations), not to something a detector can see. One video, one extra seed, 70 frames.
