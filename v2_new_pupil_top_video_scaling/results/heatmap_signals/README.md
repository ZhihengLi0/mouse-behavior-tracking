# Heatmap-shape signals for finding unreliable frames (pilot)

Follow-up of `results/error_detector/` (page section 8.2): every signal computable from the saved prediction files
saturates at about 35 % recall of the > 20 px frames at 5 % flagged. The saved files keep one coordinate and one
confidence per key point; the shape of the network's heatmap is lost. This pilot re-runs the plateau model of one
video and records a few numbers about the heatmap of each key point, to see whether they find frames that the
time-series deviation (fit_pupil) misses.

## Basic information

| item | value |
|---|---|
| video | 8 (`8_20251024_Pluto_spont1`, 71 748 frames, 928 × 736 px, 60 fps); the video with the most wrong frames (38 of 70) |
| model | the plateau-point model of video 8 (shuffle 911, 20 labels, snapshot 120), unchanged |
| inference | standard DLC 3.0.1 PyTorch runner, batch 32, device mps, no rescaling; the regressed coordinates equal the saved `.h5` of `predictions_step01` to 0.000 px |
| what is recorded | per frame and key point, from the heatmap as the predictor sees it (no sigmoid in this model's predictor; negative values set to 0; 8-px stride): `peak1` (highest value = confidence before clipping), `peak2` (highest local maximum ≥ 3 cells away), `ratio` = peak2 / peak1, `dist_px` between the two peaks, `n_peaks` (local maxima > 0.3 peak1), `entropy` (normalised, 0 = one cell, 1 = flat), `mass_near` (share of the mass within 12 px of peak1) |
| frame scores | max / mean over the 4 pupil points or all 8 points of ratio, entropy, −mass_near, n_peaks, −peak1, ratio × distance |
| truth and reference | the 70 step-8 frames of video 8 (50 test + 20 validation), frame RMSE vs human labels > 20 px / > 50 px; compared with fit_pupil, fitting, jump, min_conf of `results/unreliable_frames_sweep/signals_all_frames.csv` on the same frames; AUC and recall at 5 / 10 / 20 % flagged |
| not done | heatmaps are not stored; no training, label, model or earlier result is changed; one video only, so the numbers are a first look, not a conclusion |

Script: `scripts/heatmap_signals.py --video 8` (≈ 80–95 min on this Mac). Files: `video08_shuffle911.csv`
(per frame, local only, ≈ 40 MB), `pilot_video08.csv` (the table below), `pilot_video08_frames.csv` (the 70 frames
with every score), `run_video08.log` (local only).

## Result (video 8, 70 frames: 38 > 20 px, 9 > 50 px)

Video 8 has more wrong frames than right ones, so "recall at 5 % flagged" (3 frames) says nothing here; the AUC is the
number to read. Full table: `pilot_video08.csv`.

| score | AUC > 20 px | AUC > 50 px |
|---|---|---|
| fit_pupil (reference, best signal of section 8.2) | 0.894 | 0.729 |
| fitting (8 points) | 0.873 | 0.703 |
| min_conf (lowest confidence) | 0.830 | 0.741 |
| jump | 0.752 | 0.500 |
| heatmap entropy, mean of the 4 pupil points (best heatmap score) | 0.868 | 0.809 |
| heatmap peak ratio, mean of the 4 pupil points | 0.864 | 0.780 |
| other heatmap scores (mass near the peak, number of peaks, max over points, all 8 points) | 0.59–0.86 | 0.57–0.79 |
| rank average of fit_pupil and heatmap entropy | 0.926 | 0.798 |

- Alone, no heatmap score beats fit_pupil at > 20 px; the best one (pupil entropy) is a better "uncertain" than the
  confidence (0.868 vs 0.830) and the best single score at > 50 px (0.809, 9 frames).
- Combined with fit_pupil (rank average) the AUC at > 20 px rises by 0.031 (bootstrap over the 70 frames: 95 % interval
  −0.013 to +0.090, gain > 0 in 90 % of resamples). Of the 6 wrong frames that fit_pupil ranks outside its top 38,
  the entropy ranks 3 inside the top 38 and the combination 2; precision of the top 38 goes 0.84 → 0.87.
- The entropy is correlated with fit_pupil (Spearman 0.64) and with the confidence (0.77): partly new information,
  not an independent signal.
- The best score was picked among 16 heatmap scores on the same 70 frames, which flatters it; one video, 70 frames,
  so this is a reason to look at two more videos (1 and 4, ≈ 3 h), not a result.

Videos 1, 2, 11, 4 (2026-10-27, `pilot_videoNN.csv`, written by the `--peaks` passes): the mean pupil heatmap entropy
has AUC 0.72 / 0.86 / 0.77 / 0.43 against the > 50 px frames where fit_pupil has 0.996 / 0.985 / 0.977 / 1.000, and
0.76 / 0.71 / 0.81 / 0.63 against > 20 px (fit_pupil 0.94 / 0.998 / 0.98 / 0.75). The heatmap statistics add nothing
to the fitting signal on these videos either; the peak coordinates are used by `scripts/second_peak_repair.py`.
