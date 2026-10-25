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

## Result

(pending: the run started 2026-10-25 17:18)
