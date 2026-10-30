# Flag by the fitting deviation, repair by the second peak, accept by the same criterion, 2026-10-30

Scripts: `scripts/fit_threshold.py` (the threshold), `scripts/flag_and_repair.py --video N --accept fit|geometry` (one
video), `scripts/flag_and_repair_all_videos.py` (the nineteen videos together, `all_videos/`). Inputs: the plateau
model's whole-video predictions, the sweep's cached fitting deviations (`results/unreliable_frames_sweep/cache/`), the
heatmap second peaks (`results/heatmap_signals/videoNN_shuffle*_peaks.csv`), the 1330 truth frames. Nothing existing
is modified; the earlier pipeline in `results/second_peak_repair/` is left as it was, with a note at its top.

## Three decisions (user, 2026-10-30)

1. **The threshold is read off the frames that must be found, not set as a budget.** The earlier 44 px was the 95 %
   quantile of the score on the truth frames, i.e. "flag 5 % of frames", a number with no basis in the errors.
2. **Nothing is deleted.** A flagged frame the repair cannot fix stays flagged (`flagged_not_repaired`). Whether such
   frames are left out later is a decision for whoever uses the data, not for this pipeline.
3. **The acceptance test of a repair should be the criterion that flagged the frame**, not a separate hand-made rule.
   The geometry check of `second_peak_repair.py` (5 MAD bands, 0.6 eye width, 0.1 lid margin) was designed on the same
   32 frames it was then judged on. It is kept here only as the comparison (`--accept geometry`).

## The finder and its threshold (`results/fit_threshold/`)

Finder, unchanged: DeepLabCut's fitting outlier score, SARIMAX (2,0,0) fitted on the whole video, one-step forward
prediction, distance between predicted and actual position, largest of the 8 key points (`fit_o200_max`). Points with
likelihood below 0.01 are treated as missing, as in the sweep.

Threshold rule: over the 1330 truth frames, for every integer threshold t, recall = share of the 32 frames with error
> 50 px whose score exceeds t, false share = share of the 1175 frames with error ≤ 20 px whose score exceeds t; the
20–50 px frames are neither target nor clean and are only reported. Chosen t maximises recall minus false share (the
ROC optimum; no free constant).

| threshold | catches of the 32 > 50 px frames | flags among the 1175 ≤ 20 px frames | flags among all truth frames |
|---|---|---|---|
| **36 px (chosen)** | **31** | **2.0 %** | **6.4 %** |
| 44 px (previous, 95 % quantile) | 27 | 1.4 % | 5.0 % |

Why 36: the 32 scores, sorted, are 11 px (video 14, the frame no detector finds), then 36.3, 36.4, 36.7, 39.8 (four
marginal video-8 frames of 50–55 px), then 45 px, then 26 frames from 56 px up to 580 px. Above 36 the four video-8
frames are lost for a gain of 0.3 % fewer false flags; below 36 no further > 50 px frame is gained until 10 px, where a
quarter of the good frames would be flagged.

Leave-one-video-out (`lovo.csv`): the same rule on the other 18 videos, applied to the held-out one. 18 of 19 folds
choose 35–36 px; the fold that holds out video 8 chooses 117 px, because the frames between 36 and 56 px are all video
8's. Over the 19 folds 25 of 32 frames are caught, 21 of 1175 good frames flagged. **The 36 px value therefore rests on
video 8**; without it the data would say "anything between 45 and 117 px". That is a property of the truth set (one
video holds all the marginal errors), to be said as such.

## The repair and the two acceptance tests

On a flagged open frame, every pupil point whose deviation exceeds the threshold is replaced by its own heatmap second
peak (cell × stride + the per-point median offset, as before). Then:

- `--accept fit`: the deviation of the new position from the same one-step prediction is computed; the frame is
  "repaired" if all 8 points are now within 36 px, otherwise it stays flagged and nothing changes. If the point over the
  threshold is an eyelid or corner point nothing is tried. Limitation: the prediction comes from the model's original
  trajectory, so a repaired frame does not change the prediction for the frames after it.
- `--accept geometry`: the earlier geometry rule, with the 36 px threshold. Its "a flagged frame that already passes the
  check" is reported as `flagged_passes_check`, its "drop" as `flagged_not_repaired`.

Both tests, same 19 videos, same threshold (`all_videos/totals.csv`, `comparison.csv`):

| | accept = fitting (same criterion) | accept = geometry (comparison) |
|---|---|---|
| > 50 px frames flagged, of 32 | 31 | 31 |
| … repaired to below 50 px (below 20 px) | 6 (2) | 16 (8) |
| … flagged, not repaired | 21 | 11 |
| … left alone: closed-eye 4, unflagged 1 | 5 | 5 |
| ≤ 20 px frames repaired / made worse by > 5 px | 7 / 6 | 0 / 0 |
| ≤ 20 px frames flagged / of them still flagged | 23 / 16 | 23 / 3 |
| whole videos: flagged / repaired / still flagged | 6.3 % / 2.2 % / 3.7 % | 6.3 % / 1.6 % / 1.1 % |
| touched share per video, min / median / max | 0.3 / 3.7 / 17.2 % | 0.1 / 1.4 / 11.8 % |

**The same criterion is the worse acceptance test, on both counts.** It repairs 6 of the 32 big errors where the
geometry test repairs 16, and it turns 6 falsely flagged good frames into frames that are actually wrong, where the
geometry test touches none of them (it says "the frame passes, leave it"). The frame-by-frame comparison
(`all_videos/truth_summary_all.csv`, per-video `truth_frames.csv`) shows why, and it is the same reason both times:
the one-step prediction is built from the preceding frames. When the model is wrong for a stretch, the prediction
follows the wrong trajectory, so the correct second peak lies far from it and is refused (videos 1, 2, 11: six frames
the geometry test repairs to 7–17 px stay flagged). When a good frame is flagged because the eye really moved fast,
the prediction lags behind the true position, and a second peak that happens to sit near the lagging prediction is
accepted although it is farther from the truth (video 3 frame 70575: 9 → 24 px; video 18 frame 59921: 16 → 34 px).
Finding asks "does this frame disagree with the past"; accepting a replacement needs evidence from the frame itself,
which the time series cannot give. So the two steps cannot share one criterion, and the geometry test, hand-made as it
is, is doing the second job. Its constants remain the open point: they were set by hand and should be validated the
same way as the threshold (leave-one-video-out) before the rule is called final.

## Spot check by eye, same rule on every video

`all_videos/spotcheck_sample.csv`: per video and per acceptance test, up to 10 repaired and 10 still-flagged frames, one
per episode (touched frames at most 5 frames apart), episodes drawn at random (seed 0); sheets
`all_videos/spotcheck/videoNN_<test>.jpg` (local only; red = model, blue = accepted second peak; caption: frame, status,
points, episode length, fitting deviation). The `verdict` column is empty: real error / false alarm / cannot tell.

## Files

`videoNN_fit/`, `videoNN_geometry/`: `summary.csv` (truth frames), `whole_video.csv`, `truth_frames.csv`;
`frames_all.csv.gz` and `cache/videoNN_pred200.npy` local only. `all_videos/`: `whole_video_all.csv`,
`truth_summary_all.csv`, `totals.csv`, `comparison.csv`, `touched_structure.csv`, `share_per_video.png`,
`spotcheck_sample.csv`.
