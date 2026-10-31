# Find, repair, check: the pipeline in units of eye width, all nineteen videos, 2026-10-30

Scripts: `scripts/error_tier.py` (the error line), `scripts/fit_threshold.py` (the flagging threshold),
`scripts/acceptance_thresholds.py` (cuts of the acceptance tests), `scripts/flag_and_repair.py --video N --accept …`
(one video), `scripts/flag_and_repair_all_videos.py` (all videos, `all_videos/`). Inputs: the plateau model's whole-video
predictions, the sweep's cached fitting deviations, the heatmap second peaks, the 1330 truth frames. Nothing existing is
modified; the earlier pixel-based pipeline in `results/second_peak_repair/` is left as a record, with a note at its top.

## Decisions (user, 2026-10-30)

1. **Units.** Every error and threshold is a fraction of the video's eye width (median corner-to-corner distance of the
   plateau model's predictions, label-free; 392–493 px on these videos, `results/error_tier/eye_width.csv`), so the numbers
   carry over to another camera or distance.
2. **One line.** The histogram of the 1330 truth-frame errors (`results/error_tier/error_histogram.png`) has a smooth main
   group ending at about 13 % of the eye width and a sparse tail of frames with a point thrown far. The line is read off
   the histogram where the main group ends, **0.14 of the eye width** (55–69 px here; no frame between 12.7 % and
   15.4 %). A truth frame above it is a *wrong frame* (23 of 1330: videos 1: 8, 2: 4, 4: 2, 8: 1, 10: 3, 11: 4, 14: 1),
   every other frame is a *good frame*. The former 20-px level is label noise and is not reported any more; the former
   50-px level (32 frames) differed from this line by the nine 50–55 px frames of video 8, which sit at the end of the
   main group.
3. **Flagging threshold = the line.** A frame is flagged when the fitting deviation of one of its 8 points exceeds 0.14
   of the eye width (`results/fit_threshold/chosen.csv`). All 23 wrong frames have a deviation of at least 0.257, so
   the ROC optimum would sit at 0.257 with no margin (leave-one-video-out: 20 of 23 caught); the line itself catches
   23 of 23 in every fold and flags 2.1 % of the good frames, 3.8 % of all truth frames.
4. **Nothing is deleted.** A flagged frame the repair cannot fix stays flagged (`flagged_not_repaired`).

## The three steps

- **Find**: DeepLabCut's fitting outlier score (SARIMAX (2,0,0) on the whole video, one-step forward prediction,
  distance between predicted and actual position), largest of the 8 key points, divided by the eye width; above 0.14
  the frame is flagged. Closed-eye frames (opening below half the video's median) are left alone and counted apart.
- **Repair**: on a flagged open frame, the candidate pupil points are replaced by their own heatmap second peak.
- **Check**: an acceptance test decides whether the replacement is kept. Seven tests were run the same way on all 19
  videos (`all_videos/totals.csv`, per-video `videoNN_<test>/`):

| test | what accepts the repair | where its numbers come from |
|---|---|---|
| geometry | the hand-made rule of `triggered_repair.py`: ratio bands of 5 MAD around the video's median, inside the corners and the eyelid band, left/right and top/bottom order; a flagged frame that already passes is left alone | constants set by hand (10-26) |
| fit | the finder's own criterion: after the replacement all 8 deviations from the same one-step prediction are within 0.14 | none needed |
| conf | the second peak's heatmap value ≥ 0.238 (DeepLabCut's confidence on the candidate point) | cut read off the 37 candidate repairs on the truth frames: max (right repairs accepted − wrong repairs accepted); leave-one-video-out |
| jump | the candidate's displacement to the previous and next frame ≤ 0.485 eye widths (DeepLabCut's jump criterion) | same rule |
| labelgeo | the frame's twelve pupil-point quantities inside the range spanned by the 1330 human labels | no constant |
| labelregion0 | labelgeo in the two steps of the geometry rule: a flagged frame already inside the label range is left alone, otherwise repaired if the replacement brings it inside | no constant |
| labelregion | labelregion0 with the range widened by 42 % of each quantity's width, the smallest margin at which every held-out video's labels fall inside the range of the other 18 | leave-one-video-out coverage |

## Results on the truth frames (1330 frames; 23 wrong, 1307 good; the finder flags 23 wrong and 28 good frames)

| | geometry | fit | conf | jump | labelgeo | labelregion0 | labelregion |
|---|---|---|---|---|---|---|---|
| wrong frames repaired to good, of 23 | **12** | 5 | 10 | 5 | 10 | 10 | **12** |
| wrong frames still flagged | 7 | 14 | 9 | 14 | 9 | 9 | 7 |
| wrong frames left alone (eye judged closed) | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| good frames repaired / made worse by > 1 % of eye width | 4 / 0 | 10 / 2 | 10 / 2 | 11 / 3 | 8 / 1 | 4 / 0 | 1 / 1 |
| good frames still flagged, of 28 | 7 | 18 | 18 | 17 | 20 | 13 | **6** |
| whole videos: repaired / still flagged (share of all frames) | 1.0 % / 0.9 % | 1.0 % / 2.4 % | 1.6 % / 1.8 % | 1.2 % / 2.2 % | 1.1 % / 2.3 % | 0.8 % / 1.6 % | 0.8 % / 1.2 % |
| touched share per video, median (max) | 1.2 % (8.0 %) | 2.2 % (14.6 %) | 2.2 % (14.6 %) | 2.2 % (14.6 %) | 2.2 % (14.6 %) | 1.4 % (10.3 %) | 1.1 % (10.1 %) |

- **The finder does its job**: all 23 wrong frames are flagged, in every leave-one-video-out fold, at a cost of 28 good
  frames flagged among 1307 and 3.8 % of all frames.
- **Of the 23 wrong frames, 4 are left alone because the eyelid rule calls the eye closed** (videos 10 and 11); in two of
  them the model's eyelid points are themselves wrong, so "closed" is the model's error. The remaining 19 are
  repairable in principle; the best tests repair 12 and leave 7 flagged. The 7 are multi-point throws where the second
  peak is not the pupil either (videos 1, 2, 14) and a video-8 frame.
- **The two best acceptance tests are the hand-made geometry rule and the label range in the same two steps with the
  leave-one-video-out margin**: both repair 12 and leave 7 flagged; on the good frames the label range does slightly
  better (1 repaired, 6 still flagged, against 4 and 7), the geometry rule damages none. The label-range version has no
  hand-set constant: its range comes from the human labels and its margin from leave-one-video-out coverage. It is the
  version with a basis, and it is as good as the hand-made rule.
- **The same criterion for finding and accepting (fit) is the worst**: 5 repaired, 14 still flagged, 2 good frames made
  worse. The one-step prediction follows the wrong trajectory inside an episode of errors, so the correct second peak is
  refused, and lags behind a fast true movement, so a wrong second peak near the lag is accepted. Finding asks whether
  the frame disagrees with the past; accepting needs evidence from the frame itself.
- **Confidence and jump** (DeepLabCut's own quantities on the candidate point) are in between: conf repairs 10 but
  damages 2 good frames; jump repairs 5.
- **Whole videos**: with either of the two best tests about 1 % of all frames are repaired and about 1 % stay flagged;
  the share per video follows the video's quality (video 8 and video 1 highest).

## Error before and after the repair, all 19 videos

`scripts/repair_before_after.py` reads the per-video `truth_frames.csv` of the seven tests and puts the old and the
new error side by side: `all_videos/error_before_after.csv` (per video and test) and `all_videos/error_before_after.png`.
The comparison exists only on the 1330 truth frames, the frames with human labels; for the whole videos only the share
of frames changed can be given.

| 1330 truth frames | before | geometry | fit | conf | jump | labelgeo | labelregion0 | labelregion |
|---|---|---|---|---|---|---|---|---|
| mean frame error, % of eye width | 3.14 | 2.92 | 3.02 | 2.95 | 3.05 | 2.95 | 2.95 | **2.91** |
| median frame error, % of eye width | 2.00 | 1.99 | 2.00 | 1.99 | 2.00 | 1.99 | 1.99 | 1.99 |
| frames above the line (wrong) | 23 | 11 | 18 | 14 | 18 | 13 | 13 | **11** |
| frames changed / improved / made worse by > 1 % | | 16 / 16 / 0 | 14 / 12 / 2 | 19 / 16 / 2 | 15 / 12 / 3 | 17 / 15 / 1 | 14 / 14 / 0 | 13 / 12 / 1 |
| whole videos, share of frames changed | | 1.0 % | 1.0 % | 1.6 % | 1.2 % | 1.1 % | 0.8 % | 0.8 % |

With the adopted test only six videos change at all (1, 2, 4, 8, 10, 11, the videos that have wrong frames); video 1
goes from 6.5 % to 4.5 % mean error and from 8 to 2 wrong frames. The median does not move because the repair touches
about 1 % of the frames; the mean falls because the thrown points were far.

## Spot check by eye, same rule on every video

`all_videos/spotcheck_sample.csv`: per video and per test, up to 10 repaired and 10 still-flagged frames, one per
episode (touched frames at most 5 frames apart), episodes drawn at random (seed 0); sheets `all_videos/spotcheck/
videoNN_<test>.jpg` (local only; red = model, blue = accepted second peak; caption: frame, status, points, episode
length, fitting deviation in % of eye width). The `verdict` column is empty: real error / false alarm / cannot tell.

## Files

`videoNN_<test>/`: `summary.csv` (truth frames), `whole_video.csv`, `truth_frames.csv` (errors in eye widths and in px);
`frames_all.csv.gz` and `cache/videoNN_pred200.npy` local only. `all_videos/`: `whole_video_all.csv`,
`truth_summary_all.csv`, `totals.csv`, `comparison.csv`, `touched_structure.csv`, `error_before_after.csv`, `share_per_video.png`, `error_before_after.png`,
`spotcheck_sample.csv`. Threshold and line: `results/error_tier/`, `results/fit_threshold/`,
`results/acceptance_thresholds/`.
