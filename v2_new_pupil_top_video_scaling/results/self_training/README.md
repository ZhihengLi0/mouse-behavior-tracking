# Self-training (the loop of page step 9.4)

The first section below is the round-1 pilot as written on 2026-11-04; the later runs (experiment 1 on more videos,
the second turn, experiment 2 with repaired frames, experiment 3 with both kinds repeated) are in "Later runs".

# Self-training, round 1 (pilot and all-videos pass)

Direction 3 of the 2026-10-31 meeting: retrain the plateau model of a video on its own human labels plus
pseudo labels taken from its own predictions on frames the final pipeline passed, and see whether the retrained
model (the student) makes fewer wrong frames than the plateau model (the teacher). Design in
`docs/next_phase_analysis_2026-10-31.md`, section 3.3. Script: `scripts/self_training.py`. Run on 2026-11-02 to
2026-11-04 on the Mac (mps): videos 1, 7 and 8 with both pseudo densities first, then videos 2 and 4 with the denser
one only. Videos 10 and 11 were planned for the same night and not started, because the driver does not start a
video that cannot finish before the morning deadline. After the blind test set (`../blind_test/`) confirmed
the students' gain, the same every30 recipe is being run on all remaining videos, in order of the teacher's
flagged share (highest first), two or three videos a night: videos 15, 6 and 0 on 2026-11-05 (driver
`self_training_all.sh`, log `all_videos.log`).

## Basic information

| item | value |
|---|---|
| videos | 1 and 8 (hard, many flagged frames), 7 (clean control); then 2 and 4 (every30 only); then 15, 6 and 0 (every30 only, all-videos pass) |
| teacher | the plateau model of each video: video 1 shuffle 211, video 2 shuffle 321, video 4 shuffle 532, video 7 shuffle 811, video 8 shuffle 911, video 15 shuffle 1612, video 6 shuffle 711, video 0 shuffle 113 |
| teacher training set | video 1: 120 human frames (100 of video 0 plus 20 of video 1); video 2: 220 (prior videos plus 20 of itself); video 4: 360 (prior videos plus 40 of itself); videos 7 and 8: 420 (prior videos plus 20 of the video itself); video 15: 720 (700 carried plus 20 of itself); video 6: 400 (380 plus 20); video 0: 60 (its own first batch, no prior video); 20 validation frames each |
| pseudo labels | the teacher's own coordinates after the final pipeline (`flag_and_repair.py --finder either --accept labelregion --points all --replace smallest`) on every 30th frame (variant every30) or every 100th frame (variant every100) |
| frames excluded from pseudo labels | frames the pipeline could not pass, and 60 frames on each side of the 70 truth frames |
| per-point filter | a point is dropped when its likelihood is under 0.01, its fitting deviation is over the video's threshold, or its PCA residual is over the video's PCA threshold; a frame is kept when all four pupil points and at least six points remain |
| student training set | the teacher's human frames (read from the teacher's DeepLabCut documentation file) plus the pseudo frames; validation = the teacher's 20 validation frames |
| recipe | same as the plateau models: ResNet-50, batch 2, seed 42, two learning-rate drops at 80 and 95 percent of the epochs; epochs chosen so the number of iterations matches the teacher's |
| shuffle numbers | 4000 + 10 x video + 1 (every30) or + 2 (every100) |
| evaluation | one code path for teacher and student on whole-video predictions: error on the 70 truth frames as RMSE over the labeled points divided by the teacher's median eye width, wrong = over 0.14 eye widths; test50 median pixel RMSE; share of the video flagged by the fitting finder and by the PCA finder (same thresholds as the pipeline) |
| time per student | training 11 min (video 0), 22 min (video 1), 40 min (video 2), 66 min (video 4), 71 min (video 6), 75 min (videos 7, 8) or 126 min (video 15); whole-video prediction 87 to 90 min (video 0: 22 min, 18,014 frames) |
| human-label control | the 40-label model of video 8 (shuffle 912, prediction from the plateau run) evaluated on the same code path, row "human40" |

The every100 variant stands in for the "pseudo labels at 0.3 loss weight" variant of the design: DeepLabCut's loss
has no per-image weight, and a pseudo set about a third as dense is the SGD equivalent of weighting each pseudo
frame by about a third.

## Pseudo label sets

| video | variant | candidates | kept frames | dropped frames | points dropped by fitting | points dropped by PCA | labeled points |
|---|---|---|---|---|---|---|---|
| 0 | every30 | 394 | 368 | 26 | 0 | 33 | 2943 |
| 1 | every30 | 1935 | 1784 | 151 | 8 | 825 | 14255 |
| 1 | every100 | 583 | 544 | 39 | 4 | 200 | 4345 |
| 2 | every30 | 2065 | 2020 | 45 | 10 | 200 | 16144 |
| 4 | every30 | 2068 | 1725 | 343 | 5 | 758 | 13738 |
| 6 | every30 | 2022 | 1859 | 163 | 4 | 457 | 14872 |
| 7 | every30 | 2095 | 2030 | 65 | 0 | 91 | 16239 |
| 7 | every100 | 628 | 609 | 19 | 3 | 25 | 4872 |
| 8 | every30 | 1988 | 1829 | 159 | 2 | 431 | 14623 |
| 8 | every100 | 600 | 552 | 48 | 1 | 125 | 4411 |
| 15 | every30 | 1909 | 1776 | 133 | 1 | 260 | 14207 |

No point was dropped for low likelihood. The PCA filter removes far more points than the fitting filter; on the
clean video 7 it removes almost nothing. Video 4 loses the most frames (343 of 2068): its PCA threshold is
the widest of the five videos and its per-point PCA residuals are often over it. Full tables: `videoNN/pseudo_everyNN/pseudo_summary.csv`,
`pseudo_frames.csv`, `pseudo_labels.csv`.

## Student models

| video | variant | human | pseudo | epochs | iterations | teacher iterations |
|---|---|---|---|---|---|---|
| 0 | every30 | 60 | 368 | 17 | 3638 | 3600 |
| 1 | every30 | 120 | 1784 | 8 | 7616 | 7200 |
| 1 | every100 | 120 | 544 | 22 | 7304 | 7200 |
| 2 | every30 | 220 | 2020 | 12 | 13440 | 13200 |
| 4 | every30 | 360 | 1725 | 21 | 21903 | 21600 |
| 6 | every30 | 400 | 1859 | 21 | 23730 | 24000 |
| 7 | every30 | 420 | 2030 | 21 | 25725 | 25200 |
| 7 | every100 | 420 | 609 | 48 | 24720 | 25200 |
| 8 | every30 | 420 | 1829 | 22 | 24750 | 25200 |
| 8 | every100 | 420 | 552 | 52 | 25272 | 25200 |
| 15 | every30 | 720 | 1776 | 34 | 42092 | 42000 |

## Results

`comparison.csv`, `comparison.png`. The "teacher + repair" row is the pipeline's output on the teacher's
predictions (from the committed `truth_frames.csv` of each video); it has no whole-video flagged share of its own.
The "human40" row of video 8 is the model trained with 40 instead of 20 labels of the video (shuffle 912, no pseudo
labels), the control for "more human labels instead of self-training".

| video | model | median error, % eye width | p90 | wrong of 70 | test50 median px | flagged fitting % | flagged PCA % | flagged either % |
|---|---|---|---|---|---|---|---|---|
| 0 | teacher | 2.25 | 3.37 | 0 | 12.04 | 1.00 | 1.03 | 1.22 |
| 0 | teacher + repair | 2.25 | 3.36 | 0 | | | | |
| 0 | every30 | 2.42 | 3.81 | 5 | 12.62 | 2.55 | 2.02 | 2.63 |
| 1 | teacher | 2.94 | 19.25 | 8 | 10.37 | 15.25 | 8.53 | 15.31 |
| 1 | teacher + repair | 2.90 | 5.43 | 2 | | | | |
| 1 | every30 | 2.67 | 5.91 | 2 | 10.89 | 1.33 | 0.78 | 1.34 |
| 1 | every100 | 2.94 | 4.90 | 3 | 10.98 | 2.09 | 1.62 | 2.24 |
| 2 | teacher | 1.54 | 5.19 | 4 | 7.73 | 4.17 | 2.75 | 4.49 |
| 2 | teacher + repair | 1.54 | 4.40 | 1 | | | | |
| 2 | every30 | 1.58 | 3.35 | 1 | 7.91 | 1.65 | 0.99 | 1.76 |
| 4 | teacher | 3.36 | 6.13 | 2 | 15.01 | 3.56 | 1.55 | 3.67 |
| 4 | teacher + repair | 3.36 | 6.01 | 0 | | | | |
| 4 | every30 | 3.38 | 5.38 | 0 | 16.36 | 0.98 | 0.60 | 1.07 |
| 6 | teacher | 1.49 | 2.47 | 0 | 6.75 | 5.32 | 3.13 | 5.41 |
| 6 | teacher + repair | 1.49 | 2.47 | 0 | | | | |
| 6 | every30 | 1.44 | 2.45 | 0 | 6.51 | 0.39 | 0.34 | 0.44 |
| 7 | teacher | 1.29 | 1.80 | 0 | 5.86 | 0.65 | 0.82 | 0.99 |
| 7 | teacher + repair | 1.29 | 1.72 | 0 | | | | |
| 7 | every30 | 1.34 | 1.77 | 0 | 5.97 | 0.35 | 0.40 | 0.45 |
| 7 | every100 | 1.34 | 1.81 | 0 | 6.07 | 0.37 | 0.43 | 0.49 |
| 8 | teacher | 5.02 | 11.80 | 1 | 25.34 | 8.84 | 3.15 | 8.90 |
| 8 | teacher + repair | 5.02 | 11.69 | 0 | | | | |
| 8 | every30 | 5.26 | 11.53 | 0 | 26.04 | 3.16 | 1.12 | 3.22 |
| 8 | every100 | 5.11 | 11.63 | 0 | 30.06 | 2.46 | 1.07 | 2.51 |
| 8 | human40 | 6.10 | 11.76 | 0 | 30.03 | 2.54 | 1.44 | 2.68 |
| 15 | teacher | 2.42 | 4.34 | 0 | 9.73 | 7.48 | 4.32 | 7.67 |
| 15 | teacher + repair | 2.41 | 4.34 | 0 | | | | |
| 15 | every30 | 2.39 | 4.75 | 0 | 10.25 | 2.54 | 1.59 | 2.69 |

What the table says:

- The median truth-frame error of every student is within half a percent of eye width of its teacher. Pseudo
  labels did not change the typical accuracy, in either direction, on any of the five videos.
- The wrong frames are where the students differ from their teachers. On video 1 the teacher has eight wrong
  truth frames; the every30 student has two, the every100 student three. Six of the teacher's eight are fixed by the
  every30 student, with errors falling from a fifth or a third of the eye width to a few percent; the frame with the
  largest error is only partly fixed. The student reaches the same wrong count as the pipeline's repair of the
  teacher's output, without any repair. On video 8 the teacher's one wrong frame is gone in both students. On
  video 7 nothing was wrong before and nothing is wrong after. Video 2: four wrong frames become one, and video 4:
  two become zero.
- Student and repair fix the same frames. On video 1 the student fixes the six frames the repair fixes and leaves
  the same two (61431, 62186). On video 2 the student fixes the three frames the repair fixes and leaves the same one
  (67793, a third of the eye width off in both). On video 4 both fix both. On the truth frames, self-training and the
  repair pipeline are so far redundant; neither reaches the frames the other misses.
- The 40-label control on video 8 (human40) also has zero wrong frames, with a higher median error than the
  20-label teacher and the students. Twenty more human labels and self-training reach the same wrong count here.
- The share of the video flagged by the two finders falls to a half (video 7) or to a third or less (videos 1,
  2, 4 and 8) in every student.
- The denser pseudo set (every30) is at least as good as the sparser one (every100) on all three videos.
- All-videos pass, videos 15 and 6: the teachers have no wrong truth frame and the students keep it so, with the
  median unchanged; the flagged share falls from 7.7 to 2.7 percent (video 15) and from 5.4 to 0.4 percent
  (video 6), the largest drop so far.
- Video 0 is the exception. Its teacher is the 60-label first-batch model (3600 iterations, no carried labels),
  its pseudo set is small (368 frames of an 18,014-frame video) and its PCA threshold is the tightest of all
  videos. The student has five wrong test frames where the teacher has none, all gross errors late in the video
  (a pupil or eyelid point lands a whole eye width away, or the temporal corner two thirds of an eye width away;
  frames 16250, 16323, 16911, 17205, 17719), and its flagged share doubles (1.2 to 2.6 percent). On this one
  video self-training made the model worse; the small training set and the short schedule are the likely reason,
  and the blind-test strata have not been built for it.

Per truth frame: `videoNN/<model>/truth_errors.csv`.

## Caveats

- A lower flagged share is not proof of accuracy. The finders measure how far a prediction departs from the
  fitting and from the PCA shape model; a student trained on its own predictions is expected to agree with its own
  shape model more often. Whether the unflagged frames are also right can only be settled by a by-eye check of
  student frames the teacher had flagged, which has not been done.
- On video 8 the test50 median pixel error of the every100 student is higher than the teacher's. The same number
  of the every30 student is close to the teacher's. The 50-frame test set has wide noise, but the rise is a
  reminder that the sparse variant has no advantage.
- The 70 truth frames are the plateau model's validation and test frames; pseudo frames near them were excluded,
  but the student still saw the same video. A fully independent set would be frames labeled after the fact, as in
  the blind evaluation that was paused earlier.
- Video 7's teacher is shuffle 811, which has 20 labels of video 7, although the plateau table lists video 7 with
  zero labels; all step-9 numbers of video 7 were computed on 811, so the pilot kept it.

## Later runs (2026-11-05 to 2026-11-07)

On the summary page (step 9.4) the names are: starting model = teacher, experiment 1 = `round1_every30`,
experiment 2 = `round1_repaired`, experiment 3 = `round{r}_both`, second turn of experiment 1 = `round2_every30`.

### Experiment 1 on videos 12 and 18 (every30)

| video | human | pseudo (candidates / kept / points dropped fitting, PCA) | epochs | iterations | teacher iterations | median % | wrong of 70 | flagged either % |
|---|---|---|---|---|---|---|---|---|
| 12 | 600 | 2045 / 1953 / 8, 308 | 28 | 35756 | 36000 | 1.37 → 1.48 | 0 → 0 | 4.10 → 1.77 |
| 18 | 820 | 1982 / 1490 / 4, 875 | 42 | 48510 | 49200 | 2.57 → 2.67 | 0 → 0 | 6.60 → 5.03 |

Ten videos have an experiment-1 student (0 1 2 4 6 7 8 12 15 18); nine do not (11 14 10 3 5 9 13 17 16).

### Second turn of experiment 1 (`round2_every30`, shuffle 4000 + 10 x video + 3)

Pseudo labels from the round-1 student's own whole-video prediction (finders recomputed on it, PCA median its own),
every 30th unflagged frame, same per-point filter, same iteration matching. Videos 1 and 8.

| video | pseudo (candidates / kept) | epochs | iterations | median % | wrong of 70 | flagged either % |
|---|---|---|---|---|---|---|
| 1 | 2088 / 2064 | 7 | 7644 | 2.82 | 1 | 0.98 |
| 8 | 2040 / 1994 | 21 | 25347 | 5.06 | 0 | 2.13 |

Against the first turn: video 1 wrong 2 → 1, video 8 0 → 0; flagged share 1.34 → 0.98 and 3.22 → 2.13 percent; the
blind frames unchanged (`../blind_test/`). A second turn without repair brings back only self-prediction and
saturates.

### Experiment 2: repaired frames only (`round1_repaired`, shuffle 4000 + 10 x video + 5)

`self_training.py pseudo --source repaired`: every frame the final pipeline repaired, all 8 points at the repaired
coordinates (a point not replaced is blank only when its likelihood is under 0.01), minus frames within 60 of a
truth frame, within 60 of a drawn blind frame, or inside the held-out blocks (blocks 4, 9, 14, 19 of 20, where the
supplementary blind frames were later drawn); count capped at experiment 1's count on the video (evenly spread
indices). Videos 1, 8, 15, 6 (2026-11-05/06).

| video | repaired frames | removed: held out / near truth / near blind | kept | cap | epochs | iterations | median % | wrong of 70 | flagged either % |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4885 | 559 / 515 / 571 | 1784 | 1784 | 8 | 7616 | 2.73 | 2 | 1.51 |
| 8 | 1433 | 511 / 181 / 125 | 616 | 1829 | 48 | 24864 | 5.08 | 0 | 3.35 |
| 15 | 826 | 71 / 16 / 191 | 548 | 1776 | 65 | 40560 | 2.75 | 0 | 3.37 |
| 6 | 1460 | 100 / 4 / 226 | 1130 | 1859 | 32 | 24480 | 1.52 | 0 | 3.39 |

Same wrong counts as experiment 1 on the truth frames; medians within 0.33 percent; flagged share a little higher
than experiment 1 on every video. On the blind frames the two experiments cannot be told apart (`../blind_test/`).

### Experiment 3: both kinds of frames, repeated (`round{r}_both`, shuffle 5000 + 10 x video + r)

`self_training.py pseudo --source both --round r`, `train`, `evaluate --save-flags`, `loop-status`; driver
`self_training_loop.sh`, log `loop.log`. Frames fed back = (experiment 1's rule: every 30th checked or repaired
frame, per-point filter) ∪ (every repaired frame, all points), no cap, no held-out blocks, minus frames within 60 of
a truth frame or of any of the 60 blind frames (50 drawn + 10 supplementary). Epochs = experiment 1's epochs on the
video (8 / 22 / 34 / 21), so the iterations grow with the set; every turn trains afresh from ImageNet weights.
From round 2 the previous loop model's prediction is flagged with its own PCA median and repaired with its own second
peaks (`student_repair`: `heatmap_signals.peaks_for_frames` on the flagged frames and the frames within 0.9 of either
threshold, then `flag_and_repair.py --pred-h5 --peaks-csv --dev-npy --pca-own`, output `videoNN/round{r-1}_both/repair/`,
local only). Stop rule (`videoNN/loop_status.csv`): flagged frames not below 90 percent of the previous round's, or
wrong blind frames (24 + 10) not lower, or round 3.

| video | round | fed back (grid-only + repaired) | epochs | iterations | median % | wrong of 70 | flagged either % | flagged frames (prev → now) | blind wrong 34 (prev → now) | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 5347 (1668 + 3679) | 8 | 21872 | 2.56 | 1 | 0.83 | 10987 → 598 | 17 → 4 | continue |
| 8 | 1 | 2508 (1704 + 804) | 22 | 32208 | 4.85 | 0 | 3.30 | 6383 → 2365 | 12 → 3 | continue |
| 15 | 1 | 2254 (1688 + 566) | 34 | 50218 | 2.50 | 0 | 2.77 | 5500 → 1988 | 10 → 4 | continue |
| 6 | 1 | 2921 (1725 + 1196) | 21 | | | | | 3883 → | 9 → | |

Rows are added as rounds finish. Files: `videoNN/pseudo_both[_round{r}]/`, `videoNN/round{r}_both/{eval,truth_errors,train_info}.csv`
(`flags.csv.gz`, `pred200.npy`, `repair/` local), `videoNN/loop_status.csv`, logs `loop.log`, `pseudo_v*_both_r*.log`,
`train_v*_both_r*.log`, `eval_v*_both_r*.log`, `score_v*_both_r*.log`, `status_v*_r*.log` (local).

Figures for the page: `scripts/self_training_figures.py` → `figures/loop_{truth,blind,examples,all_videos,turns}.*`.

## Files

- `videoNN/repair/`: the final pipeline re-run with `--save-coords` (local csv.gz, not committed; the summary is
  asserted equal to the committed run)
- `videoNN/pseudo_everyNN/`: pseudo label tables; the PNGs and DeepLabCut label files are in
  `local_data/self_training/` (not committed)
- `videoNN/teacher/`, `videoNN/round1_everyNN/`: `eval.csv`, `truth_errors.csv`, `train_info.csv`
- `comparison.csv`, `comparison.png`
- `video08/human40/`: the 40-label control's `eval.csv`, `truth_errors.csv`
- logs: `round1.log`, `round1b.log`, `all_videos.log`, `train_v*_every*.log`, `eval_v*_every*.log`, `pseudo_all.log` (not committed)
- models: `dlc_projects/.../dlc-models-pytorch/iteration-0/*shuffle40NN`; whole-video predictions in
  `local_data/self_training/videoNN/round1_everyNN/predictions/`
