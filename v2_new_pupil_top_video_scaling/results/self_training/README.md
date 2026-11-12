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
| 15 | 2055 / 1958 | 32 | 42528 | 2.46 | 0 | 2.23 |
| 6 | 2103 / 2021 | 20 | 24220 | 1.43 | 0 | 0.35 |

Against the first turn: video 1 wrong 2 → 1, video 8 0 → 0; flagged share 1.34 → 0.98 and 3.22 → 2.13 percent; the
blind frames unchanged (`../blind_test/`). A second turn without repair brings back only self-prediction and
saturates.

Blind band of experiment 1 (checked 2026-11-10): experiment 1's pseudo frames avoid only the truth band, not the
blind band (the blind frames were drawn after experiment 1, excluding its pseudo frames exactly). No pseudo frame of
either turn is a blind frame, but of the 60 blind frames, those with a pseudo frame within 60 / within 15 frames:
video 1 turn 1 41 / 23, turn 2 56 / 43; video 8 52 / 29 and 56 / 40; video 15 turn 1 50 / 21; video 6 turn 1 56 / 34.
Experiments 2 and 3 exclude the band (none within 60, one frame on video 15 in experiment 2). Experiment 1's blind
counts are therefore measured on frames whose neighbours it trained on.

Video 15, second turn (2026-11-11, `tools/earlier_round2.sh`): wrong truth 0 → 0, flagged share 2.69 → 2.23 %,
median 2.39 → 2.46, blind 3 → 4 of 34 (24: 2 → 2, supplement: 1 → 2). Blind band: 51 / 34 of the 60 blind frames have
a pseudo frame within 60 / 15, and one blind frame is itself a pseudo frame: frame 17550 (`img037.png`, stratum A, one
of the 24 scored). The teacher flagged it, so turn 1 did not take it and the blind selection (which excluded turn 1's
pseudo frames) kept it; the round-1 student does not flag it, so turn 2 takes it as an every-30th frame. Every model's
error on it is 0.074 to 0.077 eye widths (right), so no wrong count changes; the frame stays in both sets (rule unchanged).

Video 6, second turn (2026-11-11): training (shuffle 4063, 20 epochs, `snapshot-best-020.pt`) finished at 12:26 after
the user's 11:58 stop, and its prediction was stopped at its start; the prediction was then run from the saved snapshot
with the same command train() hands over to (13:14 to 14:42), and `tools/finish_round_then_stop.sh` evaluated and
scored it. Wrong truth 0 → 0, flagged share 0.44 → 0.35 %, median 1.44 → 1.43, blind 4 → 2 of 34 (24: 3 → 2,
supplement: 1 → 0). The two stratum-B frames wrong in turn 1 (`img024.png`, `img030.png`) are still wrong; `img031.png`
(stratum D) and `img105.png` (supplement, E_t) are right in turn 2; no blind frame became wrong. Blind band: 59 / 37 of
the 60 blind frames have a pseudo frame within 60 / 15 (turn 1: 56 / 34); no blind frame is a pseudo frame.

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

### Second turn of experiment 2 (`round2_repaired`, shuffle 4000 + 10 x video + 7, 2026-11-10)

User 2026-11-09: after the final loop, the second turn of experiments 1 to 3 on videos 1, 8, 15, 6, to see which kind
of frames helps (driver `tools/earlier_round2.sh`, log `earlier_round2.log`). The round-1 repaired-only student's
whole-video prediction is flagged with its own PCA median and repaired with its own second peaks, as experiment 3 does
from round 2 (`student_repair`, same pipeline and acceptance as experiment 2's first turn); then experiment 2's rule
unchanged: every repaired frame, all points, minus the held-out blocks, the truth band and the blind band, capped at
experiment 1's first-turn count; iterations matched to the teacher.

| video | flagged by the round-1 student | repaired | removed: held out / near truth / near blind | kept | epochs | iterations | median % | wrong of 70 | flagged either % | blind wrong 24 + 10 (turn 1 → turn 2) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1086 | 206 | 13 / 8 / 92 | 93 | 65 | 6955 | 2.70 | 1 | 2.17 | 4 + 0 → 1 + 2 |
| 8 | 2403 | 47 | 4 / 0 / 21 | 22 | 117 | 25857 | 5.26 | 1 | 5.27 | 2 + 0 → 1 + 4 |
| 15 | 2417 | 158 | 31 / 1 / 68 | 58 | 108 | 40932 | 2.39 | 0 | 4.10 | 3 + 1 → 1 + 1 |
| 6 | 2429 | 563 | 184 / 47 / 26 | 306 | 70 | 24710 | 1.51 | 0 | 2.15 | 2 + 3 → 4 + 0 |

Video 1 against experiment 2's first turn: median 2.73 → 2.70, wrong truth frames 2 → 1, flagged share 1.51 → 2.17 %,
blind 4 → 3 of 34. Only 93 repaired frames are fed back (1784 in the first turn), so the student trains on the human
labels almost alone, with 65 epochs to match the teacher's iterations. Rows are added as the students finish.

Video 8 against experiment 2's first turn: median 5.08 → 5.26, wrong truth frames 0 → 1, flagged share 3.35 → 5.27 %,
blind 2 → 5 of 34 (24: 2 → 1, supplement: 0 → 4). The round-1 student repairs only 47 of its 2403 flagged frames and 22
remain after the exclusions (616 in the first turn), so the student trains on the human labels almost alone, with 117
epochs to match the teacher's iterations.

Video 15 against experiment 2's first turn: median 2.75 → 2.39, wrong truth frames 0 → 0, flagged share 3.37 → 4.10 %,
blind 4 → 2 of 34 (24: 3 → 1, supplement: 1 → 1). 158 of the round-1 student's 2417 flagged frames are repaired and 58
remain after the exclusions (548 in the first turn); 108 epochs to match the teacher's iterations.

Video 6 against experiment 2's first turn (2026-11-12, resumed 2026-11-11 21:36): median 1.52 → 1.51, wrong truth
frames 0 → 0, flagged share 3.39 → 2.15 %, blind 5 → 4 of 34 (24: 2 → 4, supplement: 3 → 0). 563 of the round-1
student's 2429 flagged frames are repaired and 306 remain after the exclusions (1130 in the first turn), the most of
the four videos; 70 epochs to match the teacher's iterations. In the 24 scored frames the two wrong in turn 1
(`img030.png` B, `img042.png` A) stay wrong and `img011.png` (D) and `img024.png` (B) become wrong; the three
supplement frames wrong in turn 1 (`img102`, `img106`, `img108`, all F_n) are right in turn 2.

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
wrong blind frames (24 + 10) not lower (the number of rounds is not a criterion; the driver's MAXR only limits one launch).

| video | round | fed back (grid-only + repaired) | epochs | iterations | median % | wrong of 70 | flagged either % | flagged frames (prev → now) | blind wrong 34 (prev → now) | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 5347 (1668 + 3679) | 8 | 21872 | 2.56 | 1 | 0.83 | 10987 → 598 | 17 → 4 | continue |
| 8 | 1 | 2508 (1704 + 804) | 22 | 32208 | 4.85 | 0 | 3.30 | 6383 → 2365 | 12 → 3 | continue |
| 15 | 1 | 2254 (1688 + 566) | 34 | 50218 | 2.50 | 0 | 2.77 | 5500 → 1988 | 10 → 4 | continue |
| 6 | 1 | 2921 (1725 + 1196) | 21 | | | | | 3883 → | 9 → | |
| 1 | 2 | 1922 (1903 + 19) | 8 | 8168 | 2.63 | 1 | 1.08 | 598 → 774 | 4 → 4 | stop |
| 8 | 2 | 1779 (1759 + 20) | 22 | 24200 | 5.93 | 0 | 2.10 | 2365 → 1507 | 3 → 4 | stop |
| 15 | 2 | 1789 (1745 + 44) | 34 | 42330 | 2.45 | 0 | 2.37 | 1988 → 1697 | 4 → 6 | stop |
| 6 | 1 | 2921 (1725 + 1196) | 21 | 34881 | 1.47 | 0 | 0.54 | 3883 → 387 | 9 → 3 | continue |
| 6 | 2 | 1892 (1886 + 6) | 21 | 24066 | 1.50 | 0 | 0.41 | 387 → 295 | 3 → 2 | continue (round 3 not run) |
| 6 | 3 | 1910 (1907 + 3) | 21 | 24255 | 1.49 | 0 | 0.37 | 295 → 263 | 2 → 3 | stop |

Second turn (2026-11-10, run with the second turns of experiments 1 and 2, `tools/earlier_round2.sh`): video 1 round
2 against round 1: flagged frames 598 → 774, blind 4 → 4, truth 1 → 1, median 2.56 → 2.63; both conditions fire, so the
loop stops at round 2 (19 repaired frames fed back). The status step failed in the driver: since commit 584c710 (5b)
`loop-status` takes `--source` with default `pipeline`, and `self_training_loop.sh` did not pass it; it was re-run by
hand with `--source both` (rule unchanged) and the driver now passes it. `compare` fails as well (the palette is too
short for the new models; summary figure only, not used by the page).

Video 8 round 2 against round 1 (prediction resumed 2026-11-10 22:50 after the user's stop at 08:17; the model trained
in the morning was used unchanged): flagged frames 2365 → 1507 (64 %), blind 3 → 4 (24: 2 → 2, supplement: 1 → 2),
truth 0 → 0, median 4.85 → 5.93; the blind condition fires, so the loop stops at round 2 (20 repaired frames fed back).

Video 15 round 2 against round 1: flagged frames 1988 → 1697 (85 %), blind 4 → 6 (24: 2 → 3, supplement: 2 → 3),
truth 0 → 0, median 2.50 → 2.45; the blind condition fires, so the loop stops at round 2 (44 repaired frames fed back).

Video 6 is not run: the user stopped the chain on 2026-11-11 at 11:58 ("不损失的前提下，尽快停止") during video 6's
second turn of experiment 1 (shuffle 4063); that training is let finish and its prediction is stopped at its start, so
video 6 has no second-turn result yet for any experiment (its experiment 3 round 1 is trained, not predicted).

Update 2026-11-11 14:43: the user resumed at 13:14 ("继续跑") and at 13:22 asked to finish the running round and stop
("最近一轮跑完不损失尽快结束"). Only video 6's second turn of experiment 1 was finished (row and paragraph in that
section). Video 6's second turn of experiment 2 and its experiment 3 (round 1 prediction, round 2) are not run.

Video 6 round 1 (2026-11-12 01:51, after the user's "继续开始跑" on 2026-11-11 21:36): the model trained on 2026-11-07
(shuffle 5061, final `snapshot-021.pt`) was predicted 00:24 to 01:50 by `earlier_round2.sh` before the loop driver.
Against the starting model: flagged frames 3883 → 387, blind 9 → 3 (24: 8 → 2, supplement: 1 → 1), truth 0 → 0,
median 1.49 → 1.47; neither condition fires, so the loop continues to round 2 (training since 01:53, shuffle 5062).
The video 6 round 1 row with blanks is the state on 2026-11-07 (trained, not predicted); the full row is the second to last.

Video 6 round 2 against round 1 (2026-11-12 04:33): flagged frames 387 → 295 (76 %), blind 3 → 2 (24: 2 → 2,
supplement: 1 → 0), truth 0 → 0, median 1.47 → 1.50; neither condition fires, so the decision is continue, the only
video of the four where it is. Round 3 was not run: this launch's budget was MAXR = 2 (the earlier-round-2 scope is two
turns), and the user asked to stop before 08:00. 6 repaired frames fed back. In the 24 scored frames `img033.png` (A)
becomes right and `img024.png` (B) becomes wrong, `img030.png` (B) stays wrong; supplement `img107.png` (F_t) becomes
right. `compare` failed again (palette, summary figure only).

Video 6 round 3 against round 2 (user 2026-11-12 10:26 "继续跑"; launch VIDEOS=6, round 4 not to be started, user:
"第三圈都不着急开始"): training 10:28 to 11:42 (shuffle 5063), prediction to 13:09. The evaluation at 13:09 failed
because macOS took the shell's Desktop access away (PermissionError at import); after access was restored it was
re-run at 14:31 on the same prediction (`self_training_loop.sh` VIDEOS=6 MAXR=3). Flagged frames 295 → 263 (89 %,
below the 90 % line), blind 2 → 3 (24: 2 → 3, supplement: 0 → 0), truth 0 → 0, median 1.50 → 1.49; the blind
condition fires, so the loop stops at round 3 (3 repaired frames fed back). In the 24 scored frames `img033.png` (A)
becomes wrong again (right in round 2 only); `img024.png` and `img030.png` (B) stay wrong. All four videos have now
stopped by the rule (1, 8, 15 at round 2, 6 at round 3).

Rows are added as rounds finish. Files: `videoNN/pseudo_both[_round{r}]/`, `videoNN/round{r}_both/{eval,truth_errors,train_info}.csv`
(`flags.csv.gz`, `pred200.npy`, `repair/` local), `videoNN/loop_status.csv`, logs `loop.log`, `pseudo_v*_both_r*.log`,
`train_v*_both_r*.log`, `eval_v*_both_r*.log`, `score_v*_both_r*.log`, `status_v*_r*.log` (local).

### Noise band of the stop rule: the same round with another seed (`round1_both_seed43`, 2026-11-08)

The stop rule's 90 percent was set at design time. To give it a basis, video 8's round 1 (source both, same 2508
fed-back frames, same 22 epochs) was trained again with seed 43 instead of 42 (`self_training.py train --seed 43
--shuffle 7081 --tag seed43`, then `evaluate --save-flags`; driver `seed_experiment.sh`, log `seed_experiment.log`).
Everything else is identical, so the difference between the two models is the training noise alone.

| model | median % | p90 % | wrong of 70 | flagged fit % | flagged PCA % | flagged either % | flagged frames | blind wrong (24 + 10) |
|---|---|---|---|---|---|---|---|---|
| teacher (40 labels) | 5.02 | 11.80 | 1 | 8.84 | 3.15 | 8.90 | 6383 | 12 (main set) |
| round1_both, seed 42 | 4.85 | 11.65 | 0 | 3.26 | 0.88 | 3.30 | 2365 | 2 + 1 |
| round1_both, seed 43 | 4.92 | 11.32 | 0 | 2.98 | 0.84 | 3.00 | 2150 | 2 + 0 |

- Flagged share: 3.00 against 3.30 percent, a ratio of 0.91. The two seeds flag different frames: of the fitting
  flags (2341 and 2137) only 1106 are shared; of the PCA flags (631 and 600) 561 are shared.
- Truth frames: per-frame errors differ by 0.08 percent of the eye width in the median (largest 5.6); seed 43 is
  better by more than 1 percent on 4 frames and worse on none; both have 0 wrong frames.
- Blind frames (`round1_both_seed43/blind_scores.csv`, scored with the same function as `blind_test/video08/`, the
  committed blind tables untouched): median 6.02 against 6.30 percent on the 24 main frames, 20 of 24 within 1
  percent of each other; wrong frames 2 against 2 on the main set and 0 against 1 on the supplement, so the blind
  wrong count moves by one frame from the seed alone.

Reading for the stop rule: one training's flagged share differs from another's by about 9 percent of itself when
only the seed changes (one pair measured; the round-to-round drop teacher → round 1 was 63 percent). The 90 percent
line therefore sits at the edge of the seed noise: a round whose flagged share stays above 90 percent of the previous
round's cannot be told from no change, which is what the rule means by stopping. The blind criterion "wrong frames
not lower" has a noise of one frame on 34 frames and should not stop the loop on its own when the flagged share still
falls; the formal loop keeps both criteria and records this band next to them.

### Experiment 4: repair of the closed-eye frames (2026-11-08)

The closed-eye branch of the repair, audited and tested in three parts after experiment 3 (user's naming, 2026-11-08).
Details and tables in `results/flag_and_repair/README.md`, items 3, 3b, 3c, 3d.

**4a. The closed-eye rule.** The rule in use called a frame closed when the eyelid opening was below half the video's
median (hand-set; the 1330 truth frames hold no truly closed eye, so it could not be derived from labels). The user
judged a stratified sample of 80 frames: 33 fully closed, 30 open with wrong eyelid or corner points, 16 half open
with a labelable pupil. The opening ratio does not separate closed from open eyes (no line better than 61 %); the mean
heatmap confidence of the four pupil points does: confidence < 0.40 is right on 92 % of the judged frames (32 of 33
closed caught, 5 of 46 open called closed) and on the 1318 labeled truth frames calls 4 closed, all already wrong.
Rule: closed = mean pupil confidence < 0.40 (`flag_and_repair.py --closed confidence`). Under it the 19 videos have
7358 closed frames (8058 before), 15,878 repaired (14,395), 4404 flagged and not repaired (6694); truth frames 16 of
23 wrong repaired (15), 2 still flagged (4), 1 made worse (1).

**4b. The reopened repairs judged.** 2008 frames the opening rule called closed go through the ordinary repair under
the confidence rule and pass the acceptance (video 15: 846, video 11: 628, video 1: 272). On 20 of them, one per
episode, the user found the repair right with every point in place in 18, one frame with a further uncaught wrong
point, one wrong; no truly closed eye among them. Same quality as the repair's own check by eye (87 %).

**4c. A student on the reopened frames only** (`round1_reopened`, shuffle 8000 + 10 x video + 1). Like experiment 2, a
student is trained on the human labels plus only these frames (`self_training.py pseudo --source reopened`: the
confidence run re-made into `videoNN/repair_confclosed/` with `--save-coords` and checked against the committed run;
all eight points as accepted, a non-replaced point dropped only below the likelihood bound; excluded: the truth band
and the blind band; no cap, no held-out blocks), iterations matched to the teacher. Evaluation as in experiments 1 to 3
(70 truth frames, flagged share, blind frames where a set exists), plus the number of frames each rule calls closed on
the student's prediction. Driver `exp4_reopened.sh`, log `exp4.log`; videos 15, 11, 1 in turn. Figure `figures/loop_exp4.png`
(`self_training_figures.py --audit`).
Like experiments 1 to 3, 4c repairs with the subset order in use (A); only the closed-eye rule is changed, so the
student is compared with the same video's teacher and `round1_both` on one changed variable. The subset rule D chosen
in the audit is used in the formal loop only, where every settled piece is applied together (user, 2026-11-08).

| video | reopened frames | fed back (after the bands) | epochs | iterations | median % | wrong of 70 | flagged either % | closed (opening / confidence rule) | blind wrong | 
|---|---|---|---|---|---|---|---|---|---|
| 15 | 846 | 660 (1 near truth, 185 near blind) | 60 | 40800 | 2.50 | 0 | 4.41 | teacher 3403 / 901, student 3978 / 942 | 3 + 2 |
| 11 | 628 | 512 (116 near truth, no blind set) | 65 | 35490 | 2.15 (teacher 2.12) | 0 (teacher 4) | 0.73 (teacher 4.02) | teacher 683 / 73, student 80 / 28 | no blind set |
| 1 | 272 | 98 (174 near truth or blind) | 65 | 7085 | 2.88 (teacher 2.94) | 1 (teacher 8) | 7.37 (teacher 15.31) | teacher 474 / 398, student 177 / 343 | 4 + 4 (teacher 11 + 6) |

Video 15, the reopened frames left out of the training set (186, within 60 frames of a truth or blind frame): the
teacher flags all of them (a repaired frame is a flagged frame), the student 3. They are narrowed eyes with right
eyelid points and the left pupil point thrown outside the eye: left-to-right pupil distance 1.15 eye widths on the
teacher's prediction, 0.26 on the student's. The opening rule still calls 177 of the 186 closed on the student's
prediction (the eye is narrowed), the confidence rule none; so the whole-video closed counts below are not the measure
of 4c, the held-out reopened frames are (figure `loop_exp4c.png`, panel b).

Video 11 (2026-11-08 11:12): the reopened frames are nearly all nasal-corner replacements (592 of 628 alone, 28 with a
pupil point): the corner thrown away widens the eye (corner distance 1.27 x the video median on the teacher) and the
opening, measured in eye widths, looks collapsed. Of the 116 held-out reopened frames the teacher flags all, the
student none; corner distance 0.96 x median, opening rule closed 0 of 116. Whole video: wrong truth frames 4 to 0
(7 frames better by more than 1 %, none worse), flagged 4.02 to 0.73 %, opening-rule closed 683 to 80, confidence-rule
closed 73 to 28. The teacher's evaluation of video 11 was made for this comparison (`evaluate --teacher --save-flags`).

Video 1 (2026-11-08 13:02): the reopened frames are temporal-corner replacements (249 of 272 alone, 20 with the left
pupil point); only 98 enter the training set, 174 lie within 60 frames of a truth or blind frame. Of those 174 held-out
frames the teacher flags all, the student 3; opening-rule closed on them 174 to 0. Wrong truth frames 8 to 1 (14 frames
better by more than 1 %, 3 worse), flagged 15.31 to 7.37 %, blind wrong 17 to 8 of 34 (`round1_reopened/blind_scores.csv`).
Experiment 4c is complete: on all three videos the student beats its teacher and stops flagging the held-out reopened
frames; the repaired closed-eye frames go into the final loop's training set.

Held-out reopened frames still flagged by the other students of the same video (not shown on the page): video 1
(174) every30 4, repaired only 11, experiment 3 turn 1 12, reopened only 3; video 15 (186) every30 13, repaired only
22, experiment 3 turn 1 39, reopened only 3. The students without these frames already fix most of them; the reopened
frames fix the rest. Truth frames changed by more than 1 % of the eye width (reopened student against teacher): video 1
14 better / 3 worse, video 11 7 / 0, video 15 3 / 10 (none crossing the error line): trained on one kind of frame
alone the student shifts slightly on ordinary frames, as in experiment 2, so these frames belong mixed with the grid
frames in the final loop, not on their own.

Video 15 against the other students of the same video (`videoNN/round1_*/eval.csv`, blind `round1_reopened/blind_scores.csv`,
closed counts from the prediction with both rules): teacher median 2.42 %, p90 4.34, flagged 7.67 %, blind wrong 7 + 3;
every30 2.39 / 4.75 / 2.69 / 2 + 1; repaired only 2.75 / 6.13 / 3.37 / (not scored); both 2.50 / 4.99 / 2.77 / 2 + 2;
reopened only 2.50 / 5.88 / 4.41 / 3 + 2. The 660 reopened frames do improve on the teacher (flagged share 7.67 to
4.41 %, blind wrong 10 to 5 of 34) but less than the grid frames or the repaired frames, and the number of frames the
student's prediction puts into a closed eye does not fall under either rule (opening rule 3403 to 3978, confidence rule
901 to 942). On this video the reopened frames are not a substitute for the other pseudo labels; whether they add
anything on top of them is a question for the formal loop (both sources together), not for this experiment.

Rows are added as the students finish. Files: `videoNN/pseudo_reopened/`, `videoNN/round1_reopened/{eval,truth_errors,train_info}.csv`.

### Experiment 5: the final loop version (2026-11-08)

Experiments 1 to 4 each prove one part of the loop; experiment 5 puts the settled parts together (user's naming).

**5a. Choosing the repair's subset rule.** The four variants A (order in use), B (largest deviation first), C (B with
the largest acceptance margin within the smallest passing size), D (largest margin over every size) and their
evaluation are in `results/flag_and_repair/README.md`, items 1, 1b, 1c. Summary: identical on the 1330 truth frames
(15 / 4 / 4 / 1), also with leave-one-video-out ranges; on the whole videos the same 14,395 frames repaired, 15,159
points replaced by A, B, C and 15,552 by D, replaced point sets differing in 15 (A vs B), 26 (B vs C), 374 (C vs D)
frames; judged by eye on 53 frames, every point right for D in 38 (72 %), C 24, B 10, A 6. Decision: D
(`flag_and_repair.py --order deviation --tiebreak margin-all`) in the final loop; the earlier sections keep A.
Figure `figures/loop_exp5a.png` and the example sheet `loop_exp5a_examples.jpg` (`self_training_figures.py --audit`).

**5b. The loop in its final version** (`self_training.py --source formal`, shuffle 6000 + 10 x video + round, model
`round{r}_formal`, driver `self_training_formal.sh`, log `formal_loop.log`; started 2026-11-08 14:05 on videos 1, 8,
15, 6, user's choice). Experiment 3's procedure with the settled repair (`PIPE_FORMAL` = the pipeline options plus
`--order deviation --tiebreak margin-all --closed confidence`); finders, error line, ranges and second peak unchanged.
Differences from experiment 3 besides the repair options: (1) from round 2 the previous model's prediction is flagged
with its OWN eye width, exactly as `flag_and_repair.py` judges it, and the second peaks are computed on the flagged
frames only; a flagged frame found without peaks is computed on demand and the repair rerun (the 0.9 margin of
experiment 3 is gone, user 2026-11-08); (2) `evaluate` records two independent detectors the loop never uses to pick
or repair frames, as a check that a falling flagged share means fewer errors: the share of frames whose lowest point
confidence is below the starting model's 5 % quantile, and the share whose largest frame-to-frame move exceeds the
starting model's 95 % quantile (`indep_lowconf_pct`, `indep_jump_pct`; the starting model is 5.0 / 5.0 by
construction); (3) `loop-status --source formal` writes `videoNN/loop_status_formal.csv` with those columns and the
wrong truth frames beside the stop rule (flagged frames not below 90 % of the previous round's, the blind count
watched; neither monitoring column stops the loop). Round 0 of the formal loop = `videoNN/repair_formal/` (the
settled repair on the starting model's prediction; summary and whole_video committed). Blind frames stay 24 + 10
(user: no further labeling before the loop).

| video | round | fed back (grid + repaired) | epochs | iterations | median % | wrong of 70 | flagged either % | low confidence % | jump % | blind wrong 34 (prev → now) | flagged frames (prev → now) | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 5440 (1668 + 3772) | 8 | 22240 | 2.64 | 1 | 0.80 | 0.70 | 0.42 | 17 → 4 | 10987 → 575 | continue |
| 1 | 2 | 1922 (1908 + 14) | 8 | 8168 | 2.70 | 2 | 0.97 | 0.50 | 0.56 | 4 → 5 | 575 → 698 | stop |
| 8 | 1 | 2498 (1703 + 795) | 22 | 32098 | 5.08 | 0 | 3.41 | 0.86 | 1.09 | 12 → 2 | 6383 → 2449 | continue |
| 8 | 2 | 1794 (1759 + 35) | 22 | 24354 | 4.95 | 0 | 2.56 | 0.94 | 1.05 | 2 → 4 | 2449 → 1838 | stop |
| 15 | 1 | 2934 (1734 + 1200) | 34 | 61778 | 2.32 | 0 | 2.70 | 2.08 | 1.84 | 10 → 4 | 5500 → 1935 | continue |
| 15 | 2 | 1867 (1832 + 35) | 34 | 43656 | 2.46 | 0 | 1.77 | 2.02 | 1.29 | 4 → 3 | 1935 → 1269 | continue |
| 15 | 3 | 1880 (1850 + 30) | 34 | 43860 | 2.56 | 0 | 1.71 | 1.92 | 1.26 | 3 → 3 | 1269 → 1228 | stop |
| 6 | 1 | 2922 (1725 + 1197) | 21 | 34881 | 1.42 | 0 | 0.45 | 0.32 | 1.24 | 9 → 5 | 3883 → 322 | continue |
| 6 | 2 | 1855 (1851 + 4) | 21 | 23688 | 1.45 | 0 | 0.42 | 0.29 | 0.88 | 5 → 3 | 322 → 298 | stop |

Rows are added as rounds finish.

Against experiment 3's first round of the same video (`round1_both`, same epochs; the two detectors computed for it on the same
thresholds, not stored in its eval.csv): video 1 median 2.56, wrong 1, flagged 0.83 %, low confidence 0.58 %, jump 0.42 %,
blind 4, flagged frames 598, fed back 5347 (1668 + 3679). The final round 1 is the same within the seed band (93 more
repaired frames, mostly the closed-eye ones). Closed frames under the confidence rule (`tools/closed_counts.py`): starting
model 398, experiment 3 round 1 233, final round 1 267; under the opening rule 474, 164, 102.
Video 8, against experiment 3 round 1: median 4.85, wrong 0, flagged 3.30 %, low confidence 0.78 %, jump 1.06 %, blind 3,
flagged frames 2365, fed back 2508 (1704 + 804); the same within the seed band. Video 8 has almost no reopened frames, so
this is subset rule D's effect alone: none measurable. Closed frames under the confidence rule: starting model 880,
experiment 3 round 1 510, final round 1 518; under the opening rule 213, 211, 221.
Video 15, against experiment 3 round 1: median 2.50, wrong 0, flagged 2.77 %, low confidence 2.37 %, jump 1.80 %, blind 4,
flagged frames 1988, fed back 2254 (1688 + 566). Blind, truth and flagged frames are the same within the seed band; the
median is 0.18 lower (the one seed pair differed by 0.07). The final round feeds back 634 more repaired frames, about
the 660 reopened frames experiment 4 found on this video: with the confidence rule 901 frames count as closed instead of
3403, and the whole-video repair (`repair_formal/`) repairs 1630 frames instead of 826. Closed frames under the
confidence rule: starting model 901, experiment 3 round 1 757, final round 1 654; under the opening rule 3403, 3268, 3579.
Video 6: experiment 3's round 1 was trained but never predicted (stopped by the user on 2026-11-07), so only against
the starting model: median 1.49, wrong 0, flagged 5.41 %, blind 9, flagged frames 3883. Its fed-back set is almost
experiment 3's (2921 = 1725 + 1196): the video has few closed frames (129 under the confidence rule, 122 under the
opening rule), so the confidence rule changes little, and the whole-video repair repairs 1471 frames against 1460.
Closed frames under the confidence rule: starting model 129, final round 1 107; under the opening rule 122, 119.
Video 1, round 2 against round 1: flagged frames 575 → 698, blind 4 → 5, truth 1 → 2, median 2.64 → 2.70; the
low-confidence detector falls (0.70 → 0.50 %), the jump detector rises (0.42 → 0.56 %). The stop rule fires (698 not
below 90 % of 575, blind not lower), so video 1's loop stops at round 2. The blind and truth changes are one frame each,
as much as the seed alone moves them; the flagged frames rise by 21 %, beyond the seed band. Round 2's set is smaller:
the round-1 model flags only 575 frames, so 14 repaired frames are fed back against round 1's 3772, and with the same
8 epochs it trains 8168 iterations against 22240. Rounds 2 of videos 8, 15 and 6 were not run in this launch (the
driver's estimate for video 8 ended after the 2026-11-09 08:00 deadline).
Relaunched without a deadline on 2026-11-09 08:57 (`self_training_formal.sh`, finished rounds skipped).
Video 8, round 2 against round 1: flagged frames 2449 → 1838 (75 %), blind 2 → 4, truth 0 → 0, median 5.08 → 4.95;
low confidence 0.86 → 0.94 %, jump 1.09 → 1.05 %. The stop rule fires on the blind count (4 not below 2), not on the
flagged frames, so video 8's loop stops at round 2; the seed alone moved this video's blind count by one frame.
35 repaired frames are fed back against round 1's 795.
Video 15, round 2 against round 1: flagged frames 1935 → 1269 (66 %), blind 4 → 3, truth 0 → 0, median 2.32 → 2.46;
low confidence 2.08 → 2.02 %, jump 1.84 → 1.29 %. Both conditions say continue; 35 repaired frames are fed back
against round 1's 1200.
The Mac restarted at about 18:01 on 2026-11-09 and cut video 6's round 2 at epoch 16 of 21; the interrupted shuffle 6062
was moved aside to `local_data/interrupted/shuffle6062_cut_20261109/` (not deleted) and round 2 retrained from scratch
from 22:13 (user: all four videos finish round 2; `tools/retrain_v6_r2_after_driver.sh`).
Video 15, round 3 against round 2: flagged frames 1269 → 1228 (97 %), blind 3 → 3, truth 0 → 0, median 2.46 → 2.56;
low confidence 2.02 → 1.92 %, jump 1.29 → 1.26 %. Both conditions fire, so video 15's loop stops at round 3; 30 repaired
frames are fed back.
Video 6, round 2 against round 1 (the retrained run): flagged frames 322 → 298 (93 %), blind 5 → 3, truth 0 → 0, median
1.42 → 1.45; low confidence 0.32 → 0.29 %, jump 1.24 → 0.88 %. The flagged condition fires while the blind count falls,
so video 6's loop stops at round 2; 4 repaired frames are fed back. All four loops have now stopped: videos 1, 8 and 6
at round 2, video 15 at round 3.

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
