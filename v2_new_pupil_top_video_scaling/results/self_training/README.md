# Self-training pilot, round 1

Direction 3 of the 2026-10-31 meeting: retrain the plateau model of a video on its own human labels plus
pseudo labels taken from its own predictions on frames the final pipeline passed, and see whether the retrained
model (the student) makes fewer wrong frames than the plateau model (the teacher). Design in
`docs/next_phase_analysis_2026-10-31.md`, section 3.3. Script: `scripts/self_training.py`. Run on 2026-11-02 and
2026-11-03 on the Mac (mps).

## Basic information

| item | value |
|---|---|
| videos | 1 and 8 (hard, many flagged frames), 7 (clean control) |
| teacher | the plateau model of each video: video 1 shuffle 211, video 7 shuffle 811, video 8 shuffle 911 |
| teacher training set | video 1: 120 human frames (100 of video 0 plus 20 of video 1); videos 7 and 8: 420 human frames (prior videos plus 20 of the video itself); 20 validation frames each |
| pseudo labels | the teacher's own coordinates after the final pipeline (`flag_and_repair.py --finder either --accept labelregion --points all --replace smallest`) on every 30th frame (variant every30) or every 100th frame (variant every100) |
| frames excluded from pseudo labels | frames the pipeline could not pass, and 60 frames on each side of the 70 truth frames |
| per-point filter | a point is dropped when its likelihood is under 0.01, its fitting deviation is over the video's threshold, or its PCA residual is over the video's PCA threshold; a frame is kept when all four pupil points and at least six points remain |
| student training set | the teacher's human frames (read from the teacher's DeepLabCut documentation file) plus the pseudo frames; validation = the teacher's 20 validation frames |
| recipe | same as the plateau models: ResNet-50, batch 2, seed 42, two learning-rate drops at 80 and 95 percent of the epochs; epochs chosen so the number of iterations matches the teacher's |
| shuffle numbers | 4000 + 10 x video + 1 (every30) or + 2 (every100) |
| evaluation | one code path for teacher and student on whole-video predictions: error on the 70 truth frames as RMSE over the labeled points divided by the teacher's median eye width, wrong = over 0.14 eye widths; test50 median pixel RMSE; share of the video flagged by the fitting finder and by the PCA finder (same thresholds as the pipeline) |
| time per student | training 22 min (video 1) or 75 min (videos 7, 8); whole-video prediction 87 to 90 min |

The every100 variant stands in for the "pseudo labels at 0.3 loss weight" variant of the design: DeepLabCut's loss
has no per-image weight, and a pseudo set about a third as dense is the SGD equivalent of weighting each pseudo
frame by about a third.

## Pseudo label sets

| video | variant | candidates | kept frames | dropped frames | points dropped by fitting | points dropped by PCA | labeled points |
|---|---|---|---|---|---|---|---|
| 1 | every30 | 1935 | 1784 | 151 | 8 | 825 | 14255 |
| 1 | every100 | 583 | 544 | 39 | 4 | 200 | 4345 |
| 7 | every30 | 2095 | 2030 | 65 | 0 | 91 | 16239 |
| 7 | every100 | 628 | 609 | 19 | 3 | 25 | 4872 |
| 8 | every30 | 1988 | 1829 | 159 | 2 | 431 | 14623 |
| 8 | every100 | 600 | 552 | 48 | 1 | 125 | 4411 |

No point was dropped for low likelihood. The PCA filter removes far more points than the fitting filter; on the
clean video 7 it removes almost nothing. Full tables: `videoNN/pseudo_everyNN/pseudo_summary.csv`,
`pseudo_frames.csv`, `pseudo_labels.csv`.

## Student models

| video | variant | human | pseudo | epochs | iterations | teacher iterations |
|---|---|---|---|---|---|---|
| 1 | every30 | 120 | 1784 | 8 | 7616 | 7200 |
| 1 | every100 | 120 | 544 | 22 | 7304 | 7200 |
| 7 | every30 | 420 | 2030 | 21 | 25725 | 25200 |
| 7 | every100 | 420 | 609 | 48 | 24720 | 25200 |
| 8 | every30 | 420 | 1829 | 22 | 24750 | 25200 |
| 8 | every100 | 420 | 552 | 52 | 25272 | 25200 |

## Results

`comparison.csv`, `comparison.png`. The "teacher + repair" row is the pipeline's output on the teacher's
predictions (from the committed `truth_frames.csv` of each video); it has no whole-video flagged share of its own.

| video | model | median error, % eye width | p90 | wrong of 70 | test50 median px | flagged fitting % | flagged PCA % | flagged either % |
|---|---|---|---|---|---|---|---|---|
| 1 | teacher | 2.94 | 19.25 | 8 | 10.37 | 15.25 | 8.53 | 15.31 |
| 1 | teacher + repair | 2.90 | 5.43 | 2 | | | | |
| 1 | every30 | 2.67 | 5.91 | 2 | 10.89 | 1.33 | 0.78 | 1.34 |
| 1 | every100 | 2.94 | 4.90 | 3 | 10.98 | 2.09 | 1.62 | 2.24 |
| 7 | teacher | 1.29 | 1.80 | 0 | 5.86 | 0.65 | 0.82 | 0.99 |
| 7 | teacher + repair | 1.29 | 1.72 | 0 | | | | |
| 7 | every30 | 1.34 | 1.77 | 0 | 5.97 | 0.35 | 0.40 | 0.45 |
| 7 | every100 | 1.34 | 1.81 | 0 | 6.07 | 0.37 | 0.43 | 0.49 |
| 8 | teacher | 5.02 | 11.80 | 1 | 25.34 | 8.84 | 3.15 | 8.90 |
| 8 | teacher + repair | 5.02 | 11.69 | 0 | | | | |
| 8 | every30 | 5.26 | 11.53 | 0 | 26.04 | 3.16 | 1.12 | 3.22 |
| 8 | every100 | 5.11 | 11.63 | 0 | 30.06 | 2.46 | 1.07 | 2.51 |

What the table says:

- The median truth-frame error of every student is within half a percent of eye width of its teacher. Pseudo
  labels did not change the typical accuracy, in either direction, on any of the three videos.
- The wrong frames are where the students differ from their teachers. On video 1 the teacher has eight wrong
  truth frames; the every30 student has two, the every100 student three. Six of the teacher's eight are fixed by the
  every30 student, with errors falling from a fifth or a third of the eye width to a few percent; the frame with the
  largest error is only partly fixed. The student reaches the same wrong count as the pipeline's repair of the
  teacher's output, without any repair. On video 8 the teacher's one wrong frame is gone in both students. On
  video 7 nothing was wrong before and nothing is wrong after.
- The share of the video flagged by the two finders falls to a half (video 7) or to a third or less (videos 1
  and 8) in every student.
- The denser pseudo set (every30) is at least as good as the sparser one (every100) on all three videos.

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

## Status

Round 1 is complete on all three videos and two variants. Round 2 (pseudo labels from the student) was not
started; the design asked for a second round only when the first lowers the median error and the wrong count,
which happened on video 1 only. The decision is open for the next meeting.

## Files

- `videoNN/repair/`: the final pipeline re-run with `--save-coords` (local csv.gz, not committed; the summary is
  asserted equal to the committed run)
- `videoNN/pseudo_everyNN/`: pseudo label tables; the PNGs and DeepLabCut label files are in
  `local_data/self_training/` (not committed)
- `videoNN/teacher/`, `videoNN/round1_everyNN/`: `eval.csv`, `truth_errors.csv`, `train_info.csv`
- `comparison.csv`, `comparison.png`
- logs: `round1.log`, `train_v*_every*.log`, `eval_v*_every*.log`, `pseudo_all.log` (not committed)
- models: `dlc_projects/.../dlc-models-pytorch/iteration-0/*shuffle40NN`; whole-video predictions in
  `local_data/self_training/videoNN/round1_everyNN/predictions/`
