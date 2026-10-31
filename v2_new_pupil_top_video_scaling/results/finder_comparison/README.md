# Which finder of unreliable frames: every signal at the error line, fitting against the PCA, 2026-10-31

Follow-up of `results/fit_threshold/` (report 9.2). The finder was fixed as the fitting deviation on 2026-10-30 because it
found every wrong frame at the error line. This folder asks whether anything else does as well at that line, compares the
two best on every frame of the 19 videos and by eye, and settles how the PCA finder's threshold has to be set.

## Basic information

| item | value |
|---|---|
| videos | all 19 (0–18), plateau model of each, every frame predicted (1 309 478 frames) |
| truth frames | 1330 (50 test + 20 validation per video); wrong frame = error > 0.14 of the eye width (`results/error_tier/tier.csv`), 23 frames |
| signals | `results/unreliable_frames_sweep/signals_all_frames.csv` (DeepLabCut jump / uncertain / fitting and variants, two-seed ensemble, geometry, PCA reconstruction error, combinations) and the trained detectors of `results/error_detector/scores_lovo.csv` |
| fitting finder | largest of 8 one-step SARIMAX (2,0,0) deviations / eye width, threshold = the line 0.14 |
| PCA finder | PCA (99 % variance, ≥ 2 components) of the labeled poses the plateau model was trained on (17–73 per video); frame score = RMS of the 8 reconstruction residuals / eye width; threshold = 3.93 × the video's median score (`pca_threshold_rule.csv`) |
| unit | eye width = median corner distance of the video's predictions |

## 1. Every signal at the line (`scripts/finder_comparison.py`, `signals_at_line.csv/.png`)

Threshold of each signal set to just find all 23 wrong frames; share of the 1307 right frames flagged: PCA of the video's
training poses 0.0 %, fitting mean of 8 (DLC default) 0.3 %, fitting largest of 8 order (1,0,0) 0.6 %, trained detector D1
0.6 %, fitting largest of 8 order (2,0,0) (the one used) 1.0 %, other fitting forms 1.2–1.7 %; two-seed disagreement 8.7 %,
PCA of the mouse's poses 15 %, lowest confidence 25 %, mean confidence 72 %, jump 86 %, geometry 75–91 %. Leave-one-video-out
finds 22 of 23 for most of the first tier (20 for the form used). With 23 wrong frames the first tier cannot be ranked.

## 2. Fitting against the PCA on whole videos (`whole_video_share.csv/.png`, fixed PCA threshold 3.0 % of eye width)

At matched margins the PCA flags 3.1 % of all frames, fitting 3.8 %; the PCA flags fewer on 17 of 19 videos but 18 % of
video 18 (fitting 5.7 %): narrowed-eye frames absent from its 35 training poses (`examples.jpg`, local).

## 3. By eye where the two disagree (`scripts/finder_disagreement_review.py`, `_eval.py`; `disagreement_review/`)

4 frames per video and group at random (fixed-threshold PCA), judged blind, 145 of 148 judged (`finder_disagreement_verdicts.csv`,
`result.csv/.png`): flagged by the PCA only, 70 judged: 39 really wrong, 22 fine, 9 cannot tell; flagged by fitting only,
75 judged: 8 wrong, 59 fine, 8 cannot tell. The PCA-only errors are the four pupil points bunched beside the pupil for runs
of frames, no jump; fitting's own flags are mostly fast eye movements. Weighted by each video's count of such frames
(`per_video.csv`): the PCA alone points at an estimated 1 900 wrong frames without video 18 (4 500 with it), fitting alone
at 1 800 out of 22 400 flagged.

## 4. The PCA against the number of labels (`scripts/pca_vs_labels.py`, `pca_vs_labels*.csv/.png`)

PCA fitted on the first k labels in active-learning order, truth frames excluded. Background (video median score) falls
with k and settles after 20 labels; the threshold that just finds all wrong frames falls from 12.6 % of eye width at k = 5
to 6 % at 20, 2.4 % at 40, 1 % at 80, so a threshold fixed in eye widths cannot transfer; beyond 40 labels a pooled fixed
threshold starts to misflag because the scale drifts differently per video, while within each video the separation stays
clean. Video 18 stays at 17 % flagged whatever k ≥ 30.

## 5. The threshold rule used (`scripts/pca_threshold.py`, `pca_threshold*.csv/.png`, `pca_cache/` local)

Threshold = M × the video's own median score, label-free. M = (smallest wrong truth frame's multiple, 7.22) / (margin of
the fitting line to its smallest wrong frame, 0.257 / 0.14 = 1.84) = 3.93. On the truth frames: 23 of 23 found, 14 of 1307
right frames flagged; leave-one-video-out (M from the other 18, 3.9–5.1) 23 of 23, 14 right frames. Whole videos: 2.4 %
flagged (fixed rule 3.1 %); video 18 goes from 17.6 % to 3.2 %; the most flagged is video 1 at 8.5 %, which has the most
real errors. The repair with this finder is in `results/flag_and_repair/videoNN_pca_labelregion/` and `videoNN_either_labelregion/`
(`flag_and_repair.py --finder pca | either`), compared in `results/flag_and_repair/all_videos/finders.*`.

## Conclusion

Fitting and the PCA both find every wrong frame at the line but not the same frames on the whole videos: fitting sees jumps in
time, the PCA sees implausible shapes. Fitting stays the main finder (no labels needed, threshold = the line, checked by eye on
the whole videos); the PCA is the complement, with a per-video threshold and the video's training labels as its one dependence.
