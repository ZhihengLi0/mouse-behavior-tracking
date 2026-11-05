# Blind test set: independent labels on frames the models never saw as truth

Purpose: the self-training pilot (`results/self_training/`) showed students that are flagged by the two finders a
third as often as their teachers. The finders measure agreement with a shape model, so a student trained on its own
predictions may just agree with itself. This set answers whether the unflagged frames are actually right, with
continuous paired errors instead of yes/no flags. Script: `scripts/blind_test_set.py` (`select`, `subset`,
`prelabel`, `score`, `check`). Proposed by the user on 2026-11-03; video 1 labeled and scored on 2026-11-04.

## Basic information

| item | video 1 |
|---|---|
| unit | `1_20251031_Pluto_spont_1`, 71,748 frames |
| frame universe | all frames minus 60 on each side of the 70 truth frames (val20 + test50), minus 5 on each side of every human label frame, minus 5 on each side of every pseudo frame of both students: 38,809 eligible frames |
| strata (from the whole-video flags of teacher and every30 student, same thresholds as the pipeline) | A: teacher flagged, student not (19.0 % of the video); B: both flagged (1.2 %); C: neither (79.5 %); D: student only (0.2 %) |
| draw | within a stratum, k-means in the teacher's pose space (8 points / eye width), one frame per cluster at random, seed 42; 50 frames drawn (13, 12, 13, 12), then cut to 24 for labeling by the user's decision (A 10, B 3, C 8, D 3; the lowest cluster ids kept; the other 26 PNGs in `labeling/blind_unused`) |
| weights | stratum share of the video / frames kept in the stratum; "all (weighted)" numbers refer to the whole video |
| labeling | PNGs named img001..img050 in random order, no frame number or model output shown; AI pre-labels placed from the raw PNGs only (`prelabels_ai.csv`, notes in `prelabels_ai_notes.md`); the user reviewed every point in napari on 2026-11-04 (171 of 192 points moved, median move 18 px = 4 % of eye width); one frame (img006) has 4 of 8 points, the rest 8 |
| label standard | the 8 points of the project (pupil top/bottom/left/right, eyelid top/bottom, nasal and temporal corner); not visible = empty |
| models | teacher = plateau model shuffle 211 (120 human frames); teacher + repair = the final pipeline's output on the teacher; round1_every30 / round1_every100 = the self-training students (shuffle 4011 / 4012); human40 = the 40-label model of the scaling curve (shuffle 212, prediction from `predictions_step02`) |
| error | RMSE over the labeled points of a frame, divided by the teacher's median eye width (433.3 px); wrong = over 0.14 eye widths (the project's line); per frame in `per_frame.csv` |
| paired tests | each model against the teacher on the same frames: better / worse / same within 1 % of eye width, median difference, sign test, Wilcoxon (`paired.csv`) |
| check | `check --video 1`: the scorer reproduces `results/self_training/video01/teacher/truth_errors.csv` on the 70 truth frames (max difference 5e-5) |

## Video 1 results (24 frames)

Whole-video weighted (`scores.csv`, stratum "all (weighted)"):

| model | median error, % eye width | p90 | wrong of 24 | wrong, % of video (weighted) |
|---|---|---|---|---|
| teacher | 3.76 | 36.56 | 11 | 16.4 |
| teacher + repair | 3.75 | 36.56 | 10 | 14.5 |
| round1_every30 | 3.65 | 7.73 | 4 | 1.3 |
| round1_every100 | 3.66 | 7.77 | 3 | 2.4 |
| human40 | 4.00 | 14.93 | 7 | 8.2 |

Per stratum (median error, % eye width / wrong frames):

| model | A (10 frames) | B (3) | C (8) | D (3) |
|---|---|---|---|---|
| teacher | 35.9 / 8 | 58.8 / 3 | 3.7 / 0 | 2.8 / 0 |
| teacher + repair | 35.9 / 7 | 58.8 / 3 | 3.7 / 0 | 2.8 / 0 |
| round1_every30 | 4.9 / 0 | 34.6 / 3 | 3.5 / 0 | 3.0 / 1 |
| round1_every100 | 5.7 / 1 | 7.3 / 1 | 3.5 / 0 | 3.0 / 1 |
| human40 | 8.6 / 4 | 7.3 / 1 | 3.6 / 0 | 17.4 / 2 |

Paired against the teacher (`paired.csv`, `paired.png`):

| model | stratum | better | worse | same | median difference, % eye width | sign test p |
|---|---|---|---|---|---|---|
| round1_every30 | A | 9 | 0 | 1 | -29.9 | 0.004 |
| round1_every30 | C | 2 | 0 | 6 | -0.1 | 0.5 |
| round1_every30 | all | 14 | 1 | 9 | -4.7 | 0.001 |
| round1_every100 | A | 9 | 0 | 1 | -27.7 | 0.004 |
| round1_every100 | all | 14 | 1 | 9 | -3.2 | 0.001 |
| teacher + repair | all | 2 | 0 | 22 | 0.0 | 0.5 |
| human40 | A | 7 | 1 | 2 | -10.9 | 0.07 |
| human40 | all | 10 | 4 | 10 | -0.4 | 0.18 |

What the numbers say:

- Stratum A is the question, and the answer is yes: on the ten frames the teacher flagged and the student did not,
  the teacher is wrong on eight (median error a third of the eye width) and the every30 student is wrong on none
  (median 4.9 %). Nine of ten frames are better, none worse. The drop in flagged share is real accuracy, not
  self-consistency, on this video.
- Stratum C (neither flagged, four fifths of the video) is unchanged: both models at 3.5 % median, no wrong frame.
  The student did not get worse where the teacher was fine.
- Stratum B (both flagged, 1.2 % of the video): the student is still wrong on all three frames, though with smaller
  errors; frames both finders still flag remain the hard ones. Stratum D (student only, 0.2 %): one of three frames
  wrong for the student; too few frames to say more.
- The repair pipeline does almost nothing on this set (2 frames slightly better, same wrong count minus one): on
  the 70 truth frames it had matched the student, on frames drawn by the flags it does not. The repair only
  replaces single points; on these frames the whole pupil is misplaced.
- The 40-label model (human40) is better than the teacher in stratum A (7 of 10) but still wrong on four, and
  worse than the teacher on two of the three stratum-D frames. Twenty more human labels buy less than the pseudo
  labels did on this video.
- The two pseudo densities give the same picture; every30 is slightly better in A and worse in B.

Caveats: 24 frames; strata B and D have three frames each and carry little weight; one video. The labels were
placed by one person after AI pre-labels, with every point reviewed; the pre-labels were not derived from any model
in the comparison.

## Files

- `videoNN/selection_key.csv`: blind name, frame, stratum, cluster, stratum share, weight, flags, `subset`
- `videoNN/selection_summary.csv`: universe, exclusions, strata shares, frames drawn and kept
- `videoNN/prelabels_ai.csv`, `prelabels_ai_notes.md`: the AI pre-labels and how they were placed
- `videoNN/labeling/blind/CollectedData_Zhiheng.csv`: the reviewed human labels (the `.h5` and the PNGs are local)
- `videoNN/per_frame.csv`, `scores.csv`, `paired.csv`, `paired.png`
