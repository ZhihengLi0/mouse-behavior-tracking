# Blind test set: independent labels on frames the models never saw as truth

Purpose: the self-training pilot (`results/self_training/`) showed students that are flagged by the two finders a
third as often as their teachers. The finders measure agreement with a shape model, so a student trained on its own
predictions may just agree with itself. This set answers whether the unflagged frames are actually right, with
continuous paired errors instead of yes/no flags. Script: `scripts/blind_test_set.py` (`select`, `subset`,
`prelabel`, `score`, `check`). Proposed by the user on 2026-11-03; videos 1 and 8 labeled and scored on 2026-11-04.

## Basic information

| item | video 1 | video 8 |
|---|---|---|
| unit | `1_20251031_Pluto_spont_1`, 71,748 frames | `8_20251024_Pluto_spont1`, 71,748 frames |
| frame universe | all frames minus 60 on each side of the 70 truth frames (val20 + test50), minus 5 on each side of every human label frame, minus 5 on each side of every pseudo frame of both students: 38,809 eligible frames | same rule: 39,130 eligible frames |
| strata (from the whole-video flags of teacher and every30 student, same thresholds as the pipeline) | A: teacher flagged, student not (19.0 % of the video); B: both flagged (1.2 %); C: neither (79.5 %); D: student only (0.2 %) | A 7.8 %; B 2.6 %; C 87.6 %; D 1.9 % |
| draw | within a stratum, k-means in the teacher's pose space (8 points / eye width), one frame per cluster at random, seed 42; 50 frames drawn (13, 12, 13, 12), then cut to 24 for labeling by the user's decision (A 10, B 3, C 8, D 3; the lowest cluster ids kept; the other 26 PNGs in `labeling/blind_unused`) | same (24 of 50) |
| weights | stratum share of the video / frames kept in the stratum; "all (weighted)" numbers refer to the whole video | same |
| labeling | PNGs named img001..img050 in random order, no frame number or model output shown; AI pre-labels placed from the raw PNGs only (`prelabels_ai.csv`, notes in `prelabels_ai_notes.md`); the user reviewed every point in napari on 2026-11-04 (171 of 192 points moved, median move 18 px = 4 % of eye width); one frame (img006) has 4 of 8 points, the rest 8 | same procedure (133 of 192 points moved, median 9 px = 2 % of eye width); img031 has 4 points |
| label standard | the 8 points of the project (pupil top/bottom/left/right, eyelid top/bottom, nasal and temporal corner); not visible = empty | same |
| models | teacher = plateau model shuffle 211 (120 human frames); teacher + repair = the final pipeline's output on the teacher; round1_every30 / round1_every100 = the self-training students (shuffle 4011 / 4012); human40 = the 40-label model of the scaling curve (shuffle 212, prediction from `predictions_step02`) | teacher shuffle 911 (420 human frames); students 4081 / 4082; human40 = shuffle 912 |
| error | RMSE over the labeled points of a frame, divided by the teacher's median eye width (433.3 px); wrong = over 0.14 eye widths (the project's line); per frame in `per_frame.csv` | eye width 428.4 px |
| paired tests | each model against the teacher on the same frames: better / worse / same within 1 % of eye width, median difference, sign test, Wilcoxon (`paired.csv`) | same |
| check | `check --video 1`: the scorer reproduces `results/self_training/video01/teacher/truth_errors.csv` on the 70 truth frames (max difference 5e-5) | `check --video 8` passed the same way |

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

Caveats: 24 frames; strata B and D have three frames each and carry little weight. The labels were
placed by one person after AI pre-labels, with every point reviewed; the pre-labels were not derived from any model
in the comparison.

## Video 8 results (24 frames)

Whole-video weighted:

| model | median error, % eye width | p90 | wrong of 24 | wrong, % of video (weighted) |
|---|---|---|---|---|
| teacher | 4.09 | 12.62 | 9 | 7.3 |
| teacher + repair | 4.07 | 12.35 | 4 | 3.3 |
| round1_every30 | 4.12 | 12.18 | 2 | 1.8 |
| round1_every100 | 4.13 | 12.31 | 2 | 1.8 |
| human40 | 4.39 | 12.37 | 2 | 1.8 |

Per stratum (median error, % eye width / wrong frames):

| model | A (10 frames) | B (3) | C (8) | D (3) |
|---|---|---|---|---|
| teacher | 23.7 / 7 | 27.7 / 2 | 3.9 / 0 | 6.4 / 0 |
| teacher + repair | 6.7 / 2 | 27.7 / 2 | 3.9 / 0 | 6.4 / 0 |
| round1_every30 | 6.3 / 0 | 50.0 / 2 | 4.1 / 0 | 6.1 / 0 |
| round1_every100 | 6.4 / 0 | 32.6 / 2 | 4.1 / 0 | 7.7 / 0 |
| human40 | 7.2 / 0 | 33.6 / 2 | 4.2 / 0 | 9.6 / 0 |

Paired against the teacher in stratum A: teacher + repair 5 better / 0 worse / 5 same (p 0.06); round1_every30
7 / 0 / 3 (p 0.016); round1_every100 8 / 0 / 2 (p 0.008); human40 7 / 0 / 3 (p 0.016). Stratum C: every model
identical to the teacher on all eight frames (within 1 % of eye width).

What the numbers say:

- Stratum A again: the teacher is wrong on seven of ten, both students and the 40-label model on none. The
  flagged-share drop is real on this video too.
- Stratum C unchanged, as on video 1.
- Unlike video 1, the repair pipeline helps here: it fixes five of the seven wrong stratum-A frames (two remain).
  Students fix all seven. On video 8 the teacher's errors are single misplaced points more often than whole-pupil
  misses, which is what the repair can handle.
- Also unlike video 1, the 40-label model equals the students on this video (same wrong count in every stratum).
  Twenty more human labels and 1,800 pseudo labels buy the same here.
- Stratum B (both flagged): two of three frames wrong for every model; the hard frames stay hard.

## Across the two videos

In the decisive stratum (teacher flagged, student not) the teacher is wrong on 15 of 20 frames and the every30
student on 0 of 20; the student is never worse than the teacher on any of the 16 neither-flagged frames. The
repair pipeline recovers 5 of the 15 (all on video 8); the 40-label model 11 of the 15 (4 left on video 1). The
pseudo-label density does not matter. Frames both finders flag are wrong for every model (5 of 6).

## Videos 15 and 6 (24 frames each, labeled 2026-11-06) and the loop models

Same procedure (strata from the teacher's and the every30 student's flags, 50 drawn, 24 labeled after AI
pre-labels). Models scored on all four videos since then: `round1_repaired` (experiment 2 of the page),
`round2_every30` (videos 1, 8), `round1_both` and later `round{r}_both` (experiment 3, every turn). Wrong frames
per stratum A / B / C / D and the paired comparison against the teacher (`scores.csv`, `paired.csv`):

| video | teacher | teacher + repair | round1_every30 | round1_repaired | round1_both |
|---|---|---|---|---|---|
| 1 | 8/3/0/0 | 7/3/0/0 | 0/3/0/1 (14 better, 1 worse) | 1/1/0/2 (13, 2) | 0/1/0/1 (13, 1) |
| 8 | 7/2/0/0 | 2/2/0/0 | 0/2/0/0 (9, 2) | 0/2/0/0 (9, 4) | 0/2/0/0 (9, 2) |
| 15 | 5/2/0/0 | 2/2/0/0 | 0/2/0/0 (8, 3) | 1/2/0/0 (8, 6) | 0/2/0/0 (8, 3) |
| 6 | 7/1/0/0 | 4/1/0/0 | 0/2/0/1 (11, 2) | 1/1/0/0 (12, 1) | |

Over the 40 stratum-A frames: teacher wrong 27, teacher + repair 15, round1_every30 0, round1_repaired 3,
round1_both 0 of 30 (video 6 pending). Experiment 2 against experiment 1 paired on the 24 frames: 2/7, 4/2, 3/5,
3/5 better/worse on videos 1, 8, 15, 6, none significant.

## Supplementary set (`videoNN/supplement/`, 10 frames per video, labeled 2026-11-07)

`blind_test_set.py supplement`: frames on which the every30 and the repaired students disagree (either flags the
frame and the other does not, or their predictions differ by more than 0.14 eye widths), drawn only inside the
held-out blocks 4, 9, 14, 19 of 20 that experiment 2 never trained on, more than 60 frames from truth and drawn blind
frames; cells E_t / E_n (every30 flags, repaired not; t = teacher also flagged, n = not) and F_t / F_n (the reverse);
k-means per cell, 10 frames, AI pre-labels from the images only, the user reviewed every point. Scored with
`score --video V --supplement` into `supplement/{scores,paired,per_frame}.csv`, `paired.png`.

Wrong of 10 per video (1 / 8 / 15 / 6): teacher 6 / 3 / 3 / 1, teacher + repair 4 / 1 / 2 / 0, round1_every30
3 / 1 / 1 / 1, round1_repaired 0 / 0 / 1 / 3, round1_both 2 / 1 / 2 / (pending). The direction between the two
first experiments changes with the video (video 1 favours repaired frames, video 6 checked frames: its three
repaired-student errors are pupil_left throws on frames it flags itself).

## Files

- `videoNN/selection_key.csv`: blind name, frame, stratum, cluster, stratum share, weight, flags, `subset`
- `videoNN/supplement/`: the same files for the supplementary set (`selection_key.csv`, `selection_summary.csv`,
  `prelabels_ai.csv`, `prelabels_ai_notes.md`, `labeling/blind/CollectedData_Zhiheng.csv`, `scores.csv`,
  `paired.csv`, `per_frame.csv`, `paired.png`)
- `videoNN/selection_summary.csv`: universe, exclusions, strata shares, frames drawn and kept
- `videoNN/prelabels_ai.csv`, `prelabels_ai_notes.md`: the AI pre-labels and how they were placed
- `videoNN/labeling/blind/CollectedData_Zhiheng.csv`: the reviewed human labels (the `.h5` and the PNGs are local)
- `videoNN/per_frame.csv`, `scores.csv`, `paired.csv`, `paired.png`
