# Three directions after the 2026-10-31 meeting: what the existing data says

Written 2026-10-31 night, from data already on disk. No new training, no new labels, no change to the page. Everything
below reuses the whole-video outputs of the pipeline of step 9 (`results/flag_and_repair/videoNN_labelregion` for the
fitting finder, `videoNN_pca_labelregion` for the PCA finder), the 1330 truth frames with their human labels, the two
by-eye reviews, the heatmap peak files and the models themselves.

| | |
|---|---|
| Videos | 19, every video at its plateau model (`results/labels_to_plateau.csv`) |
| Frames | 1,310,718 whole-video frames; 1330 truth frames with human labels, never trained on |
| Finders | fitting (8-point maximum deviation from the one-step forecast > 14 % of eye width); PCA (reconstruction error > 3.93 × the video's median) |
| Repair | second heatmap peak of the candidate pupil point(s), accepted by the human-label range in two steps |
| By-eye material | 355 judged frames of the repair spot check (`results/flag_and_repair/spotcheck_review`); 145 judged disagreement frames (`results/finder_comparison/disagreement_review`) |
| Scripts | `scripts/repair_principle.py`, `scripts/episode_structure.py`, `scripts/pseudo_label_noise.py` |
| Results | `results/repair_principle/`, `results/episode_structure/`, `results/pseudo_label_noise/` |

The meeting also raised, and then withdrew, the idea of restricting the heatmap to a region before taking the second
peak: it would treat the model as untrustworthy and would use the same information as the acceptance test. It is not
analysed here.

## Direction 1. Why a repair falls short, why a frame is sent back

### 1.1 How DeepLabCut turns a heatmap into a coordinate (read in the installed package, `pose_estimation_pytorch`)

- The network has two output heads per key point. The heatmap head gives one score per cell of 8 × 8 px (stride 8),
  trained with a weighted mean-squared error (weight 1) against a Gaussian centred on the label. The reported
  confidence is the score of the highest cell.
- The location-refinement head gives two numbers per cell, the vector from the cell centre to the label, divided by
  7.28. It is trained with a Huber loss (weight 0.05) only on cells within 17 px of the label. Nothing scores or
  checks this offset at prediction time.
- The reported coordinate is `cell × 8 + 4 + offset × 7.28` at the highest cell. So DeepLabCut is precise to the
  pixel at the highest cell only. Our second peak is our own rule (the highest local maximum at least 3 cells from the
  highest cell, `heatmap_signals.py`); the peaks file keeps its cell, and the repaired point sits at the cell centre
  plus the video's median offset, i.e. it is precise to the cell.
- Does that matter? `peak2_locref.csv`: on the 23 truth-frame repairs of both routes the second peak was recomputed
  with the network's own offset at its cell. The point moves by 5 px on average, the error against the human label
  changes by 0.1 to 0.5 % of eye width in either direction (median after repair 3.7 % against 3.7 % on the fitting
  route, 3.2 % against 3.0 % on the PCA route). Storing the offset of the second peak would be free, but it changes
  nothing at the scale of a wrong frame.

### 1.2 Why a flagged frame stays flagged (`reasons.csv`, `reasons_total.csv`, `reasons.png`)

Every frame with status flagged-not-repaired was replayed through the decision of `flag_and_repair.py`:

| reason, in the order the code decides | fitting route (15,125 frames) | PCA route (10,788 frames) |
|---|---|---|
| no candidate: an eye-outline point is over the threshold (fitting) or carries the largest residual (PCA), the repair is not attempted | 86.1 % | 48.0 % |
| a candidate pupil point has no second peak | 0 | 0 |
| the frame as it is fails the label range, and so does the frame with the candidate moved to its second peak | 13.9 % | 52.0 % |

Of the frames in the last row, the replaced point itself is still outside the range in 85 % (fitting) and 74 %
(PCA); in the rest another pupil point or only the shape ratios fail. A different single pupil point or a pair would
have passed the range in 19 % (fitting) and 25 % (PCA) of them, so a wider search over which point to move would
repair about one in five of those frames, if the range can be trusted on them.

Joined to the by-eye verdicts of the 184 still-flagged frames of the spot check (fitting route,
`reasons_by_verdict.csv`, `reasons_worst_by_verdict.csv`):

| reason | really wrong | false alarm | cannot tell |
|---|---|---|---|
| no candidate, an eye-outline point over the threshold | 71 | 69 | 10 |
| of these, the worst point is an eye corner | 21 | 59 | 7 |
| of these, the worst point is an eyelid point | 28 | 6 | 1 |
| second peak tried and rejected by the range | 30 | 0 | 2 |

Findings:

1. The repair never touches the eye outline. Every false alarm of the fitting finder that survives the acceptance
   test is a frame where an outline point, in 59 of 69 cases an eye corner, moved more than the threshold. The
   finder is right that the corner jumped, the human judged the frame fine, so the corner prediction jitters at the
   scale of the threshold without being wrong. These frames are the 1.2 % "flagged, not repaired" of the whole videos;
   they would disappear from that class if corner points were given their own, wider threshold, or if the corner
   deviation were measured against the eye width differently. That is a finder question, not a repair question.
2. The second-peak attempt itself is reliable: when a second peak was tried and the range rejected it, the frame was
   really wrong in 30 of 32 judged cases. The range is not sending back good repairs; the second peak is not on the
   pupil in those frames.
3. Real errors on outline points (49 of the 184 judged frames: 21 corners, 28 eyelids) are the largest unrepaired
   class after the clustered pupil errors. The pipeline has no repair for them because a moved corner also moves the
   eye frame in which every pupil quantity is measured. A second-peak repair of eyelid points, checked by the eyelid
   band of the human labels, would be the natural extension; corners would need a check that does not depend on the
   corners themselves (for example the corner-to-corner distance against the video's median eye width).

### 1.3 Why a repaired frame is "closer, but still not right" (`repaired_by_verdict.csv`, `heatmap_stats.png`, `heatmap_maxima.csv`, galleries local)

The 174 repaired frames of the spot check carry 178 replaced points: 152 judged right, 20 closer but not right, 4 not
better. From the peaks file of the replaced point:

| | right | closer, not right | not better |
|---|---|---|---|
| distance peak 1 to peak 2 (eye widths), median | 0.82 | 0.44 | 0.22 |
| peak 2 / peak 1, median | 0.89 | 0.81 | 0.88 |
| local maxima above 0.3 × peak 1, median | 4 | 4 | 3 |
| score of peak 2, median | 0.30 | 0.21 | 0.40 |
| fitting deviation of the frame (eye widths), median | 0.73 | 0.57 | 0.22 |

The full heatmaps of the 24 not-right points and of 23 right ones were recomputed (CPU) and every local maximum near
the eye listed. Both groups have the same number of maxima (3.5 on average) and no useful third peak (best third
maximum 0.11 against 0.14). The one thing that separates them is distance: in a right repair the thrown point is far
away (0.8 eye widths) and the second peak is a clean peak on the pupil; in a not-right repair peak 1 and peak 2 are
both near the pupil, 0.3 to 0.4 eye widths apart, in a low-contrast eye (videos 14, 17, 18 hold 11 of the 20), and
the heatmap is a smear along the pupil edge rather than two peaks. Moving to the second peak then moves the point
along the smear, closer but not to the ellipse end point the label convention asks for.

Finding 4: the second peak repairs "thrown far" errors and cannot repair "smeared" ones; the two are told apart by
the distance between the peaks, not by the scores. The 4 "not better" frames have the smallest peak distance of all
(0.22) and the smallest fitting deviation (0.22), i.e. they sit just above the line. A rule "do not move a point to a
second peak closer than about 0.3 eye widths" would have avoided most not-right repairs at the price of some right
ones (the right group has 22 of 152 points below 0.3); it is a tunable, not a fix, and the smear itself is a model
limit at the current label count.

## Direction 2. Flagged frames come in episodes; what the good frames around an episode give

### 2.1 Episodes (`episodes_by_video.csv`, `episode_lengths.png`, `timeline.png`)

| | fitting | PCA | either |
|---|---|---|---|
| flagged frames, 19 videos | 49,975 | 31,369 | 54,962 |
| episodes (runs of consecutive flagged frames) | 15,662 | 11,506 | 16,213 |
| episodes of a single frame | 58 % | 62 % | 57 % |
| share of flagged frames in episodes of 10 frames or more | 39 % | 36 % | 43 % |
| share in episodes of 50 or more | 7.8 % | 7.8 % | 10.6 % |
| longest episode (frames) | 159 | 151 | 274 |

Most episodes are a single frame, but most flagged frames sit in runs: four in ten are in runs of at least a sixth
of a second. The timeline shows the runs cluster in particular stretches of a video (video 1: the first and the last
tenth; video 6: 40 to 50 %; video 8: 70 to 90 %; video 18: 10 to 15 %), not uniformly. Of the 19 wrong truth frames
the fitting finder flags, the median episode length is 10 frames (up to 89); for the PCA finder it is 3 (up to 58).
Right truth frames it flags sit in episodes of 3 (fitting) and 9 (PCA).

### 2.2 Filling a flagged frame from the frames around its episode (`temporal_fill_truth.csv`, `temporal_fill_summary.csv`, `temporal_fill.png`)

Three fills were tried on every flagged truth frame, each moving only the flagged points of the flagged frame and
never touching an unflagged frame: linear interpolation between the last unflagged frame before the episode and the
first after; the nearest of those two copied; the one-step forecast the fitting finder already computes. Error
against the human label, in eye widths, next to the second-peak repair on the same frames:

| 19 wrong truth frames, fitting route | median error | still wrong after |
|---|---|---|
| before | 0.269 | 19 |
| second-peak repair (current) | 0.056 | 7 |
| interpolation across the episode | 0.046 | 2 |
| nearest unflagged frame copied | 0.047 | 2 |
| one-step forecast | 0.087 | 5 |
| interpolation of all 8 points | 0.038 | 1 |

| 19 wrong truth frames, PCA route | median error | still wrong after |
|---|---|---|
| second-peak repair (current) | 0.071 | 8 |
| interpolation across the episode | 0.054 | 6 |
| interpolation of all 8 points | 0.039 | 1 |

On the flagged right frames (28 fitting, 14 PCA) interpolation makes one frame wrong on the fitting route (the
second peak none), none on the PCA route. By episode length (fitting route): episodes of 11 to 30 frames are where
the second peak fails (median 0.198 after) and interpolation does not (0.058); in episodes over 30 frames both are
fine (0.037 and 0.038), in short ones both are fine.

### 2.3 What interpolation costs, without labels (`displacement_vs_gap.csv`, `displacement_vs_gap.png`)

On unflagged stretches of every video, the pupil points at the middle of a gap of L frames were predicted by linear
interpolation from the two ends and compared with the model's own prediction there:

| gap L (frames) | 1 | 5 | 10 | 30 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|
| frame error, median (eye widths) | 0.004 | 0.005 | 0.005 | 0.006 | 0.007 | 0.009 | 0.011 |
| 90th percentile, worst video | 0.024 | 0.028 | 0.028 | 0.028 | 0.030 | 0.033 | 0.037 |

The pupil moves slowly against the 14 % line: even across 200 frames (3.3 s) the interpolation error is a quarter of
the model's own noise on good frames, and no gap of 200 or less exceeds the line in more than 0.3 % of cases.

Findings:

5. For the errors the finders catch, the frames on both sides of the episode are a better source of the pupil
   position than the heatmap's second peak: 2 of 19 wrong truth frames stay wrong instead of 7, with the same
   number of good frames made worse (1 of 28 against 0). Interpolating all eight points of the flagged frame is
   better still (1 of 19), which says the errors of a flagged frame are seldom confined to the one point over the
   threshold.
6. This is not smoothing. Unflagged frames keep the model's coordinates to the pixel; only frames the finder already
   flagged change, the same set the second-peak repair changes now. What changes is where the replacement comes from.
   The change should still be reported per frame for the alignment with the neural signal (which frames, how long the
   episode, how far the point moved).
7. The limit is the anchors: interpolation assumes the first unflagged frame on each side is right. Where the finders
   miss a clustered error (the frames only the PCA finder caught in the disagreement review) the anchor is wrong and
   the fill copies the error. Using the union of both finders for the anchors, and the rule that an anchor must be
   at least k unflagged frames into its stretch, are the two obvious guards; neither was tested here.
8. What remains unrepaired either way: the 4 closed-eye frames, and the wrong frames inside episodes longer than the
   pupil's own time scale once the eye actually moves (none in the truth set, but the 274-frame episode of video 18
   is 4.6 s).

## Direction 3. Self-training

### 3.1 The three references (summarised from the papers; details in the meeting notes)

- **Noisy Student (Xie et al., CVPR 2020).** Teacher trained on labeled data labels a large unlabeled pool on clean
  images; a student of equal or larger size is trained on labeled plus pseudo-labeled data under noise (RandAugment,
  dropout, stochastic depth); the student becomes the teacher; three rounds. Error accumulation is held by:
  confidence filtering (below 0.3 dropped) and per-class balancing; soft labels; noise on the student only; the
  labeled data kept in every round. Gains shrink per round (87.6, 88.1, 88.4 %).
- **Meta Pseudo Labels (Pham et al., CVPR 2021).** Teacher and student trained together; the teacher is updated by
  how much its pseudo labels improve the student's loss on the labeled set (one-step meta-gradient). The labeled set
  is the arbiter of the pseudo labels. Needs a rewrite of the training loop; the feedback signal from 20 labels per
  video would be very noisy.
- **CVPR 2026 poster 36408 (Sheng et al., "Revisiting Learning with Noisy Labels: Active Forgetting and Noise
  Suppression").** Not a self-training paper: supervised learning where some given labels are wrong. Two stages
  observed, the net first fits clean and noisy alike, then overfits the noisy ones. Suspect samples, picked by the
  base method's selection rule, get a negative cross-entropy (unlearning) and a complementary-label loss instead of
  positive training. For a heatmap network the analogue is to down-weight or mask the loss on frames or points judged
  suspect, not to train on them.

Common ground: the human labels stay in every round; suspect labels are treated differently from trusted ones; noise
or regularisation stops the student from copying the teacher; few rounds.

### 3.2 The noise a first round would start from (`noise_total.csv`, `noise.csv`, `noise_points.csv`, `noise.png`)

A round would train on the model's own coordinates on frames no finder flagged, plus the repaired frames, and
exclude still-flagged and closed frames. On the truth frames those coordinates have a known error:

| class of frame | share of the whole videos | truth frames | median error | 90th percentile | over the 14 % line |
|---|---|---|---|---|---|
| no finder flagged it | 95.7 % | 1267 | 2.0 % | 4.4 % | 0 |
| flagged, repaired | 0.7 % | 12 | 3.2 % | 7.0 % | 0 |
| flagged, judged plausible, left alone | 0.4 % | 12 | 4.6 % | 5.9 % | 0 |
| flagged, not repaired (excluded) | 2.6 % | 35 | 6.5 % | 27 % | 8 |
| eye closed (excluded) | 0.6 % | 4 | 62 % | | 4 |

Findings:

9. The pseudo labels of a first round would be as accurate as the plateau model itself: median 2 % of eye width,
   9 in 10 under 4.4 %, none over the line on the truth frames. Per video the median runs from 0.8 % (video 3) to
   4.4 % (video 8). The human relabel repeatability was never measured (the 40-frame relabel test of 2026-10-26 was
   prepared and not carried out), so whether 2 % is above or below label noise is not known; the labeler's own
   standard is "the structure is right", which the 14 % line encodes, and under that standard 0 of 1291 usable truth
   frames carry a wrong pseudo label.
10. The frame-level line hides single-point errors. Among the usable truth frames, 43 have the pupil_left point
    more than 14 % of eye width from its label while the frame as a whole is under the line (10 for pupil_right, 4
    each for top and bottom). A pseudo label should therefore be filtered per point, not per frame: the finder's
    per-point deviation and the PCA residual are already available for that.
11. Only 4 % of frames would be excluded, so the pseudo-labeled set is 95 % of each video at the same accuracy as the
    model. That is the Noisy Student situation of a teacher that is already good on most of the data; the question is
    whether retraining on it moves the hard 4 %, which are hard because they look different (low contrast, narrowed
    eye, fur), not because they are under-sampled. The two-seed pilot of 2026-10-25 (two models wrong on the same
    frames) suggests the errors are systematic, which self-training alone does not cure.

### 3.3 A pilot that would answer it (not run; for decision)

- Videos: 1 (8 wrong truth frames, 17 labels, worst flagged share) and 8 (the unrepairable clustered errors), plus
  one clean video (7) as a control that nothing gets worse.
- Teacher: the plateau model. Pseudo labels: every 30th frame of the usable classes (unflagged by either finder, or
  repaired), excluding the 70 truth frames and 60 frames around each, filtered per point (no point over 14 % on the
  fitting deviation or the PCA residual); about 2000 frames per video.
- Student: the same DLC configuration retrained from ImageNet weights on the human labels plus the pseudo labels
  (DLC's augmentation on, as now); a second variant with the pseudo-labeled frames down-weighted by 0.3 in the loss.
- Evaluation: the 70 truth frames, same metric, plus the whole-video flagged share of both finders. Round 2 only if
  round 1 lowers the median truth error and the wrong-frame count; stop at any rise. Two rounds at most.
- Cost: one plateau-size training per video and round took about the time of one step of the scaling pipeline on the
  Linux machine; with 2000 pseudo frames the epoch is 100 times larger, so the training schedule (iterations, not
  epochs) must be fixed at the current value to keep the cost comparable.

## What each direction needs next

- Direction 1: decide whether to extend the repair to eyelid points (second peak, checked by the eyelid band) and to
  give corner points their own threshold; both are small changes to `flag_and_repair.py` with the same evaluation.
- Direction 2: decide whether to replace the second peak by interpolation across the episode as the source of the
  repaired position (with the acceptance test unchanged), and how the anchors are guarded; then run it on the 19
  videos and re-judge a spot check by eye.
- Direction 3: decide whether to run the pilot above on videos 1, 8 and 7.

## Addendum, same night: the repair extended to all eight points

Decision (user): the fixed pipeline for self-training is finder = fitting or PCA (union), repair = second peak,
acceptance = two-step human-label range, and the repair is extended from the four pupil points to all eight
(`flag_and_repair.py --points all`; outline range from `scripts/outline_region.py`, margin 0.23 by leave-one-video-out).
Results on the 19 videos (`results/flag_and_repair/all_videos/full_repair.csv`, `full_repair.png`): the truth-frame
numbers do not move (12 of 23 wrong frames repaired to good, 1 good frame made worse), one wrong eyelid frame is gained
and one wrong frame is lost because its eyelid's second peak fails the outline range and the whole replacement is
reverted; on the whole videos the still-flagged share falls from 1.18 % to 0.66 %, mostly into "plausible as it is,
left alone" (1.74 % to 2.11 %) and partly into repaired (0.80 % to 0.96 %); 959 eyelid points and 1916 corner points
are put in. A by-eye page of the outline repairs is prepared (`results/flag_and_repair/outline_review/`), verdicts
pending. One refinement worth testing: when the full replacement fails the range, fall back to the subsets of the
candidate points (single points, then pairs) and accept the smallest that passes; the lost frame of video 4 would be
recovered.

By-eye verdicts on the all-point repair (90 frames, 139 replaced points, `results/flag_and_repair/outline_review/`):
eyelid points right in 39 of 41 judged, corner points right in 31 of 35 judged (18 of 53 could not be judged from
the crop), pupil points moved in the same frames right in 18 of 32 judged with 10 that were right before and were
moved for nothing. So the outline repair itself works at the same rate as the pupil repair did (87 %); the cost is
that the frame-level replacement moves pupil candidates that were fine. Next change: replace the smallest set of
candidate points that makes the frame pass (outline points first), instead of all candidates at once.
