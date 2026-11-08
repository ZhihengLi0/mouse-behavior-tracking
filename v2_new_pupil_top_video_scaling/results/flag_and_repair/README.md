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

## Spot check by eye, same rule on every video: the check of the repair on the whole videos

Sample: `all_videos/spotcheck_sample.csv`, per video and per test up to 10 repaired and 10 still-flagged frames, one per
episode (touched frames at most 5 frames apart), episodes drawn at random (seed 0). For the adopted test
(`labelregion`, 358 frames) `scripts/spotcheck_review.py` writes a review page (`spotcheck_review/spotcheck_review.html`,
crops in `spotcheck_review/items/`, local only): for a repaired frame the frame, the model's points before and the point
put in after; for a still-flagged frame the frame and the model's points. One question per frame, nothing else shown
(no fitting value, no episode length). The labeler judged 355 of the 358 frames on 2026-10-30
(`spotcheck_review/spotcheck_review_verdicts.csv`); `scripts/spotcheck_review_eval.py` gives
`spotcheck_review/spotcheck_review_result.{csv,png}`. None of these frames took part in setting the line, the
threshold or the label range, and the frames are one per episode, so the shares are shares of episodes.

| judged by eye | frames shown | verdict |
|---|---|---|
| repaired frames: is the point put in on the pupil where it belongs? | 174 | **149 yes (87 % of the judged)**, 19 closer but still not right, 4 not better, 2 cannot tell |
| still-flagged frames: is a model point really wrong? | 184 | **101 yes (59 % of the judged)**, 69 false alarm, 14 cannot tell |

- **The repair is right in about seven of eight episodes it touches**, and it almost never makes a frame worse (4 of
  174). The 19 "closer but not right" frames sit mostly in the Terra videos 14, 17 and 18 (10 of the 19): the second peak
  lands on the iris or the eyelid instead of the pupil edge. This matches the truth frames, where 12 of 13 repairs were
  right.
- **Of the frames that stay flagged, about three in five are real errors and two in five are false alarms.** On the
  truth frames the still-flagged set was 7 wrong and 6 good, the same proportion. The false alarms concentrate in
  videos 5, 7, 11, 12 and 15 (7 or 8 of 10 each; video 11: 8 of 8 judged); why these videos is not examined yet.
- For a user of the data: the repaired frames (0.8 % of all frames) can be used; the still-flagged frames (1.2 %)
  contain about 0.7 % real errors and 0.5 % right frames, and that is where human time goes if it goes anywhere.

## Three finders: fitting, PCA, either (`flag_and_repair.py --finder`, 2026-10-31)

`--finder pca` flags by the PCA reconstruction error of `results/finder_comparison/pca_threshold.py` (threshold = 3.93 × the
video's median score) and replaces the pupil point with the largest residual (of four candidate rules tried on videos 1, 8,
12, 15 the only one that repairs anything, see the function `candidates_pca`); `--finder either` flags when either finder
does and replaces the union of the two candidate sets. Repair and acceptance (human-label range) unchanged. Results in
`videoNN_pca_labelregion/`, `videoNN_either_labelregion/`, compared in `all_videos/finders.csv`, `finders_by_video.csv`,
`finders.png`, `finders_share_by_video.png` (`scripts/finder_repair_comparison.py`).

| finder | wrong flagged / 23 | repaired to good | wrong after | good made worse | flagged, all frames | repaired | flagged, not repaired | flagged, passes check |
|---|---|---|---|---|---|---|---|---|
| fitting | 23 | 12 | 11 | 1 | 3.82 % | 0.78 % | 1.16 % | 1.42 % |
| PCA | 23 | 11 | 12 | 0 | 2.39 % | 0.72 % | 0.82 % | 0.42 % |
| either | 23 | 12 | 11 | 1 | 4.20 % | 0.80 % | 1.18 % | 1.74 % |

On the truth frames the three are the same within one frame. On the whole videos the PCA flags fewer frames for almost the
same number of repairs; the union flags more but repairs barely more: the errors the PCA alone finds (bunched pupil points,
`results/finder_comparison/README.md` §3) have no usable second peak or fail the check and stay flagged. A gain in finding
that is not yet a gain in repair.

## Acceptance tests under the PCA finder (`scripts/acceptance_by_finder.py`, `all_videos/acceptance_by_finder.*`)

Same repair, frames flagged by the PCA (`videoNN_pca_<accept>/`; the test "fit" is defined only for the fitting finder).
23 wrong frames repaired to good / flagged right frames (14) repaired, made worse: label range two steps with margin 11 /
1, 0; geometry 12 / 5, 2; label range two steps no margin 10 / 4, 0; confidence 9 / 11, 3; label range one step 10 / 13, 4;
jump 4 / 13, 4. The ranking is the one of the fitting route.

## Files

`videoNN_<test>/`: `summary.csv` (truth frames), `whole_video.csv`, `truth_frames.csv` (errors in eye widths and in px);
`frames_all.csv.gz` and `cache/videoNN_pred200.npy` local only. `all_videos/`: `whole_video_all.csv`,
`truth_summary_all.csv`, `totals.csv`, `comparison.csv`, `touched_structure.csv`, `error_before_after.csv`, `share_per_video.png`, `error_before_after.png`,
`spotcheck_sample.csv`; `spotcheck_review/`: `items.csv`, `spotcheck_review_verdicts.csv`, `spotcheck_review_result.{csv,png}`, `heatmap_examples.png` (`scripts/heatmap_example_figure.py`: three repaired frames with the full heatmap of the thrown point, both peaks marked), `review_examples.jpg` (`scripts/review_examples_figure.py`: before / after pairs of judged frames); page and crops local only. Threshold and line: `results/error_tier/`, `results/fit_threshold/`,
`results/acceptance_thresholds/`.

## All eight points (2026-10-31 night)

The repair above touches pupil points only: a frame whose over-threshold points include an eyelid or a corner is left
flagged. `flag_and_repair.py --points all` (default `pupil`, old outputs byte-identical) repairs every point:
candidates = every point over the threshold (fitting), the point with the largest residual whichever it is (PCA), the
union (either); accepted by the two-step human-label range on the pupil quantities and on eight outline quantities
(`scripts/outline_region.py` → `results/acceptance_thresholds/label_region_outline.csv`: corner distance, angle and
midpoint against the video's label-free medians, eyelid positions and opening in the frame's eye frame; own
leave-one-video-out margin 0.23). Folders `videoNN_either_labelregion_all/`; comparison
`scripts/full_repair_comparison.py` → `all_videos/full_repair.csv`, `full_repair_by_video.csv`, `full_repair.png`.

| | pupil points only | all eight points |
|---|---|---|
| 23 wrong truth frames: flagged / repaired to good / still flagged / closed | 23 / 12 / 7 / 4 | 23 / 12 / 7 / 4 |
| 40 right truth frames flagged: repaired / made worse / left alone as plausible | 2 / 1 / 32 | 2 / 1 / 38 |
| whole videos: repaired / still flagged / left alone as plausible (% of frames) | 0.80 / 1.18 / 1.74 | 0.96 / 0.66 / 2.11 |
| points put in: pupil / eyelid / corner | 11072 / 0 / 0 | 11140 / 959 / 1916 |

On the truth frames the two runs differ on eight frames: six good frames whose corner tripped the finder are now
judged plausible as they are and left alone instead of staying flagged; one wrong eyelid frame (video 2, 68086) is
repaired to good; one wrong frame (video 4, 62186, eyelid_bottom worst) that the pupil-only run repaired through its
PCA candidate is now sent back, because the eyelid's second peak fails the outline range and the whole replacement is
reverted. Whole videos: the still-flagged share halves, most of it moving to "plausible as it is", the rest repaired.
By-eye check of the outline repairs: `scripts/outline_review.py` → `outline_review/` (page and crops local,
`items.csv` committed), verdicts pending.

By-eye verdicts (2026-10-31 night, `outline_review/outline_review_result.csv`, `scripts/outline_review_eval.py`): 90
repaired frames with an eyelid or corner among the replaced points, 5 per video, one verdict per replaced point, 139
points.

| replaced point | right | closer, not right | not better | was right, moved for nothing | cannot tell |
|---|---|---|---|---|---|
| eyelid (43) | 39 | 2 | 0 | 0 | 2 |
| corner (53) | 31 | 2 | 0 | 2 | 18 |
| pupil, in the same frames (43) | 18 | 4 | 0 | 10 | 11 |

Eyelid repairs are right in 39 of 41 judged; corner repairs in 31 of 35 judged, but a third of the corners could not
be judged from the crop. The pupil points moved in these frames are the weak part: 10 of 32 judged were right before
and were moved for nothing. These are frames the pupil-only run left flagged (an outline point over the threshold);
with all points allowed the frame fails the outline range, every candidate point is replaced at once, and a pupil
candidate that was fine moves with the rest. The fix to test is the subset fallback: replace the outline candidates
first and keep a pupil candidate only if the frame does not pass without moving it.

**Smallest subset (`--replace smallest`, folders `videoNN_either_labelregion_all_smallest/`).** Instead of replacing
every candidate at once, the smallest subset of candidate points that makes the frame pass both ranges is put in,
outline points tried first, singles before pairs. Truth frames: 15 of 23 wrong frames repaired to good (12 before),
still flagged 4 (7), good frames made worse 1 (1); the three gained are the eyelid frames of video 2 (68086, 68379)
and video 4 (70429). Whole videos: repaired 1.10 % (0.96 %), still flagged 0.51 % (0.66 %), left alone 2.11 %;
1878 frames repaired that the whole replacement sent back, none lost; of the 12,517 frames both variants repair, 805
keep a pupil point the whole replacement would have moved (841 points). Points put in: 11,571 pupil, 1295 eyelid,
2293 corner. This is the variant to carry forward; the by-eye count of pupil points moved for nothing (10 of 32)
was taken on the whole replacement and should be repeated on this one.

## Audit of the repair, 2026-11-07 (subset order, leave-one-video-out ranges, closed-eye frames)

After the meeting of 2026-11-07 the chain is audited before the formal self-training loop: every number must have a
stated basis. Three items concern the repair. Scripts: `label_region_lovo_ranges.py`, `flag_and_repair.py` options
`--order deviation`, `--tiebreak margin`, `--lovo-range` (defaults unchanged: video 0 rerun with defaults is byte-identical),
`subset_order_comparison.py`, `closed_frames_check.py`. Pop-up pages and their items are local only; the verdict CSVs
and every table are committed.

**1. Order of the smallest-subset search** (`all_videos/subset_order_*.csv`). Variants on all 19 videos: A = in use
(outline points first, then pupil points, singles before pairs, first passing subset), B = the point of largest
deviation first, then the pairs that contain it (`--order deviation`), C = B with a tie-break among the passing subsets
of the smallest size: the one farthest inside the label ranges (`--tiebreak margin`, `range_margin`). On the 1330 truth
frames the three are identical (15 of 23 wrong frames repaired, 4 still flagged, 4 called closed, 1 good frame made
worse). On the whole videos all three repair the same 14,395 frames; the replaced point sets differ in 35 frames (A vs B
15, B vs C 26); more than one subset of the chosen size passes in 46 frames only, so the tie-break hardly ever acts.
The earlier count "9.8 % of the repaired frames do not replace the largest-deviation point" (1398 frames) is the same
with B: in those frames that point's second peak was tried alone and in pairs and failed the ranges, another point's
replacement passed. Judged by eye by the user (`all_videos/subset_review_verdicts.csv`, `subset_order_verdicts.csv`,
`subset_order_worst_left_verdicts.csv`, 2026-11-08 00:05):
- Part 2, 20 of those 1398 frames: the untouched largest-deviation point was right where it was in 19, and the point
  the repair moved instead was right in 19. Leaving the point whose second peak fails the ranges is the correct
  behaviour; the finder had flagged a right point.
- Part 1, the 35 frames where the variants disagree: C has every point right in 21, B in 5, A in 3; in 2 the model
  needed no repair, in 7 more than one of the shown replacements was needed together (the frame passed the ranges
  with one point moved while another candidate was still wrong), in 1 a point the finders did not catch was still
  wrong, in 4 none was right. Where the tie-break acts, the subset farthest inside the label ranges is the right
  one far more often than the first passing subset.

**2. Leave-one-video-out ranges** (`acceptance_thresholds/label_region_lovo_ranges.csv`,
`label_region_outline_lovo_ranges.csv`, runs `videoNN_..._lovo`). With the ranges built from the other 18 videos'
labels (same margins 0.42 and 0.23) the truth-frame outcome is identical for every variant, and the whole videos
change by 8 repaired frames (14,403 against 14,395) and 22 still-flagged frames (6716 against 6694) in 1.31 million.
The acceptance does not depend on the video's own labels.

**3. Closed-eye frames** (`closed_frames/`). The rule in use calls a frame closed when the eyelid opening is below
0.5 x the video's median opening (hand-set). Facts: 8058 such frames (0.59 %, 2417 runs, 1310 of them one frame long);
the 1330 truth frames hold no truly closed eye (the 4 frames the rule calls closed all have human pupil labels; humans
labeled pupils down to an opening ratio of 0.36), so the line cannot be derived from the labels. The user judged a
stratified sample of 80 frames (`sample.csv`, `closed_review_verdicts.csv`; opening-ratio bins < 0.2 / 0.2-0.35 /
0.35-0.5 / 0.5-0.7 x eyelid point flagged or not, 10 each; 79 judged): 33 fully closed, 30 open with wrong eyelid or
corner points (the pupil right in 20 of them), 16 half open with a labelable pupil. Findings (`verdict_summary.csv`,
`feature_crosstabs.csv`, `line_candidates.csv`, `confidence_line_candidates.csv`):
- The opening ratio does not separate a closed eye from an open eye with collapsed eyelid points: no line between
  0.1 and 0.7 does better than 61 % on the judged frames. Among the frames the rule calls closed, about half are
  open eyes with wrong outline points.
- The finders flag all of them (fitting and PCA flag every judged closed-now frame); an eyelid point over the
  threshold marks a real blink (28 of 35 flagged frames closed) rather than a wrong eyelid (7 of 30 wrong-eyelid
  frames flagged): the finder cannot tell the two apart.
- The mean heatmap confidence of the four pupil points does: closed frames 0.20 (median; max 0.67), wrong-eyelid
  frames 0.64, half-open frames 0.60. The line confidence < 0.40 calls 32 of 33 closed frames closed and 5 of 46 open
  frames (accuracy 92 %, the best of 0.20 ... 0.70 together with 0.35); on the 1318 labeled-pupil truth frames it
  calls 4 closed, all four wrong frames the model had already lost. This is the same construction as step 11's
  not-open rule for implausible frames (confidence < 0.527 on judged frames), with its own judged set.
Decision for the formal pipeline (to be implemented as an option, default unchanged until then): closed = mean pupil
confidence < 0.40 (calibrated as above); a frame with a collapsed opening but a confident pupil is an open eye with
wrong outline points and goes through the ordinary flagged path (candidates = points over the threshold, smallest
subset, both ranges); no separate "reopen" branch and no new hand-set number.

**1b. Variant D** (`--tiebreak margin-all`: the largest range margin among the passing subsets of ANY size, runs
`videoNN_..._devorder_marginall`, 2026-11-08 00:15): truth frames identical to C (15 / 4 / 1); same 14,395 repaired
frames; replaces more points than C in 374 frames (15,552 against 15,159 points). Of the 7 frames the user judged
"both changes needed", D moves both points in 4. Whether the extra points are right is for a by-eye check
(`subset_review/subset_d_review.html`, 20 frames); verdicts pending.

**3b. The confidence closed-eye rule in the repair** (`flag_and_repair.py --closed confidence`, line 0.40 in
`closed_frames/closed_line_chosen.csv`, runs `videoNN_..._confclosed` and `_devorder_margin_confclosed`, 2026-11-08
00:15). Against the opening rule: truth frames 16 of 23 wrong frames repaired (15), 2 still flagged (4), 5 called
closed (4; the 4 labeled ones are the same already-wrong frames, the fifth has no pupil label), 1 good frame made worse
(1). Whole videos: 15,878 frames repaired (14,395), 4404 flagged and not repaired (6694), 7358 closed (8058). Of the
8058 frames the opening rule called closed, 2008 are now repaired, 1793 pass without a flag, 469 are flagged and
plausible as they are, 575 stay flagged, 3213 are still closed. Of the 7358 the confidence rule calls closed, 2865
were "flagged, not repaired" before (no visible pupil: nothing to repair) and 525 were repaired before; those 525 are
for a by-eye check (`subset_review/closed_conf_review.html`, 20 frames); verdicts pending.
