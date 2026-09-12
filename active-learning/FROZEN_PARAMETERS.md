# Active Learning: Frozen Parameters

Frozen 2026-09-12 00:45 CDT, before round 1, per the advisor's requirement.
Nothing below may change during the experiment. The final-minute curve is
never used to adapt any of it.

## Selection provenance
- Architecture: **ResNet-50** (shuffle 27 weights are round-0). Selected by the
  pre-declared internal three-tier rule: (1) accuracy on the 20 validation
  frames - five-way statistical tie (loss span 1.4%; paired frame-level
  bootstrap, 10000 resamples, all 20 pairwise 95% CIs contain 0);
  (2) tie-break = validation-frame confidence: ResNet-50 117/160 points at
  likelihood >= 0.6 (73%) vs 61% next; (3) speed tier not needed. The
  report-only final-minute set independently shows ResNet-50 lowest
  (20.10 px). User confirmed 2026-09-12.
- Batch size 2: selected 2026-09-11 by lowest minimum internal validation
  loss; external independently agreed (results in ../batch-size-selection/).

## Fixed configuration
- Seed labels: the reviewed 100-frame tables (copies in training-data/).
- Split: temporal block 80 train / 20 validation; validation NEVER grows.
- Training: 100 epochs, batch 2, CPU, LR milestones [80, 95], snapshots every
  10 epochs (retention 12), from scratch each round (--no-resume), DLC seed 42.
- Rounds: 5. Frames per round per branch: 20. Cumulative training frames per
  branch: 80 -> 180. x-axis of every convergence curve = cumulative
  human-reviewed TRAINING frames (validation's 20 not counted).
- Branches (independent seed copies; no branch inherits another's frames):
  extractionalgorithm = "kmeans" for all three;
  outlieralgorithm = "uncertain" (p_bound = 0.6)
                   | "jump"      (epsilon = 20 px)
                   | "fitting"   (epsilon = 20 px, DLC default AR/MA degrees).
  Candidate pool: frames 0-14399 of face.mp4 only (first 4 minutes).
- k-means: DLC's built-in implementation as shipped; numframes2pick = 20.
  numpy seed 42 set before each call (DLC does not expose its own seed - the
  residual nondeterminism is recorded here as a limitation).
- Duplicate policy: a selected frame that is already labeled/reviewed in that
  branch is replaced by the next k-means candidate.
- Occlusion policy: fully occluded keypoints are left unlabeled; partially
  visible ones are labeled at best anatomical estimate.
- Human review: all 8 keypoints on every selected frame; machine predictions
  are corrected, never accepted merely because the frame was selected.
- Evaluation: after each round's retraining, the branch model is evaluated
  ONCE on the reviewed final-minute 100 frames (report-only). Checkpoint for
  evaluation: snapshot-best by internal mAP on the fixed 20 validation frames.
- Plateau: judged retrospectively after round 5 as two consecutive rounds with
  < 5% external improvement. Never used to stop early or tune anything.
- Leakage rule: first 4 minutes = selection/labeling/training; final minute =
  report only, never selects, tunes, or stops anything.
