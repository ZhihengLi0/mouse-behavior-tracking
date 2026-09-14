# Changelog

Experiment phases for the mouse eye keypoint pipeline. Each version marks a
state where results were audited and published, so a later phase can be
compared against exactly what was known before it.

Only aggregate tables, figures, reports, and code are versioned here. Videos,
frames, labels, model weights, and predictions stay in ignored local
directories.

## v0.2.0 - 2026-09-10 - Architecture screening, and why it has to be redone

Tag: `v0.2.0`

This is the end of the first experimental era and the point at which the
hyperparameter selection restarts on a rebuilt validation set. Everything below
was produced with a 95 training / 5 internal validation split and a 200-epoch
budget.

**Added**

- `results/02_architecture_sweep/`: five bottom-up backbones (HRNet-W18,
  HRNet-W32, HRNet-W48, ResNet-50, CSPNeXt-S) on the identical 95/5 split,
  identical labels, batch 2, 200 epochs, CPU, seed 42, evaluated on the same
  100 final-minute frames. RTMPose-S excluded by design as a top-down model
  with an SSDLite detector.
- `results/03_checkpoint_matrix/`: every architecture evaluated at every
  surviving checkpoint, 25 evaluations, no retraining. Reports four views of
  the same data and claims a ranking only where two error ranges do not
  overlap.
- `scripts/publish_eye_architecture_sweep.py`,
  `scripts/publish_eye_checkpoint_matrix.py`,
  `scripts/run_eye_checkpoint_matrix.sh`.

**Findings that invalidate the architecture ranking**

- Each model was tested at the checkpoint DeepLabCut selects by maximum
  internal mAP over five validation frames. Those five labels carry a slight
  offset, noticed mid-sweep and deliberately left unchanged so the comparison
  variables stayed fixed.
- Checkpoint choice alone moves a single model by up to 16 px, which is larger
  than several architecture gaps. The published one-number-per-architecture
  table was therefore not measuring architecture.
- CSPNeXt-S is not a 70 px model. That figure came from an epoch-10 checkpoint.
  Across epochs 100-200 it sits at 25.8 px and is the most stable backbone
  tested, SD 0.10 px.
- The five-frame rule cost ResNet-50 9.75 px and CSPNeXt-S 44.40 px, but cost
  HRNet-W32 only 0.16 px. HRNet-W32 led the table largely because the selection
  rule happened to suit it.
- On the mean over the five checkpoints every model kept, which selects nothing
  and never reads the validation labels, ResNet-50 (20.47 px, SD 1.72) and
  HRNet-W32 (21.57 px, SD 2.25) are within one standard deviation and cannot be
  separated.
- HRNet-W48 cannot be formally ruled out, because its comparable checkpoint no
  longer exists. Its training was interrupted twice and resuming destroyed the
  best snapshot: DeepLabCut's snapshot manager restarts with no memory of the
  best metric, so the first evaluation after a resume writes a new best; when
  that epoch matched the existing best epoch the manager overwrote the file and
  then deleted it as a stale copy. No later epoch strictly exceeded the previous
  best mAP, so it was never recreated, and the epochs holding the maximum
  internal mAP had already been pruned. Evidence does not suggest it was
  underrated: on the leakage-free mean it is second worst, and in the other
  models the early internally selected checkpoints were usually worse
  externally, sometimes much worse.

**Changed**

- `scripts/train_eye_model.py` refuses to retrain over weights that already
  reached the epoch budget but lost their best snapshot, instead of silently
  discarding finished work. It also subtracts the resume epoch from the epoch
  target, because DeepLabCut treats `epochs` as additional epochs beyond a
  resumed snapshot.
- `scripts/evaluate_eye_test_set.py` accepts `--snapshot-index` so a specific
  numbered snapshot can be evaluated.

**Known limitations carried forward**

- Single run per configuration; only large gaps are resolvable.
- The final-minute 100 frames have been inspected and relabeled during earlier
  debugging, so they are a controlled comparison set rather than a pristine test
  set. Any new video needs a fresh untouched confirmation set.
- Batch 2 was chosen because it had the lowest final-minute error, which used
  the report-only set for a selection decision. This is a known protocol
  deviation, so batch 2 is provisional.
- No backbone produces enough points above likelihood 0.6 for a hard `pcutoff`
  filter, which constrains the `uncertain` outlier detector.

## v0.1.0 - 2026-09-06 - Batch-size sweep

Tag: `v0.1.0`

**Added**

- `results/01_batch_size_sweep/`: HRNet-W32 at batch 1, 2, 4, 8, 16, 200
  epochs, identical 95/5 split, same 100 final-minute frames.
- Restartable sweep runner with an `fcntl` lock, resource watchdog, and status
  script.

**Findings**

- The pre-specified internal rule selected batch 16; the final-minute set was
  best at batch 2. First time the internal and external signals disagreed.
- Minimum internal validation losses span only 5.8% across all five runs while
  external error spans 156%, so the internal signal had no separating power.
- Batch 32 was stopped before epoch 1 because swap reached about 14.5 GiB on a
  16 GiB Mac. No valid result; batch 64 excluded.

## Next phase, not yet released

Hyperparameters are being settled again before the active-learning experiment,
agreed with the advisor:

1. All 100 training-pool labels reviewed, including the five offset validation
   frames.
2. Internal split changes from 95/5 to 80/20 so internal validation can
   separate models.
3. Epoch budget drops from 200 to 100, because validation loss bottoms out near
   epoch 50-75.

Two configuration changes are required for the shorter schedule to stay a valid
experiment:

- Learning-rate milestones scale from `[160, 190]` to `[80, 95]`. Left
  unchanged they never fire in a 100-epoch run, so the model would train at the
  initial rate throughout and never reach its fine-tuning phase.
- `save_epochs` drops from 25 to 10 with raised retention, because the
  surviving checkpoints in v0.2.0 were too coarse to locate a real optimum.

Interrupted runs are restarted from scratch rather than resumed, to avoid the
best-snapshot destruction described above.

## v0.3.0 - 2026-09-11 - Second-era batch sweep: internal and external agree

Tag: `v0.3.0`

First result of the second era (reviewed labels, temporal block 80/20 split,
100 epochs, milestones [80, 95], snapshots every 10 epochs, no-resume policy).

- Internal rule (declared before results): within-run snapshot-best by
  internal mAP; across batches lowest minimum validation loss. It selected
  **batch 2** (0.01107 at epoch 50).
- The reviewed final-minute set, used report-only, independently ranks batch 2
  lowest: 21.07 px overall RMSE, 11.08 px pupil-center RMSE, 375/800 points at
  likelihood >= 0.6 (era-1 best was 59/800).
- **Internal and external agree for the first time**, the direct evidence that
  the rebuilt validation set resolves what the five-frame one could not.
- Best snapshots land mid-training (epochs 90/60/50/50), not at epoch 10-20.
- The rescaled LR milestones fired on schedule (loss drop at epoch 80).
- Published: `results/batch-size-selection/`. Not comparable with v0.1.0.

Architecture comparison at batch 2 started the same morning (shuffles 25-28
plus shuffle 21 reused for HRNet-W32).

## v0.4.0 - 2026-09-12 - Active learning complete: a decisive null result

Tag: `v0.4.0`

Five rounds, three branches (uncertain / jump / fitting), 300 human-reviewed
frames added under the frozen protocol, every point scored once on the
report-only final-minute set.

- Median frame error flat for every branch and round (16.3-17.5 px, inside
  the measured ±2.4 px single-run noise band); no mean finished below the
  17.70 px baseline. The frozen plateau rule fires at round 2 everywhere.
- Detectors statistically indistinguishable; the branch means' excursions
  (up to 48 px) trace to 3-6 blink/occlusion frames - all branches kept
  rediscovering the same 160.5-160.7 s event from different suspicion logics.
- Conclusion delivered by the curves: ~80 consistent labels saturate
  typical-frame accuracy here; the residual budget is label-noise-order floor
  plus ill-posed occluded frames, so the next lever is likelihood-gated
  blink/occlusion handling, not more keypoint labels.
- Operationally: MPS training (9x) plus interleaved human/machine scheduling
  compressed the planned 5-day experiment into ~30 hours; 15/15 label batches
  audited; per-round provenance in active-learning/branches (local).

## v0.5.0 - 2026-09-14

Active learning extension (rounds 6-11) and closure. Round 6, the first
round labeled under the amended blink-eyelid rule, dropped all three branch
medians together (17 -> 14-15 px floor); rounds 7-11 confirmed a second
saturation by pre-registered referee rules, and the experiment closed at
round 11 (cap 12). New units: time-series-analysis (unsupervised error
mining: 120 events, 11.4% of frames, blink signals) and
active-learning/analysis-120frame-spike (the round-2 mean spike autopsied
to 3-5 teleporting frames). Central finding upgraded: labeling standards,
not label count, move the error floor; detector choice never mattered.
