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
