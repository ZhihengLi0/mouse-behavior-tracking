# Mouse Behavior Tracking

Local DeepLabCut workflow for mouse eye, pupil, blink, face, ear, and forepaw tracking.

This repository tracks code, configuration templates, and notes only.
Raw videos, PDFs, trained models, extracted labels, and generated results stay local.

## Repository Logic

There are two layers:

1. GitHub/repository layer: code, configuration, documentation, and reproducible commands.
2. Local research-data layer: raw videos, PDFs, extracted frames, labels, model weights, predictions, and plots.

The local research-data layer is intentionally ignored by Git. This keeps private or large files out of GitHub.

Main local inputs:

- `face.mp4`: eye video for pupil and blink tracking.
- `body.mp4`: deferred face/body video for facial features, ears, and forepaws.
- `*.pdf`: local reading material.
- `local_data/`: temporary files and local results.

## Current Eye Project

Active DeepLabCut project:

```text
dlc_projects/EyePupilBlink-Zhiheng-2026-08-17/config.yaml
```

Tracked eye keypoints:

```text
pupil_top
pupil_bottom
pupil_left
pupil_right
eyelid_top
eyelid_bottom
eye_nasal_corner
eye_temporal_corner
```

Current train/test split:

```text
training pool: first 4 minutes of face.mp4
held-out test pool: last 1 minute of face.mp4
```

## Current Status

The first 20/50/100-frame scaling study and the 100-frame ResNet-50 versus
HRNet-W32 comparison are complete. The controlled HRNet-W32 batch-size
sweep is also complete for batch sizes `1, 2, 4, 8, 16`.

A controlled batch-size `32` extension was attempted with the same
HRNet-W32 model and split (`shuffle9`). It was stopped before epoch 1
because swap grew to about 14.5 GB on this 16 GB Mac. It has no valid
result and is excluded together with batch 64.

Every batch-size run used the same 100-frame label pool from the first
four minutes, the same 95/5 internal split, 200 epochs, and the same
locked 100-frame final-minute test set.

| Batch | DLC best epoch | Minimum internal valid loss | External overall RMSE | Pupil center RMSE | Pupil width MAE |
|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 0.00836 | 51.31 px | 15.76 px | 32.95 px |
| 2 | 90 | 0.00824 | 20.06 px | 9.53 px | 20.11 px |
| 4 | 20 | 0.00820 | 24.31 px | 10.53 px | 22.86 px |
| 8 | 10 | 0.00806 | 25.68 px | 11.96 px | 26.31 px |
| 16 | 10 | 0.00790 | 38.61 px | 16.68 px | 37.99 px |

The pre-specified internal-validation rule selects batch size **16**.
Batch size **2** has the lowest external-test errors, but that final-minute
result is report-only and must not be used retroactively for tuning. The
disagreement matters because internal validation contains only five
images and all five runs were performed once.

Audited, publication-safe outputs are in
[`results/01_batch_size_sweep/`](results/01_batch_size_sweep/).
The detailed interpretation is in the
[result report](results/01_batch_size_sweep/README.md).

### Architecture Comparison At Batch 2

Five bottom-up backbones were trained with the identical 95/5 split, identical
labels, batch 2, 200 epochs, CPU, and seed 42, then evaluated on the same
100 final-minute frames. RTMPose-S is excluded by design: DeepLabCut configured
it as a top-down model with an SSDLite detector, so it is not controlled against
these bottom-up single-eye models.

Results are in [`results/02_architecture_sweep/`](results/02_architecture_sweep/).

**That comparison does not support an architecture ranking, and the reason is
measured rather than assumed.** Each model was tested at the checkpoint
DeepLabCut selected by maximum internal mAP over five validation frames. Those
five labels are imperfect, and evaluating every surviving checkpoint of every
architecture showed that checkpoint choice alone moves a single model by up to
16 px, which is larger than several architecture gaps.

The full architecture-by-checkpoint matrix is in
[`results/03_checkpoint_matrix/`](results/03_checkpoint_matrix/). It reports
four views of the same data and claims a ranking only where two error ranges do
not overlap. Two corrections came out of it:

- CSPNeXt-S is not a 70 px model. That number came from an epoch-10 checkpoint.
  Across epochs 100-200 it sits at 25.8 px and is the most stable backbone
  tested (SD 0.10 px).
- The five-frame rule cost ResNet-50 9.75 px and CSPNeXt-S 44.40 px, but cost
  HRNet-W32 only 0.16 px. HRNet-W32 led the published table largely because the
  selection rule happened to work for it.

On the mean over the five checkpoints every model kept, which selects nothing
and never touches the validation labels, ResNet-50 (20.47 px) and HRNet-W32
(21.57 px) are within one standard deviation of each other and cannot be
separated.

### Next: Rebuilt Validation Set

Agreed with the advisor, the hyperparameters are being settled again before the
active-learning experiment, with three changes:

1. All 100 training-pool labels are reviewed, including the five validation
   frames that carried a slight offset and were deliberately left unchanged
   mid-sweep to hold the comparison variables fixed.
2. The internal split changes from 95/5 to 80/20, so internal validation can
   actually separate models.
3. The epoch budget drops from 200 to 100, because validation loss bottoms out
   near epoch 50-75 and everything after that was overfitting.

Two configuration changes are required for the shorter schedule to remain a
valid experiment, and they are easy to miss:

- The learning-rate milestones must scale from `[160, 190]` to `[80, 95]`.
  Left unchanged they never fire in a 100-epoch run, so the model would train at
  the initial rate throughout and never reach its fine-tuning phase.
- `save_epochs` drops from 25 to 10 and snapshot retention is raised, because
  the surviving checkpoints in the matrix above were too coarse to locate a real
  optimum.

Interrupted runs are restarted from scratch rather than resumed. Resuming resets
DeepLabCut's memory of the best metric, which is how HRNet-W48's best checkpoint
was destroyed: the first evaluation after a resume writes a new best snapshot,
and when that epoch matches the existing best epoch the manager overwrites the
file and then deletes it as a stale copy. At 100 epochs a restart is cheaper
than the risk.

## Where To Look

- `CHANGELOG.md`: experiment phases and what each tagged version established.
- `docs/`: learning workflow and metric definitions.
- `scripts/`: extraction, labeling, training, evaluation, and publishing.
- `results/`: versioned aggregate tables, figures, and reports.
- `local_data/`: ignored local predictions, outlier montages, and test assets.
- `dlc_projects/`: DeepLabCut project; generated models and labels are ignored.

Next scientific work:

1. Review all 100 training-pool labels and re-split 80/20.
2. Rerun the architecture comparison at batch 2, 100 epochs, with rescaled
   learning-rate milestones and denser snapshots.
3. Calibrate DLC likelihood before applying a confidence cutoff; no backbone
   currently produces enough points above 0.6 for a hard filter.
4. Run the active-learning comparison using K-means frame selection and
   the `uncertain`, `jump`, and `fitting` outlier methods.

## Useful Commands

Check the environment:

```bash
bash scripts/check_setup.sh
```

Open training-frame labeling:

```bash
bash scripts/label_eye_frames.sh
```

Extract the fixed 100-frame eye test set:

```bash
/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python scripts/extract_eye_test_frames.py
```

Open test-frame labeling:

```bash
bash scripts/label_eye_test_frames.sh
```

Evaluate the current model on the test set:

```bash
/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python scripts/evaluate_eye_test_set.py
```

Regenerate the current result plot:

```bash
/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python scripts/plot_eye_test_results.py
```

Regenerate outlier tables and montages:

```bash
/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python scripts/plot_eye_outliers.py --train-frames 100
/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python scripts/plot_eye_outliers.py --train-frames 100 --model-label hrnet_w32
```

## Documentation

Start with:

```text
docs/01_local_setup.md
docs/02_eye_project.md
docs/04_metrics_and_outputs.md
```

The body video workflow is documented but deferred until the eye pipeline is more mature.

## HRNet-W32 Batch-Size Sweep

The sweep completed on 2026-09-08. Existing `shuffle4` is batch size 8;
`shuffle5-8` are batch sizes 1, 2, 4, and 16. All five model runs,
external evaluations, and plots completed successfully.

Regenerate and audit the versioned result package:

```bash
/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python scripts/publish_eye_batch_sweep.py
```

The command validates completion, architecture, configured batch sizes,
tested snapshot epochs, and identical external frame order before writing
the result package. Full local artifacts remain in:

```text
local_data/experiments/01_batch_size_sweep/
local_data/test_sets/eye_last_minute_100/predictions_100train_hrnet_w32_batch*/
```

These local directories contain predictions and logs and remain ignored
by Git. Only aggregate result tables, plots, and reports are published.

The batch-32 script now refuses to rerun unless its explicit
`--force-memory-risk` flag is passed on a machine with substantially
more RAM. Inspect the local attempt:

```bash
bash scripts/status_eye_batch32_extension.sh
```

The architecture sweep fixes `batch_size=2` and uses the same 95/5
split for HRNet-W18, HRNet-W32, HRNet-W48, ResNet-50, and CSPNeXt-S.
Check the serial queue:

```bash
bash scripts/status_eye_architecture_queue.sh
```
