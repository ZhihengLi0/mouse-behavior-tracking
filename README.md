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

**Second experimental era, in progress.** All 100 training-pool labels were
re-reviewed on 2026-09-10. The review moved `pupil_top` systematically by about
20 px (86% of points in the same direction), which is a definition correction,
not noise removal. Every model, prediction, and error number produced before
the review is therefore measured against labels that no longer exist and must
not be compared with anything produced after it.

First-era results were removed from `main` and remain retrievable, with the
READMEs explaining their limitations, at:

- tag `v0.1.0`: HRNet-W32 batch-size sweep (95/5 split, 200 epochs)
- tag `v0.2.0`: five-backbone architecture sweep and the
  architecture-by-checkpoint matrix that showed why its ranking could not be
  trusted: checkpoint choice alone moved one model 16 px, more than the
  architecture gaps being measured, because checkpoints were selected on five
  suspect validation frames.

The second era changes, agreed with the advisor:

- reviewed labels for all 100 training-pool frames
- internal split 80/20 instead of 95/5, so validation can separate models
- 100 epochs instead of 200; validation loss bottomed out near epoch 50-75
- learning-rate milestones rescaled from `[160, 190]` to `[80, 95]` (left
  unchanged they would never fire in a 100-epoch run)
- snapshots every 10 epochs with retention raised, so checkpoint analysis is
  no longer limited to five coarse survivors
- interrupted runs restart from scratch; resuming resets DeepLabCut's
  best-metric memory and destroyed a best snapshot in the first era

Currently running: HRNet-W32 batch-size sweep at batch 16, 8, 4, 2
(shuffles 24, 23, 22, 21) under `local_data/experiments/04_batch_sweep_80_20/`.
The final-minute test labels are being re-reviewed to the same standard in
parallel; no external evaluation happens until that review is done.

## Where To Look

- `CHANGELOG.md`: experiment phases and what each tagged version established.
- `archive/`: superseded-era artifacts (own inner git repo; ignored, never pushed).
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
bash environment/check_setup.sh
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
environment/01_local_setup.md
docs/02_eye_project.md
docs/04_metrics_and_outputs.md
```

The body video workflow is documented but deferred until the eye pipeline is more mature.

## Second-Era Commands

Recreate the 80/20 split and shuffles (idempotent, refuses mismatched splits):

```bash
/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python scripts/prepare_eye_split_80_20.py
```

Run the batch sweep queue (fcntl-locked, no-resume, training only):

```bash
nohup caffeinate -i /Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python \
  scripts/run_eye_batch_sweep_80_20.py >> \
  local_data/experiments/04_batch_sweep_80_20_stdout.log 2>&1 &
nohup bash scripts/watch_eye_batch_sweep_80_20.sh >/dev/null 2>&1 &
```

The split record with filenames is `local_data/experiments/split_80_20.json`.
External evaluation deliberately waits for the reviewed final-minute labels.
