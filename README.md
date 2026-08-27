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

The eye scaling experiment has been run with 20, 50, and 100 training frames from the first 4 minutes. All three models were evaluated on the same 100 manually labeled frames from the final minute.

The current results are organized locally under:

```text
local_data/test_sets/eye_last_minute_100/deliverables/
```

The curve is not yet a reliable plateau: the overall error is not monotonically decreasing with training-frame count. The next step is to audit the remaining outliers and verify that the training subsets are comparable before adding 150 frames or testing other architectures.

Current test metrics:

```text
20 frames:  overall RMSE 36.17 px, pupil center RMSE 16.50 px, pupil width MAE 40.44 px
50 frames:  overall RMSE 42.14 px, pupil center RMSE 17.18 px, pupil width MAE 25.41 px
100 frames: overall RMSE 56.83 px, pupil center RMSE 35.17 px, pupil width MAE 29.44 px
```

## Where To Look

Use `docs/` for the learning workflow and `scripts/` for executable steps. Use the local `deliverables/` folder for advisor-facing outputs:

```text
01_summary_figures/  plots and curves
02_outlier_checks/   worst-frame visual inspections
03_share_videos/     videos with prediction overlays
04_tables/           metrics and audit CSV files
05_reference_inputs/ source test clip
```

Next scaling step:

```bash
/Users/lizhiheng/miniforge3/envs/DEEPLABCUT/bin/python scripts/expand_eye_training_frames.py --target 50
bash scripts/label_eye_frames.sh
```

After labeling the added frames, recreate the training dataset, train the next model, and evaluate it on the same fixed 100-frame test set.

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

## Documentation

Start with:

```text
docs/01_local_setup.md
docs/02_eye_project.md
docs/04_metrics_and_outputs.md
```

The body video workflow is documented but deferred until the eye pipeline is more mature.
