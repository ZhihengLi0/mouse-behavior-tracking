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

The curve is not yet a reliable plateau: the overall error is not monotonically decreasing with training-frame count. The remaining outliers have been regenerated from the latest manual labels, and a controlled 100-frame architecture comparison has also been completed.

Current test metrics:

```text
ResNet-50, 20 frames:  overall RMSE 35.48 px, pupil center RMSE 16.79 px, pupil width MAE 43.10 px
ResNet-50, 50 frames:  overall RMSE 42.25 px, pupil center RMSE 17.45 px, pupil width MAE 28.74 px
ResNet-50, 100 frames: overall RMSE 56.82 px, pupil center RMSE 35.77 px, pupil width MAE 32.48 px
HRNet-W32, 100 frames: overall RMSE 25.68 px, pupil center RMSE 11.96 px, pupil width MAE 26.31 px
```

The ResNet-50 and HRNet-W32 100-frame runs use the same 95 internal training images, the same 5 internal validation images, and the same fixed 100-frame held-out test set. HRNet-W32 has substantially lower raw-coordinate error, but all 800 held-out predictions have likelihood below `0.6`. This confidence-calibration problem must be reported alongside the error improvement.

## Where To Look

Use `docs/` for the learning workflow and `scripts/` for executable steps. Use the local `deliverables/` folder for advisor-facing outputs:

```text
01_summary_figures/  plots and curves
02_outlier_checks/   worst-frame visual inspections
03_share_videos/     videos with prediction overlays
04_tables/           metrics and audit CSV files
05_reference_inputs/ source test clip
```

Next scientific checks:

```text
1. Inspect the current ResNet-50 and HRNet-W32 outlier montages.
2. Generate and inspect a final-minute HRNet-W32 labeled video with all points visible.
3. Diagnose HRNet-W32 likelihood calibration before choosing a confidence cutoff.
4. Only then decide whether to add more training frames or another architecture.
```

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
