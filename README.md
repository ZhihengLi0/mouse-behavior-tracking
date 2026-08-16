# Mouse Behavior Tracking

Local DeepLabCut workflow for mouse eye, pupil, blink, face, ear, and forepaw tracking.

This repository tracks code, configuration templates, and notes only.
Raw videos, PDFs, trained models, extracted labels, and generated results stay local.

## Current Local Data

These files are intentionally ignored by git:

- `face.mp4`: eye close-up video for pupil and blink tracking.
- `body.mp4`: face/body video for facial features, ears, and forepaws.
- `*.pdf`: local reading materials and unpublished/review materials.

## Project Plan

1. Set up a DeepLabCut environment.
2. Create the eye project and label frames from `face.mp4`.
3. Train a first eye model.
4. Analyze `face.mp4` and inspect the labeled video.
5. Add post-processing for pupil size and blink detection.
6. Create the face/body project from `body.mp4`.
7. Train and evaluate face/body tracking.

Start with `docs/01_local_setup.md`.
