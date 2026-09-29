# Scaling Curve: Training-Frame Growth Study (20 / 50 / 100)

The project's first study (August 2026): how does final-minute error change as
the manually labelled training pool grows from 20 to 50 to 100 frames?

- `scripts/` — the exact tooling of that era, recovered verbatim from the main
  repo's `v0.1.0` tag: frame expansion, test-set extraction, labeling
  launchers, evaluation, scaling/comparison/outlier plotting. Training itself
  used `deeplabcut.train_network` directly; the commands are recorded in
  `../AI_HANDOFF.md`.
- `results/` — the reconstructed tables and the study's findings. The original
  figures and the model weights (shuffles 1-4) were deleted in the 2026-09-10
  cleanup and cannot be regenerated; the numbers survive in full. Original
  figures may still exist as attachments in the advisor Slack thread
  (2026-08-22 to 2026-08-28).

Everything here is measured against era-0 labels and the old 95/5 split, and
is **not comparable** with anything after the 2026-09-10 label review. Its
value: it documents the first recorded internal/external disagreement and the
pupil_left failure cluster that motivated the switch to HRNet-W32, the outlier
tooling, and eventually the active-learning design.
