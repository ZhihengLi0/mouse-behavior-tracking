# Blind re-label test (prepared 2026-10-26, not carried out)

`scripts/relabel_test.py make` selected 40 of the 1330 truth frames (20 with frame RMSE 20–50 px from videos 1, 4, 8,
10, 15, 18; 10 above 50 px; 10 below 10 px; `selection_key.csv`) and prepared a blind labeling folder (images only, local).
The purpose was a number for the labeler's own repeatability, to decide whether the 20-px tier is within label noise.

The labeler did label the 40 frames once, but the save failed (the label file prepared by the script used the path
level `labels/…` while the plugin addresses rows by the folder's real path `labeling/…`; the plugin refused the merge
and the points were lost when the window closed). The script has been corrected (`write_empty_label_file`), the test
was not repeated.

Decision instead (labeler + Kaiwen, Slack 2026-10-26): the 20–50 px tier is judged by eye on the sampled contact sheets
of `results/wrong_frames_gallery/` (frames 20–50 px: the labeler cannot place the pupil edge better than that, the
dark core vs the wider ring; frames > 50 px: visibly wrong). The "must fix" tier is 50 px; 20 px is reported only.
`scripts/relabel_test.py eval` remains usable if the test is ever run.
