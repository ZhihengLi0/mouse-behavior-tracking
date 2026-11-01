# Flagged frames come in episodes; what the frames around an episode give (2026-10-31)

Direction 2 of the 2026-10-31 meeting. Script `scripts/episode_structure.py`; write-up in
`docs/next_phase_analysis_2026-10-31.md`, section 2.

| file | content |
|---|---|
| `episodes_by_video.csv`, `episode_lengths.png`, `timeline.png` | runs of consecutive flagged frames per video and finder (fit, pca, either): count, length, share of flagged frames in long runs; where they are in each video |
| `truth_in_episodes.csv` | every flagged truth frame of both routes: wrong or right, episode length and position |
| `temporal_fill_truth.csv`, `temporal_fill_summary.csv`, `temporal_fill_by_length.csv`, `temporal_fill.png` | on the same frames, the error against the human label when the flagged points are filled from the frames around the episode (interpolation, nearest unflagged frame, one-step forecast; and interpolation of all 8 points), next to the second-peak repair |
| `displacement_vs_gap.csv`, `displacement_vs_gap_by_video.csv`, `displacement_vs_gap.png` | label-free bound: error of interpolating the pupil points across a gap of L frames on unflagged stretches |

Only flagged frames are ever changed; unflagged frames keep the model's coordinates (no temporal smoothing, the rule
for the alignment with the neural signal). Main numbers: 19 wrong truth frames the fitting finder flags, still wrong
after the second-peak repair 7, after interpolation across the episode 2, after interpolation of all 8 points 1;
interpolation across 200 frames costs 1.1 % of eye width at the median.
