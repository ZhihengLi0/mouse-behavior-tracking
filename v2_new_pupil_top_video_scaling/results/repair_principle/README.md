# Why a second-peak repair falls short, why a flagged frame is sent back (2026-10-31)

Direction 1 of the 2026-10-31 meeting: look at the mechanism, add no rule. Script `scripts/repair_principle.py`
(parts `reasons`, `repaired`, `locref`); write-up with the numbers in `docs/next_phase_analysis_2026-10-31.md`,
section 1.

| file | content |
|---|---|
| `reasons.csv`, `reasons_total.csv`, `reasons.png` | every still-flagged frame of the 19 videos, fitting and PCA route, by the reason the code did not repair it (no candidate because an eye-outline point tripped the finder; no second peak; second peak tried and rejected by the label range) |
| `reasons_by_verdict.csv`, `reasons_sub_by_verdict.csv`, `reasons_worst_by_verdict.csv` | the 184 still-flagged frames of the by-eye spot check by reason, by which quantity failed, and by the kind of point that tripped the finder, against the verdicts |
| `repaired_points.csv`, `repaired_by_verdict.csv`, `heatmap_stats.png` | the 178 replaced points of the spot check with the heatmap statistics of the peaks run, by verdict |
| `heatmap_maxima.csv`, `heatmap_maxima_summary.csv` | full heatmaps recomputed for the 24 points judged not right and 23 judged right: every local maximum near the eye |
| `peak2_locref.csv` | the 23 truth-frame repairs with the second peak at the cell centre plus median offset (current) and with the network's own location-refinement offset |
| local only | `reasons_frames.csv.gz` (per frame), `closer_not_right.jpg`, `right_for_comparison.jpg` (galleries), logs |

Main numbers: fitting route 86 % of still-flagged frames never had a repair attempted because an outline point was
over the threshold (PCA route 48 %); of the by-eye false alarms among still-flagged frames 59 of 69 were tripped by an
eye corner; when a second peak was tried and rejected, the frame was really wrong in 30 of 32 judged cases; a
not-right repair has peak 1 and peak 2 only 0.4 eye widths apart (right repairs 0.8); the sub-pixel offset of the
second peak changes the error by under 0.5 % of eye width.
