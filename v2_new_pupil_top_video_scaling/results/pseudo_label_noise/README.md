# Noise a self-training round would start from (2026-10-31)

Direction 3 of the 2026-10-31 meeting. Script `scripts/pseudo_label_noise.py`; write-up and the pilot design in
`docs/next_phase_analysis_2026-10-31.md`, section 3.

| file | content |
|---|---|
| `noise_total.csv` | the 1330 truth frames by class (no finder flagged it; repaired; judged plausible as it is; still flagged; closed): share of the whole videos, error of the would-be pseudo label against the human label |
| `noise.csv` | the same per video |
| `noise_points.csv` | per pupil point on the usable classes |
| `noise.png` | error histogram by class; per-video median and 90th percentile of the unflagged frames |

Main numbers: 95.7 % of all frames are unflagged, their truth-frame error has median 2.0 % of eye width, 90th
percentile 4.4 %, none over the 14 % line; but 43 usable truth frames have the pupil_left point alone more than 14 %
off, so a pseudo label should be filtered per point.
