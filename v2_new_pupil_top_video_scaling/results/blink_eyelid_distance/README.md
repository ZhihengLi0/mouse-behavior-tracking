# Eye closure from the eyelid distance: first round, waiting for the human spot check (2026-10-04)

First round agreed with Kaiwen on 2026-10-04: judge closure from the distance between the upper and the lower eyelid
point, run it on the videos, then spot-check part of the result by eye. The finer classes (blink / closed / squint) come
after this round. **No conclusion yet: the spot check has not been done.**

**Basic information.** Videos 0-12 (finished videos). Model of a video = the model at its plateau point (plateau at 0
labels: the 20-label model), ResNet-50, final snapshot, whole-video prediction, no confidence cut-off. Rule:
eye opening `O = y(eyelid_bottom) - y(eyelid_top)`; relative opening `r = O / median of O over the video`; closure frame
when `r < 0.70`; closure frames less than 5 frames (83 ms) apart form one event. The threshold 0.70 was set before the
spot check and is to be revised by it. Script: `../../scripts/blink_by_eyelid_distance.py`.

| file | content |
|---|---|
| `blink_eyelid_events_all_videos.csv` | all 733 events: video, start, end, deepest frame, duration, minimum relative opening |
| `blink_eyelid_summary_all_videos.csv` | per video: events, events per minute, median duration, share of closure frames |
| `blink_eyelid_spotcheck_all_videos.csv` | the 140 sampled items (101 events spread over the event depth, 39 "near misses" with 0.70 <= minimum r < 0.85 that the rule does not flag); column `human_verdict` is empty |
| `blink_eyelid_spotcheck.html`, `item_*.jpg`, `blink_eyelid_spotcheck_videoNN.jpg` | review page and images (5 frames per item: 100 ms before, start, deepest, end, 100 ms after). The images are not in the git repository (36 MB); the script regenerates them |

Per video the rule gives 5 to 130 events (0.25 to 6.5 per minute), median duration 100-308 ms, 0.15-3.8% of the frames.
183 of the 733 events last less than 50 ms (1-2 frames) and 37 last more than 1 s.

Seen while checking that the sheets are readable (4 items of video 11, not a spot check): the two single-frame events
were not closures - one eyelid point jumped away for one frame while the eye was open; in one real closure the eyelid
points left the eye at the deepest frame (relative opening -2.5). Both suggest that the rule needs a minimum duration and
a check that the eyelid points are plausible; to be decided from the human spot check.
