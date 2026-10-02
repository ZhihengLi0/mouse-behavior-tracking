# Video 10: 2025-10-22 Pluto 1 (mouse B, 19.9 min; index 10) - labeling convention v2

Label scaling under convention v2, continuing the v2 line after video 9 (finished at 80 labels); the last Pluto video of
the sequence. This video had no earlier human labels: test50, val20 and batch01 were pre-labeled with the video-9
80-label model (1014) and corrected by the labeler on 2026-10-01 (test50: 234 of 400 points moved; val20: 96 of 160;
batch01: 73 of 160), then the test set was frozen. Training set of every step = the carried labels of the earlier videos
(videos 0-3: 100 + 100 + 60 + 60; video 4 run 2 batches 1-2: 40; videos 5, 6, 8 batch 1: 20 each; video 9 batches 1-4:
80; video 7 contributed none; 500 in total) + this video's labels; shuffles 1111, 1112, ...

| labels of this video | median | p90 | frames > 50 px | keypoints conf >= 0.6 | pool frames flagged by the jump rule |
|---|---|---|---|---|---|
| 0 (video-9 80-label model 1014, applied unchanged) | 18.13 px | 32.0 | 8% | 60% | – |
| 20 (shuffle 1111) | 11.83 px | 27.7 | 2% | 72% | 9003 of 57158 (15.8%) |
