# Error modes from the model's own output (frame selection by cluster), 2026-10-26

Script: `scripts/error_modes.py --video N`. Per video one folder `videoNN/` with `clusters_summary.csv`, `cluster_features.csv`
(cluster mean of every feature, z-scored), `silhouette_by_k.csv`, `selection.csv` (the frames to label) and, local only,
`frames_clusters.csv` (every frame) and one contact sheet per cluster `cluster_KK.jpg`.

Question (user's proposal): instead of a fixed number of frames per key point, let the data say how many ways the model
goes wrong, and label one or two frames per way. Every frame of the whole video is described by what the model produced on
it — the 8 points in the eye frame (origin midpoint of the corners, x along the corners, unit eye width), pupil width /
height / aspect / axes offset / opening, the 8 confidences, 4 heatmap numbers per point (entropy, second-peak ratio, number
of peaks, mass near the peak; `results/heatmap_signals/`), the one-step fitting deviation per point — 70 features,
standardised, PCA to 95 % of the variance, k-means with k from the silhouette (k = 4 … 14). The 70 truth frames of the
video are used only to judge the clusters afterwards.

## Video 8 (71 748 frames, 33 principal components, k = 4, silhouette 0.164 — the next best k = 7 has 0.124)

| cluster | frames | share | mean suspicion | fit_pupil median | truth frames | > 20 px | > 50 px | area error > 20 % | median RMSE | what sets it apart |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 722 | 1.0 % | 0.99 | 187 px | 0 | – | – | – | – | all fitting deviations and entropies extreme (blinks, fast motion) |
| 1 | 27 496 | 38.3 % | 0.31 | 3.0 px | 3 | 0 | 0 | 0 | 7.0 px | eye wide open, pupil top higher, bright reflections |
| 2 | 21 427 | 29.9 % | 0.80 | 9.9 px | 49 | 35 | 9 | 28 | 27.6 px | pupil_left: low confidence, flat heatmap with several peaks |
| 3 | 22 103 | 30.8 % | 0.43 | 3.3 px | 18 | 3 | 0 | 5 | 13.7 px | confident eyelid_top and pupil_right |

- **The clusters are eye states that come in blocks of time, not "shapes of error".** Cluster 1 is minutes 0–8 of the video
  (eye wide open, two large reflections), cluster 3 minutes 8–13 and 17–18, cluster 2 minutes 13–19 (eye narrower, the
  left side of the pupil blends into the dark eye region). Inside a cluster the frames look alike and the model's
  points sit at the same places relative to the eye; between clusters they differ (contact sheets).
- **The wrong frames sit in one state**: 35 of the 38 truth frames above 20 px and all 9 above 50 px are in cluster 2,
  whose signature is the pupil_left heatmap (confidence −1.3 SD, entropy +1.2 SD, more peaks). Cluster 1 has no wrong
  frame among its 3 truth frames, cluster 3 has 3 of 18.
- **Why the truth frames are 70 % cluster 2 although the cluster is 30 % of the video**: by design of `select_frames.py`
  the test window is the last 10 % of every video and the validation window the 10 % before it (minutes 15.9–19.9 of a
  20-min video), and the training pool is everything earlier. In video 8 the eye state of the last six minutes
  (cluster 2) is under-represented in the pool (12 923 of its 57 158 frames, mostly minutes 8–15) and absent from the
  first eight minutes, where the jump rule + k-means of the main line drew most of its 60 labels (batches 1–3: median
  minute 8.6–9.6). The plateau at 25 px of this video is therefore at least partly a state the labels never covered,
  not only pupil-edge ambiguity. The same windows hold for every video (test 17.9–19.9 min, val 15.9–17.9), so the
  step-8 truth measures "the last four minutes after training on the earlier part" in all of them.
- **Frames selected to label** (`selection.csv`, 20, pool only, quota ∝ size × suspicion, medoids of sub-clusters):
  cluster 0: 1, cluster 1: 5, cluster 2: 9 (minutes 8.4–14.8), cluster 3: 5. No label and no truth frame was used.

## Next

Label the 20 frames (model pre-labels, human corrects), retrain on 420 + 20 (`scale_step.py --tag modes`), compare on the
70 truth frames with the plateau model (median 25.3 / 21.5 px on the 70 frames, 38 > 20 px) and with the existing
40-label and 60-label models of the main line (same label count, old selection rule).
