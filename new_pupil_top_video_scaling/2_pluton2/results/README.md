# Video 2: 2_pluton2 (19.9 min, second recording of the mouse of video 1)

Scale curve within this video under the new pupil standard. Training = ALL 100 labels of video 0
(`0_first5minvedio`) + ALL 100 labels of video 1 (`1_20251031_Pluto_spont_1`) + N labels of this video, from
scratch; validation = this video's val20; test = this video's 50 frozen test frames. The x = 0 point is the
video-1 step-5 model (shuffle 225) applied unchanged.

Video file: `2_pluton2/2_pluton2.mp4` (a copy of `pluton2.mp4`, local only).

Frame extraction (2026-09-22): test50 / val20 / batch01 by the same model-free protocol as videos 0 and 1
(k-means on appearance fingerprints, medoid per cluster). Pre-labels for the labeler come from the video-2
step-5 model; they only save dragging time and do not influence which frames were chosen.
