# Video 3: 3_pluton2 (19.9 min, second recording of the mouse of video 2)

Scale curve within this video under the new pupil standard. Training = ALL 100 labels of video 1
(`1_first5minvedio`) + ALL 100 labels of video 2 (`2_20251031_Pluto_spont_1`) + N labels of this video, from
scratch; validation = this video's val20; test = this video's 50 frozen test frames. The x = 0 point is the
video-2 step-5 model (shuffle 225) applied unchanged.

Frame extraction (2026-09-22): test50 / val20 / batch01 by the same model-free protocol as videos 1 and 2
(k-means on appearance fingerprints, medoid per cluster). Pre-labels for the labeler come from the video-2
step-5 model; they only save dragging time and do not influence which frames were chosen.
