# Video 2: 20251031_Pluto_spont_1 (19.9 min, a different mouse)

Scale curve within this video under the new pupil standard. Training = ALL 100 labels of video 1
(`first5minvedio`, batches 01-05) + N labels of this video, from scratch; validation = this video's val20;
test = this video's 50 frozen test frames. The point at N = 0 is the video-1 model applied unchanged.
Reference from the old label standard (not comparable in px): trained with 676 old-mouse labels, this video
reached its plateau at about 80 labels.
