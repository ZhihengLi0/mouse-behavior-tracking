# v3 mouse-eye keypoints -- one-page quick reference

Frame = 928 x 736, 60 fps. Eight points: `pupil_top, pupil_bottom, pupil_left, pupil_right, eyelid_top, eyelid_bottom, eye_nasal_corner, eye_temporal_corner`.
Nasal = LEFT of the image, temporal = RIGHT (all videos: the nose is on the left of the frame, the LED glint on the right of the pupil; v2 human labels: nasal corner x≈290-370, temporal corner x≈730-860). See Fig. 2.

## The pupil = the flat BLACK core, as an ellipse

1. The pupil is the region that is uniformly black (no texture). The grey ring around it is the IRIS ("halo"), even where it looks dark.
2. Imagine/fit the smooth ellipse that follows the visible black arcs. The four pupil points are the LEFT-most, RIGHT-most, TOP-most and BOTTOM-most points of that ellipse in image coordinates (not the ends of the ellipse axes -- Fig. 10).
3. Parts of the ellipse hidden under a lid or under an LED reflection are extrapolated: keep the curvature of the visible arcs (Fig. 5a, Fig. 4a).
4. Edge = where the uniform black stops (first grey step of ~4-5 grey levels), NOT the outer edge of any dark-grey zone (Fig. 3, Fig. 6).

| point | click here | never here |
|---|---|---|
| pupil_left | left end of the black ellipse; in Pluto/Terra the nasal side fades into dark-grey iris -- stop at the end of the flat black | end of the dark-grey zone / dark nasal patch (Fig. 6), iris stripes (Fig. 8) |
| pupil_right | right end of the ellipse; if the LED glint overlaps the edge the point is INSIDE the glint where the ellipse says (Fig. 4) | the far (right) border of the glint (old error, +25..+55 px) |
| pupil_top | top of the ellipse; usually UNDER the upper lid (extrapolated). If the black core top is visible with grey iris above it, click the top of the black (Fig. 5b) | the lid margin, the iris ring, a stripe tip (old error, 45-60 px too high) |
| pupil_bottom | bottom of the ellipse, normally visible; in Terra it is next to the big lower glint -> extrapolate | the lower lid margin (old error in video 6, 57 px too high) |
| eyelid_top | upper lid margin (skin/fur edge) on the vertical line through the pupil centre | a hair, the iris ring |
| eyelid_bottom | lower lid margin on the same vertical line | the iris ring |
| eye_nasal_corner | left tip of the palpebral fissure (where upper and lower lid margins meet) | fur shadow further left |
| eye_temporal_corner | right tip of the fissure | glints / whisker reflections |

## Blink / closed / slit

* Lids closed, or the visible black is a flat slit (width/height > 2.8), or less than ~40% of the normal lid opening: leave ALL FOUR pupil points EMPTY. Still label the lids (at their current position) and the corners (Fig. 7).
* Lid cuts only the top (or only the bottom): label the pupil from the visible arcs, extrapolate the hidden point (Fig. 5a, 7d).
* Lids cut top AND bottom but the black is still tall (W/H < 2.8): fixed-shape ellipse, width from the visible black, height = width / 1.45 (Pluto, Terra) or / 1.15 (mouse A); mark it low confidence.

## Contrast: stretch the image!

Look at the frame with a strong stretch (black = pupil level, white = pupil level + ~40..60 grey). The pupil is the region that stays black; the iris ring turns grey; the lids turn white. In a normal view the ring and the pupil look alike.

## Old human-label errors to avoid (v2 audit)

1. pupil_right on the far side of the LED glint (+25..+55 px).
2. pupil_top on the lid margin / iris ring (45-60 px too high) -- the most common error.
3. pupil_bottom on the lower lid margin when the eye is half closed (video 6).
4. Half-closed eye labelled with a flat 3:1 "pupil" spanning the lid slit.
5. pupil_left at the outer end of the dark nasal zone in dilated Pluto frames (10-30 px too far left).

Figures: `fig01`..`fig12` in this folder; full rules in `LABELING_GUIDE.md`.
