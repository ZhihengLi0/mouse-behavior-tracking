# v3 mouse-eye keypoint labelling guide (DeepLabCut, 8 points)

| version | date | change | basis |
|---|---|---|---|
| v0.1 | 2026-09-28 | first complete draft | full read of videos 0-15, v2 human labels, earlier analyses |
| v0.2 | 2026-09-29 | **pupil definition reversed** (see the decision box below): the light band around the flat black core IS pupil; the core+4.5 rule of Parts A/B is withdrawn; the labeler's outer-edge placement is the reference | Kaiwen Sheng + Zhiheng Li, after the video-0 re-check (fur fringe, IR-LED lightening) |

> **DECISION 2026-09-29 (supersedes Parts A, B.2-B.7 and D.1-D.3 below until they are rewritten).** The grey /
> lighter band that surrounds the flat black core is **pupil**, not iris or halo: it is pupil seen through fur and
> eyelashes hanging over the eye, and pupil lightened by the two infrared LEDs (their reflections sit at the lower-left
> and right of the pupil, and the areas they illuminate look paler). The pupil boundary is therefore the **outer
> boundary of the darkened region**, including the lightened parts, and the four pupil points are the extremes of the
> ellipse through that outer boundary (pupil_top extrapolated under the lid; pupil_right at the far side of the
> lightened zone next to the right LED reflection). The human labels of the v2 project follow this definition and are
> the reference; the "flat black core" pre-labels made for v3 video 0 on 2026-09-28 are withdrawn. The AI procedure
> (`scripts/label_procedure.py`) must be re-derived from the human labels before v3 labelling starts. Evidence for the
> decision: the video-0 re-check (fur strands 5-10 px wide anchored at the lid ending exactly where the flat black
> starts; the flat-black right edge fixed at x≈650 while the eye is still, i.e. an imposed boundary, not a margin) and
> Kaiwen Sheng's assessment of the IR-LED lightening.

**Principle 1 - label the anatomy, not the optics.** Every point marks a real structure of the eye (pupil margin, lid margin, fissure corner). Optical effects - the grey halo around the pupil, LED glints, reflections, shadows, image noise - are never labelled; where they hide a structure, the point is placed where the structure is (extrapolated from its visible shape and, if needed, from the neighbouring frames).

**Status of the open decisions (D.4):** until the user's review says otherwise, the defaults in force are: half-closed slits -> pupil empty; nasal edge -> end of the flat black (core+4.5); lids and corners as defined in Part 0. Corrections made during review update this guide (new version row above), the figures and `scripts/label_procedure.py`, and are pushed before the next training step.

Written 2026-09-28 from a full read of the v3 videos 0-15 (mouse A, Pluto, Terra; video 16 not yet available), the v2-project human labels and the earlier analyses (`true_pupil/`, `ghost_dynamics/`, `label_audit/`, `label_review/`).
In the repository: figures `guide_figures/fig01`..`fig12`, `QUICK_REFERENCE.md` (one page), `video_notes.csv` (measured facts per video), `../scripts/label_procedure.py` (reference implementation of Part A). Supporting material (`store/` measurements, contact sheets, kymographs) is local only.

Contents

* Part 0 - Definitions (what each of the 8 points is)
* Part A - Procedure for the AI (executable, with numbers)
* Part B - Instructions for human reviewers (plain language, with pictures)
* Part C - Per-mouse / per-video notes
* Part D - Answers to the open questions, confidence of each rule, what the human must decide
* Appendix - Figure list, files, methods

Coordinate conventions: image 928 x 736 px, x to the right, y downwards. **Nasal = left side of the image, temporal = right side**, in all videos: the camera looks at the animal's right eye from the side with the nose on the left of the frame (the bright fur/whisker pad and the nasal fur shadow are on the left, the LED glint is on the right of the pupil). This matches the v2-project human labels: `eye_nasal_corner` x = 293-367, `eye_temporal_corner` x = 732-857 in 100 % of the 740 human frames.

---

## Part 0 - Definitions of the 8 keypoints

Anatomy (Fig. 1): the **pupil** is the uniformly black region (no texture; grey level = "core level", 14-20 on the 8-bit scale in all videos). The **iris** is the grey ring around it (core + 8 .. + 40 grey), 20-55 px wide depending on pupil size; parts of it are dark grey and look like a "halo" - it is still iris. The **LED reflections (glints)** are the saturated white blobs on the cornea. The **lids** are the bright skin/fur regions; the **lid margins** are the sharp bright/dark edges bounding the eye opening (palpebral fissure).

The pupil is labelled as an **ellipse**. The ellipse is the smooth closed curve that follows the boundary of the black core wherever that boundary is visible, and continues with the same curvature where it is hidden (under a lid, under a glint). All four pupil points are read off this ellipse:

| point | definition |
|---|---|
| `pupil_left` | the point of the ellipse with the smallest x (left-most point in image coordinates) |
| `pupil_right` | the point with the largest x |
| `pupil_top` | the point with the smallest y (highest), even if it lies under the upper lid |
| `pupil_bottom` | the point with the largest y (lowest) |
| `eyelid_top` | the upper lid margin (edge between bright skin/fur and the dark eye) on the vertical line through the pupil centre |
| `eyelid_bottom` | the lower lid margin on the same vertical line |
| `eye_nasal_corner` | the left tip of the palpebral fissure, where the upper and lower lid margins meet |
| `eye_temporal_corner` | the right tip of the palpebral fissure |

Because the pupil ellipse is often tilted (mouse A by ~28 deg, Pluto/Terra by 5-25 deg), the four pupil points are the image-axis extremes, **not** the end points of the ellipse axes (Fig. 10). `pupil_left` and `pupil_right` are then not at the same height, and `pupil_top`/`pupil_bottom` not at the same x.

Closed eye / blink: all four pupil points are left empty (NaN). Lids and corners are always labelled.

---

## Part A - Procedure for the AI (Fable)

This is the procedure implemented in `scripts/label_procedure.py` (`label_frame`). All numbers are 8-bit grey levels and full-resolution pixels. Running it on the same frame always gives the same points; the human reviewer sees its `confidence` and `flags`.

### A.1 Read the frame

1. Read frames f-1, f, f+1 as grayscale (all three colour channels are identical) and average them (float32). If the mean absolute difference between f-1 and f+1 inside the eye box (x 150-900, y 200-720) exceeds 6 grey (the eye moved / blinked), use frame f alone.
2. `g` = Gaussian blur of the image, sigma = 3.0 px. `g1` = Gaussian blur, sigma = 1.5 px (for glint detection only).

### A.2 Seed and core level

3. Seed = pupil centre of the previous labelled frame; for the first frame, the darkest point of `g` blurred with sigma 8 inside the eye box. Refine: the minimum of `g` (sigma 5) within +-40 px of the seed.
4. `core` = 5th percentile of `g` in a disc of radius 25 px around the refined seed. (The pupil plateau then lies at core .. core+2; noise after the sigma-3 blur is about +-1 grey.)
5. `skin` = 80th percentile of `g` inside the eye box (used for the lid test).

### A.3 Glint mask

6. Big glints: connected components of `g1 > core + 45` with area < 3000 px, dilated by 6 px (larger bright areas are skin, not glints). Small reflections: local maxima of `g1` (15 x 15 window) with `g1 > core + 12` whose sigma-8 surround is `< core + 20`, dilated by 12 px. `gm` = union.
7. `gs` = glint-inpainted image: normalised convolution of the image with `gm` masked out (sigma 3); undefined where the mask weight < 0.3.

### A.4 Dark-core mask and contour

8. Core mask `m` = (`gs < core + 4.5`) OR `gm`, restricted to the eye box, holes filled; keep the connected component containing the seed. **Threshold T = core + 4.5** is the rule that defines the pupil boundary ("where the uniform black stops"); see D.1 for why.
9. Leak guard (dark videos, half-closed dark eyes): if more than 50 % of the boundary is bordered by dark shadow (level at 25 px outward < core + 8), re-segment with T = core + 3.0 and flag `re-segmented at core+3`. If the mask touches the eye box, flag `leak` (confidence low).
10. Eye region `er` (for lids and corners) = (`g < core + 22`) OR `gm`, holes filled, morphological opening 4 iterations, component containing the seed.
11. Contour `c` = outer contour of `m` (cv2.findContours, CHAIN_APPROX_NONE), as float (x, y).

### A.5 Classify the contour points

For every contour point compute the outward normal direction (from the contour centroid) and sample `g`:

12. **lid point**: `g` at 10 px outward > core + max(20, 0.35 x (skin - core)). The boundary here is the lid or skin, not the pupil edge. (Red dots in the figures.)
13. **glint point**: within 6 px of the glint mask. The boundary here is the glint.
14. **weak point**: `g` at 25 px outward < core + 8 and not lid/glint. The boundary here is bordered by dark shadow / dark nasal iris, not by the grey iris ring; it is unreliable.
15. Fit points = all other contour points ("pupil arcs").

### A.6 Ellipse fit

16. If >= 40 fit points: `e = cv2.fitEllipse(points)`. Then twice: compute the distance of every fit point to the ellipse, drop points farther than max(6 px, 80th percentile), refit. Record `res` = median distance (px) and `cover` = fraction of the 24 x 15-degree angular sectors (around the ellipse centre) that contain fit points.
17. Accept the free fit if all of: 0.75 <= W/H <= 1.9 (W, H = extents of the ellipse along x and y); cover >= 0.45; weak points <= 50 % of the contour; not (upper-half lid points > 40 % AND lower-half lid points > 40 %).
18. Otherwise (top and bottom both hidden, or the shape is implausible) use the **fixed-shape fallback**: re-segment at core + 3.0 to get the flat plateau; W = its horizontal extent (extend to the free fit's right extreme if that lies inside a glint); H = W / aspect with aspect = 1.45 (Pluto, Terra) or 1.15 (mouse A); centre = centre of the plateau's bounding box; tilt 0. Confidence = low; flag `fixed-aspect ellipse`.
19. The four pupil points = the extreme x / y points of the ellipse (sample the ellipse at 721 angles and take argmin/argmax).

### A.7 Closed eye, blink, slit

20. Pupil points are **empty** (confidence `none`) if the lid aperture (height of the eye region `er` on the pupil column) < 40 % of the video's median lid aperture (from a first pass over the video, `store/<unit>_meas.csv`), or the core mask's width/height > 2.8 (a flat slit - too little visible to extrapolate), or the core mask is < 25 px tall with width/height > 2. A small but round pupil (H 30-60 px, Terra) is NOT closed. Lids and corners are still produced.

### A.8 Lids and corners

21. `eyelid_top` / `eyelid_bottom` = top / bottom of the eye region `er` on the column through the ellipse centre (columns cx-3 .. cx+3, any).
22. Corners: restrict `er` to rows within +-150 px of the pupil centre; keep only the contiguous run of columns (containing the pupil) whose eye-region height >= 12 px; `eye_nasal_corner` = (left-most such column, median row of the region there) - the LEFT corner; `eye_temporal_corner` = right-most column - the RIGHT corner. Lid/corner quality is not yet validated against the human labels quantitatively (see D.4); the v2 human lid/corner labels are the better reference and the AI points are a starting position for the reviewer.

### A.9 Confidence and temporal consistency

23. Confidence: `high` if free fit with res <= 4 px and cover >= 0.6; `medium` if free fit otherwise; `low` if fixed fallback or leak; `none` if closed.
24. Label frames in temporal order, propagating the seed. When a frame is `low`/`medium`, look at the fits at f-5 and f+5 (or the nearest high-confidence frames within +-30 frames = 0.5 s): if both are `high` and their ellipse centres agree within 10 px and their W within 15 %, replace the frame's ellipse by the frame-wise linear interpolation of (cx, cy, W, H, tilt) and set confidence `medium (interpolated)`. Pupil size changes at most ~2 px/frame and eye position at most ~10 px/frame outside saccades, so this is safe when the neighbours agree. Never interpolate across a blink (a `none` frame in between).
25. Never label pupil points on a `none` frame, and never take W/H outside 0.75-1.9 as final without the `low` flag.

### A.10 What the AI reports per frame

x, y of the 8 points (pupil NaN when `none`), `confidence`, `mode` (free / fixed / none), `flags`, `res`, `cover`, W, H, core level. The human reviews everything with confidence `low`/`medium` first.

---

## Part B - Instructions for human reviewers

Read `QUICK_REFERENCE.md` first (one page). This part explains each rule with a picture. Green circles in the figures are correct positions, red X are places people clicked in the v2 project that are wrong, the green line is the outline of the black core (core + 4.5 grey), the dashed yellow line is the fitted ellipse.

### B.1 Set the display contrast first

The pupil and the dark iris look alike at normal contrast. Stretch the display so that the pupil level is black and pupil level + 40..60 grey is white (in DLC: reduce the max of the colour range; in an image viewer: levels 15..70). Fig. 3 (top row) shows the same frame stretched this way: the black core is the pupil, the grey ring is the iris, white is lid/skin. At the bottom the ring is thin (12-25 px), at the top and on the nasal side it is wider (30-55 px), and it gets thinner when the pupil dilates (Fig. 3a vs 3b).

### B.2 The pupil ellipse and its four points (Fig. 2, Fig. 10)

1. Find the black core. Follow its edge with your eye: the edge is where the uniform black turns grey (Fig. 3, bottom row: the profile jumps from 0 to +10..+25 within about 10 px on the temporal, top and bottom sides).
2. Imagine the ellipse that passes through the visible black arcs. Where the arcs are interrupted (lid, glint, dark nasal patch), continue the curve smoothly.
3. Click the left-most, right-most, top-most and bottom-most point of that ellipse (Fig. 2). For a tilted ellipse these are not on the axes (Fig. 10a: mouse A, tilt ~28 deg; the red X on Fig. 10 are the axis end points - wrong). `pupil_left` and `pupil_right` are normally at different heights.

### B.3 pupil_right and the LED glint (Fig. 4)

The big LED reflection sits immediately to the right of the pupil at pupil height. When the pupil is large it overlaps the right edge. The correct `pupil_right` is where the ellipse through the arcs above and below the glint reaches its right-most point - usually **inside** the white blob (Fig. 4a), or on the near (left) border of the blob when it only touches (Fig. 4b). In mouse A the iris ring separates pupil and glint (Fig. 4c). Never click the far (right) border of the glint: that was the standard v2-project error (+25..+55 px, Fig. 11a, 11b, 11f).

### B.4 pupil_top under the upper lid (Fig. 5)

In most frames the upper lid covers the top of the pupil. `pupil_top` is then the top of the extrapolated ellipse, under the lid (Fig. 5a), typically 10-40 px below the lid margin. It is NOT the lid margin and NOT the top of the grey iris ring (Fig. 5b, 11b, 11d): the v2-project labels put it 45-60 px too high in every Pluto video and in mouse A. When the black core's top is visible with grey iris above it (small pupil, wide-open eye, Fig. 5b), click the top of the black.

### B.5 pupil_bottom

Usually visible, on the black/grey edge (the ring is thin there). Errors in the v2 project: bottom on the lower lid margin when the eye was half closed (video 6: 57 px too high, Fig. 11d). In Terra a big glint sits on the lower-left edge: extrapolate the ellipse through it (Fig. 9a).

### B.6 pupil_left, dark nasal iris and the "ramp" (Fig. 6, Fig. 9b/c)

In Pluto and Terra the nasal (left) iris is dark. Fig. 6 shows for three videos a small-pupil frame (left) and a large-pupil frame (middle) with a stretch of only 16 grey levels, and the intensity profiles through the centre (right).

* When the pupil is small, a separate dark patch (as dark as the pupil, level +1..+3) sits about 100-130 px nasal of the pupil edge, separated from the pupil by brighter iris (+11..+13). Its position is fixed to the eye (about 315 px nasal of the LED glint in all Pluto videos) - it is **iris**, not pupil.
* When the pupil dilates (W > 200) the pupil edge reaches that patch, and the pupil seems to "fade" into a dark-grey zone 30-90 px wide (Terra: 70-90 px). The pupil ends where the **flat black stops** (first step of ~4-5 grey), which is where the ellipse continuing the top/bottom arcs lands (verified: the ellipse fitted without the nasal sector predicts the nasal edge within 5-30 px of the flat-plateau edge and 40-90 px inside the end of the dark zone, Terra videos 9/10).
* So: click `pupil_left` at the end of the flat black, keeping the ellipse shape; do not follow the dark-grey tail. The v2-project left labels sat 10-30 px outside this in dilated Pluto frames (mid-ramp), and 25-70 px inside it in video 3 test50/val20 (too narrow).

### B.7 Iris stripes and poor image quality (Fig. 8)

Video 2 (10-30) is noisy with vertical dark iris stripes reaching the pupil (Fig. 8a; 8b is the same frame with a strong stretch). Stripes are iris texture; where a stripe touches the pupil edge the edge is the ellipse through the neighbouring arcs, not the stripe tip. The v2-project left labels of this video were placed inside the striped zone (red X). Video 7 has a ridged upper lid margin with hairs (Fig. 8c): `pupil_top` from the ellipse, `eyelid_top` on the skin edge, not on a hair. When a single frame is ambiguous, scrub +-10 frames: the pupil edge moves smoothly and the stripes/hairs do not.

### B.8 Blinks, closed eyes, half-closed eyes (Fig. 7)

* Lids closed or nearly closed (Fig. 7a, 7e, 7f): pupil points **empty**; label `eyelid_top`, `eyelid_bottom` at the current lid positions and the two corners.
* Lid slit with a flat black band (width/height > 2.8; Fig. 7b, 7c): pupil points empty. (These occur in long episodes in videos 1, 2, 6, 8 and 10.) If you are confident about the pupil width, you may add a fixed-shape ellipse (H = W / 1.45 for Pluto/Terra, W / 1.15 for mouse A) but mark the frame low confidence; the AI does not do this by itself.
* Only the top (or only the bottom) hidden (Fig. 7d, Fig. 5a): label normally, extrapolate the hidden point.
* During the lid movement of a blink (2-6 frames) the image is smeared: pupil empty.

### B.9 Lids and corners (Fig. 1, Fig. 12)

`eyelid_top`/`eyelid_bottom`: the edge between bright lid skin/fur and the dark eye, on the vertical line through the pupil centre. If the lid margin is ridged (video 7) take the skin edge, not a hair or the dark shadow under the lid. `eye_nasal_corner` (LEFT tip, nose side) / `eye_temporal_corner` (RIGHT tip): the tips of the lens-shaped eye opening (Fig. 2). On the nasal side the fur shadow continues further left - stop where the upper and lower lid margins meet. On the temporal side ignore the small whisker reflections beyond the corner (Terra). The v2-project human lid/corner labels were checked visually on ~50 frames per video and look right; keep them.

### B.10 Before/after examples (Fig. 11)

Six v2-project frames with the old human labels (red +) and the corrected points (green): a) mouse A: right beyond the glint, top on the lid; b) video 3: top on the lid margin, right beyond the glint; c) video 1 half-closed: top/bottom on the lids; d) video 6: bottom on the lower lid, top on the lid, right beyond the glint; e) video 2: top, right and bottom pulled outward; f) video 5: right beyond the glint, top on the lid. The lid and corner labels (orange circles) are fine in all six.

### B.11 Review checklist (per frame)

1. Contrast stretched? Black = pupil only.
2. Is the eye open enough (visible black taller than ~40 % of the normal opening, not a flat slit)? If not: pupil empty.
3. Does the ellipse follow the black arcs on every visible side, with W/H between 0.75 and 1.9?
4. `pupil_right`: inside/at the near border of the glint, never beyond it.
5. `pupil_top`: under the lid on the ellipse, not on the lid margin; or on the black top if visible.
6. `pupil_left`: end of the flat black, not the end of the dark nasal zone; not on a stripe.
7. `pupil_bottom`: on the black/grey edge, not the lower lid margin; through the Terra glint.
8. Lids on the skin edges through the pupil centre; corners at the fissure tips.
9. If unsure, scrub +-10 frames: the correct point moves smoothly with the pupil.

---

## Part C - Per-mouse and per-video notes

Measured facts (pupil size ranges, lid aperture, grey levels, sharpness, blink counts, glint overlap fractions, old-label error sizes) are in `video_notes.csv`; here are the practical notes. Fig. 12 shows one correctly labelled frame per video.

**Mouse A, video 0 (5 min)**: big bright eye, slightly soft image; pupil W 218-323 px, ellipse tilted ~28 deg (major axis up-right), W/H ~1.13. Upper lid covers the top in almost every frame; no blinks. Main glint 65 px, separated from the core by the iris ring; second glint lower-left touches the bottom-left edge only when W > 280. old (v2-project) label errors: right +45..+55 (beyond the glint) in 60-100 % of frames, top -55 (on the lid); left/bottom good. Lids/corners good.

**Pluto (videos 1-8, one session per day, 20 min each)**: darker eye; the pupil is a horizontal ellipse (W/H 1.36-1.48, tilt 5-10 deg, temporal end slightly lower); W 100-280 px. Main glint 80-130 px directly right of the pupil at pupil height: overlaps the right edge in 8-50 % of frames (video 4: 50 %, video 3: 36 %). Second glint lower-left, overlaps the bottom-left edge when W > 200-230. One or two tiny reflections inside the pupil near the centre (ignore; the AI masks them). Dark nasal iris patch (B.6) in every video. Blinks 0.1-2 per minute, in series.

* video 1 (10-31): sharp; long half-closed slits 15.5-19.5 min. old (v2-project) label errors: top -53, right +25..+35.
* video 2 (10-30): POOR quality (noise, stripes, low contrast), many partial closures (7.6 % of frames), slit episodes at 6-7 and 15-16 min. old (v2-project) label errors: top -55, right +17..+29.
* video 3 (10-29): clearest edges of the set, only 2 blinks; the black top is visible in many frames (Fig. 5b). old (v2-project) label errors: top -43..-54, bottom -20..-35 (test50/val20), left far too narrow in test50/val20 (human W 155 vs core 205).
* video 4 (10-28): glint overlap in 50 % of frames; slit episodes at 0.5-1 and 10.5-11.5 min. Only machine labels in the v2 project (already on the inner edge).
* video 5 (10-27): glint touching rather than overlapping (6 %); blink series around 6 min. old (v2-project) label errors: top -48..-61, right +5..+29.
* video 6 (10-24): half-open, flat slit during the whole test50 period 18-20 min (W/H > 2.5); 2 blinks/min. old (v2-project) label errors: top -61, bottom -57 (on the lower lid), right +26.
* video 7 (10-23): ridged, hairy upper lid margin; blink series; whisker reflections lower right. No human labels.
* video 8 (10-22): DARK video, small pupil (W ~150), long dark half-closed periods (2, 7-8, 10, 16-19 min) in which the whole eye opening is nearly as dark as the pupil - the AI re-segments at core+3, flags leaks, and leaves slits empty; expect many `low` frames for the reviewer.

**Terra (videos 9-15 read; 16 pending)**: darker, sharper images (highest Laplacian variance), lid aperture smaller (150-175 px), pupil W 115-310, W/H 1.43-1.70, tilt 10-27 deg (major axis up-left to down-right = down-nasal gaze). Glints differ from Pluto: the main right glint is small (60-110 x 50-60 px) and separated, but a **large very bright glint sits at the bottom-left directly on the lower pupil edge** in most frames (Fig. 9a): `pupil_bottom` must be extrapolated through it. A row of small whisker reflections lies lower right of the eye (not the temporal corner). The nasal iris is very dark: dark ramp 70-90 px (Fig. 9b/c) - pupil ends at the flat black. Blinks 0.2-2 per minute.

* video 9 (10-17): darkest, small pupil, 7.6 % partial closures. video 10 (10-16): brighter, W up to 274, dilated slit episodes at 5-6 and 11 min. video 11 (10-15): large pupil (W 158-310), lid aperture 154 px so the top is almost always cut; 4 blinks. video 12 (10-14): glint overlap on the right in 37 % of frames; blink series at 6 min. video 13 (10-13): same appearance as v12 but very blinky (97 closures, ~3-5 blinks/min, several closure series) and a very variable pupil (W 130-265, H down to 50): many empty-pupil frames; spot-checked on `sheet3_13.png`. video 14 (10-10): brightest and cleanest Terra video, wide-open eye (aperture 182 px), W 146-252, tilt ~20 deg, only 2 closures in 20 min; spot-checked on `sheet3_14.png`. video 15 (10-09): bright, wide-open eye (aperture 224 px), W 145-291, tilt ~20 deg, 1.7 blinks/min but 12.8 % of frames partially closed (the most of any video; half-closed episodes around 1 min and 8 min) and a bright dot inside the pupil lower right (masked; not an edge); spot-checked on `sheet3_15.png`. video 16 (10-08): not available at the time of writing (row kept in `video_notes.csv` as pending).

**Terra (videos 9-15)** also differ from Pluto in that the eye opening is smaller relative to the pupil, so `pupil_top` is extrapolated in most large-pupil frames, and the tilt is larger (10-27 deg): check that `pupil_left` sits higher than `pupil_right` (Fig. 10b).

---

## Part D - Open questions, confidence, decisions for the human

### D.1 Where exactly is the pupil boundary? (threshold core + 4.5)

Evidence: (i) the intensity profile across the temporal, top and bottom edges is a step from the plateau (core .. core+2) to the iris (core+10 .. +25) within ~10 px (Fig. 3); (ii) an ellipse fitted to the contour at core+4.5 has a median residual of 2-3.5 px in all videos (`store/*_fits.csv`); (iii) on the nasal side, the ellipse fitted to the other arcs predicts the nasal extreme at the core+4.5 crossing (Terra: -5 and -31 px) and far from the end of the dark zone (+40..+89 px) (`store/ellipse_consistency.csv`); (iv) the "halo" ring has iris texture and scales like iris (earlier `ghost_dynamics`, `true_pupil`). The threshold value itself (4.5 vs 3 or 6) moves the edge by only 2-4 px on a sharp edge; on the nasal ramp it moves it by up to 10-15 px. Confidence: **high** for right/top/bottom, **medium** for the nasal side in dilated Pluto/Terra frames.

### D.2 Is the extra dark region left/below the human ellipse pupil or iris? (open question from label_review)

* Left/nasal: **iris** (dark medial iris patch). When the pupil is small it is a separate patch, level +1.5..+3 above the pupil core, ~120 px nasal of the pupil edge, separated by brighter iris (+11..+13), at a fixed place relative to the LED glint (about -315 px) in videos 1, 2, 3, 5; when the pupil dilates the pupil edge reaches it and the two merge (`store/medial_zone_summary.csv`, Fig. 6). A region that exists independently of the pupil and is separated from it by iris cannot be pupil. Confidence **high** that the patch is not pupil; **medium** for the exact edge position inside the merged zone (rule: end of the flat black, ellipse-consistent).
* Bottom: there is **no** extra dark region; the +4.5 to +8 transition is 6-9 px wide everywhere (edge blur). The impression of a lower dark extension came from human bottom points that were 20-60 px too high (videos 3 test50/val20, 6). Confidence **high**.
* Video 2 (10-30) stripes: the vertical iris stripes are iris; the pupil boundary is the ellipse through the arcs between stripes. Confidence **medium-high** (poor image; the AI's residual there is 3-5 px).

### D.3 The old human errors (confirmed on 740 human frames)

Median offsets human - correct (px): right +25..+55 (beyond glint) in videos 0, 1, 3, 5, 6; top -43..-61 (too high) in all videos with human labels; bottom -57 in video 6, -20..-35 in video 3 test50/val20, within 5 px elsewhere; left within +-10..30 px (mid-ramp, slightly outside) except video 3 test50/val20 (+60..+70, far too narrow). Lids and corners: no systematic problem found on the ~300 frames inspected.

### D.4 What the human must decide

1. **Half-closed slits (W/H > 2.8)**: the AI leaves the pupil empty. Alternative: fixed-shape ellipse from the visible width. This affects long episodes (videos 1, 2, 6, 8, 10) and the entire video-6 test50 set. Decide whether these frames enter the training set as empty, as low-confidence fixed-shape, or are excluded.
2. **Nasal edge in dilated Pluto/Terra frames**: rule = end of the flat black (core+4.5). The mid-ramp position (old human habit) is 10-30 px further left. If you prefer mid-ramp, the threshold in A.4 becomes core+8 for the nasal sector only; everything else stays.
3. **Fixed-shape aspect** for the fallback (1.45 Pluto/Terra, 1.15 mouse A): measured from good open-eye fits; under half-closed lids the pupil is usually dilated, so the true H may be larger (aspect closer to 1.3). Low-confidence frames anyway.
4. **Lids/corners from the AI** (A.8) are a starting position only; the v2-project human values are better. For new videos the reviewer should adjust them.
5. **Blink margin**: frames within 2 frames of a closure are left empty by rule 20 only if the mask is a slit; smeared frames just before/after a blink may need to be emptied by hand.

### D.5 Confidence summary of the rules

| rule | confidence |
|---|---|
| pupil = flat black core, iris ring is not pupil | high |
| pupil_right inside/at the near border of the glint (ellipse extreme) | high |
| pupil_top under the lid = ellipse top; not the lid margin | high |
| pupil_bottom on the black edge; extrapolate through the Terra glint | high |
| pupil_left = end of the flat black, ellipse-consistent | medium (Pluto/Terra dilated), high otherwise |
| dark nasal patch is iris | high |
| empty pupil for closed eyes and slits (W/H > 2.8, aperture < 40 %) | high (closed), medium (slit threshold) |
| fixed-shape fallback with aspect 1.45 / 1.15 | low |
| lids and corners from the AI | low-medium (use human labels where they exist) |
| temporal interpolation across +-5 frames when neighbours agree | medium |

---

## Appendix

### Figures

* `fig01_anatomy.png` - Pluto, mouse A, Terra: pupil core, iris ring, glints, lid margins, corners (natural contrast).
* `fig02_eight_points.png` - the 8 points on a normal frame with the typical wrong clicks.
* `fig03_ring_is_iris.png` - stretched frames + intensity profiles: black plateau, iris ring, lid; ring thickness vs pupil size.
* `fig04_glint_right_edge.png` - pupil_right with an overlapping / touching / separated glint.
* `fig05_top_under_lid.png` - pupil_top hidden under the lid vs visible.
* `fig06_nasal_ramp_dynamics.png` - nasal dark zone at small vs large pupil, with profiles (videos 1, 2, 6).
* `fig07_blink_halfclosed.png` - closed, slit, half-closed, top-only cut, dark video.
* `fig08_iris_stripes.png` - video 2 stripes / poor quality, video 7 ridged lid.
* `fig09_terra.png` - Terra: glint on the lower edge, dark nasal iris.
* `fig10_tilted_ellipse.png` - image-axis extremes vs ellipse-axis ends.
* `fig11_before_after.png` - six v2 frames: old human labels (red) vs correct (green).
* `fig12_gallery_all_videos.png` - one labelled frame per video (0-11).
* Supporting material (not numbered): `ts_<unit>.png` (whole-video time series of pupil W/H, position, lid aperture), `kymo_*.png` (60 Hz space-time strips through the pupil), `sheet*_<unit>*.png` (contact sheets of the sampled frames with the procedure's output), `labels_<unit>.png` (human-label sheets), `band_profiles.png`, `shoulder_*.png`, `sanity_1.png`.

### Files

* `../scripts/label_procedure.py` (copied from the analysis folder's `procedure.py`) - reference implementation of Part A (`label_frame(im, seed, prior, ap_median)`); `montage2.py` renders its output on chosen frames.
* `video_notes.csv` - per-video measured facts + notes (one row per video, including the pending ones).
* `store/<unit>_meas.csv` - coarse per-15-frame measurements (half resolution); `store/<unit>_fits.csv` - full-res ellipse fits every 150 frames plus all human-labelled frames; `store/<unit>_kymo.npz` - 60 Hz strips; `store/plateau_check.csv`, `store/ellipse_consistency.csv`, `store/medial_zone_summary.csv`, `store/shoulder_regress*.csv` - the analyses behind Part D; `v2_labels_all.csv` - all v2 labels of the mapped units (human and machine).

### Method summary

Every video was read completely once at 60 Hz (space-time strips through the tracked pupil centre) and once every 15 frames at half resolution (core mask, ellipse, glints, lid aperture), plus full-resolution fits every 150 frames and on every human-labelled frame; blinks were counted from the 60 Hz lid aperture. Frames were then chosen at the 2/15/50/85/98 percentiles of pupil area, the gaze extremes and the closure episodes, and inspected visually (contact sheets). The pupil-boundary questions were answered by intensity profiles on 7-frame averaged images, by regressions of edge positions on pupil width, and by ellipse-consistency tests (fit without the nasal sector, predict the nasal edge).
