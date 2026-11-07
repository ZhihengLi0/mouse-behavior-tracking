# Blind pre-labels, video 6 supplement (img101-img110): low-confidence images

Placed by looking at the 10 PNG frames only (brightened, with a coordinate grid; no model output used).
Same standard as the first blind set: pupil = four end points of the pupil ellipse on the outer edge of the
dark disc (the darkest core, not the dark grey iris around it), extrapolated under the lids; eyelid_top/bottom
= lid margins on the vertical through the pupil centre; corners = nasal (left) and temporal (right) tips of
the eye opening.

- Points placed: 80 (10 frames with all 8 points).
- Images with no point: none.

Low-confidence images:
- img104, img105, img107: eye narrowed (opening about 150 px); pupil_top / pupil_bottom extrapolated outside
  the lids, pupil_left in the nasal shadow, nasal corner uncertain
- img101, img102, img103, img106, img108, img109, img110: eye open, large pupil (about 270 px); the boundary
  between the pupil and the dark iris is diffuse on the left side, so pupil_left is uncertain by about 20 px;
  pupil_top sits at the upper lid margin
