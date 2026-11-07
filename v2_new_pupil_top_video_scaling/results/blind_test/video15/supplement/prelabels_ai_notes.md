# Blind pre-labels, video 15 supplement (img101-img110): low-confidence images

Placed by looking at the 10 PNG frames only (brightened, with a coordinate grid; no model output used).
Same standard as the first blind set: pupil = four end points of the pupil ellipse on the outer edge of the
dark disc, extrapolated under the lids; eyelid_top/bottom = lid margins on the vertical through the pupil
centre; corners = nasal (left) and temporal (right) tips of the eye opening.

- Points placed: 68 (7 frames with all 8 points, 3 closed-eye frames with 4 points).
- Images with no point: none.
- Images with no pupil points: img101, img102, img104 (eye closed to a dark slit, no corneal reflection;
  the whole image is overexposed).

Low-confidence images:
- img101, img102, img104: closed; lids and corners placed on the dark slit; the nasal corner is uncertain
  because the slit continues to the left edge of the image
- img108, img109: eye narrowed (about 160 px), only a faint reflection; pupil ellipse guessed from the dark
  core, pupil_top / pupil_bottom extrapolated far outside the lids; least certain frames of the set
- img103, img105, img106, img107, img110: eye open with a large pupil (about 360 px wide); pupil_top at the
  upper lid margin, the temporal corner near the right image border; the upper lid edge is ragged (fur), so
  eyelid_top is uncertain by about 15 px
