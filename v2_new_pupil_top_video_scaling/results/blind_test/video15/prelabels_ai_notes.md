# Blind pre-labels, video15: low-confidence images

Placed by looking at the 24 PNG frames only (enlarged, brightened crops; no model output used).
Labeling standard: pupil = four end points of the pupil ellipse on the outer edge of the dark disc
(extrapolated under the upper lid, ellipse height taken as about 0.9 x width, as in the reference
frames img002265 / img008765); eyelid_top/bottom = lid margins on the vertical through the pupil
centre; corners = nasal (left) and temporal (right) tips of the eye opening.

- Points placed: 184 (22 frames with all 8 points, 2 closed-eye frames with 4 points each).
- Images with no point: none.
- Images with no pupil points: img006, img031 (eye closed).

Low-confidence images:
- img006: eye closed - only eyelids/corners placed on the closure line (dark band), pupil omitted; corners uncertain
- img031: eye closed - only eyelids/corners placed on the closure line (dark band), pupil omitted; corners uncertain
- img013: eye narrowed, diffuse pupil edge, no clear glint - pupil ellipse (about 160 px wide) and lids uncertain
- img018: eye narrowed, blurry, diffuse pupil edge - pupil (about 150 px wide) uncertain, pupil_top near lid
- img022: squint, eye nearly closed, pupil only partly visible between the lids - pupil ellipse guessed from the darkest core (about 150 px), corners uncertain
- img032: squint, pupil partly hidden by both lids - pupil ellipse guessed from the darkest core; corners uncertain
- img035: squint, pupil partly hidden by both lids - pupil ellipse guessed from the darkest core; corners uncertain
- img040: squint, pupil partly hidden by both lids - pupil ellipse guessed from the darkest core; corners uncertain
- img045: squint, pupil partly hidden by both lids - pupil ellipse guessed from the darkest core; corners uncertain
- img007: upper lid low, slightly blurred - pupil_top extrapolated well above the lid; pupil_right uncertain (right edge diffuse)
- img015: upper lid low (y about 440) - pupil_top extrapolated about 40 px above the lid; uncertain
- img026: eye half open - pupil_top extrapolated above the lid; pupil_right next to a glint, uncertain
- img030: eye half open - pupil_top extrapolated above the lid; right pupil edge diffuse
- img034: eye narrowed, bright fur/lash specks along the upper lid - pupil_top and eyelid_top uncertain
- img044: eye half open - pupil_top extrapolated above the lid; diffuse edges
- img050: eye half open with a large pupil - pupil_top extrapolated about 65 px above the lid (y about 452), least certain point of the set

Fully open frames with a large pupil (img002, img003, img008, img017, img037, img041, img042, img043):
pupil_top is hidden under the upper lid/fur in all of them and was extrapolated (as in the reference labels);
left/right/bottom and lids were visible.
