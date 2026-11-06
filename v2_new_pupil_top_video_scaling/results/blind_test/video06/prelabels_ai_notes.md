# Blind pre-labels, video06: low-confidence images

Points placed: 184 on 24 images (22 images with all 8 landmarks, 2 blink images with 4 lid/corner points each). No image was left without points.

Method: each frame was viewed as an enlarged, brightness-stretched crop with a pixel grid (plus smoothed iso-intensity contours as a viewing aid); the three human-labeled reference frames (img000835, img004460, img006925) were overlaid first to calibrate where the human puts each landmark relative to the visible edges. No model output was used or read.

General caveats (apply to every image):
- eye_nasal_corner: no sharp feature is visible under the fur in any frame; placed by the reference convention (x about 280-285, about 30 px below the upper-lid line at that x).
- pupil_right: in this video the corneal reflection sits on the temporal pupil margin, so pupil_right is placed at the edge seen just above the reflection, as in the references.
- pupil_top: the upper lid covers the top of the pupil in most frames; pupil_top is the extrapolated ellipse top (as the human does in the references), so its error is larger than for the other pupil points.

Images with no pupil points:
- img024: blink (upper lid half down, no corneal reflection, pupil edge not visible) - only eyelids/corners placed on the slit; eyelid_bottom and temporal corner uncertain.
- img030: blink/closed (lid ridge runs diagonally across the eye, no corneal reflection) - only eyelids/corners placed on the closure band; eyelid_bottom and both corners uncertain.

Other low-confidence images:
- img002: squinting eye (lids 432-635), upper ~40% of pupil under the lid - pupil_top extrapolated, pupil_left on a faint nasal notch.
- img008: wide open, large pupil - pupil_top extrapolated above the lid; pupil_right partly under the reflection.
- img011: squint, narrow opening (455-625) - pupil_top extrapolated far above lid; pupil_left faint.
- img012: squint - same as img011; pupil_bottom edge diffuse.
- img013: squint - same as img011.
- img014: large pupil - pupil_top extrapolated above the lid; pupil_right on reflection edge.
- img018: large pupil, low nasal contrast - pupil_left faint, pupil_top extrapolated well above the lid.
- img020: squint - pupil_top extrapolated, pupil_left faint.
- img022: squint - as img020.
- img025: squint - as img020; pupil_bottom diffuse.
- img026: squint - as img020.
- img028: squint (slightly more open) - pupil_top extrapolated.
- img031: very dark frame, fully dilated pupil, lower lid barely visible - pupil_right beyond the reflection, pupil_top extrapolated, eyelid_bottom and both corners uncertain.
- img032: large pupil - pupil_top extrapolated above the lid.
- img033: very dark frame like img031 - pupil_left/right faint, eyelid_bottom and corners uncertain.
- img034: squint - pupil_top extrapolated.
- img039: squint; pupil looks smaller and shifted nasally - pupil_left uncertain, pupil_top extrapolated.
- img042: squint - pupil_top extrapolated, pupil_left faint.
- img043: squint - as img042.
- img044: squint - as img042.
- img046: squint - as img042.
- img048: large pupil - pupil_top extrapolated above the lid.
