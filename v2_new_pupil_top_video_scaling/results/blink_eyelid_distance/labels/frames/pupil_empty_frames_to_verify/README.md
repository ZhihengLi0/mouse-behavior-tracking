# The 43 labeled frames whose four pupil points were all left empty

Part B of blink_and_area.py uses these frames as "eye not open" (truth for the 0.527 pupil-confidence fallback of section 10).
Leaving the four pupil points empty only means the labeler could not place them; it does not by itself mean the eye is closed.
So the labeler looked at each of the 43 frames again on 2026-10-24 and judged it open / not open, the same two states as
the blind single-frame judgments. Result: all 43 judged not open (verdicts.csv). The truth of part B therefore stands.

Files: verdicts.csv (one row per frame: video, set, frame, verdict); contact_sheet.jpg (all 43 with their numbers);
the 43 original frames (NN_videoVV_set_frameF.png) are kept locally and not committed (54 MB; they are copies of
<unit>/training-data/labels/<set>/imgF.png).
Source list: results/blink_and_area/blink_area_closure_frames_all_videos.csv, rows with closed = True.
