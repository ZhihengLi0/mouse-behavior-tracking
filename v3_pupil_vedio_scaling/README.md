# v3: labels needed per video, one recording day at a time (from 2026-09-28)

Same experiment as `../v2_new_pupil_top_video_scaling` (how many human labels a new video needs), redone on one
consistent pupil definition: **the pupil is the flat black core; its four points are the extreme points of the
ellipse through the visible black arcs; optical effects (halo, glints, shadows) are never labelled.** The full
standard is `docs/LABELING_GUIDE.md` (procedure for the AI + instructions for human reviewers, with figures);
`docs/QUICK_REFERENCE.md` is the one-page version. Every set is pre-labelled by Fable 5.1 following the guide,
then reviewed and corrected by a person; corrections feed back into the guide (revision table at its top).

Videos (one per recording day, `spont_1` of each day; names = the lab Google Drive names, byte-checked):
`0_first5minvedio` (mouse A), `1_20251031` ... `8_20251022` (Pluto), `9_20251017` ... `16_20251008` (Terra).
Each video folder has `results/` (tracked) and `training-data/` (local: frames, labels, predictions).
Label steps per video: 5 / 10 / 20 / 40 labels, each scored on the same 50 frozen test frames; the reviewer
checks every batch before it is used.
