"""Contact sheets of the single-frame 'not open' spot check: the sampled frame crops with the labeler's verdict on each
(for the appendix of the summary page). One PNG per video: results/blink_eyelid_distance/blink_notopen_sheet_videoNN.png.
Green = eye open, red = not open (closed, or the lid covers part of the pupil), grey = cannot tell. Frames are shown in
the order of the blind sheet; the eye opening r of the model is printed small for reference (it was hidden when judging)."""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_step import HERE  # noqa: E402
OUT = HERE / "results" / "blink_eyelid_distance"
Q = pd.read_csv(OUT / "blink_notopen_frames_all_videos.csv")
COL = {"eye open": (60, 160, 60), "not open": (40, 40, 220), "cannot": (140, 140, 140)}
for v, g in Q.groupby("video"):
    tiles = []
    for q in g.itertuples():
        im = cv2.imread(str(OUT / f"{q.item}.jpg"))
        if im is None: continue
        im = cv2.resize(im, (420, int(im.shape[0] * 420 / im.shape[1])))
        verdict = str(q.human_verdict); key = "eye open" if verdict.startswith("eye open") else ("not open" if verdict.startswith("not open") else "cannot")
        c = COL[key]; cv2.rectangle(im, (0, 0), (im.shape[1] - 1, im.shape[0] - 1), c, 6)
        band = np.full((30, im.shape[1], 3), c, np.uint8)
        cv2.putText(band, f"{key.upper()}   frame {q.frame}   r = {q.r:.2f}" + ("" if q.trusted else "  (eyelid points implausible)"), (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([im, band]))
    if not tiles: continue
    h = max(t.shape[0] for t in tiles); tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    cols = 6; rows_ = [np.hstack(tiles[i:i + cols] + [np.full((h, 420, 3), 255, np.uint8)] * (cols - len(tiles[i:i + cols]))) for i in range(0, len(tiles), cols)]
    sheet = np.vstack(rows_); head = np.full((44, sheet.shape[1], 3), 255, np.uint8)
    n_no = sum(str(x).startswith("not open") for x in g.human_verdict)
    cv2.putText(head, f"Video {v}: {len(g)} single frames judged blind; {n_no} not open, {len(g) - n_no} open (green = open, red = not open)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(OUT / f"blink_notopen_sheet_video{v:02d}.png"), np.vstack([head, sheet]))
print("sheets written")
