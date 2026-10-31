#!/usr/bin/env python3
"""Before / after pairs of repaired frames from the check by eye, for the report.

    python review_examples_figure.py

Takes the crops of results/flag_and_repair/spotcheck_review/items/ (frame, before, after) for repaired frames and
puts the before and after panels side by side: eight frames judged "on the pupil where it belongs" (one per video,
different points, largest deviation first) and two judged "closer, but still not right". Output:
results/flag_and_repair/spotcheck_review/review_examples.jpg. Nothing existing is modified."""
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_peak_repair import RES  # noqa: E402

D = RES / "flag_and_repair" / "spotcheck_review"
PW, PH, PAD = 400, 300, 10
try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    BOLD = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
except OSError:
    FONT = BOLD = ImageFont.load_default()


def pairs(sel):
    out = []
    for r in sel.itertuples():
        im = Image.open(D / "items" / f"{r.item}.jpg")
        before = im.crop((PW + PAD, 22, 2 * PW + PAD, 22 + PH))
        after = im.crop((2 * (PW + PAD), 22, 3 * PW + 2 * PAD, 22 + PH))
        P = Image.new("RGB", (2 * PW + 4, PH + 20), (255, 255, 255))
        P.paste(before, (0, 20))
        P.paste(after, (PW + 4, 20))
        ImageDraw.Draw(P).text((2, 2), f"video {r.video}, frame {r.frame}: before (red = model)  |  after (blue = {r.points.replace('pupil_', '').replace('+', ' + ')})", fill=(0, 0, 0), font=FONT)
        out.append(P)
    return out


def main():
    I = pd.read_csv(D / "items.csv").merge(pd.read_csv(D / "spotcheck_review_verdicts.csv"), on="item")
    R = I[I.status == "repaired"].sort_values("fit_rel", ascending=False)
    yes = R[R.verdict == "yes"].drop_duplicates("video")
    # one per video, prefer variety of points
    chosen, seen = [], set()
    for r in yes.itertuples():
        if r.points not in seen or len(chosen) >= 4:
            chosen.append(r.Index)
            seen.add(r.points)
        if len(chosen) == 8:
            break
    Y = R.loc[chosen]
    N = R[R.verdict == "better, but still not right"].drop_duplicates("video").head(2)
    py, pn = pairs(Y), pairs(N)
    COLS = 2
    cw, ch = 2 * PW + 4, PH + 20
    rows_y = (len(py) + COLS - 1) // COLS
    rows_n = (len(pn) + COLS - 1) // COLS
    S = Image.new("RGB", (COLS * cw + (COLS - 1) * PAD, 30 + rows_y * (ch + PAD) + 34 + rows_n * (ch + PAD)), (255, 255, 255))
    dr = ImageDraw.Draw(S)
    dr.text((4, 6), "Repaired frames judged by eye: the point put in is on the pupil where it belongs", fill=(0, 0, 0), font=BOLD)
    for j, P in enumerate(py):
        S.paste(P, ((j % COLS) * (cw + PAD), 30 + (j // COLS) * (ch + PAD)))
    y2 = 30 + rows_y * (ch + PAD)
    dr.text((4, y2 + 8), "Judged closer, but still not right: the second peak is not on the pupil edge either", fill=(0, 0, 0), font=BOLD)
    for j, P in enumerate(pn):
        S.paste(P, ((j % COLS) * (cw + PAD), y2 + 34 + (j // COLS) * (ch + PAD)))
    S.save(D / "review_examples.jpg", quality=88)
    print(len(py), len(pn), "->", D / "review_examples.jpg")
    print(Y[["item", "points", "fit_rel"]].to_string(index=False))
    print(N[["item", "points", "fit_rel"]].to_string(index=False))


if __name__ == "__main__":
    main()
