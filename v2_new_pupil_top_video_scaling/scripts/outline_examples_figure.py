#!/usr/bin/env python3
"""Before / after pairs of frames in which the repair replaced an eyelid or a corner point, for the report (page 9.3,
"all eight points").

    python outline_examples_figure.py

Takes the crops of results/flag_and_repair/outline_review/items/ (frame, BEFORE, AFTER strips made by outline_review.py
on the whole-replacement run) for four repaired frames whose every replaced point was judged "yes" in the by-eye check
(outline_review_verdicts.csv): one temporal corner, one nasal corner, one upper eyelid, one lower eyelid, from four
different videos. Output: results/flag_and_repair/outline_review/outline_examples.jpg. Nothing existing is modified."""
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_peak_repair import RES  # noqa: E402

D = RES / "flag_and_repair" / "outline_review"
PW, PH, PAD = 400, 300, 10
PICK = [("video02_f059597", "temporal corner"), ("video06_f027171", "nasal corner"),
        ("video07_f009443", "upper eyelid"), ("video03_f001268", "lower eyelid")]
try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
except OSError:
    FONT = ImageFont.load_default()


def main():
    I = pd.read_csv(D / "items.csv").set_index("item")
    V = pd.read_csv(D / "outline_review_verdicts.csv")
    V["item0"] = V.item.str.split("__").str[0]
    panels = []
    for item, what in PICK:
        assert set(V[V.item0 == item].verdict) == {"yes"}, item
        r = I.loc[item]
        im = Image.open(D / "items" / f"{item}.jpg")
        before = im.crop((PW + PAD, 22, 2 * PW + PAD, 22 + PH))
        after = im.crop((2 * (PW + PAD), 22, 3 * PW + 2 * PAD, 22 + PH))
        P = Image.new("RGB", (2 * PW + 4, PH + 20), (255, 255, 255))
        P.paste(before, (0, 20))
        P.paste(after, (PW + 4, 20))
        pts = str(r.points).replace("eye_", "").replace("_", " ").replace("+", " + ")
        ImageDraw.Draw(P).text((2, 2), f"video {int(r.video)}, frame {int(r.frame)}, {what}: before (red = model)  |  after (blue = {pts})",
                               fill=(0, 0, 0), font=FONT)
        panels.append(P)
    cw, ch = 2 * PW + 4, PH + 20
    F = Image.new("RGB", (2 * cw + PAD, 2 * ch + PAD), (255, 255, 255))
    for k, P in enumerate(panels):
        F.paste(P, ((k % 2) * (cw + PAD), (k // 2) * (ch + PAD)))
    out = D / "outline_examples.jpg"
    F.save(out, quality=90)
    print("written", out, F.size)


if __name__ == "__main__":
    main()
