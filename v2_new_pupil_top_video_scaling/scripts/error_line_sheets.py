#!/usr/bin/env python3
"""Contact sheets as evidence for the error line (report 8.1), in units of eye width: every truth frame above the line
(the 23 wrong frames) and every truth frame in the upper part of the main group (7 % to the line). Green ring = human
label, red dot = model, yellow line joins the two where a point's error exceeds half the line. Output
results/error_tier/sheet_above_line.jpg, sheet_below_line.jpg (English captions). Nothing existing is modified."""
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from second_peak_repair import BPS, RES, load

TW, TH, COLS = 300, 225, 5
FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
FONT_B = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
E = pd.read_csv(RES / "error_tier" / "truth_errors.csv")
line = float(pd.read_csv(RES / "error_tier" / "tier.csv").iloc[0].line_eye_widths)
cache = {}


def frame_tile(v, f, err):
    if v not in cache:
        rr, d, P, H, truth, fwd, sig = load(v)
        xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
        Wm = float(np.nanmedian(np.linalg.norm(xy["eye_temporal_corner"] - xy["eye_nasal_corner"], axis=1)))
        cache[v] = (xy, Wm, cv2.VideoCapture(str(RES.parent / rr.unit / f"{rr.unit}.mp4")), truth)
    xy, Wm, cap, truth = cache[v]
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    assert ok, (v, f)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    hum = {b: np.array([truth.loc[f, f"human_{b}_x"], truth.loc[f, f"human_{b}_y"]], float) for b in BPS}
    pts = np.array([hum[b] for b in BPS] + [xy[b][f] for b in BPS])
    pts = pts[np.isfinite(pts).all(1)]
    x0, y0 = pts.min(0) - 30
    x1, y1 = pts.max(0) + 30
    w, h = max(x1 - x0, 1.3 * Wm), max(y1 - y0, 1.3 * Wm * TH / TW)
    if w / h > TW / TH:
        h = w * TH / TW
    else:
        w = h * TW / TH
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))
    crop = img.crop(box).resize((TW, TH))
    sc = TW / (box[2] - box[0])
    cd = ImageDraw.Draw(crop)
    worst, werr = None, 0
    for b in BPS:
        hx, hy = (hum[b] - box[:2]) * sc
        px, py = (xy[b][f] - box[:2]) * sc
        e = np.hypot(*(hum[b] - xy[b][f])) / Wm
        if np.isfinite(e) and e > werr:
            worst, werr = b, e
        if np.isfinite(e) and e > line / 2:
            cd.line([hx, hy, px, py], fill=(255, 220, 0), width=2)
        if np.isfinite(hx) and np.isfinite(hy):
            cd.ellipse([hx - 5, hy - 5, hx + 5, hy + 5], outline=(60, 255, 60), width=3)
        if np.isfinite(px) and np.isfinite(py):
            cd.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(255, 60, 60))
    cd.rectangle([0, 0, TW, 18], fill=(0, 0, 0))
    cd.text((3, 2), f"v{v:02d} f{f}  error {100 * err:.0f}%  worst {worst.replace('pupil_', 'pupil ').replace('eye_', '').replace('_', ' ')} {100 * werr:.0f}%", fill=(255, 255, 255), font=FONT)
    return crop


def sheet(D, title, name):
    D = D.sort_values("error_rel", ascending=False)
    rows_n = int(np.ceil(len(D) / COLS))
    S = Image.new("RGB", (COLS * TW, 30 + rows_n * TH), (255, 255, 255))
    ImageDraw.Draw(S).text((6, 7), title, fill=(0, 0, 0), font=FONT_B)
    for j, r in enumerate(D.itertuples()):
        S.paste(frame_tile(int(r.video), int(r.frame), r.error_rel), ((j % COLS) * TW, 30 + (j // COLS) * TH))
    S.save(RES / "error_tier" / name, quality=88)
    print("saved", name, len(D), "frames")


sheet(E[E.error_rel > line], f"Above the line: every truth frame with error > {100 * line:.0f}% of eye width ({int((E.error_rel > line).sum())} frames). green ring = human label, red = model, yellow line = point error > {50 * line:.0f}% of eye width", "sheet_above_line.jpg")
sheet(E[(E.error_rel <= line) & (E.error_rel >= 0.07)], f"Below the line, upper part of the main group: every truth frame with error 7% to {100 * line:.0f}% of eye width ({int(((E.error_rel <= line) & (E.error_rel >= 0.07)).sum())} frames). Same colours", "sheet_below_line.jpg")
for v in cache:
    cache[v][2].release()
