#!/usr/bin/env python3
"""Example truth frames at increasing error, as evidence for the error line (report 8.1): below the line the model's
points (red) and the human labels (green rings) differ within labeling noise; above the line a point is visibly thrown.
Frames drawn from results/error_tier/truth_errors.csv, one per error band, seed 0. Output
results/error_tier/error_line_examples.jpg (English captions). Nothing existing is modified."""
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from second_peak_repair import BPS, RES, load

BANDS_BELOW = [(0.01, 0.02), (0.03, 0.045), (0.06, 0.08), (0.10, 0.13)]
BANDS_ABOVE = [(0.15, 0.22), (0.25, 0.35), (0.38, 0.55), (0.7, 1.0)]
TW, TH = 320, 240
FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
FONT_B = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
rng = np.random.default_rng(0)
E = pd.read_csv(RES / "error_tier" / "truth_errors.csv")
line = float(pd.read_csv(RES / "error_tier" / "tier.csv").iloc[0].line_eye_widths)


def pick(bands):
    out, used = [], set()
    for lo, hi in bands:
        c = E[(E.error_rel >= lo) & (E.error_rel < hi) & ~E.video.isin(used)]
        if c.empty:
            c = E[(E.error_rel >= lo) & (E.error_rel < hi)]
        r = c.iloc[rng.integers(len(c))]
        used.add(int(r.video))
        out.append(r)
    return out


sel = [(0, r) for r in pick(BANDS_BELOW)] + [(1, r) for r in pick(BANDS_ABOVE)]
S = Image.new("RGB", (4 * TW, 2 * (TH + 32) + 8), (255, 255, 255))
dr = ImageDraw.Draw(S)
dr.text((6, 6), f"Below the line (error under {100 * line:.0f}% of eye width): red = model, green ring = human label; the difference is within labeling noise", fill=(0, 0, 0), font=FONT_B)
dr.text((6, TH + 40), f"Above the line (error over {100 * line:.0f}% of eye width): one or more points thrown far from the eye", fill=(0, 0, 0), font=FONT_B)
cache = {}
for col, (row, r) in enumerate(sel):
    v, f = int(r.video), int(r.frame)
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
    cx, cy = np.nanmedian([hum[b] for b in BPS], axis=0)
    w = max(1.6 * Wm, 240)
    h = w * TH / TW
    box = (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))
    crop = img.crop(box).resize((TW, TH))
    sc = TW / (box[2] - box[0])
    cd = ImageDraw.Draw(crop)
    for b in BPS:
        hx, hy = (hum[b] - box[:2]) * sc
        if np.isfinite(hx) and np.isfinite(hy):
            cd.ellipse([hx - 6, hy - 6, hx + 6, hy + 6], outline=(60, 255, 60), width=3)
        px, py = (xy[b][f] - box[:2]) * sc
        if np.isfinite(px) and np.isfinite(py):
            px, py = np.clip(px, 5, TW - 5), np.clip(py, 25, TH - 5)
            cd.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(255, 60, 60))
    cd.rectangle([0, 0, TW, 20], fill=(0, 0, 0))
    cd.text((4, 2), f"video {v}, frame {f}: error {100 * r.error_rel:.1f}% of eye width ({r.frame_rmse_px:.0f} px)", fill=(255, 255, 255), font=FONT)
    S.paste(crop, ((col % 4) * TW, 30 + row * (TH + 32)))
for v in cache:
    cache[v][2].release()
out = RES / "error_tier" / "error_line_examples.jpg"
S.save(out, quality=90)
print("saved", out)
