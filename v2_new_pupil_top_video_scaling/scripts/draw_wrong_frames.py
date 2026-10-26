#!/usr/bin/env python3
"""Contact sheets of the wrong frames: human labels (green) and the plateau model's prediction (red) on the eye crop,
one group per error level (frame RMSE > 50 px; 20-50 px). Reads results/error_detector/truth_pairs.csv and the
label PNGs; writes results/wrong_frames_gallery/."""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_step import BPS, HERE  # noqa: E402

OUT = HERE / "results" / "wrong_frames_gallery"
OUT.mkdir(parents=True, exist_ok=True)
TILE_W, TILE_H, COLS, ROWS = 360, 270, 5, 4
GREEN, RED, YELLOW = (40, 220, 40), (255, 60, 60), (255, 220, 0)
try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    FONT_B = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
except OSError:
    FONT = FONT_B = ImageFont.load_default()


def png_of(unit, s, frame):
    hits = glob.glob(str(HERE / unit / "training-data" / "labels" / s / f"*{frame:06d}*.png"))
    return hits[0] if hits else None


def tile(row):
    img = Image.open(png_of(row.unit, row.set, int(row.frame))).convert("RGB")
    H = np.array([[row[f"human_{b}_x"], row[f"human_{b}_y"]] for b in BPS], float)
    P = np.array([[row[f"pred_{b}_x"], row[f"pred_{b}_y"]] for b in BPS], float)
    pts = np.vstack([H, P]); pts = pts[np.isfinite(pts).all(1)]
    x0, y0 = pts.min(0) - 40; x1, y1 = pts.max(0) + 40
    w, h = max(x1 - x0, 240), max(y1 - y0, 180)
    if w / h > TILE_W / TILE_H: h = w * TILE_H / TILE_W
    else: w = h * TILE_W / TILE_H
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))
    crop = img.crop(box).resize((TILE_W, TILE_H))
    sc = TILE_W / (box[2] - box[0])
    d = ImageDraw.Draw(crop)
    for k, b in enumerate(BPS):
        hx, hy = (H[k] - box[:2]) * sc; px, py = (P[k] - box[:2]) * sc
        err = row[f"err_{b}"]
        if np.isfinite(err) and err > 20:
            d.line([hx, hy, px, py], fill=YELLOW, width=2)
        if np.isfinite(H[k]).all():
            d.ellipse([hx - 5, hy - 5, hx + 5, hy + 5], outline=GREEN, width=3)
        if np.isfinite(P[k]).all():
            d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=RED)
    errs = {b: row[f"err_{b}"] for b in BPS if np.isfinite(row[f"err_{b}"])}
    worst = max(errs, key=errs.get)
    cap = f"v{int(row.video):02d} f{int(row.frame)}  RMSE {row.frame_rmse_px:.0f} px  worst {worst} {errs[worst]:.0f}"
    d.rectangle([0, 0, TILE_W, 20], fill=(0, 0, 0))
    d.text((4, 2), cap, fill=(255, 255, 255), font=FONT)
    return crop


def sheets(D, name, title):
    D = D.sort_values(["video", "frame"]).reset_index(drop=True)
    per = COLS * ROWS
    n_sheets = int(np.ceil(len(D) / per))
    for si in range(n_sheets):
        part = D.iloc[si * per:(si + 1) * per]
        sheet = Image.new("RGB", (COLS * TILE_W, ROWS * TILE_H + 40), (30, 30, 30))
        d = ImageDraw.Draw(sheet)
        d.text((8, 8), f"{title}  -  {len(D)} frames, sheet {si + 1}/{n_sheets}.  green ring = human label, red dot = model, yellow line = point error > 20 px", fill=(255, 255, 255), font=FONT_B)
        for j, (_, row) in enumerate(part.iterrows()):
            sheet.paste(tile(row), ((j % COLS) * TILE_W, 40 + (j // COLS) * TILE_H))
        f = OUT / f"{name}_sheet{si + 1}.jpg"
        sheet.save(f, quality=88)
        print("written", f)
    D[["video", "unit", "set", "frame", "frame_rmse_px"] + [f"err_{b}" for b in BPS]].round(1).to_csv(OUT / f"{name}_frames.csv", index=False)


def main():
    T = pd.read_csv(HERE / "results" / "error_detector" / "truth_pairs.csv")
    T = T[T["eval"] == True]  # noqa: E712
    assert len(T) == 1330
    sheets(T[T.frame_rmse_px > 50], "rmse_over50", "Frame RMSE > 50 px")
    sheets(T[(T.frame_rmse_px > 20) & (T.frame_rmse_px <= 50)], "rmse_20to50", "Frame RMSE 20-50 px")


if __name__ == "__main__":
    main()
