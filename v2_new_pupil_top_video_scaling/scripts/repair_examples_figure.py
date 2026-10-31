#!/usr/bin/env python3
"""Example frames for the report (section 8.3): wrong truth frames repaired by the pipeline and wrong truth frames that
stay flagged, from results/flag_and_repair/videoNN_labelregion/truth_frames.csv. Crop around the eye; red = model,
blue = the accepted second peak (repaired frames), green ring = human label. Output results/flag_and_repair/all_videos/
repair_examples.jpg (English captions). Nothing existing is modified."""
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from second_peak_repair import BPS, P4, RES, Geometry, load, peak2_pixels, pooled_bands

FAR = RES / "flag_and_repair"
TEST, N_EACH, TW, TH = "labelregion", 5, 320, 240
FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
FONT_B = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)

line = float(pd.read_csv(RES / "error_tier" / "tier.csv").iloc[0].line_eye_widths)
rows = []
for d in sorted(FAR.glob(f"video*_{TEST}")):
    v = int(d.name[5:7])
    T = pd.read_csv(d / "truth_frames.csv")
    T["video"] = v
    rows.append(T[T.error_before > line])
W = pd.concat(rows)
rep = W[W.status == "repaired"].sort_values("error_before", ascending=False)
rep = rep.groupby("video").head(2).head(N_EACH)
left = W[W.status == "flagged_not_repaired"].sort_values("error_before", ascending=False).groupby("video").head(2).head(N_EACH)
sel = pd.concat([rep.assign(row=0), left.assign(row=1)])

S = Image.new("RGB", (N_EACH * TW, 2 * (TH + 30) + 10), (255, 255, 255))
dr = ImageDraw.Draw(S)
dr.text((6, 6), "Wrong frames repaired: red = model, blue = accepted second peak, green ring = human label", fill=(0, 0, 0), font=FONT_B)
dr.text((6, TH + 40), "Wrong frames that stay flagged: several points thrown at once, the second peak is not on the pupil either", fill=(0, 0, 0), font=FONT_B)
cache = {}
for r in sel.itertuples():
    v, f = int(r.video), int(r.frame)
    if v not in cache:
        rr, d, P, H, truth, fwd, sig = load(v)
        xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
        p2, off = peak2_pixels(d, P)
        B, scale = pooled_bands(RES / "second_peak_repair")
        G = Geometry(xy, scale)
        cap = cv2.VideoCapture(str(RES.parent / rr.unit / f"{rr.unit}.mp4"))
        cache[v] = (xy, p2, G, cap, truth)
    xy, p2, G, cap, truth = cache[v]
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    assert ok, (v, f)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    hum = {b: np.array([truth.loc[f, f"human_{b}_x"], truth.loc[f, f"human_{b}_y"]], float) for b in BPS}
    cx, cy = np.nanmedian([hum[b] for b in BPS], axis=0)
    w = max(1.6 * G.Wm, 240)
    h = w * TH / TW
    box = (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))
    crop = img.crop(box).resize((TW, TH))
    sc = TW / (box[2] - box[0])
    cd = ImageDraw.Draw(crop)
    pts = [p for p in str(r.points).split("+") if p and p != "nan"]
    for b in BPS:
        hx, hy = (hum[b] - box[:2]) * sc
        if np.isfinite(hx) and np.isfinite(hy):
            cd.ellipse([hx - 6, hy - 6, hx + 6, hy + 6], outline=(60, 255, 60), width=3)
        px, py = (xy[b][f] - box[:2]) * sc
        cd.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(255, 60, 60))
        if r.status == "repaired" and b in pts:
            nx, ny = (p2[b][f] - box[:2]) * sc
            cd.line([px, py, nx, ny], fill=(80, 120, 255), width=2)
            cd.ellipse([nx - 4, ny - 4, nx + 4, ny + 4], fill=(80, 120, 255))
    cd.rectangle([0, 0, TW, 20], fill=(0, 0, 0))
    tag = f"video {v}, frame {f}: error {100 * r.error_before:.0f}% -> {100 * r.error_after:.0f}% of eye width" if r.status == "repaired" else f"video {v}, frame {f}: error {100 * r.error_before:.0f}% of eye width, flagged"
    cd.text((4, 2), tag, fill=(255, 255, 255), font=FONT)
    j = list(sel.index).index(r.Index) if False else None
    col = (rep.index.get_loc(r.Index) if r.row == 0 else left.index.get_loc(r.Index))
    S.paste(crop, (col * TW, 30 + r.row * (TH + 30)))
for v in cache:
    cache[v][3].release()
out = FAR / "all_videos" / "repair_examples.jpg"
S.save(out, quality=90)
print("saved", out, len(rep), "repaired,", len(left), "still flagged")
