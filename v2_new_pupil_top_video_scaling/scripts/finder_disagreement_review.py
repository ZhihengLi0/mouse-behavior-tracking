#!/usr/bin/env python3
"""By-eye review of the frames on which the two finders disagree (report 9.2): frames flagged by the PCA reconstruction
error but not by fitting, and frames flagged by fitting but not by the PCA, thresholds as in finder_comparison.py.

    python finder_disagreement_review.py [--html-only]

Sample: up to 4 frames of each group per video, drawn at random (seed 1), so that video 18 (where the PCA flags 14 % of
the frames on its own) does not dominate. The page shows the frame with the model's 8 points and the frame number only;
which finder flagged it is hidden from the judge and kept in items.csv. Question: is one of the points really wrong,
off the structure it belongs to? Verdicts are kept in the browser and downloaded as a CSV.
Outputs in results/finder_comparison/disagreement_review/: items.csv, items/*.jpg, review.html. Nothing existing is modified."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finder_comparison as F  # noqa: E402
from second_peak_repair import BPS, HERE, predictions  # noqa: E402

OUT = F.OUT / "disagreement_review"
ITEMS = OUT / "items"
PW, PH = 520, 390
PER = 4
FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
rng = np.random.default_rng(1)


def tile(v, f, d, cap):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
    ok, img = cap.read()
    assert ok, (v, f)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    xy = {b: np.array([d[b]["x"].iloc[f], d[b]["y"].iloc[f]], float) for b in BPS}
    pts = np.array([xy[b] for b in BPS])
    pts = pts[np.isfinite(pts).all(1)]
    cx, cy = pts.mean(0)
    wdt = max(1.8 * F.W[v], np.ptp(pts[:, 0]) + 80)
    hgt = max(wdt * PH / PW, np.ptp(pts[:, 1]) + 80)
    wdt = max(wdt, hgt * PW / PH)
    hgt = wdt * PH / PW
    box = (int(cx - wdt / 2), int(cy - hgt / 2), int(cx + wdt / 2), int(cy + hgt / 2))
    crop = img.crop(box).resize((PW, PH))
    sc = PW / wdt
    cd = ImageDraw.Draw(crop)
    for b in BPS:
        px, py = (xy[b] - box[:2]) * sc
        if np.isfinite(px):
            cd.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(255, 60, 60))
    plain = img.crop(box).resize((PW, PH))
    sheet = Image.new("RGB", (2 * PW + 6, PH + 20), (255, 255, 255))
    sheet.paste(plain, (0, 20))
    sheet.paste(crop, (PW + 6, 20))
    sd = ImageDraw.Draw(sheet)
    sd.text((4, 3), f"video {v}  frame {f}  (left: the frame; right: the model's 8 points)", fill=(0, 0, 0), font=FONT)
    return sheet


def main():
    ITEMS.mkdir(parents=True, exist_ok=True)
    if "--html-only" in sys.argv and (OUT / "items.csv").exists():
        P = pd.read_csv(OUT / "items.csv")
    else:
        T, M = F.truth_table()
        S, scores = F.whole_video(M)
        rows = []
        for v, (pca_f, fit_f) in scores.items():
            for group, mask in (("pca_only", pca_f & ~fit_f), ("fit_only", fit_f & ~pca_f)):
                cand = np.flatnonzero(mask)
                for f in np.sort(rng.choice(cand, size=min(PER, len(cand)), replace=False)):
                    rows.append({"item": f"video{v:02d}_f{f:06d}", "video": v, "frame": int(f), "group": group, "candidates": len(cand)})
        P = pd.DataFrame(rows).sort_values(["video", "frame"]).reset_index(drop=True)
        P.to_csv(OUT / "items.csv", index=False)
        for v, g in P.groupby("video"):
            rr, d = predictions(int(v))
            cap = cv2.VideoCapture(str(HERE / rr.unit / f"{rr.unit}.mp4"))
            for r in g.itertuples():
                tile(int(v), r.frame, d, cap).save(ITEMS / f"{r.item}.jpg", quality=90)
            print("video", v, len(g))
    opts = ["yes, a point is wrong", "no, all points are fine", "cannot tell"]
    parts, last = [], None
    for r in P.itertuples():
        if r.video != last:
            parts.append(f"<h2>video {r.video}</h2>")
            last = r.video
        parts.append(f'<div class="it" data-item="{r.item}"><div class="h">frame {r.frame}</div><img src="items/{r.item}.jpg" loading="lazy"><div class="b"><span class="q">Is one of the red points really wrong?</span>' +
                     "".join(f'<label><input type="radio" name="{r.item}" value="{o}"> {o}</label>' for o in opts) + "</div></div>")
    n = len(P)
    (OUT / "review.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Finder disagreement review</title>
<style>body{{font:15px -apple-system,Arial,sans-serif;margin:16px;background:#fafafa;color:#222}}.it{{margin:0 0 22px;padding:8px;background:#fff;border:1px solid #ddd;max-width:{2 * PW + 40}px}}
img{{max-width:100%;display:block}}.h{{font-weight:600;margin-bottom:4px}}.b{{margin-top:8px;display:flex;gap:18px;flex-wrap:wrap;align-items:center}}.q{{font-weight:600;margin-right:6px}}
.done{{border-color:#2a9d8f;background:#f3fbf9}}#bar{{position:sticky;top:0;background:#fafafa;padding:8px 0;border-bottom:1px solid #ddd;margin-bottom:12px;z-index:2}}
button{{font:inherit;padding:4px 12px}}h2{{font-size:17px;margin:26px 0 10px}}</style>
<div id="bar"><b>Frames on which the two finders disagree</b> ({n} frames; which finder flagged each one is hidden). Left: the frame. Right: the model's 8 points in red. Question: is one of the red points really wrong, that is, off the structure it belongs to (pupil edge, lid margin, corner)? A point that merely sits a little off along its structure is not wrong. Answer from the images only.<br>
每帧两张图：原帧、模型的 8 个点（红）。问：有没有哪个红点真的错了，也就是离开了它该在的结构（瞳孔边缘、睑缘、眼角）？沿着结构滑了一点不算错。只看图回答。
&nbsp; <span id="n"></span> <button onclick="save()">Download verdicts (CSV)</button></div>
{"".join(parts)}
<script>
const K='finder_disagreement';let V={{}};try{{V=JSON.parse(localStorage.getItem(K)||'{{}}')}}catch(e){{}}
function upd(){{document.getElementById('n').textContent=Object.keys(V).length+' of {n} judged.';document.querySelectorAll('.it').forEach(d=>d.classList.toggle('done',!!V[d.dataset.item]))}}
document.querySelectorAll('input[type=radio]').forEach(r=>{{if(V[r.name]===r.value)r.checked=true;r.addEventListener('change',()=>{{V[r.name]=r.value;try{{localStorage.setItem(K,JSON.stringify(V))}}catch(e){{}}upd()}})}});upd();
function save(){{const rows=['item,verdict'].concat(Object.keys(V).sort().map(k=>k+',"'+V[k]+'"'));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')],{{type:'text/csv'}}));a.download='finder_disagreement_verdicts.csv';a.click()}}
</script>""", encoding="utf-8")
    print(n, "items ->", OUT / "review.html")
    print(P.group.value_counts().to_dict())


if __name__ == "__main__":
    main()
