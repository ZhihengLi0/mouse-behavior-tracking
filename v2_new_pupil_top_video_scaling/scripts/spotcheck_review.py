#!/usr/bin/env python3
"""Review page for the spot check of the adopted repair (acceptance = labelregion): one frame at a time, yes / no.

    python spotcheck_review.py [--html-only]

Takes the frames of results/flag_and_repair/all_videos/spotcheck_sample.csv for the adopted test (up to 10 repaired and
10 still-flagged frames per video, one per episode, drawn at random by flag_and_repair_all_videos.py) and writes for
each one a pair of crops, the frame as it is and the frame with the model's points (red) and, for a repaired frame, the
accepted second peak (blue), plus an HTML page with one question per frame:
  repaired frame      Is the blue point on the right place of the pupil?   yes / better, but still not right / no / cannot tell
  still-flagged frame Is the red point really wrong?                       yes / no / cannot tell
The page shows nothing else (no fitting value, no episode length). Verdicts are kept in the browser and downloaded as a
CSV with the button at the top; spotcheck_review_eval.py reads that CSV. None of these frames took part in setting the
line, the threshold or the label range, so the verdicts are the check of the repair on the whole videos.
Outputs (results/flag_and_repair/spotcheck_review/, local only): items/videoNN_fFFFFFF.jpg, spotcheck_review.html,
items.csv. Nothing existing is modified."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_peak_repair import BPS, RES, Geometry, load, peak2_pixels, pooled_bands  # noqa: E402

TEST = "labelregion"
FAR = RES / "flag_and_repair"
OUT = FAR / "spotcheck_review"
ITEMS = OUT / "items"
PW, PH = 400, 300
try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
except OSError:
    FONT = ImageFont.load_default()


def render(v, rows):
    rr, d, Pk, H, truth, fwd, sig = load(v)
    xy = {b: np.c_[d[b]["x"].to_numpy(float), d[b]["y"].to_numpy(float)] for b in BPS}
    p2, off = peak2_pixels(d, Pk)
    B, scale = pooled_bands(RES / "second_peak_repair")
    G = Geometry(xy, scale)
    Wd = np.linalg.norm(xy["eye_temporal_corner"] - xy["eye_nasal_corner"], axis=1)
    cap = cv2.VideoCapture(str(RES.parent / rr.unit / f"{rr.unit}.mp4"))
    for r in rows:
        f = int(r.frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        assert ok, (v, f)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cx, cy = G.C[f]
        w = max(1.7 * Wd[f], 240)
        h = w * PH / PW
        box = (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))
        raw = img.crop(box).resize((PW, PH), Image.LANCZOS)
        sc = PW / (box[2] - box[0])
        pts = [p for p in str(r.points).split("+") if p and p != "nan"]

        def panel(after):
            im = raw.copy()
            dr = ImageDraw.Draw(im)
            for b in BPS:
                rep = after and r.status == "repaired" and b in pts
                src = p2[b][f] if rep else xy[b][f]
                px, py = (src - box[:2]) * sc
                col = (80, 120, 255) if rep else (255, 60, 60)
                if 0 <= px < PW and 0 <= py < PH:
                    dr.ellipse([px - 5, py - 5, px + 5, py + 5], outline=col, width=2)
                else:
                    cx0, cy0 = PW / 2, PH / 2
                    dx, dy = px - cx0, py - cy0
                    t = min((PW / 2 - 12) / abs(dx) if dx else 9e9, (PH / 2 - 12) / abs(dy) if dy else 9e9)
                    ex, ey = cx0 + dx * t, cy0 + dy * t
                    dr.line([ex - dx * 0.08 if abs(dx) > abs(dy) else ex, ey - dy * 0.08 if abs(dy) >= abs(dx) else ey, ex, ey], fill=col, width=3)
                    dr.ellipse([ex - 6, ey - 6, ex + 6, ey + 6], fill=col)
                    dist = float(np.hypot(*(src - G.C[f])) / Wd[f])
                    lx = min(max(ex - 60, 2), PW - 150)
                    ly = ey + 10 if ey < PH / 2 else ey - 24
                    dr.text((lx, ly), f"{b.replace('pupil_', '')}: outside, {100 * dist:.0f}% of eye width away", fill=col, font=FONT)
            return im

        panels = [("frame as it is", raw)]
        if r.status == "repaired":
            panels += [("BEFORE the repair: red = model's 8 points", panel(False)), ("AFTER the repair: blue = the point(s) the repair put in", panel(True))]
        else:
            panels += [("red = model's 8 points (not repaired)", panel(False))]
        S = Image.new("RGB", (len(panels) * PW + 10 * (len(panels) - 1), PH + 22), (255, 255, 255))
        d2 = ImageDraw.Draw(S)
        for k, (title, im) in enumerate(panels):
            S.paste(im, (k * (PW + 10), 22))
            d2.text((k * (PW + 10) + 4, 3), title, fill=(0, 0, 0), font=FONT)
        S.save(ITEMS / f"video{v:02d}_f{f:06d}.jpg", quality=90)
    cap.release()


def main():
    ITEMS.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(FAR / "all_videos" / "spotcheck_sample.csv", dtype={"points": str})
    P = P[P.accept == TEST].sort_values(["video", "status", "frame"], ascending=[True, False, True]).reset_index(drop=True)
    P["item"] = [f"video{v:02d}_f{f:06d}" for v, f in zip(P.video, P.frame)]
    P[["item", "video", "frame", "status", "points", "episode_len", "fit_rel", "worst_point"]].to_csv(OUT / "items.csv", index=False)
    for v, g in P.groupby("video"):
        if all((ITEMS / f"{it}.jpg").exists() for it in g.item) and "--html-only" in sys.argv:
            continue
        render(int(v), list(g.itertuples()))
        print("video", v, len(g))
    Q = {"repaired": ("Is the blue point on the right place of the pupil?", ["yes", "better, but still not right", "no", "cannot tell"]),
         "flagged_not_repaired": ("Is the red point really wrong?", ["yes", "no", "cannot tell"])}
    parts, last = [], None
    for r in P.itertuples():
        if r.video != last:
            parts.append(f'<h2>video {r.video}</h2>')
            last = r.video
        q, opts = Q[r.status]
        parts.append(f'<div class="it" data-item="{r.item}"><div class="h">{"REPAIRED" if r.status == "repaired" else "STILL FLAGGED"} &nbsp; frame {r.frame}</div>'
                     f'<img src="items/{r.item}.jpg" loading="lazy"><div class="b"><span class="q">{q}</span>' +
                     "".join(f'<label><input type="radio" name="{r.item}" value="{o}"> {o}</label>' for o in opts) + "</div></div>")
    n = len(P)
    (OUT / "spotcheck_review.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Repair spot check</title>
<style>body{{font:15px -apple-system,Arial,sans-serif;margin:16px;background:#fafafa;color:#222}}.it{{margin:0 0 22px;padding:8px;background:#fff;border:1px solid #ddd;max-width:{3 * PW + 40}px}}
img{{max-width:100%;display:block}}.h{{font-weight:600;margin-bottom:4px}}.b{{margin-top:8px;display:flex;gap:18px;flex-wrap:wrap;align-items:center}}.q{{font-weight:600;margin-right:6px}}
.done{{border-color:#2a9d8f;background:#f3fbf9}}#bar{{position:sticky;top:0;background:#fafafa;padding:8px 0;border-bottom:1px solid #ddd;margin-bottom:12px;z-index:2}}
button{{font:inherit;padding:4px 12px}}h2{{font-size:17px;margin:26px 0 10px}}</style>
<div id="bar"><b>Spot check of the repair</b> ({n} frames, acceptance = human-label range). Repaired frame, three panels: the frame, BEFORE (red = the model's 8 points; a point thrown outside the crop is shown as an arrow at the border), AFTER (blue = the point the repair put in). Question: is the blue point where the pupil point belongs? "yes" only if it sits on the pupil edge where that point belongs; closer than before but not on the pupil = "better, but still not right".
Still-flagged frame, two panels: the frame and the model's points. Question: is a red point really wrong (not on the pupil)? Answer from the images only.<br>
修复帧三张图：原帧、修复前（红 = 模型的 8 个点，被甩到框外的点在边上用箭头标出）、修复后（蓝 = 修复换上去的点），问蓝点是不是在瞳孔该在的位置：落在瞳孔边缘该点的位置才选 yes，比原来近了但还不在瞳孔上选 better, but still not right。仍标出帧两张图：原帧、模型的点，问红点是不是真的错了。
&nbsp; <span id="n"></span> <button onclick="save()">Download verdicts (CSV)</button></div>
{"".join(parts)}
<script>
const K='repair_spotcheck_{TEST}';let V={{}};try{{V=JSON.parse(localStorage.getItem(K)||'{{}}')}}catch(e){{}}
function upd(){{document.getElementById('n').textContent=Object.keys(V).length+' of {n} judged.';document.querySelectorAll('.it').forEach(d=>d.classList.toggle('done',!!V[d.dataset.item]))}}
document.querySelectorAll('input[type=radio]').forEach(r=>{{if(V[r.name]===r.value)r.checked=true;r.addEventListener('change',()=>{{V[r.name]=r.value;try{{localStorage.setItem(K,JSON.stringify(V))}}catch(e){{}}upd()}})}});upd();
function save(){{const rows=['item,verdict'].concat(Object.keys(V).sort().map(k=>k+',"'+V[k]+'"'));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')],{{type:'text/csv'}}));a.download='spotcheck_review_verdicts.csv';a.click()}}
</script>""", encoding="utf-8")
    print(n, "items ->", OUT / "spotcheck_review.html")


if __name__ == "__main__":
    main()
