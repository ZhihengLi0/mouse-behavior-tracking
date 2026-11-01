#!/usr/bin/env python3
"""Review page for the repair of all eight points (--points all, finder either, acceptance human-label range): one frame
at a time, yes / no, the same form as spotcheck_review.py.

    python outline_review.py [--html-only]

Frames: from results/flag_and_repair/videoNN_either_labelregion_all/frames_all.csv.gz, per video up to 5 repaired frames
whose replaced points include an eyelid or a corner (the new part of the repair), one per episode (frames within 5 of
each other form an episode), random episodes, seed 0. Still-flagged frames are not shown (N_REP, WITH_FLAGGED): they
were judged in the earlier spot check and the new repair only moves some of them to "left alone" or "repaired". For each frame a strip of
crops: the frame as it is; BEFORE (red = the model's 8 points); for a repaired frame AFTER (blue = every point the
repair put in, pupil or outline). Questions:
  repaired frame      Are the blue points on the right place (pupil edge, eyelid, eye corner)?
                      yes / better, but still not right / no / already right before, should not have moved / cannot tell
  still-flagged frame Is a red point really wrong?   yes / no / cannot tell
Outputs (results/flag_and_repair/outline_review/, local only except items.csv): items/videoNN_fFFFFFF.jpg,
outline_review.html, items.csv. Verdicts are kept in the browser and downloaded as outline_review_verdicts.csv.
Nothing existing is modified."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_peak_repair import BPS, RES, Geometry, load, peak2_pixels, pooled_bands  # noqa: E402
from second_peak_repair import PL  # noqa: E402

FAR = RES / "flag_and_repair"
OUT = FAR / "outline_review"
ITEMS = OUT / "items"
PW, PH = 400, 300
GAP, N_PER, SEED = 5, 10, 0
N_REP, WITH_FLAGGED = 5, False  # user, 2026-10-31 night: 5 repaired frames per video, no still-flagged frames (those were judged in the spot check already)
OUTLINE = ("eyelid_top", "eyelid_bottom", "eye_nasal_corner", "eye_temporal_corner")
SHORT = {"pupil_top": "pupil top", "pupil_bottom": "pupil bottom", "pupil_left": "pupil left", "pupil_right": "pupil right",
         "eyelid_top": "eyelid top", "eyelid_bottom": "eyelid bottom", "eye_nasal_corner": "nasal corner", "eye_temporal_corner": "temporal corner"}
try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
except OSError:
    FONT = ImageFont.load_default()


def sample():
    rng = np.random.default_rng(SEED)
    rows = []
    for v in sorted(PL[PL.status == "final"].video.astype(int)):
        F = pd.read_csv(FAR / f"video{v:02d}_either_labelregion_all" / "frames_all.csv.gz", dtype={"points": str})
        T = F[F.status.isin(["repaired", "flagged_not_repaired"])].copy()
        T["episode"] = np.cumsum(np.r_[True, np.diff(T.frame.to_numpy()) > GAP]) if len(T) else np.array([], int)
        T["outline"] = T.points.fillna("").apply(lambda s: any(b in s.split("+") for b in OUTLINE))
        for st, sel, k in (("repaired", (T.status == "repaired") & T.outline, N_REP), ("flagged_not_repaired", T.status == "flagged_not_repaired", N_PER if WITH_FLAGGED else 0)):
            S = T[sel]
            if S.empty or k == 0:
                continue
            eps = S.episode.unique()
            rng.shuffle(eps)
            for e in eps[:k]:
                E = S[S.episode == e]
                r = E.iloc[rng.integers(len(E))]
                rows.append({"video": v, "frame": int(r.frame), "status": st, "points": r.points if isinstance(r.points, str) else "", "episode_len": int((T.episode == e).sum()),
                             "fit_rel": round(float(r.fit_rel), 3), "worst_point": r.worst_point})
    P = pd.DataFrame(rows).sort_values(["video", "status", "frame"], ascending=[True, False, True]).reset_index(drop=True)
    P["item"] = [f"video{v:02d}_f{f:06d}" for v, f in zip(P.video, P.frame)]
    return P


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
                if rep:  # a thin line from where the point was to where the repair put it
                    ox, oy = (xy[b][f] - box[:2]) * sc
                    dr.line([ox, oy, px, py], fill=(80, 120, 255), width=1)
                    dr.ellipse([ox - 2, oy - 2, ox + 2, oy + 2], fill=(255, 60, 60))
                if 0 <= px < PW and 0 <= py < PH:
                    if b in OUTLINE:
                        dr.rectangle([px - 5, py - 5, px + 5, py + 5], outline=col, width=2)
                    else:
                        dr.ellipse([px - 5, py - 5, px + 5, py + 5], outline=col, width=2)
                    if rep or (not after and r.status == "repaired" and b in pts):
                        dr.text((min(px + 7, PW - 90), max(py - 16, 2)), SHORT[b], fill=col, font=FONT)
                else:
                    cx0, cy0 = PW / 2, PH / 2
                    dx, dy = px - cx0, py - cy0
                    t = min((PW / 2 - 12) / abs(dx) if dx else 9e9, (PH / 2 - 12) / abs(dy) if dy else 9e9)
                    ex, ey = cx0 + dx * t, cy0 + dy * t
                    dr.ellipse([ex - 6, ey - 6, ex + 6, ey + 6], fill=col)
                    dist = float(np.hypot(*(src - G.C[f])) / Wd[f])
                    lx = min(max(ex - 60, 2), PW - 150)
                    ly = ey + 10 if ey < PH / 2 else ey - 24
                    dr.text((lx, ly), f"{b}: outside, {100 * dist:.0f}% of eye width away", fill=col, font=FONT)
            return im

        panels = [("frame as it is", raw)]
        if r.status == "repaired":
            panels += [("BEFORE: red = model's 8 points (circle pupil, square outline); named = will be moved", panel(False)), (f"AFTER: blue = put in by the repair ({', '.join(pts)})", panel(True))]
        else:
            panels += [("red = model's 8 points, not repaired (circle pupil, square outline)", panel(False))]
        S = Image.new("RGB", (len(panels) * PW + 10 * (len(panels) - 1), PH + 22), (255, 255, 255))
        d2 = ImageDraw.Draw(S)
        for k, (title, im) in enumerate(panels):
            S.paste(im, (k * (PW + 10), 22))
            d2.text((k * (PW + 10) + 4, 3), title, fill=(0, 0, 0), font=FONT)
        S.save(ITEMS / f"video{v:02d}_f{f:06d}.jpg", quality=90)
    cap.release()


def main():
    ITEMS.mkdir(parents=True, exist_ok=True)
    P = sample()
    P[["item", "video", "frame", "status", "points", "episode_len", "fit_rel", "worst_point"]].to_csv(OUT / "items.csv", index=False)
    print(P.groupby(["status"]).size().to_string())
    for v, g in P.groupby("video"):
        if all((ITEMS / f"{it}.jpg").exists() for it in g.item) and "--html-only" in sys.argv:
            continue
        render(int(v), list(g.itertuples()))
        print("video", v, len(g))
    OPTS_REP = ["yes", "better, but still not right", "no", "already right before, should not have moved", "cannot tell"]
    Q = {"flagged_not_repaired": ("Is a red point really wrong?", ["yes", "no", "cannot tell"])}
    parts, last = [], None
    for r in P.itertuples():
        if r.video != last:
            parts.append(f'<h2>video {r.video}</h2>')
            last = r.video
        if r.status == "repaired":
            pts = [p for p in str(r.points).split("+") if p and p != "nan"]
            qs = "".join(f'<div class="b" data-q="{r.item}__{b}"><span class="q">{SHORT[b]} (blue): is it on the right place?</span>' +
                         "".join(f'<label><input type="radio" name="{r.item}__{b}" value="{o}"> {o}</label>' for o in OPTS_REP) + "</div>" for b in pts)
            parts.append(f'<div class="it" data-item="{r.item}" data-n="{len(pts)}"><div class="h">REPAIRED &nbsp; frame {r.frame} &nbsp; ({len(pts)} point{"s" if len(pts) > 1 else ""} put in)</div>'
                         f'<img src="items/{r.item}.jpg" loading="lazy">{qs}</div>')
        else:
            q, opts = Q[r.status]
            parts.append(f'<div class="it" data-item="{r.item}" data-n="1"><div class="h">STILL FLAGGED &nbsp; frame {r.frame}</div>'
                         f'<img src="items/{r.item}.jpg" loading="lazy"><div class="b"><span class="q">{q}</span>' +
                         "".join(f'<label><input type="radio" name="{r.item}" value="{o}"> {o}</label>' for o in opts) + "</div></div>")
    n = len(P)
    (OUT / "outline_review.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Repair of all eight points: by-eye check</title>
<style>body{{font:15px -apple-system,Arial,sans-serif;margin:16px;background:#fafafa;color:#222}}.it{{margin:0 0 22px;padding:8px;background:#fff;border:1px solid #ddd;max-width:{3 * PW + 40}px}}
img{{max-width:100%;display:block}}.h{{font-weight:600;margin-bottom:4px}}.b{{margin-top:8px;display:flex;gap:18px;flex-wrap:wrap;align-items:center}}.q{{font-weight:600;margin-right:6px}}
.done{{border-color:#2a9d8f;background:#f3fbf9}}#bar{{position:sticky;top:0;background:#fafafa;padding:8px 0;border-bottom:1px solid #ddd;margin-bottom:12px;z-index:2}}
button{{font:inherit;padding:4px 12px}}h2{{font-size:17px;margin:26px 0 10px}}</style>
<div id="bar"><b>Repair of all eight points, by-eye check</b> ({n} frames; finder = fitting or PCA, acceptance = human-label range on pupil and outline). Repaired frame, three panels: the frame, BEFORE (red = the model's 8 points; circle = pupil point, square = eyelid or corner), AFTER (blue = every point the repair put in). One question per blue point (its name is written next to it): is it where it belongs (pupil edge, eyelid, eye corner)? "yes" if it sits where it belongs; closer than before but not there = "better, but still not right"; the red points were already right and the repair moved them for nothing = "already right before, should not have moved".
Still-flagged frame, two panels: the frame and the model's points. Question: is a red point really wrong? Answer from the images only.<br>
修复帧三张图：原帧、修复前（红 = 模型的 8 个点，圆 = 瞳孔点，方 = 眼睑或眼角点）、修复后（蓝 = 修复换上去的点，可能是瞳孔点也可能是眼睑、眼角点），每个蓝点一个问题（图上蓝点旁写了名字），问它是不是在该在的位置（瞳孔边缘、眼睑、眼角）：落在该在的位置选 yes，近了但还不到位选 better, but still not right；红点本来就对、修复是白动的，选 already right before, should not have moved。仍标出帧两张图：原帧、模型的点，问红点是不是真的错了。
&nbsp; <span id="n"></span> <button onclick="save()">Download verdicts (CSV)</button></div>
{"".join(parts)}
<script>
const K='outline_review_all_v2';let V={{}};try{{V=JSON.parse(localStorage.getItem(K)||'{{}}')}}catch(e){{}}
function upd(){{let done=0;document.querySelectorAll('.it').forEach(d=>{{const names=[...d.querySelectorAll('input[type=radio]')].map(r=>r.name);const u=[...new Set(names)];const ok=u.every(k=>V[k]);d.classList.toggle('done',ok);if(ok)done++}});document.getElementById('n').textContent=done+' of {n} frames judged.'}}
document.querySelectorAll('input[type=radio]').forEach(r=>{{if(V[r.name]===r.value)r.checked=true;r.addEventListener('change',()=>{{V[r.name]=r.value;try{{localStorage.setItem(K,JSON.stringify(V))}}catch(e){{}}upd()}})}});upd();
function save(){{const rows=['item,verdict'].concat(Object.keys(V).sort().map(k=>k+',"'+V[k]+'"'));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')],{{type:'text/csv'}}));a.download='outline_review_verdicts.csv';a.click()}}
</script>""")
    print("page", OUT / "outline_review.html", n, "frames")


if __name__ == "__main__":
    main()
